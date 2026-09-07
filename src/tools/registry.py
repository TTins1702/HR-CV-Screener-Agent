"""The five deterministic tools, wrapped for LangGraph and for slide 2.

The tool functions themselves take and return pydantic models, which is what the
graph's own python code wants. A model calling a tool wants flat JSON. The thin
adapters below are the only place that translation is allowed to happen -- no
tool logic lives in this module.

`TOOL_RATIONALE` is the "why is this a tool and not the LLM's job" column of the
spec's tool table, kept next to the code so the slide cannot drift from reality.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from langchain_core.tools import StructuredTool

from src.contracts.rubric import JDRubric
from src.contracts.screening import CriterionScore
from src.tools.evidence import DEFAULT_MAX_RESULTS, DEFAULT_MIN_SCORE, search_evidence
from src.tools.experience import calculate_experience
from src.tools.injection import scan_injection
from src.tools.scorecard import aggregate_scorecard
from src.tools.skills import normalize_skill

TOOL_RATIONALE: dict[str, str] = {
    "calculate_experience": (
        "Date arithmetic over gaps and overlaps. 198 of the 290 dev resumes with "
        "parseable dates have overlapping roles, so the naive sum of durations is "
        "wrong on two thirds of real CVs."
    ),
    "normalize_skill": (
        "Turns string comparison into concept comparison: React, ReactJS and "
        "React.js are one skill, and a criterion must not miss a CV that spells it "
        "differently."
    ),
    "search_evidence": (
        "Traces every score back to the exact characters that justify it. The "
        "dataset drops spaces between sentences, so naive substring matching fails; "
        "this is the foundation of the evidence-linked scoring claim."
    ),
    "scan_injection": (
        "Detects instructions hidden in an untrusted CV before that CV reaches the "
        "model's context, and feeds the guard edge. Measured at zero false positives "
        "across the 300 dev resumes."
    ),
    "aggregate_scorecard": (
        "The weighted sum and the two threshold cuts must be reproducible to the "
        "decimal and auditable, because they decide the reject_fast and deep_review "
        "branches."
    ),
}


def _search_evidence_tool(
    text: str,
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[dict[str, Any]]:
    """Find where in the CV text a quote appears, with character offsets."""
    hits = search_evidence(text, query, max_results=max_results, min_score=min_score)
    return [hit.model_dump(mode="json") for hit in hits]


def _calculate_experience_tool(text: str, today: str | None = None) -> dict[str, Any]:
    """Total years of experience in a CV, with overlapping roles counted once."""
    reference = date.fromisoformat(today) if today else None
    return calculate_experience(text, today=reference).model_dump(mode="json")


def _normalize_skill_tool(raw: str) -> dict[str, Any]:
    """Resolve a skill string to its canonical name."""
    return normalize_skill(raw).model_dump(mode="json")


def _scan_injection_tool(text: str) -> dict[str, Any]:
    """Scan a CV for hidden instructions aimed at the screening model."""
    report = scan_injection(text)
    payload = report.model_dump(mode="json")
    payload["flags"] = report.flags  # a property, so model_dump omits it
    return payload


def _aggregate_scorecard_tool(
    criterion_scores: list[dict[str, Any]], rubric: dict[str, Any]
) -> dict[str, Any]:
    """Combine per-criterion scores into an overall score and a fit label."""
    scores = [CriterionScore.model_validate(entry) for entry in criterion_scores]
    return aggregate_scorecard(scores, JDRubric.model_validate(rubric)).model_dump(mode="json")


def _build(function: Any, name: str) -> StructuredTool:
    """Wrap an adapter, using its docstring plus the rationale as the description."""
    return StructuredTool.from_function(
        func=function,
        name=name,
        description=f"{(function.__doc__ or '').strip()} {TOOL_RATIONALE[name]}".strip(),
    )


SCREENER_TOOLS: list[StructuredTool] = [
    _build(_search_evidence_tool, "search_evidence"),
    _build(_calculate_experience_tool, "calculate_experience"),
    _build(_normalize_skill_tool, "normalize_skill"),
    _build(_scan_injection_tool, "scan_injection"),
    _build(_aggregate_scorecard_tool, "aggregate_scorecard"),
]

_BY_NAME = {tool.name: tool for tool in SCREENER_TOOLS}


def get_tool(name: str) -> StructuredTool:
    """The registered tool called `name`."""
    if name not in _BY_NAME:
        raise KeyError(f"no tool named {name!r}; known tools: {sorted(_BY_NAME)}")
    return _BY_NAME[name]
