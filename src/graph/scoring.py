"""score_criteria: where three of the five tools meet the model.

`expand_skill` finds each skill criterion's surface forms in the CV before the model
is asked anything, longest form first so a React Native CV is not scored as plain
React. `search_evidence` resolves every quote the model writes back to character
offsets -- measured on 122 real LLM-written quotes, 95 resolved exactly, 24 only
fuzzily, and 3 not at all; a naive `str.find` would have failed on 27 of them.
`calculate_experience` does not just correct the profile, it sets the score for
every experience_years criterion, which is how the tool's contribution reaches the
final number instead of stopping at a field the model can ignore.

A quote that will not resolve is dropped. Storing it would put offsets in the record
that do not slice back to the quote, and E1 would stop being an invariant the moment
the first one landed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from src.contracts.rubric import JDRubric
from src.contracts.screening import CriterionScore, Evidence
from src.contracts.state import ScreeningState
from src.contracts.trace import NodeTrace
from src.graph.rubric_nodes import required_years
from src.tools.evidence import search_evidence
from src.tools.skills import expand_skill

SCORE_SYSTEM = (
    "Score the resume against each criterion from 0.0 to 1.0. For every criterion, "
    "copy 1 to 3 quotes VERBATIM from the resume that justify the score. Copy the "
    "characters exactly as they appear, including missing spaces between words. If "
    "nothing in the resume supports the criterion, score 0.0 and return no quotes."
)


class RawScore(BaseModel):
    """One criterion's score as the model writes it."""

    criterion_id: str
    score: float = Field(description="0.0 to 1.0")
    quotes: list[str] = Field(description="verbatim spans copied from the resume")
    reasoning: str


class RawScores(BaseModel):
    """The score_criteria node's response schema."""

    scores: list[RawScore]


def deterministic_skill_evidence(
    cv_text: str, rubric: JDRubric
) -> dict[str, list[Evidence]]:
    """Find each skill criterion's terms in the CV, without asking the model.

    `expand_skill` returns surface forms longest first, so "react native" is tried
    before "react" and the first hit wins. A criterion with no hit is absent from
    the mapping rather than present with an empty list, so callers can tell
    "searched and found nothing" from "not a skill criterion".
    """
    hits: dict[str, list[Evidence]] = {}
    for criterion in rubric.criteria:
        if criterion.kind != "skill" or not criterion.skill_terms:
            continue
        found: list[Evidence] = []
        for term in criterion.skill_terms:
            for surface in expand_skill(term):
                spans = search_evidence(cv_text, surface, max_results=1)
                if spans:
                    found.append(spans[0])
                    break
        if found:
            hits[criterion.id] = found
    return hits


def _dedupe(spans: list[Evidence]) -> list[Evidence]:
    """Drop repeated spans, keeping first-seen order."""
    seen: set[tuple[int, int]] = set()
    unique: list[Evidence] = []
    for span in spans:
        key = (span.start, span.end)
        if key not in seen:
            seen.add(key)
            unique.append(span)
    return unique


def _score_prompt(
    state: ScreeningState, rubric: JDRubric, tool_hits: dict[str, list[Evidence]]
) -> str:
    """The criteria, their weights, what the tools already found, then the resume."""
    lines: list[str] = []
    for criterion in rubric.criteria:
        marker = ", MUST HAVE" if criterion.must_have else ""
        lines.append(
            f"- {criterion.id} (weight {criterion.weight:.2f}{marker}): "
            f"{criterion.description}"
        )
        for span in tool_hits.get(criterion.id, []):
            lines.append(f"    already found in the resume: {span.quote!r}")
    return "CRITERIA:\n" + "\n".join(lines) + f"\n\nRESUME:\n{state.cv_text}"


def _build_scores(
    state: ScreeningState,
    rubric: JDRubric,
    raw: RawScores,
    tool_hits: dict[str, list[Evidence]],
) -> tuple[dict[str, CriterionScore], int, int]:
    """Resolve the model's quotes and keep only criteria the rubric actually has."""
    known = {criterion.id for criterion in rubric.criteria}
    by_id: dict[str, CriterionScore] = {}
    resolved = dropped = 0

    for item in raw.scores:
        if item.criterion_id not in known:
            continue
        evidence = list(tool_hits.get(item.criterion_id, []))
        for quote in item.quotes:
            spans = search_evidence(state.cv_text, quote, max_results=1)
            if not spans:
                dropped += 1
                continue
            resolved += 1
            evidence.append(spans[0])
        by_id[item.criterion_id] = CriterionScore(
            criterion_id=item.criterion_id,
            score=min(max(item.score, 0.0), 1.0),
            evidence=_dedupe(evidence),
            reasoning=item.reasoning,
            tool_used=(
                "normalize_skill+search_evidence"
                if item.criterion_id in tool_hits
                else "search_evidence"
            ),
        )
    return by_id, resolved, dropped


def _override_experience_scores(
    by_id: dict[str, CriterionScore], state: ScreeningState, rubric: JDRubric
) -> int:
    """Let `calculate_experience` set the score for every experience_years criterion."""
    profile = state.profile
    if profile is None or profile.total_experience_years is None:
        return 0

    overridden = 0
    for criterion in rubric.criteria:
        if criterion.kind != "experience_years":
            continue
        required = required_years(criterion.description)
        if required is None or required <= 0:
            continue
        existing = by_id.get(criterion.id)
        by_id[criterion.id] = CriterionScore(
            criterion_id=criterion.id,
            score=round(min(1.0, profile.total_experience_years / required), 4),
            evidence=existing.evidence if existing else [],
            reasoning=(
                f"calculate_experience found {profile.total_experience_years:.2f} "
                f"years against a stated requirement of {required:.0f}"
            ),
            tool_used="calculate_experience",
        )
        overridden += 1
    return overridden


def make_score_criteria_node(llm: Any) -> Callable[[ScreeningState], dict]:
    """Build the `score_criteria` node bound to a model client."""

    def score_criteria(state: ScreeningState) -> dict:
        started = time.perf_counter()
        rubric = state.rubric
        if rubric is None:
            raise ValueError("score_criteria reached without a rubric")

        tool_hits = deterministic_skill_evidence(state.cv_text, rubric)
        raw, usage = llm.parse(
            system=SCORE_SYSTEM,
            user=_score_prompt(state, rubric, tool_hits),
            schema=RawScores,
        )
        by_id, resolved, dropped = _build_scores(state, rubric, raw, tool_hits)
        overridden = _override_experience_scores(by_id, state, rubric)

        return {
            "path_taken": ["score_criteria"],
            # Rubric order, so a run is comparable with any other run of the same rubric.
            "criterion_scores": [
                by_id[criterion.id] for criterion in rubric.criteria if criterion.id in by_id
            ],
            "node_traces": [
                NodeTrace.of(
                    "score_criteria",
                    started,
                    [usage],
                    note=(
                        f"scored={len(by_id)} quotes_resolved={resolved} "
                        f"quotes_dropped={dropped} "
                        f"tool_hits={sum(len(v) for v in tool_hits.values())} "
                        f"experience_overrides={overridden}"
                    ),
                )
            ],
        }

    return score_criteria
