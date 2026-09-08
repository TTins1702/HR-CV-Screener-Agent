"""The job-description half of the graph: load_rubric, must_have_check, reject_fast.

Spec section 6 wants the rubric to be data a recruiter edits, not a prompt. That is
right for the demo and leaves a hole for the eval, where 500 test rows carry job
descriptions nobody will hand-author rubrics for. The hole is closed by deriving a
rubric once per distinct job description and writing it to YAML, where a recruiter
can open and edit it exactly like a hand-authored one.

Keying the YAML on a hash of the job description text is what makes that affordable:
`test_500.jsonl` holds 500 rows but only 69 distinct job descriptions, and
`dev_300.jsonl` holds 159 across 300.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.state import ScreeningState
from src.contracts.trace import NodeTrace
from src.rubric.loader import load_rubric as read_rubric
from src.rubric.loader import save_rubric as write_rubric

DERIVED_RUBRIC_DIR = Path("data/rubrics/derived")

VALID_KINDS = frozenset({"skill", "experience_years", "education", "domain", "other"})

RUBRIC_SYSTEM = (
    "Turn the job description into 4 to 8 weighted screening criteria. Mark a "
    "criterion must_have only if the job description states it as a hard requirement. "
    "For kind=skill criteria, list the concrete skill names in skill_terms; leave "
    "skill_terms empty for every other kind. Weights are relative importance and "
    "should sum to about 1.0."
)


class RawCriterion(BaseModel):
    """One criterion as the model writes it, before validation."""

    id: str = Field(description="snake_case identifier, unique within the rubric")
    description: str
    weight: float = Field(description="relative importance, 0 to 1")
    must_have: bool
    kind: str = Field(
        description="one of: skill, experience_years, education, domain, other"
    )
    skill_terms: list[str] = Field(
        description="concrete skill names for kind=skill; empty for other kinds"
    )


class RawRubric(BaseModel):
    """The load_rubric node's response schema."""

    job_title: str
    criteria: list[RawCriterion]


def jd_fingerprint(jd_text: str) -> str:
    """A stable 16-hex-character id for one job description."""
    return hashlib.sha256(jd_text.encode("utf-8")).hexdigest()[:16]


def repair_rubric(raw: RawRubric) -> JDRubric:
    """Force a model-written rubric into a valid `JDRubric`.

    `JDRubric` requires weights summing to 1.0 within 1e-6, and the model managed
    that unaided on only 16 of 20 real job descriptions (worst sum 1.100), so
    renormalising is what stops the eval dying halfway through with a
    ValidationError. Ids and kinds are coerced for the same reason: neither went
    wrong in the sample, but neither is guaranteed, and a coerced value beats a
    crashed run.
    """
    incoming = raw.criteria or [
        RawCriterion(
            id="overall_fit",
            description="Overall fit for the role",
            weight=1.0,
            must_have=False,
            kind="other",
            skill_terms=[],
        )
    ]
    total = sum(max(item.weight, 0.0) for item in incoming) or float(len(incoming))

    seen: set[str] = set()
    criteria: list[Criterion] = []
    for item in incoming:
        identifier = item.id.strip() or "criterion"
        suffix = 2
        while identifier in seen:
            identifier = f"{item.id.strip() or 'criterion'}_{suffix}"
            suffix += 1
        seen.add(identifier)
        criteria.append(
            Criterion(
                id=identifier,
                description=item.description.strip() or identifier,
                weight=max(item.weight, 0.0) / total,
                must_have=item.must_have,
                kind=item.kind if item.kind in VALID_KINDS else "other",
                skill_terms=[term for term in item.skill_terms if term.strip()],
            )
        )

    # Absorb float drift (and the all-zero-weights case) into the first criterion.
    drift = 1.0 - sum(item.weight for item in criteria)
    criteria[0] = criteria[0].model_copy(
        update={"weight": min(1.0, max(0.0, criteria[0].weight + drift))}
    )
    return JDRubric(
        job_title=raw.job_title.strip() or "Unspecified role", criteria=criteria
    )


def make_load_rubric_node(
    llm: Any, *, derived_dir: Path | str = DERIVED_RUBRIC_DIR
) -> Callable[[ScreeningState], dict]:
    """Build the `load_rubric` node: supplied, then cached on disk, then derived."""

    def load_rubric(state: ScreeningState) -> dict:
        started = time.perf_counter()

        if state.rubric is not None:
            return {
                "path_taken": ["load_rubric"],
                "node_traces": [
                    NodeTrace.of("load_rubric", started, note="supplied by caller")
                ],
            }

        path = Path(derived_dir) / f"{jd_fingerprint(state.jd_text)}.yaml"
        if path.exists():
            rubric = read_rubric(path)
            return {
                "path_taken": ["load_rubric"],
                "rubric": rubric,
                "node_traces": [
                    NodeTrace.of("load_rubric", started, note=f"reused {path.name}")
                ],
            }

        raw, usage = llm.parse(
            system=RUBRIC_SYSTEM, user=state.jd_text, schema=RawRubric
        )
        rubric = repair_rubric(raw)
        write_rubric(rubric, path)
        return {
            "path_taken": ["load_rubric"],
            "rubric": rubric,
            "node_traces": [
                NodeTrace.of(
                    "load_rubric",
                    started,
                    [usage],
                    note=(
                        f"derived {len(rubric.criteria)} criteria, "
                        f"{len(rubric.must_haves())} must-have"
                    ),
                )
            ],
        }

    return load_rubric
