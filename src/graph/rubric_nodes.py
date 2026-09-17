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
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.screening import CandidateProfile, FitLabel, ScreeningResult
from src.contracts.state import ScreeningState
from src.contracts.trace import NodeTrace, trace_totals
from src.rubric.loader import load_rubric as read_rubric
from src.rubric.loader import save_rubric as write_rubric
from src.tools.evidence import search_evidence
from src.tools.skills import expand_skill, skills_match

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


_YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years|yrs)")

# A gate is allowed to be generous: a missed skill costs a wrong rejection with no
# score at all, while a spurious match only costs the criterion being scored normally
# by `score_criteria` a moment later. `search_evidence`'s own default is
# DEFAULT_MIN_SCORE = 0.75, and this is deliberately BELOW it -- anything above would
# tighten the gate rather than loosen it.
#
# Measured on dev_300, 2026-09-17: at 0.75 the skill half of the gate rejected 78 rows
# and 43 of them were not truly `No Fit`, a precision of 0.45. Lowering the floor to
# 0.60 took macro-F1 from 0.3721 to 0.4155 for 9% more tokens -- better *and* cheaper
# than switching the gate off entirely (0.4052 at +19.8% tokens), because the gate
# still rejects true `No Fit` rows that scoring would have mislabelled. 0.55 scored
# +0.0017 higher, which on 300 rows is noise, so 0.60 is the knee rather than the peak.
FUZZY_GATE_FLOOR = 0.60


def required_years(description: str) -> float | None:
    """The number of years a criterion description asks for, if it names one."""
    match = _YEARS_RE.search(description.lower())
    return float(match.group(1)) if match else None


def _has_skill(terms: list[str], profile: CandidateProfile, cv_text: str) -> bool:
    """Does the candidate have any of `terms`? Extracted skills first, then the text.

    The text fallback is worth 12 percentage points: on 24 real pairs the gate fired
    on 54% using extracted skills alone and 42% with the fallback, rescuing 8
    criteria. The extractor drops skills that are plainly in the CV, and asking
    `expand_skill` plus `search_evidence` directly finds them again.
    """
    for term in terms:
        for held in profile.skills:
            if skills_match(term, held):
                return True
    for term in terms:
        for surface in expand_skill(term):
            if search_evidence(
                cv_text, surface, max_results=1, min_score=FUZZY_GATE_FLOOR
            ):
                return True
    return False


def blocking_must_haves(state: ScreeningState) -> list[str]:
    """Must-have criteria the candidate demonstrably fails, before any scoring.

    Only `skill` criteria that name `skill_terms` and `experience_years` criteria
    that name a number can be judged without a score. Everything else is left to
    `score_criteria` -- a gate that guesses is worse than a gate that abstains.

    This is NOT `Scorecard.missing_must_haves`: that one is computed after scoring
    and feeds `decide`. Confusing the two puts the whole rubric behind a gate that
    has no scores to read.
    """
    rubric, profile = state.rubric, state.profile
    if rubric is None or profile is None:
        return []

    blocking: list[str] = []
    for criterion in rubric.must_haves():
        if criterion.kind == "skill" and criterion.skill_terms:
            if not _has_skill(criterion.skill_terms, profile, state.cv_text):
                blocking.append(criterion.id)
        elif criterion.kind == "experience_years":
            required = required_years(criterion.description)
            if required is None:
                continue
            have = max(
                profile.total_experience_years or 0.0, profile.llm_declared_years or 0.0
            )
            if have + 1e-9 < required:
                blocking.append(criterion.id)
    return blocking


def must_have_check(state: ScreeningState) -> dict:
    """Run the pre-scoring gate and record what it found."""
    started = time.perf_counter()
    blocking = blocking_must_haves(state)
    checked = len(state.rubric.must_haves()) if state.rubric else 0
    return {
        "path_taken": ["must_have_check"],
        "blocking_must_haves": blocking,
        "node_traces": [
            NodeTrace.of(
                "must_have_check",
                started,
                note=f"must_haves={checked} blocking={','.join(blocking) or 'none'}",
            )
        ],
    }


def reject_fast(state: ScreeningState) -> dict:
    """Terminal node for a candidate missing a hard requirement.

    The shortcut is the point: no scoring call is made, which is where the token
    saving in spec section 7's cost argument comes from.
    """
    started = time.perf_counter()
    reason = "missing must-have criteria: " + ", ".join(state.blocking_must_haves)
    return {
        "path_taken": ["reject_fast"],
        "result": ScreeningResult(
            overall_score=0.0,
            label=FitLabel.NO_FIT,
            rejected_reason=reason,
            path_taken=[*state.path_taken, "reject_fast"],
            node_traces=list(state.node_traces),
            **trace_totals(state.node_traces),
        ),
        "node_traces": [NodeTrace.of("reject_fast", started, note=reason)],
    }
