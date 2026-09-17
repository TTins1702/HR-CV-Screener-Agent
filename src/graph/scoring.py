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
from src.tools.scorecard import aggregate_scorecard
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


# A criterion scored inside this band has not really been decided, so it is what a
# gray-zone second look should spend its tokens on.
SECOND_LOOK_BAND = (0.3, 0.7)

DEEP_REVIEW_SYSTEM = (
    "A first pass scored this resume against the criteria below and the overall score "
    "landed close to a decision threshold. Re-examine only the criteria listed, using "
    "the quotes already found and the resume text. Return a revised score from 0.0 to "
    "1.0 for each, and say briefly what changed your mind or confirmed the score."
)


class RawRevision(BaseModel):
    """One revised score from the second look."""

    criterion_id: str
    score: float = Field(description="0.0 to 1.0")
    reasoning: str


class RawRevisions(BaseModel):
    """The deep_review node's response schema."""

    revisions: list[RawRevision]


def aggregate(state: ScreeningState) -> dict:
    """Combine the criterion scores deterministically.

    A thin wrapper on purpose: the weighted sum and both threshold cuts decide the
    `reject_fast` and `deep_review` branches, so they have to be reproducible to the
    decimal and auditable outside the graph. That is why `aggregate_scorecard` is a
    tool and not a paragraph of prompt.
    """
    started = time.perf_counter()
    rubric = state.rubric
    if rubric is None:
        raise ValueError("aggregate reached without a rubric")

    card = aggregate_scorecard(state.criterion_scores, rubric)
    return {
        "path_taken": ["aggregate"],
        "scorecard": card,
        "node_traces": [
            NodeTrace.of(
                "aggregate",
                started,
                note=(
                    f"overall={card.overall_score:.4f} label={card.label.value} "
                    f"gray={card.in_gray_zone} "
                    f"missing_must_haves={len(card.missing_must_haves)} "
                    f"unscored={len(card.unscored_criteria)}"
                ),
            )
        ],
    }


def needs_second_look(scores: list[CriterionScore]) -> list[CriterionScore]:
    """Criteria worth re-asking about: no evidence, or a score in the middle band.

    Deterministic on purpose -- which criteria the second pass looks at must not
    itself depend on a model call, or the branch stops being reproducible.
    """
    low, high = SECOND_LOOK_BAND
    return [
        score
        for score in scores
        if not score.evidence or low <= score.score <= high
    ]


def _revision_prompt(state: ScreeningState, thin: list[CriterionScore]) -> str:
    descriptions = {
        criterion.id: criterion.description
        for criterion in (state.rubric.criteria if state.rubric else [])
    }
    lines: list[str] = []
    for score in thin:
        lines.append(
            f"- {score.criterion_id}: {descriptions.get(score.criterion_id, '')} "
            f"(current score {score.score:.2f})"
        )
        for span in score.evidence:
            lines.append(f"    quote already found: {span.quote!r}")
        if not score.evidence:
            lines.append("    no supporting quote was found")
    return (
        "CRITERIA TO RE-EXAMINE:\n"
        + "\n".join(lines)
        + f"\n\nRESUME:\n{state.cv_text}"
    )


def make_deep_review_node(llm: Any) -> Callable[[ScreeningState], dict]:
    """Build the `deep_review` node: one extra pass over the undecided criteria.

    It runs at most once per screening and then edges unconditionally to `decide`,
    re-aggregating internally rather than looping back to `aggregate`. That is why
    there is no cycle to guard here and no second gray-zone test to oscillate on.
    """

    def deep_review(state: ScreeningState) -> dict:
        started = time.perf_counter()
        rubric = state.rubric
        if rubric is None:
            raise ValueError("deep_review reached without a rubric")

        thin = needs_second_look(state.criterion_scores)
        if not thin:
            return {
                "path_taken": ["deep_review"],
                "node_traces": [
                    NodeTrace.of(
                        "deep_review", started, note="skipped: nothing thin to revise"
                    )
                ],
            }

        raw, usage = llm.parse(
            system=DEEP_REVIEW_SYSTEM,
            user=_revision_prompt(state, thin),
            schema=RawRevisions,
        )

        by_id = {score.criterion_id: score for score in state.criterion_scores}
        revised = 0
        for revision in raw.revisions:
            existing = by_id.get(revision.criterion_id)
            if existing is None:
                continue
            by_id[revision.criterion_id] = existing.model_copy(
                update={
                    "score": min(max(revision.score, 0.0), 1.0),
                    "reasoning": revision.reasoning,
                    "tool_used": "deep_review",
                }
            )
            revised += 1

        scores = [
            by_id[criterion.id] for criterion in rubric.criteria if criterion.id in by_id
        ]
        card = aggregate_scorecard(scores, rubric)
        return {
            "path_taken": ["deep_review"],
            "criterion_scores": scores,
            "scorecard": card,
            "node_traces": [
                NodeTrace.of(
                    "deep_review",
                    started,
                    [usage],
                    note=(
                        f"reviewed={len(thin)} revised={revised} "
                        f"overall={card.overall_score:.4f} label={card.label.value}"
                    ),
                )
            ],
        }

    return deep_review
