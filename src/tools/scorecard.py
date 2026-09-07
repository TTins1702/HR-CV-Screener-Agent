"""Combine per-criterion scores into one number, one label, and two branch signals.

The weighted sum is a tool and not an LLM job for the dullest possible reason:
the same inputs must always produce the same total, and a model that adds six
weighted decimals will occasionally not. It also decides two of the graph's
conditional edges, so it has to be auditable.

Spec rules encoded here, do not let them drift:
  * A missing must-have routes to `reject_fast`, but every other criterion is
    still scored, so the UI can explain the rejection (spec 2). This function
    therefore never short-circuits.
  * The gray-zone margin around either threshold is the `deep_review` control
    parameter (spec 6).
"""

from __future__ import annotations

from src.contracts.rubric import JDRubric
from src.contracts.screening import CriterionScore, FitLabel
from src.contracts.tools import Scorecard

#: Decimal places kept in every reported number, so reruns compare equal.
ROUNDING = 4


def aggregate_scorecard(scores: list[CriterionScore], rubric: JDRubric) -> Scorecard:
    """Weighted total, label, and the two branch signals, for one candidate.

    An unscored criterion counts as 0.0 and is named in `unscored_criteria`
    rather than being quietly dropped from the denominator -- rescaling by the
    covered weight would let a scoring failure look like a good candidate.
    """
    by_id = _index_scores(scores, rubric)

    contributions: dict[str, float] = {}
    total = 0.0
    for criterion in rubric.criteria:
        score = by_id.get(criterion.id, 0.0)
        contribution = criterion.weight * score
        contributions[criterion.id] = round(contribution, ROUNDING)
        total += contribution

    overall = round(min(1.0, max(0.0, total)), ROUNDING)

    return Scorecard(
        overall_score=overall,
        label=_label_for(overall, rubric),
        in_gray_zone=_in_gray_zone(overall, rubric),
        missing_must_haves=sorted(
            criterion.id
            for criterion in rubric.must_haves()
            if by_id.get(criterion.id, 0.0) < rubric.must_have_min_score
        ),
        unscored_criteria=sorted(
            criterion.id for criterion in rubric.criteria if criterion.id not in by_id
        ),
        weighted_contributions=contributions,
    )


def _index_scores(scores: list[CriterionScore], rubric: JDRubric) -> dict[str, float]:
    """Scores by criterion id, rejecting duplicates and ids the rubric lacks."""
    known = {criterion.id for criterion in rubric.criteria}
    by_id: dict[str, float] = {}
    for entry in scores:
        if entry.criterion_id not in known:
            raise ValueError(
                f"score for {entry.criterion_id!r} has no matching criterion in rubric "
                f"{rubric.job_title!r}"
            )
        if entry.criterion_id in by_id:
            raise ValueError(f"criterion {entry.criterion_id!r} was scored more than once")
        by_id[entry.criterion_id] = entry.score
    return by_id


def _label_for(overall: float, rubric: JDRubric) -> FitLabel:
    """Which of the three streams this total falls into. Both bounds inclusive."""
    if overall >= rubric.good_fit_threshold:
        return FitLabel.GOOD_FIT
    if overall >= rubric.potential_fit_threshold:
        return FitLabel.POTENTIAL_FIT
    return FitLabel.NO_FIT


def _in_gray_zone(overall: float, rubric: JDRubric) -> bool:
    """True when the total sits within the margin of either cut-off.

    A margin of 0.0 disables the `deep_review` branch entirely, which is how the
    ablation in spec 7 turns it off.
    """
    margin = rubric.gray_zone_margin
    if margin <= 0.0:
        return False
    return any(
        abs(overall - threshold) <= margin
        for threshold in (rubric.good_fit_threshold, rubric.potential_fit_threshold)
    )
