import pytest

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.screening import CriterionScore, FitLabel
from src.tools.scorecard import aggregate_scorecard


def _rubric(**overrides) -> JDRubric:
    defaults = dict(
        job_title="Backend Engineer",
        criteria=[
            Criterion(id="language", description="Backend language", weight=0.5, must_have=True),
            Criterion(id="database", description="SQL", weight=0.3),
            Criterion(id="cloud", description="Cloud", weight=0.2),
        ],
        good_fit_threshold=0.70,
        potential_fit_threshold=0.40,
    )
    return JDRubric(**{**defaults, **overrides})


def _scores(**by_id) -> list[CriterionScore]:
    return [CriterionScore(criterion_id=key, score=value) for key, value in by_id.items()]


def test_overall_score_is_the_weighted_sum():
    card = aggregate_scorecard(_scores(language=1.0, database=1.0, cloud=1.0), _rubric())
    assert card.overall_score == 1.0


def test_weighted_contributions_are_reported_per_criterion():
    card = aggregate_scorecard(_scores(language=0.8, database=0.5, cloud=0.0), _rubric())
    assert card.weighted_contributions == {"language": 0.4, "database": 0.15, "cloud": 0.0}
    assert card.overall_score == 0.55


def test_a_score_at_or_above_the_good_threshold_is_a_good_fit():
    card = aggregate_scorecard(_scores(language=1.0, database=1.0, cloud=0.0), _rubric())
    assert card.overall_score == 0.8
    assert card.label is FitLabel.GOOD_FIT


def test_the_good_threshold_boundary_is_inclusive():
    card = aggregate_scorecard(_scores(language=1.0, database=0.0, cloud=1.0), _rubric())
    assert card.overall_score == 0.7
    assert card.label is FitLabel.GOOD_FIT


def test_a_middling_score_is_a_potential_fit():
    card = aggregate_scorecard(_scores(language=1.0, database=0.0, cloud=0.0), _rubric())
    assert card.overall_score == 0.5
    assert card.label is FitLabel.POTENTIAL_FIT


def test_a_low_score_is_no_fit():
    card = aggregate_scorecard(_scores(language=0.2, database=0.2, cloud=0.2), _rubric())
    assert card.label is FitLabel.NO_FIT


def test_an_unscored_criterion_counts_as_zero_and_is_named():
    card = aggregate_scorecard(_scores(language=1.0), _rubric())
    assert card.overall_score == 0.5
    assert card.unscored_criteria == ["cloud", "database"]
    assert card.weighted_contributions["database"] == 0.0


def test_a_must_have_below_its_minimum_is_reported():
    card = aggregate_scorecard(_scores(language=0.2, database=1.0, cloud=1.0), _rubric())
    assert card.missing_must_haves == ["language"]
    # The rest is still scored, so the UI can explain the rejection.
    assert card.overall_score == 0.6
    assert card.weighted_contributions["database"] == 0.3


def test_a_must_have_at_its_minimum_is_not_reported():
    rubric = _rubric(must_have_min_score=0.5)
    card = aggregate_scorecard(_scores(language=0.5, database=1.0, cloud=1.0), rubric)
    assert card.missing_must_haves == []


def test_an_unscored_must_have_is_reported_as_missing():
    card = aggregate_scorecard(_scores(database=1.0, cloud=1.0), _rubric())
    assert card.missing_must_haves == ["language"]
    assert "language" in card.unscored_criteria


def test_a_score_clear_of_both_thresholds_is_not_in_the_gray_zone():
    rubric = _rubric(gray_zone_margin=0.05)
    card = aggregate_scorecard(_scores(language=0.68, database=0.5, cloud=0.0), rubric)
    assert card.overall_score == 0.49  # 0.34 + 0.15; 0.09 clear of both cut-offs
    assert card.in_gray_zone is False


def test_the_gray_zone_straddles_the_potential_threshold():
    rubric = _rubric(gray_zone_margin=0.05)
    card = aggregate_scorecard(_scores(language=0.84, database=0.0, cloud=0.0), rubric)
    assert card.overall_score == 0.42
    assert card.in_gray_zone is True


def test_the_gray_zone_straddles_the_good_threshold():
    rubric = _rubric(gray_zone_margin=0.05)
    card = aggregate_scorecard(_scores(language=1.0, database=0.7, cloud=0.0), rubric)
    assert card.overall_score == 0.71
    assert card.in_gray_zone is True


def test_a_zero_margin_disables_the_gray_zone():
    rubric = _rubric(gray_zone_margin=0.0)
    card = aggregate_scorecard(_scores(language=1.0, database=0.7, cloud=0.0), rubric)
    assert card.in_gray_zone is False


def test_a_score_for_a_criterion_the_rubric_does_not_have_is_rejected():
    with pytest.raises(ValueError, match="ghost"):
        aggregate_scorecard(_scores(language=1.0, ghost=1.0), _rubric())


def test_two_scores_for_the_same_criterion_are_rejected():
    scores = [
        CriterionScore(criterion_id="language", score=1.0),
        CriterionScore(criterion_id="language", score=0.0),
    ]
    with pytest.raises(ValueError, match="language"):
        aggregate_scorecard(scores, _rubric())


def test_no_scores_at_all_yields_a_zero_no_fit_scorecard():
    card = aggregate_scorecard([], _rubric())
    assert card.overall_score == 0.0
    assert card.label is FitLabel.NO_FIT
    assert card.missing_must_haves == ["language"]
    assert card.unscored_criteria == ["cloud", "database", "language"]


def test_the_result_is_rounded_for_reproducibility():
    rubric = _rubric(
        criteria=[
            Criterion(id="a", description="a", weight=1 / 3),
            Criterion(id="b", description="b", weight=1 / 3),
            Criterion(id="c", description="c", weight=1 / 3),
        ]
    )
    card = aggregate_scorecard(_scores(a=0.7, b=0.7, c=0.7), rubric)
    assert card.overall_score == 0.7


def test_is_deterministic():
    scores = _scores(language=0.61, database=0.42, cloud=0.13)
    rubric = _rubric()
    assert aggregate_scorecard(scores, rubric) == aggregate_scorecard(scores, rubric)


def test_it_works_with_the_committed_backend_rubric():
    from src.rubric.loader import load_rubric

    rubric = load_rubric("data/rubrics/backend_engineer.yaml")
    scores = [CriterionScore(criterion_id=c.id, score=1.0) for c in rubric.criteria]
    card = aggregate_scorecard(scores, rubric)
    assert card.overall_score == 1.0
    assert card.label is FitLabel.GOOD_FIT
    assert card.missing_must_haves == []
