import pytest
from pydantic import ValidationError

from src.contracts.rubric import Criterion, JDRubric


def _criteria(*weights: float) -> list[Criterion]:
    return [
        Criterion(id=f"c{i}", description=f"criterion {i}", weight=w)
        for i, w in enumerate(weights)
    ]


def test_rubric_accepts_weights_that_sum_to_one():
    rubric = JDRubric(job_title="Backend Engineer", criteria=_criteria(0.5, 0.3, 0.2))
    assert len(rubric.criteria) == 3


def test_rubric_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1.0"):
        JDRubric(job_title="Backend Engineer", criteria=_criteria(0.5, 0.3))


def test_rubric_tolerates_float_rounding_in_the_weight_sum():
    rubric = JDRubric(
        job_title="Backend Engineer",
        criteria=_criteria(1 / 3, 1 / 3, 1 / 3),
    )
    assert len(rubric.criteria) == 3


def test_rubric_rejects_duplicate_criterion_ids():
    duplicated = [
        Criterion(id="python", description="Python", weight=0.5),
        Criterion(id="python", description="Python again", weight=0.5),
    ]
    with pytest.raises(ValidationError, match="unique"):
        JDRubric(job_title="Backend Engineer", criteria=duplicated)


def test_rubric_rejects_thresholds_that_are_not_ordered():
    with pytest.raises(ValidationError, match="greater than"):
        JDRubric(
            job_title="Backend Engineer",
            criteria=_criteria(1.0),
            good_fit_threshold=0.3,
            potential_fit_threshold=0.6,
        )


def test_rubric_requires_at_least_one_criterion():
    with pytest.raises(ValidationError):
        JDRubric(job_title="Backend Engineer", criteria=[])


def test_must_haves_returns_only_the_flagged_criteria():
    rubric = JDRubric(
        job_title="Backend Engineer",
        criteria=[
            Criterion(id="python", description="Python", weight=0.6, must_have=True),
            Criterion(id="k8s", description="Kubernetes", weight=0.4),
        ],
    )
    assert [c.id for c in rubric.must_haves()] == ["python"]


def test_default_thresholds_are_ordered():
    rubric = JDRubric(job_title="Backend Engineer", criteria=_criteria(1.0))
    assert rubric.good_fit_threshold > rubric.potential_fit_threshold


def test_criterion_carries_the_concrete_skill_terms_the_gate_checks():
    from src.contracts.rubric import Criterion

    plain = Criterion(id="db", description="SQL", weight=1.0)
    typed = Criterion(
        id="db", description="SQL", weight=1.0, kind="skill", skill_terms=["postgresql", "mysql"]
    )

    assert plain.skill_terms == []
    assert typed.skill_terms == ["postgresql", "mysql"]
