import pytest

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.state import ScreeningState
from src.graph.rubric_nodes import (
    RawCriterion,
    RawRubric,
    jd_fingerprint,
    make_load_rubric_node,
    repair_rubric,
)
from tests.graph.fixtures.poisoned import CLEAN_CV
from tests.graph.stub import StubLLM

JD = "We need a Backend Engineer with 3 years of Python and PostgreSQL."


def state(jd: str = JD) -> ScreeningState:
    return ScreeningState(cv_text=CLEAN_CV, jd_text=jd)


def raw(*criteria: RawCriterion, job_title: str = "Backend Engineer") -> RawRubric:
    return RawRubric(job_title=job_title, criteria=list(criteria))


def criterion(identifier: str, weight: float, **overrides) -> RawCriterion:
    base = dict(
        id=identifier,
        description=f"{identifier} requirement",
        weight=weight,
        must_have=False,
        kind="skill",
        skill_terms=[],
    )
    base.update(overrides)
    return RawCriterion(**base)


def test_the_fingerprint_is_stable_and_jd_specific():
    assert jd_fingerprint(JD) == jd_fingerprint(JD)
    assert jd_fingerprint(JD) != jd_fingerprint(JD + " Remote.")
    assert len(jd_fingerprint(JD)) == 16


def test_weights_that_do_not_sum_to_one_are_renormalised():
    rubric = repair_rubric(raw(criterion("a", 0.6), criterion("b", 0.5)))

    assert sum(c.weight for c in rubric.criteria) == pytest.approx(1.0, abs=1e-9)
    assert rubric.criteria[0].weight == pytest.approx(0.6 / 1.1, abs=1e-6)


def test_weights_that_already_sum_to_one_are_left_alone():
    rubric = repair_rubric(raw(criterion("a", 0.7), criterion("b", 0.3)))

    assert rubric.criteria[0].weight == pytest.approx(0.7, abs=1e-9)
    assert rubric.criteria[1].weight == pytest.approx(0.3, abs=1e-9)


def test_all_zero_weights_still_produce_a_valid_rubric():
    rubric = repair_rubric(raw(criterion("a", 0.0), criterion("b", 0.0)))

    assert sum(c.weight for c in rubric.criteria) == pytest.approx(1.0, abs=1e-9)


def test_duplicate_ids_are_disambiguated_rather_than_rejected():
    rubric = repair_rubric(raw(criterion("python", 0.5), criterion("python", 0.5)))

    assert [c.id for c in rubric.criteria] == ["python", "python_2"]


def test_an_unknown_kind_falls_back_to_other():
    rubric = repair_rubric(raw(criterion("a", 1.0, kind="hard_skill")))

    assert rubric.criteria[0].kind == "other"


def test_blank_skill_terms_are_dropped():
    rubric = repair_rubric(raw(criterion("a", 1.0, skill_terms=["Python", "  ", ""])))

    assert rubric.criteria[0].skill_terms == ["Python"]


def test_an_empty_criteria_list_becomes_a_single_overall_criterion():
    rubric = repair_rubric(raw())

    assert len(rubric.criteria) == 1
    assert rubric.criteria[0].weight == pytest.approx(1.0, abs=1e-9)


def test_a_recruiter_supplied_rubric_skips_the_model_entirely(tmp_path):
    supplied = JDRubric(
        job_title="Supplied",
        criteria=[Criterion(id="only", description="only", weight=1.0)],
    )
    before = state()
    before.rubric = supplied
    stub = StubLLM([])

    update = make_load_rubric_node(stub, derived_dir=tmp_path)(before)

    assert update["path_taken"] == ["load_rubric"]
    assert "rubric" not in update
    assert stub.calls == []
    assert "supplied" in update["node_traces"][0].note


def test_a_derived_rubric_is_written_to_yaml_for_the_recruiter(tmp_path):
    stub = StubLLM([raw(criterion("python", 0.6, must_have=True, skill_terms=["Python"]),
                        criterion("sql", 0.4, skill_terms=["PostgreSQL"]))])

    update = make_load_rubric_node(stub, derived_dir=tmp_path)(state())

    written = tmp_path / f"{jd_fingerprint(JD)}.yaml"
    assert written.exists()
    assert update["rubric"].job_title == "Backend Engineer"
    assert len(update["rubric"].criteria) == 2
    assert update["node_traces"][0].llm_calls == 1


def test_a_second_row_with_the_same_jd_reads_the_yaml_instead_of_calling_the_model(tmp_path):
    first = StubLLM([raw(criterion("python", 1.0, skill_terms=["Python"]))])
    make_load_rubric_node(first, derived_dir=tmp_path)(state())

    second = StubLLM([])
    update = make_load_rubric_node(second, derived_dir=tmp_path)(state())

    assert second.calls == []
    assert update["rubric"].criteria[0].id == "python"
    assert update["rubric"].criteria[0].skill_terms == ["Python"]
    assert "reused" in update["node_traces"][0].note
    assert update["node_traces"][0].llm_calls == 0


def test_a_different_jd_does_not_reuse_another_jds_rubric(tmp_path):
    make_load_rubric_node(
        StubLLM([raw(criterion("python", 1.0))]), derived_dir=tmp_path
    )(state())

    stub = StubLLM([raw(criterion("java", 1.0), job_title="Java Dev")])
    update = make_load_rubric_node(stub, derived_dir=tmp_path)(state(jd="Java role."))

    assert update["rubric"].job_title == "Java Dev"
    assert len(stub.calls) == 1


def test_the_model_sees_the_job_description(tmp_path):
    stub = StubLLM([raw(criterion("python", 1.0))])

    make_load_rubric_node(stub, derived_dir=tmp_path)(state())

    assert stub.calls[0]["user"] == JD
    assert stub.calls[0]["schema"] == "RawRubric"
