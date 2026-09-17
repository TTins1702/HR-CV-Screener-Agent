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


from src.contracts.screening import CandidateProfile, FitLabel
from src.graph.rubric_nodes import (
    blocking_must_haves,
    must_have_check,
    reject_fast,
    required_years,
)
from src.graph.routes import route_must_have


def profile(skills: list[str], years: float | None = 5.0) -> CandidateProfile:
    return CandidateProfile(
        raw_text=CLEAN_CV,
        skills=skills,
        total_experience_years=years,
        extraction_confidence=0.9,
    )


def gated(rubric: JDRubric, skills: list[str], years: float | None = 5.0) -> ScreeningState:
    before = state()
    before.rubric = rubric
    before.profile = profile(skills, years)
    return before


def rubric_of(*criteria: Criterion) -> JDRubric:
    return JDRubric(job_title="Backend Engineer", criteria=list(criteria))


def skill_must_have(identifier: str, *terms: str, weight: float = 1.0) -> Criterion:
    return Criterion(
        id=identifier,
        description=f"Experience with {', '.join(terms)}",
        weight=weight,
        must_have=True,
        kind="skill",
        skill_terms=list(terms),
    )


def test_required_years_reads_the_number_out_of_a_description():
    assert required_years("At least 3 years of professional experience") == 3.0
    assert required_years("5+ yrs backend") == 5.0
    assert required_years("Strong backend experience") is None


def test_a_present_skill_does_not_block():
    rubric = rubric_of(skill_must_have("lang", "Python"))

    assert blocking_must_haves(gated(rubric, ["Python", "Django"])) == []


def test_an_alias_of_a_present_skill_does_not_block():
    rubric = rubric_of(skill_must_have("frontend", "React"))

    assert blocking_must_haves(gated(rubric, ["ReactJS"])) == []


def test_a_skill_the_extractor_missed_is_rescued_from_the_raw_text():
    # Django is in CLEAN_CV but deliberately absent from the extracted skills.
    rubric = rubric_of(skill_must_have("web", "Django"))

    assert blocking_must_haves(gated(rubric, ["Python"])) == []


def test_a_skill_that_is_nowhere_in_the_cv_blocks():
    rubric = rubric_of(skill_must_have("infra", "Kubernetes"))

    assert blocking_must_haves(gated(rubric, ["Python"])) == ["infra"]


def test_any_one_of_several_skill_terms_is_enough():
    rubric = rubric_of(skill_must_have("lang", "Go", "Python"))

    assert blocking_must_haves(gated(rubric, ["Python"])) == []


def test_too_few_years_blocks():
    rubric = rubric_of(
        Criterion(
            id="seniority",
            description="At least 8 years of experience",
            weight=1.0,
            must_have=True,
            kind="experience_years",
        )
    )

    assert blocking_must_haves(gated(rubric, ["Python"], years=3.0)) == ["seniority"]
    assert blocking_must_haves(gated(rubric, ["Python"], years=8.0)) == []


def test_a_must_have_with_nothing_checkable_is_left_to_the_scorer():
    rubric = rubric_of(
        Criterion(
            id="culture",
            description="Strong communication skills",
            weight=1.0,
            must_have=True,
            kind="other",
        )
    )

    assert blocking_must_haves(gated(rubric, [])) == []


def test_a_skill_must_have_with_no_skill_terms_is_left_to_the_scorer():
    rubric = rubric_of(
        Criterion(id="vague", description="Backend skills", weight=1.0, must_have=True, kind="skill")
    )

    assert blocking_must_haves(gated(rubric, [])) == []


def test_a_criterion_that_is_not_a_must_have_never_blocks():
    rubric = rubric_of(
        Criterion(
            id="nice", description="Kubernetes", weight=1.0, kind="skill", skill_terms=["Kubernetes"]
        )
    )

    assert blocking_must_haves(gated(rubric, ["Python"])) == []


def test_must_have_check_writes_the_blocking_ids_onto_the_state():
    rubric = rubric_of(skill_must_have("infra", "Kubernetes"))

    update = must_have_check(gated(rubric, ["Python"]))

    assert update["path_taken"] == ["must_have_check"]
    assert update["blocking_must_haves"] == ["infra"]
    assert "blocking=infra" in update["node_traces"][0].note


def test_reject_fast_produces_a_terminal_result_naming_the_criteria():
    before = gated(rubric_of(skill_must_have("infra", "Kubernetes")), ["Python"])
    before.blocking_must_haves = ["infra"]
    before.path_taken = ["ingest", "guard", "extract", "load_rubric", "must_have_check"]

    result = reject_fast(before)["result"]

    assert result.label is FitLabel.NO_FIT
    assert result.overall_score == 0.0
    assert "infra" in result.rejected_reason
    assert result.path_taken[-1] == "reject_fast"


def test_route_must_have_rejects_when_something_blocks():
    before = state()
    before.blocking_must_haves = ["infra"]

    assert route_must_have(before) == "reject_fast"


def test_route_must_have_scores_when_nothing_blocks():
    assert route_must_have(state()) == "score_criteria"


def test_reject_fast_reports_the_tokens_the_shortcut_already_spent():
    """Spec section 7 argues the agent is cheaper because of its shortcuts. That
    argument needs the short-cut rows to report what they actually spent, not zero."""
    from src.contracts.trace import NodeTrace

    before = gated(rubric_of(skill_must_have("infra", "Kubernetes")), ["Python"])
    before.blocking_must_haves = ["infra"]
    before.node_traces = [
        NodeTrace(node="extract", latency_ms=2000.0, prompt_tokens=1200,
                  completion_tokens=250, llm_calls=1),
        NodeTrace(node="load_rubric", latency_ms=900.0, prompt_tokens=600,
                  completion_tokens=120, llm_calls=1),
    ]

    result = reject_fast(before)["result"]

    assert result.prompt_tokens == 1800
    assert result.completion_tokens == 370
    assert result.llm_calls == 2
    assert [t.node for t in result.node_traces] == ["extract", "load_rubric"]
