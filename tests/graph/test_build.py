from datetime import date
from pathlib import Path

import pytest

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.screening import FitLabel
from src.graph.build import NODE_NAMES, build_graph, graph_mermaid, screen
from src.graph.extract import RawExtraction, RawPeriod
from src.graph.rubric_nodes import RawCriterion, RawRubric
from src.graph.scoring import RawScore, RawScores
from tests.graph.fixtures.poisoned import CLEAN_CV, poisoned_cvs
from tests.graph.stub import StubLLM

TODAY = date(2026, 9, 7)
JD = "Backend Engineer with 3 years of Python."


def extraction(**overrides) -> RawExtraction:
    base = dict(
        skills=["Python", "PostgreSQL"],
        work_periods=[
            RawPeriod(title="Backend Engineer", company="Acme Corp",
                      start="2019-06", end="2022-12")
        ],
        degrees=["B.S. Computer Science"],
        certifications=[],
        total_experience_years=2.0,
        extraction_confidence=0.95,
    )
    base.update(overrides)
    return RawExtraction(**base)


def rubric(*, must_have_skill: str = "Python") -> RawRubric:
    return RawRubric(
        job_title="Backend Engineer",
        criteria=[
            RawCriterion(id="lang", description="Python in production", weight=0.5,
                         must_have=True, kind="skill", skill_terms=[must_have_skill]),
            RawCriterion(id="seniority", description="At least 3 years of experience",
                         weight=0.5, must_have=False, kind="experience_years",
                         skill_terms=[]),
        ],
    )


def all_scores(lang: float = 0.9) -> RawScores:
    return RawScores(scores=[
        RawScore(criterion_id="lang", score=lang,
                 quotes=["Built payment APIs in Python and PostgreSQL"], reasoning="r"),
        RawScore(criterion_id="seniority", score=0.5, quotes=[], reasoning="r"),
    ])


def run(responses, cv: str = CLEAN_CV, jd: str = JD, tmp_path: Path | None = None,
        **kwargs):
    return screen(cv, jd, llm=StubLLM(responses), today=TODAY,
                  derived_dir=tmp_path or Path("data/rubrics/derived"), **kwargs)


def test_the_graph_has_the_thirteen_nodes_the_spec_names():
    assert NODE_NAMES == (
        "ingest", "guard", "quarantine", "extract", "repair", "load_rubric",
        "must_have_check", "reject_fast", "score_criteria", "aggregate",
        "deep_review", "decide", "rank",
    )
    drawn = set(build_graph(StubLLM([])).get_graph().nodes)
    assert set(NODE_NAMES) <= drawn


def test_the_happy_path_visits_the_nodes_in_order(tmp_path):
    result = run([extraction(), rubric(), all_scores()], tmp_path=tmp_path)

    assert result.path_taken == [
        "ingest", "guard", "extract", "load_rubric", "must_have_check",
        "score_criteria", "aggregate", "decide", "rank",
    ]
    assert result.label in set(FitLabel)


def test_the_happy_path_produces_verifiable_evidence(tmp_path):
    result = run([extraction(), rubric(), all_scores()], tmp_path=tmp_path)

    spans = [e for s in result.criterion_scores for e in s.evidence]
    assert spans
    for evidence in spans:
        assert CLEAN_CV[evidence.start : evidence.end] == evidence.quote


def test_the_experience_criterion_is_scored_by_the_tool(tmp_path):
    result = run([extraction(), rubric(), all_scores()], tmp_path=tmp_path)

    seniority = next(s for s in result.criterion_scores if s.criterion_id == "seniority")
    assert seniority.tool_used == "calculate_experience"
    assert seniority.score == 1.0  # 3.5 measured years against a stated 3


@pytest.mark.parametrize("rule_id,cv", poisoned_cvs())
def test_a_poisoned_cv_either_quarantines_or_is_flagged(rule_id, cv, tmp_path):
    result = run([extraction(), rubric(), all_scores()], cv=cv, tmp_path=tmp_path)

    quarantined = result.path_taken == ["ingest", "guard", "quarantine"]
    assert quarantined or result.path_taken[-1] == "rank"
    if quarantined:
        assert rule_id in result.rejected_reason


def test_an_empty_cv_is_quarantined_without_a_single_model_call(tmp_path):
    stub = StubLLM([])

    result = screen("   ", JD, llm=stub, today=TODAY, derived_dir=tmp_path)

    assert result.path_taken == ["ingest", "guard", "quarantine"]
    assert "empty_document" in result.rejected_reason
    assert stub.calls == []


def test_a_broken_date_sends_the_run_through_repair_once(tmp_path):
    broken = extraction(work_periods=[
        RawPeriod(title="Backend Engineer", company="Acme Corp", start="2019-06", end="null")
    ])

    result = run([broken, extraction(), rubric(), all_scores()], tmp_path=tmp_path)

    assert result.path_taken[:5] == [
        "ingest", "guard", "extract", "repair", "load_rubric",
    ]


def test_the_repair_loop_stops_at_the_cap(tmp_path):
    broken = extraction(work_periods=[
        RawPeriod(title="Backend Engineer", company="Acme Corp", start="2019-06", end="null")
    ])

    result = run([broken, broken, broken, rubric(), all_scores()], tmp_path=tmp_path)

    assert result.path_taken.count("repair") == 2
    assert "load_rubric" in result.path_taken


def test_a_missing_must_have_short_circuits_before_any_scoring(tmp_path):
    stub = StubLLM([extraction(skills=["COBOL"]), rubric(must_have_skill="Kubernetes")])

    result = screen(CLEAN_CV, JD, llm=stub, today=TODAY, derived_dir=tmp_path)

    assert result.path_taken == [
        "ingest", "guard", "extract", "load_rubric", "must_have_check", "reject_fast",
    ]
    assert result.label is FitLabel.NO_FIT
    assert "lang" in result.rejected_reason
    assert len(stub.calls) == 2  # extract and load_rubric only -- no scoring call


def test_a_recruiter_supplied_rubric_skips_derivation(tmp_path):
    supplied = JDRubric(
        job_title="Supplied",
        criteria=[Criterion(id="lang", description="Python", weight=1.0, kind="skill",
                            skill_terms=["Python"])],
    )
    stub = StubLLM([
        extraction(),
        RawScores(scores=[RawScore(criterion_id="lang", score=0.9, quotes=[], reasoning="r")]),
    ])

    result = screen(CLEAN_CV, JD, llm=stub, rubric=supplied, today=TODAY,
                    derived_dir=tmp_path)

    assert [call["schema"] for call in stub.calls] == ["RawExtraction", "RawScores"]
    assert result.path_taken[-1] == "rank"


def test_the_run_totals_add_up_to_the_traces(tmp_path):
    result = run([extraction(), rubric(), all_scores()], tmp_path=tmp_path)

    assert result.prompt_tokens == sum(t.prompt_tokens for t in result.node_traces)
    assert result.llm_calls == 3
    assert result.latency_ms > 0


def test_the_mermaid_source_marks_the_conditional_edges_as_dotted():
    diagram = graph_mermaid()

    assert "guard -.-> quarantine" in diagram
    assert "guard -.-> extract" in diagram
    assert "ingest --> guard" in diagram
    assert "repair -.-> repair" in diagram


def test_openai_is_imported_in_exactly_one_module():
    root = Path("src")
    importers = sorted(
        path.as_posix()
        for path in root.rglob("*.py")
        if "openai" in path.read_text(encoding="utf-8")
    )

    assert importers == ["src/llm/client.py"]
