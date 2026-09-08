import pytest

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.screening import CriterionScore, FitLabel
from src.contracts.state import ScreeningState
from src.contracts.trace import NodeTrace
from src.graph.decide import decide, rank
from src.graph.scoring import aggregate
from tests.graph.fixtures.poisoned import CLEAN_CV

RUBRIC = JDRubric(
    job_title="Backend Engineer",
    criteria=[
        Criterion(id="small", description="Nice to have", weight=0.2),
        Criterion(id="big", description="Core requirement", weight=0.8, must_have=True),
    ],
)


def decided(small: float = 1.0, big: float = 1.0) -> ScreeningState:
    before = ScreeningState(cv_text=CLEAN_CV, jd_text="jd")
    before.rubric = RUBRIC
    before.criterion_scores = [
        CriterionScore(criterion_id="small", score=small, reasoning="r"),
        CriterionScore(criterion_id="big", score=big, reasoning="r"),
    ]
    before.scorecard = aggregate(before)["scorecard"]
    before.path_taken = ["ingest", "guard", "extract", "load_rubric"]
    before.node_traces = [
        NodeTrace(node="extract", latency_ms=2000.0, prompt_tokens=1200,
                  completion_tokens=250, llm_calls=1),
        NodeTrace(node="score_criteria", latency_ms=3000.0, prompt_tokens=1600,
                  completion_tokens=300, cached_calls=1),
    ]
    return before


def test_decide_copies_the_scorecard_verdict():
    result = decide(decided())["result"]

    assert result.overall_score == 1.0
    assert result.label is FitLabel.GOOD_FIT
    assert result.rejected_reason is None


def test_decide_sums_the_traces_into_the_run_totals():
    result = decide(decided())["result"]

    assert result.prompt_tokens == 2800
    assert result.completion_tokens == 550
    assert result.latency_ms == pytest.approx(5000.0)
    assert result.llm_calls == 1
    assert result.cached_calls == 1
    assert [t.node for t in result.node_traces] == ["extract", "score_criteria"]


def test_decide_records_the_path_including_itself():
    result = decide(decided())["result"]

    assert result.path_taken[-1] == "decide"
    assert result.path_taken[0] == "ingest"


def test_decide_explains_a_must_have_that_scored_too_low():
    result = decide(decided(big=0.1))["result"]

    assert result.label is FitLabel.NO_FIT
    assert "big" in result.rejected_reason


def test_decide_needs_a_scorecard():
    bare = ScreeningState(cv_text="cv", jd_text="jd")

    with pytest.raises(ValueError, match="scorecard"):
        decide(bare)


def test_rank_orders_criteria_by_what_they_contributed():
    before = decided()
    before.result = decide(before)["result"]

    result = rank(before)["result"]

    assert [s.criterion_id for s in result.criterion_scores] == ["big", "small"]
    assert result.path_taken[-1] == "rank"


def test_rank_puts_a_high_weight_zero_score_last():
    before = decided(small=1.0, big=0.0)
    before.result = decide(before)["result"]

    result = rank(before)["result"]

    assert [s.criterion_id for s in result.criterion_scores] == ["small", "big"]


def test_rank_is_a_no_op_without_a_result():
    bare = ScreeningState(cv_text="cv", jd_text="jd")

    assert rank(bare)["path_taken"] == ["rank"]
    assert "result" not in rank(bare)
