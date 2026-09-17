import pytest

from eval.records import (
    RowRecord,
    blocking_from_traces,
    read_records,
    record_from_result,
    scored_pairs,
    write_records,
)
from src.contracts.screening import FitLabel, ScreeningResult
from src.contracts.trace import NodeTrace


def result(**kwargs) -> ScreeningResult:
    defaults = dict(
        overall_score=0.62,
        label=FitLabel.POTENTIAL_FIT,
        path_taken=["ingest", "guard", "extract", "load_rubric", "must_have_check"],
        prompt_tokens=900,
        completion_tokens=120,
        latency_ms=1234.5,
        llm_calls=3,
        cached_calls=1,
    )
    return ScreeningResult(**{**defaults, **kwargs})


def test_blocking_must_haves_are_read_back_out_of_the_gate_trace():
    traced = result(
        node_traces=[
            NodeTrace(node="must_have_check", note="must_haves=3 blocking=c1,c4")
        ]
    )

    assert blocking_from_traces(traced) == ["c1", "c4"]


def test_a_gate_that_found_nothing_reads_as_an_empty_list_not_as_the_word_none():
    traced = result(
        node_traces=[NodeTrace(node="must_have_check", note="must_haves=3 blocking=none")]
    )

    assert blocking_from_traces(traced) == []


def test_a_run_that_never_reached_the_gate_has_no_blocking_criteria():
    assert blocking_from_traces(result(node_traces=[])) == []


def test_a_record_carries_what_the_run_cost_and_where_it_went():
    record = record_from_result(
        7,
        "Good Fit",
        result(node_traces=[NodeTrace(node="must_have_check", note="must_haves=2 blocking=c1")]),
        system="agent",
        config="shipped",
    )

    assert record.row_index == 7
    assert record.true_label == "Good Fit"
    assert record.predicted_label == "Potential Fit"
    assert record.overall_score == pytest.approx(0.62)
    assert record.blocking_must_haves == ["c1"]
    assert record.prompt_tokens == 900
    assert record.completion_tokens == 120
    assert record.llm_calls == 3
    assert record.cached_calls == 1
    assert record.error is None
    assert record.took("must_have_check", "reject_fast") is False


def test_a_record_knows_which_branch_it_took():
    record = record_from_result(
        0,
        "No Fit",
        result(path_taken=["ingest", "guard", "extract", "load_rubric",
                           "must_have_check", "reject_fast"]),
        system="agent",
        config="shipped",
    )

    assert record.took("must_have_check", "reject_fast") is True
    assert record.took("guard", "quarantine") is False


def test_records_survive_a_round_trip_through_jsonl(tmp_path):
    records = [
        record_from_result(0, "Good Fit", result(), system="agent", config="shipped"),
        record_from_result(1, "No Fit", result(), system="agent", config="shipped"),
    ]
    path = tmp_path / "records.jsonl"

    write_records(records, path)

    assert read_records(path) == records


def test_a_row_that_blew_up_is_recorded_rather_than_dropped():
    record = RowRecord(
        row_index=3,
        system="agent",
        config="shipped",
        true_label="Good Fit",
        predicted_label=None,
        error="RuntimeError: the graph ended without a result",
    )

    assert record.predicted_label is None


def test_evidence_coverage_is_the_share_of_criteria_backed_by_a_quote():
    from src.contracts.screening import CriterionScore, Evidence

    quoted = CriterionScore(
        criterion_id="lang",
        score=0.9,
        evidence=[Evidence(quote="Python", start=0, end=6)],
    )
    bare = CriterionScore(criterion_id="seniority", score=0.5)

    record = record_from_result(
        0, "Good Fit", result(criterion_scores=[quoted, bare]),
        system="agent", config="shipped",
    )

    assert record.criteria_scored == 2
    assert record.criteria_with_evidence == 1
    assert record.evidence_coverage == pytest.approx(0.5)


def test_a_system_that_answers_with_a_bare_label_scores_zero_on_evidence():
    """The baselines produce no evidence at all, and must not divide by zero."""
    record = RowRecord(
        row_index=0,
        system="baseline_naive",
        config="single_call",
        true_label="Good Fit",
        predicted_label="Good Fit",
    )

    assert record.evidence_coverage == 0.0


def test_failed_rows_are_excluded_from_the_pairs_the_metrics_see():
    """A crashed row must not become a silent No Fit.

    Scoring it as anything at all would be inventing a prediction the system
    never made, which is the one thing a measurement must not do.
    """
    good = record_from_result(0, "Good Fit", result(), system="agent", config="shipped")
    broken = RowRecord(
        row_index=1,
        system="agent",
        config="shipped",
        true_label="No Fit",
        predicted_label=None,
        error="boom",
    )

    assert scored_pairs([good, broken]) == [("Good Fit", "Potential Fit")]
