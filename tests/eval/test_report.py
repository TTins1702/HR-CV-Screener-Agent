import pytest

from eval.records import RowRecord, write_records
from eval.report import (
    collapse_warning,
    compare,
    main,
    gate_diagnosis,
    render_gate,
    render_run,
    summarise_records,
)

GOOD, POTENTIAL, NO = "Good Fit", "Potential Fit", "No Fit"

HAPPY = ["ingest", "guard", "extract", "load_rubric", "must_have_check",
         "score_criteria", "aggregate", "decide", "rank"]
REJECTED = ["ingest", "guard", "extract", "load_rubric", "must_have_check", "reject_fast"]


def record(index, true_label, predicted, path=None, *, config="shipped",
           blocking=(), tokens=1000, error=None) -> RowRecord:
    return RowRecord(
        row_index=index,
        system="agent",
        config=config,
        true_label=true_label,
        predicted_label=predicted,
        path_taken=list(path if path is not None else HAPPY),
        blocking_must_haves=list(blocking),
        prompt_tokens=tokens,
        completion_tokens=0,
        error=error,
    )


def test_a_summary_carries_the_headline_number_and_the_counts_behind_it():
    summary = summarise_records([
        record(0, GOOD, GOOD),
        record(1, GOOD, NO, REJECTED, blocking=["c1"]),
        record(2, NO, NO, REJECTED, blocking=["c2"]),
        record(3, POTENTIAL, POTENTIAL),
    ])

    assert summary.rows == 4
    assert summary.scored == 4
    assert summary.errors == 0
    # Good Fit F1 = 2/3, Potential Fit = 1.0, No Fit = 2/3; mean = 7/9.
    assert summary.macro_f1 == pytest.approx(7 / 9, abs=1e-6)
    assert summary.tokens == 4000
    assert summary.confusion[GOOD][NO] == 1


def test_errored_rows_are_counted_and_kept_out_of_the_metric():
    summary = summarise_records([
        record(0, GOOD, GOOD),
        record(1, NO, None, error="boom"),
    ])

    assert summary.rows == 2
    assert summary.scored == 1
    assert summary.errors == 1


def test_branch_traffic_is_recomputed_from_the_recorded_paths():
    summary = summarise_records([
        record(0, GOOD, GOOD),
        record(1, NO, NO, REJECTED, blocking=["c1"]),
    ])

    assert summary.branches["must_have_check -> reject_fast"].count == 1
    assert summary.branches["must_have_check -> reject_fast"].pct == 50.0
    assert summary.branches["guard -> quarantine"].count == 0


def test_the_gate_diagnosis_says_who_the_gate_actually_rejected():
    diagnosis = gate_diagnosis([
        record(0, GOOD, NO, REJECTED, blocking=["c1"]),
        record(1, GOOD, NO, REJECTED, blocking=["c1", "c3"]),
        record(2, NO, NO, REJECTED, blocking=["c2"]),
        record(3, POTENTIAL, POTENTIAL),
    ])

    assert diagnosis.rejected == 3
    assert diagnosis.true_labels == {GOOD: 2, NO: 1}
    assert diagnosis.wrongly_rejected == 2
    assert diagnosis.precision == pytest.approx(1 / 3)
    assert diagnosis.blocking_counts == {"c1": 2, "c2": 1, "c3": 1}


def test_a_gate_that_never_fired_reports_zero_rather_than_dividing_by_zero():
    diagnosis = gate_diagnosis([record(0, GOOD, GOOD)])

    assert diagnosis.rejected == 0
    assert diagnosis.precision == 0.0


def test_rendering_a_run_produces_the_markdown_the_slide_needs():
    text = render_run(summarise_records([record(0, GOOD, GOOD), record(1, NO, NO)]))

    assert "macro-F1" in text
    assert "| Truth \\ Predicted |" in text
    assert GOOD in text


def test_rendering_the_gate_names_the_criteria_doing_the_blocking():
    text = render_gate(gate_diagnosis([
        record(0, GOOD, NO, REJECTED, blocking=["c1"]),
    ]))

    assert "c1" in text


def test_comparing_two_configurations_shows_what_the_branch_bought():
    shipped = [
        record(0, GOOD, NO, REJECTED, blocking=["c1"], tokens=500),
        record(1, NO, NO, tokens=1000),
    ]
    ablated = [
        record(0, GOOD, GOOD, config="no_must_have_gate", tokens=2000),
        record(1, NO, NO, config="no_must_have_gate", tokens=1000),
    ]

    text = compare(shipped, ablated)

    assert "shipped" in text
    assert "no_must_have_gate" in text
    # The gate saved 1500 tokens and cost one correct Good Fit.
    assert "1500" in text.replace(",", "") or "-1500" in text.replace(",", "")


def test_comparing_requires_the_two_runs_to_cover_the_same_rows():
    with pytest.raises(ValueError, match="same rows"):
        compare([record(0, GOOD, GOOD)], [record(5, GOOD, GOOD, config="no_guard")])


def test_a_system_that_answers_almost_everything_the_same_way_is_flagged():
    """A collapsed baseline flatters whatever it is compared against.

    Spec section 7 asks for a *strong* rubric-in-prompt baseline. The first draft
    of its prompt told the model that a missing must-have rules out Good Fit, and
    it answered `No Fit` on 297 of 300 rows. Reporting that as "the strong baseline
    loses badly" would have been a claim about the prompt, not about the agent.
    """
    collapsed = [record(i, GOOD if i < 3 else NO, NO) for i in range(20)]

    warning = collapse_warning(summarise_records(collapsed))

    assert warning is not None
    assert "20 of 20" in warning
    assert "100%" in warning
    assert NO in warning


def test_the_collapse_threshold_is_a_cliff_not_a_slope():
    """19 of 20 is 95% and flagged; 18 of 20 is 90% and is merely lopsided."""
    at_threshold = [record(i, NO, NO if i < 19 else GOOD) for i in range(20)]
    below = [record(i, NO, NO if i < 18 else GOOD) for i in range(20)]

    assert collapse_warning(summarise_records(at_threshold)) is not None
    assert collapse_warning(summarise_records(below)) is None


def test_a_healthy_spread_of_predictions_is_not_flagged():
    mixed = [record(0, GOOD, GOOD), record(1, NO, NO), record(2, POTENTIAL, POTENTIAL)]

    assert collapse_warning(summarise_records(mixed)) is None


def test_the_cli_regenerates_a_measurement_file_from_records(tmp_path):
    """`docs/measurements/*.md` went stale because nothing regenerated them.

    Five files kept quoting a pre-gate-fix run for a day after the records had
    been replaced. The renderer was always a pure function of the records; what
    was missing was a command that ran it.
    """
    records = [record(0, GOOD, GOOD), record(1, NO, NO, REJECTED, blocking=["c1"])]
    source = tmp_path / "agent__shipped.jsonl"
    write_records(records, source)
    out = tmp_path / "out.md"

    assert main(["--records", str(source), "--out", str(out)]) == 0

    text = out.read_text(encoding="utf-8")
    assert "# agent / shipped over 2 rows" in text
    assert "macro-F1" in text


def test_the_cli_appends_the_gate_report_when_asked(tmp_path):
    records = [record(0, GOOD, NO, REJECTED, blocking=["needs_clearance"])]
    source = tmp_path / "agent__shipped.jsonl"
    write_records(records, source)
    out = tmp_path / "out.md"

    main(["--records", str(source), "--out", str(out), "--gate"])

    text = out.read_text(encoding="utf-8")
    assert "The must-have gate, judged against the labels" in text
    assert "needs_clearance" in text


def test_the_cli_refuses_to_invent_a_report_for_records_that_are_not_there(tmp_path):
    with pytest.raises(SystemExit):
        main(["--records", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path / "o.md")])
