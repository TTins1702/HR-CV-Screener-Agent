from src.contracts.screening import FitLabel, ScreeningResult
from src.contracts.trace import NodeTrace
from scripts.export_graph_diagram import main as export_main
from scripts.measure_branch_traffic import summarise


def result(path: list[str], label: FitLabel = FitLabel.NO_FIT, note: str = "") -> ScreeningResult:
    return ScreeningResult(
        overall_score=0.5,
        label=label,
        path_taken=path,
        prompt_tokens=1000,
        completion_tokens=200,
        llm_calls=2,
        node_traces=[NodeTrace(node="extract", note=note)] if note else [],
    )


HAPPY = ["ingest", "guard", "extract", "load_rubric", "must_have_check",
         "score_criteria", "aggregate", "decide", "rank"]


def test_a_branch_nothing_took_is_reported_as_zero_not_omitted():
    summary = summarise([result(HAPPY)])

    assert summary["branches"]["guard -> quarantine"] == {"count": 0, "pct": 0.0}
    assert summary["branches"]["aggregate -> deep_review"]["pct"] == 0.0


def test_traffic_is_counted_per_branch():
    summary = summarise([
        result(HAPPY),
        result(["ingest", "guard", "quarantine"]),
        result(["ingest", "guard", "extract", "repair", "load_rubric",
                "must_have_check", "reject_fast"]),
        result(HAPPY[:7] + ["deep_review", "decide", "rank"]),
    ])

    branches = summary["branches"]
    assert branches["guard -> quarantine"] == {"count": 1, "pct": 25.0}
    assert branches["extract -> repair"] == {"count": 1, "pct": 25.0}
    assert branches["must_have_check -> reject_fast"] == {"count": 1, "pct": 25.0}
    assert branches["aggregate -> deep_review"] == {"count": 1, "pct": 25.0}
    assert summary["rows"] == 4


def test_the_experience_correction_is_pulled_out_of_the_traces():
    summary = summarise([
        result(HAPPY, note="years_llm=2.00 years_tool=3.50 correction=1.50"),
        result(HAPPY, note="years_llm=5.00 years_tool=12.92 correction=7.92"),
        result(HAPPY, note="years_llm=None years_tool=None"),
    ])

    correction = summary["experience_correction"]
    assert correction["compared"] == 2
    assert correction["median"] == 4.71
    assert correction["max"] == 7.92


def test_labels_and_cost_are_summarised():
    summary = summarise([result(HAPPY, label=FitLabel.GOOD_FIT), result(HAPPY)])

    assert summary["labels"]["Good Fit"] == 1
    assert summary["labels"]["No Fit"] == 1
    assert summary["tokens"]["total"] == 2400
    assert summary["tokens"]["per_row"] == 1200.0


def test_summarising_nothing_does_not_divide_by_zero():
    summary = summarise([])

    assert summary["rows"] == 0
    assert summary["branches"]["guard -> quarantine"]["pct"] == 0.0


def test_the_diagram_script_writes_mermaid_without_calling_a_model(tmp_path):
    out = tmp_path / "graph.mmd"

    assert export_main(["--out", str(out)]) == 0

    text = out.read_text(encoding="utf-8")
    assert "graph TD" in text
    assert "guard -.-> quarantine" in text
