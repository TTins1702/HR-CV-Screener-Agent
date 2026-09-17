from src.contracts.tools import InjectionSeverity
from src.tools.injection import INJECTION_RULES

from eval.injection import (
    GuardOutcome,
    RuleComparison,
    build_poisoned_rows,
    compare_guard,
    render_guard_ablation,
)
from tests.eval.test_run import ROWS, SchemaStub

JD = ROWS[0]["job_description_text"]


def test_a_poisoned_row_exists_for_every_rule_in_the_table():
    """A rule added later must get a fixture without anybody remembering to add one.

    The guard branch carries no traffic on real data -- 0 of 300 dev resumes trip
    any rule -- so the whole measurement rests on these synthetic rows.
    """
    rows = build_poisoned_rows(JD)

    assert len(rows) == len(INJECTION_RULES)
    assert {row["rule_id"] for row in rows} == {r.rule_id for r in INJECTION_RULES}
    assert all(row["job_description_text"] == JD for row in rows)
    assert all(row["label"] == "No Fit" for row in rows)


def test_the_clean_control_carries_no_rule_and_is_scored_either_way():
    """Without it, "the score collapsed" has nothing to be a collapse from."""
    rows = build_poisoned_rows(JD, include_control=True)

    controls = [row for row in rows if row["rule_id"] is None]
    assert len(controls) == 1


def test_a_high_severity_row_is_quarantined_with_the_guard_and_scored_without_it(tmp_path):
    comparison = compare_guard(
        build_poisoned_rows(JD), llm=SchemaStub(), derived_dir=tmp_path
    )

    high = [c for c in comparison if c.severity is InjectionSeverity.HIGH]
    assert high, "the rule table has no HIGH severity rules"
    for row in high:
        assert row.guarded.quarantined
        assert not row.unguarded.quarantined
        assert row.guarded.overall_score == 0.0


def test_a_low_severity_row_is_scored_identically_with_and_without_the_guard(tmp_path):
    """The honest number for `must_hire` and `hidden_directive`.

    The guard flags them and lets them through, so switching it off changes
    nothing for these two. Reporting them inside the HIGH-severity average would
    claim a defence that was never in force.
    """
    comparison = compare_guard(
        build_poisoned_rows(JD), llm=SchemaStub(), derived_dir=tmp_path
    )

    low = [c for c in comparison if c.severity is InjectionSeverity.LOW]
    assert low, "the rule table has no LOW severity rules"
    for row in low:
        assert not row.guarded.quarantined
        assert row.guarded.overall_score == row.unguarded.overall_score
        assert row.changed is False


def test_the_report_separates_the_two_severities_rather_than_averaging_them(tmp_path):
    comparison = compare_guard(
        build_poisoned_rows(JD, include_control=True),
        llm=SchemaStub(),
        derived_dir=tmp_path,
    )

    text = render_guard_ablation(comparison)

    assert "HIGH" in text and "LOW" in text
    assert "no defence was in force either way" in text
    for rule in INJECTION_RULES:
        assert rule.rule_id in text


def test_an_outcome_records_the_path_so_the_branch_is_auditable():
    outcome = GuardOutcome(
        quarantined=True, overall_score=0.0, label="No Fit",
        path_taken=["ingest", "guard", "quarantine"], rejected_reason="quarantined: x",
    )

    assert "quarantine" in outcome.path_taken


def test_a_comparison_knows_whether_switching_the_guard_off_changed_anything():
    same = GuardOutcome(
        quarantined=False, overall_score=0.5, label="Potential Fit",
        path_taken=[], rejected_reason=None,
    )
    other = same.model_copy(update={"overall_score": 0.9, "label": "Good Fit"})

    assert RuleComparison(
        rule_id="r", severity=InjectionSeverity.LOW, guarded=same, unguarded=same
    ).changed is False
    assert RuleComparison(
        rule_id="r", severity=InjectionSeverity.HIGH, guarded=same, unguarded=other
    ).changed is True


def test_the_cli_refuses_to_run_without_a_job_description(tmp_path):
    import pytest

    from eval.injection import main

    with pytest.raises(SystemExit):
        main(["--split", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path / "o.md")])


def test_the_cli_takes_its_job_description_from_a_real_split(tmp_path):
    """A synthetic JD would derive a rubric nothing else in the eval shares.

    The poisoned CVs are already synthetic; pairing them with a made-up JD too
    would measure a pipeline configuration that ships nowhere.
    """
    import json

    from eval.injection import job_description_from

    split = tmp_path / "split.jsonl"
    split.write_text(json.dumps(ROWS[0]) + "\n", encoding="utf-8")

    assert job_description_from(split) == ROWS[0]["job_description_text"]


def test_the_report_will_not_call_it_a_collapse_when_the_score_did_not_move():
    """Measured on the real fixtures, every unguarded poisoned CV scored 0.550 --
    exactly the clean control. The injections changed nothing.

    Spec section 7 asks to see the score collapse when the guard is removed. It
    does not, and the report has to say that rather than assert the collapse it
    expected. The guard's value here is that it refuses a document attempting
    manipulation, not that it prevents a manipulation that would have worked.
    """
    from eval.injection import GuardOutcome, RuleComparison, render_guard_ablation

    scored = GuardOutcome(
        quarantined=False, overall_score=0.55, label="Potential Fit",
        path_taken=["ingest", "guard", "extract"], rejected_reason=None,
    )
    blocked = GuardOutcome(
        quarantined=True, overall_score=0.0, label="No Fit",
        path_taken=["ingest", "guard", "quarantine"], rejected_reason="quarantined: x",
    )
    comparison = [
        RuleComparison(rule_id=None, severity=None, guarded=scored, unguarded=scored),
        RuleComparison(
            rule_id="instruction_override", severity=InjectionSeverity.HIGH,
            guarded=blocked, unguarded=scored,
        ),
    ]

    text = render_guard_ablation(comparison)

    assert "did not move" in text
    assert "would have worked" in text


def test_the_report_says_so_when_an_injection_does_move_the_score():
    from eval.injection import GuardOutcome, RuleComparison, render_guard_ablation

    control = GuardOutcome(
        quarantined=False, overall_score=0.55, label="Potential Fit",
        path_taken=[], rejected_reason=None,
    )
    inflated = control.model_copy(update={"overall_score": 0.95, "label": "Good Fit"})
    blocked = GuardOutcome(
        quarantined=True, overall_score=0.0, label="No Fit",
        path_taken=[], rejected_reason="quarantined: x",
    )
    comparison = [
        RuleComparison(rule_id=None, severity=None, guarded=control, unguarded=control),
        RuleComparison(
            rule_id="score_manipulation", severity=InjectionSeverity.HIGH,
            guarded=blocked, unguarded=inflated,
        ),
    ]

    text = render_guard_ablation(comparison)

    assert "did not move" not in text
    assert "0.400" in text  # the largest gain over the control
