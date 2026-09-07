from datetime import date

import pytest
from pydantic import ValidationError

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.screening import Evidence, FitLabel
from src.contracts.tools import (
    DateRange,
    ExperienceReport,
    InjectionFinding,
    InjectionReport,
    InjectionSeverity,
    Scorecard,
    SkillMatch,
)


def _evidence() -> Evidence:
    return Evidence(quote="01/2020to03/2022", start=10, end=26)


def test_evidence_defaults_to_a_perfect_score():
    assert Evidence(quote="x", start=0, end=1).score == 1.0


def test_evidence_accepts_a_fuzzy_score():
    assert Evidence(quote="x", start=0, end=1, score=0.83).score == 0.83


def test_evidence_rejects_a_score_above_one():
    with pytest.raises(ValidationError):
        Evidence(quote="x", start=0, end=1, score=1.5)


def test_date_range_carries_its_source_span():
    span = DateRange(
        start=date(2020, 1, 1), end=date(2022, 3, 1), is_current=False, source=_evidence()
    )
    assert span.source.start == 10
    assert span.is_current is False


def test_date_range_rejects_an_end_before_its_start():
    with pytest.raises(ValidationError):
        DateRange(start=date(2022, 1, 1), end=date(2020, 1, 1), source=_evidence())


def test_experience_report_defaults_to_empty():
    report = ExperienceReport(total_years=0.0)
    assert report.ranges == []
    assert report.overlaps_merged == 0
    assert report.self_declared_years is None
    assert report.self_declared_evidence == []


def test_experience_report_rejects_negative_years():
    with pytest.raises(ValidationError):
        ExperienceReport(total_years=-1.0)


def test_skill_match_reports_an_unknown_skill():
    match = SkillMatch(raw="Rust", canonical="rust", known=False)
    assert match.matched_alias is None


def test_injection_report_is_clean_by_default():
    report = InjectionReport()
    assert report.is_suspicious is False
    assert report.severity is InjectionSeverity.NONE
    assert report.findings == []
    assert report.flags == []


def test_injection_report_flags_are_the_rule_ids_in_order():
    report = InjectionReport(
        is_suspicious=True,
        severity=InjectionSeverity.HIGH,
        findings=[
            InjectionFinding(
                rule_id="instruction_override",
                severity=InjectionSeverity.HIGH,
                evidence=_evidence(),
            ),
            InjectionFinding(
                rule_id="must_hire", severity=InjectionSeverity.LOW, evidence=_evidence()
            ),
        ],
    )
    assert report.flags == ["instruction_override", "must_hire"]


def test_scorecard_holds_the_label_and_the_gray_zone_flag():
    card = Scorecard(
        overall_score=0.72,
        label=FitLabel.GOOD_FIT,
        in_gray_zone=True,
        weighted_contributions={"backend_language": 0.3},
    )
    assert card.label is FitLabel.GOOD_FIT
    assert card.missing_must_haves == []
    assert card.unscored_criteria == []


def test_rubric_exposes_the_branch_control_parameters_with_defaults():
    rubric = JDRubric(
        job_title="Backend Engineer",
        criteria=[Criterion(id="only", description="d", weight=1.0)],
    )
    assert rubric.gray_zone_margin == 0.05
    assert rubric.must_have_min_score == 0.5


def test_rubric_rejects_a_gray_zone_margin_above_the_threshold_gap():
    with pytest.raises(ValidationError):
        JDRubric(
            job_title="Backend Engineer",
            criteria=[Criterion(id="only", description="d", weight=1.0)],
            good_fit_threshold=0.70,
            potential_fit_threshold=0.65,
            gray_zone_margin=0.20,
        )
