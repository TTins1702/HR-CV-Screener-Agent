import pytest
from pydantic import ValidationError

from src.contracts.screening import (
    CandidateProfile,
    CriterionScore,
    Evidence,
    FitLabel,
    ScreeningResult,
    WorkPeriod,
)


def test_fit_label_values_match_the_dataset_strings_exactly():
    assert FitLabel.GOOD_FIT.value == "Good Fit"
    assert FitLabel.POTENTIAL_FIT.value == "Potential Fit"
    assert FitLabel.NO_FIT.value == "No Fit"
    assert {label.value for label in FitLabel} == {"Good Fit", "Potential Fit", "No Fit"}


def test_fit_label_parses_from_the_raw_dataset_string():
    assert FitLabel("Potential Fit") is FitLabel.POTENTIAL_FIT


def test_evidence_accepts_a_forward_span():
    evidence = Evidence(quote="7+ years", start=7, end=15)
    assert evidence.end > evidence.start


def test_evidence_rejects_an_empty_quote():
    with pytest.raises(ValidationError):
        Evidence(quote="", start=0, end=5)


def test_evidence_rejects_a_non_forward_span():
    with pytest.raises(ValidationError):
        Evidence(quote="x", start=10, end=10)


def test_candidate_profile_defaults_to_empty_collections():
    profile = CandidateProfile(raw_text="cv", extraction_confidence=0.5)
    assert profile.skills == []
    assert profile.work_periods == []
    assert profile.degrees == []
    assert profile.certifications == []
    assert profile.missing_fields == []
    assert profile.total_experience_years is None


def test_candidate_profile_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        CandidateProfile(raw_text="cv", extraction_confidence=1.5)


def test_work_period_allows_an_open_ended_current_job():
    period = WorkPeriod(title="Backend Engineer", company="Base", start=None, end=None)
    assert period.end is None


def test_criterion_score_rejects_a_score_above_one():
    with pytest.raises(ValidationError):
        CriterionScore(criterion_id="python", score=1.5)


def test_screening_result_carries_the_label_and_defaults_usage_to_zero():
    result = ScreeningResult(overall_score=0.81, label=FitLabel.GOOD_FIT)
    assert result.label is FitLabel.GOOD_FIT
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.latency_ms == 0.0
    assert result.rejected_reason is None
