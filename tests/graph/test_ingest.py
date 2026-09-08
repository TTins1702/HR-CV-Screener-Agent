import pytest

from src.contracts.state import ScreeningState
from src.contracts.tools import InjectionSeverity
from src.graph.ingest import EMPTY_DOCUMENT_FLAG, guard, ingest, quarantine
from src.graph.routes import route_guard
from tests.graph.fixtures.poisoned import CLEAN_CV, poisoned_cvs


def state(cv: str = CLEAN_CV, jd: str = "Backend Engineer wanted.") -> ScreeningState:
    return ScreeningState(cv_text=cv, jd_text=jd)


def test_ingest_records_itself_and_leaves_the_text_byte_identical():
    update = ingest(state())

    assert update["path_taken"] == ["ingest"]
    assert "cv_text" not in update  # the text is never rewritten
    assert update["node_traces"][0].node == "ingest"
    assert "cv_chars=" in update["node_traces"][0].note


def test_ingest_quarantines_a_blank_document_instead_of_crashing():
    update = ingest(state(cv="   \n  "))

    assert update["quarantined"] is True
    assert update["injection_flags"] == [EMPTY_DOCUMENT_FLAG]


def test_ingest_quarantines_a_blank_job_description_too():
    assert ingest(state(jd=""))["quarantined"] is True


def test_ingest_passes_a_normal_pair_through():
    update = ingest(state())

    assert update.get("quarantined") is None


def test_guard_leaves_a_clean_cv_alone():
    update = guard(state())

    assert update["quarantined"] is False
    assert update["injection_flags"] == []
    assert "severity=none" in update["node_traces"][0].note


@pytest.mark.parametrize("rule_id,cv", poisoned_cvs(InjectionSeverity.HIGH))
def test_guard_quarantines_every_high_severity_rule(rule_id, cv):
    update = guard(state(cv=cv))

    assert update["quarantined"] is True
    assert rule_id in update["injection_flags"]


@pytest.mark.parametrize("rule_id,cv", poisoned_cvs(InjectionSeverity.LOW))
def test_guard_records_but_does_not_quarantine_a_low_severity_rule(rule_id, cv):
    update = guard(state(cv=cv))

    assert update["quarantined"] is False
    assert rule_id in update["injection_flags"]


def test_guard_keeps_the_flag_ingest_already_set():
    before = state(cv="")
    before.quarantined = True
    before.injection_flags = [EMPTY_DOCUMENT_FLAG]

    update = guard(before)

    assert update["quarantined"] is True
    assert update["injection_flags"] == [EMPTY_DOCUMENT_FLAG]


def test_quarantine_produces_a_terminal_result_naming_the_reason():
    before = state()
    before.injection_flags = ["instruction_override"]
    before.path_taken = ["ingest", "guard"]

    result = quarantine(before)["result"]

    assert result.overall_score == 0.0
    assert result.label.value == "No Fit"
    assert "instruction_override" in result.rejected_reason
    assert result.rejected_reason.startswith("quarantined:")
    assert result.path_taken == ["ingest", "guard", "quarantine"]


def test_route_guard_sends_a_quarantined_document_to_quarantine():
    before = state()
    before.quarantined = True

    assert route_guard(before) == "quarantine"


def test_route_guard_sends_a_clean_document_to_extract():
    assert route_guard(state()) == "extract"
