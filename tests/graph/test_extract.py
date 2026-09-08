from datetime import date

from src.contracts.state import ScreeningState
from src.graph.extract import (
    RawExtraction,
    RawPeriod,
    build_profile,
    make_extract_node,
    make_repair_node,
    parse_month,
    unusable_date_fields,
)
from src.graph.routes import route_repair
from tests.graph.fixtures.poisoned import CLEAN_CV
from tests.graph.stub import StubLLM

TODAY = date(2026, 9, 7)


def state() -> ScreeningState:
    return ScreeningState(cv_text=CLEAN_CV, jd_text="Backend Engineer wanted.")


def extraction(**overrides) -> RawExtraction:
    base = dict(
        skills=["Python", "Django", "PostgreSQL"],
        work_periods=[
            RawPeriod(title="Backend Engineer", company="Acme Corp", start="2019-06", end="2022-12")
        ],
        degrees=["B.S. Computer Science"],
        certifications=[],
        total_experience_years=2.0,
        extraction_confidence=0.95,
    )
    base.update(overrides)
    return RawExtraction(**base)


def test_parse_month_accepts_the_documented_formats():
    assert parse_month("2019-06") == date(2019, 6, 1)
    assert parse_month("2019") == date(2019, 1, 1)
    assert parse_month("2019-06-15") == date(2019, 6, 15)


def test_parse_month_rejects_the_tokens_the_model_actually_emits():
    for token in ("null", "None", "N/A", "present", "Current", "", "  ", "sometime"):
        assert parse_month(token) is None, token


def test_parse_month_passes_a_genuine_null_through():
    assert parse_month(None) is None


def test_unusable_date_fields_ignores_a_genuine_null():
    raw = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="2021-01", end=None)]
    )

    assert unusable_date_fields(raw) == []


def test_unusable_date_fields_names_the_field_the_value_and_the_role():
    raw = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="2021-01", end="null")]
    )

    bad = unusable_date_fields(raw)

    assert len(bad) == 1
    assert "work_periods[0].end" in bad[0]
    assert "'null'" in bad[0]
    assert "Consultant" in bad[0]


def test_the_tool_overrides_what_the_model_claimed():
    profile, delta = build_profile(state(), extraction(total_experience_years=2.0), today=TODAY)

    assert profile.llm_declared_years == 2.0
    assert profile.total_experience_years == 3.5  # 06/2019 to 12/2022, from the raw text
    assert delta == 1.5


def test_the_model_is_kept_when_the_tool_finds_no_dates():
    bare = ScreeningState(cv_text="No dates here at all.", jd_text="jd")

    profile, delta = build_profile(bare, extraction(total_experience_years=4.0), today=TODAY)

    assert profile.total_experience_years == 4.0
    assert profile.llm_declared_years == 4.0
    assert delta == 0.0


def test_unparseable_model_dates_are_dropped_from_the_profile_not_guessed():
    raw = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="present", end="null")]
    )

    profile, _ = build_profile(state(), raw, today=TODAY)

    assert profile.work_periods[0].start is None
    assert profile.work_periods[0].end is None
    assert len(profile.missing_fields) == 2


def test_extract_records_the_correction_in_its_trace():
    node = make_extract_node(StubLLM([extraction()]), today=TODAY)

    update = node(state())

    assert update["path_taken"] == ["extract"]
    assert update["profile"].total_experience_years == 3.5
    note = update["node_traces"][0].note
    assert "years_llm=2.00" in note
    assert "years_tool=3.50" in note
    assert "correction=1.50" in note
    assert update["node_traces"][0].llm_calls == 1


def test_extract_shows_the_model_the_raw_cv():
    stub = StubLLM([extraction()])

    make_extract_node(stub, today=TODAY)(state())

    assert stub.calls[0]["user"] == CLEAN_CV
    assert stub.calls[0]["schema"] == "RawExtraction"


def test_repair_names_the_broken_fields_in_its_prompt_and_counts_the_attempt():
    broken = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="2021-01", end="null")]
    )
    before = state()
    before.profile = build_profile(before, broken, today=TODAY)[0]
    stub = StubLLM([extraction()])

    update = make_repair_node(stub, today=TODAY)(before)

    assert update["path_taken"] == ["repair"]
    assert update["repair_attempts"] == 1
    assert "work_periods[0].end" in stub.calls[0]["user"]
    assert CLEAN_CV in stub.calls[0]["user"]
    assert update["profile"].missing_fields == []
    assert "before=1 after=0" in update["node_traces"][0].note


def test_route_repair_sends_a_broken_profile_to_repair():
    broken = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="x", end=None)]
    )
    before = state()
    before.profile = build_profile(before, broken, today=TODAY)[0]

    assert route_repair(before) == "repair"


def test_route_repair_gives_up_at_the_cap():
    broken = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="x", end=None)]
    )
    before = state()
    before.profile = build_profile(before, broken, today=TODAY)[0]
    before.repair_attempts = 2

    assert route_repair(before) == "load_rubric"


def test_route_repair_moves_on_when_the_profile_is_clean():
    before = state()
    before.profile = build_profile(before, extraction(), today=TODAY)[0]

    assert route_repair(before) == "load_rubric"


def test_route_repair_moves_on_when_there_is_no_profile_at_all():
    assert route_repair(state()) == "load_rubric"
