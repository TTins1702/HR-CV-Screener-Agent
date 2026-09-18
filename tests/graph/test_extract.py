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


def test_parse_month_rejects_the_sentinel_date_the_model_uses_for_unknown():
    """Measured on the first live run: `0001-01` was the single most common date
    token the model wrote (8 of 182 fields). It parses as year 1, so without a
    plausibility floor it flows into a WorkPeriod as a real date."""
    assert parse_month("0001-01") is None
    assert parse_month("0001-01-01") is None
    assert parse_month("1900-05") is None
    assert parse_month("1959-12") is None


def test_parse_month_keeps_dates_a_real_career_could_contain():
    assert parse_month("1960-01") == date(1960, 1, 1)
    assert parse_month("2024-11") == date(2024, 11, 1)


def test_an_implausible_date_is_flagged_for_repair():
    raw = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="0001-01", end=None)]
    )

    bad = unusable_date_fields(raw)

    assert len(bad) == 1
    assert "'0001-01'" in bad[0]


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


def test_nothing_in_the_graph_routes_on_extraction_confidence():
    """The field is reported, never acted on.

    It is a four-value self-report that mostly restates whether a total years
    figure came out, and the pipeline can check that directly. Routing on it
    would put a model's opinion of its own work in the control flow.
    """
    import inspect

    from src.graph import routes

    source = inspect.getsource(routes)

    assert "extraction_confidence" not in source


def test_extraction_confidence_stays_in_the_response_schema():
    """Dropping it is not the cheap tidy-up it looks like.

    The field is part of `RawExtraction`, and the cache key covers the response
    schema, so removing it invalidates every one of the extraction answers on
    disk and turns the next run cold. Keeping an unused field is the cheaper of
    the two mistakes; this test is here so the choice is made deliberately.
    """
    assert "extraction_confidence" in RawExtraction.model_fields


def test_a_confidence_the_model_invents_is_still_clamped_to_the_contract():
    profile, _ = build_profile(
        state(), extraction(extraction_confidence=7.5), today=TODAY
    )

    assert profile.extraction_confidence == 1.0


# A CV whose degree is written as a range, which is where the regexes alone
# over-count: the degree runs before the career rather than alongside it, so the
# union of the two is longer than the career by the whole length of the degree.
DEGREE_AND_JOB_CV = (
    "Education B.S. Computer Science, State University, 2012 - 2016. "
    "Experience Backend Engineer at Acme Corp, 06/2019to12/2022. Built payment APIs."
)


def test_a_degree_range_is_not_counted_as_professional_experience():
    degree_state = ScreeningState(cv_text=DEGREE_AND_JOB_CV, jd_text="jd")

    profile, _ = build_profile(degree_state, extraction(), today=TODAY)

    assert profile.total_experience_years == 3.5  # the Acme role, not the degree


def test_the_uncounted_degree_years_are_still_available_to_report():
    degree_state = ScreeningState(cv_text=DEGREE_AND_JOB_CV, jd_text="jd")

    profile, _ = build_profile(degree_state, extraction(), today=TODAY)

    assert profile.excluded_years == 4.92  # 2012-01 to 2016-12


def test_the_tool_still_counts_everything_when_the_model_found_no_dated_roles():
    """No dated role means nothing to classify against, not a zero-experience CV."""
    degree_state = ScreeningState(cv_text=DEGREE_AND_JOB_CV, jd_text="jd")
    undated = extraction(
        work_periods=[RawPeriod(title="Backend Engineer", company="Acme", start=None, end=None)]
    )

    profile, _ = build_profile(degree_state, undated, today=TODAY)

    assert profile.total_experience_years == 8.42  # degree union role


def test_the_model_is_kept_when_its_roles_match_nothing_in_the_text():
    """Roles that line up with no range in the CV must not zero the total.

    Excluding every range would otherwise read as "this candidate has never
    worked", which is a extraction failure reported as a fact about the person.
    """
    degree_state = ScreeningState(cv_text=DEGREE_AND_JOB_CV, jd_text="jd")
    elsewhere = extraction(
        total_experience_years=4.0,
        work_periods=[RawPeriod(title="Analyst", company="Old Co", start="1995-01", end="1998-01")],
    )

    profile, _ = build_profile(degree_state, elsewhere, today=TODAY)

    assert profile.total_experience_years == 4.0
