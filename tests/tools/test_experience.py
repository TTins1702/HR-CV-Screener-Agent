from datetime import date

import pytest

from src.contracts.tools import ExperienceReport
from src.tools.experience import calculate_experience, merge_spans, months_between

TODAY = date(2026, 9, 7)


def test_months_between_counts_whole_months():
    assert months_between(date(2020, 1, 1), date(2020, 7, 1)) == 6
    assert months_between(date(2020, 1, 1), date(2022, 1, 1)) == 24
    assert months_between(date(2022, 1, 1), date(2020, 1, 1)) == 0


def test_merge_spans_joins_overlapping_intervals():
    merged = merge_spans(
        [
            (date(2020, 1, 1), date(2022, 1, 1)),
            (date(2021, 6, 1), date(2023, 1, 1)),
        ]
    )
    assert merged == [(date(2020, 1, 1), date(2023, 1, 1))]


def test_merge_spans_keeps_a_real_gap():
    merged = merge_spans(
        [
            (date(2015, 1, 1), date(2016, 1, 1)),
            (date(2020, 1, 1), date(2021, 1, 1)),
        ]
    )
    assert len(merged) == 2


def test_merge_spans_swallows_a_contained_interval():
    merged = merge_spans(
        [
            (date(2010, 1, 1), date(2020, 1, 1)),
            (date(2012, 1, 1), date(2014, 1, 1)),
        ]
    )
    assert merged == [(date(2010, 1, 1), date(2020, 1, 1))]


def test_parses_a_glued_slash_range_the_dataset_actually_contains():
    report = calculate_experience("Professional Experience01/2007to01/2011Analyst", today=TODAY)
    assert len(report.ranges) == 1
    assert report.ranges[0].start == date(2007, 1, 1)
    assert report.ranges[0].end == date(2011, 1, 1)
    assert report.total_years == 4.0


def test_parses_a_glued_current_range_and_marks_it_current():
    report = calculate_experience("Data Analyst II12/2011toPresentProduced reports", today=TODAY)
    assert len(report.ranges) == 1
    assert report.ranges[0].is_current is True
    assert report.ranges[0].end == TODAY
    assert report.total_years == pytest.approx(14.75, abs=0.01)


def test_parses_a_hyphenated_current_range():
    report = calculate_experience("Engineer 10/2016-Current", today=TODAY)
    assert report.ranges[0].is_current is True


def test_parses_a_month_name_range():
    report = calculate_experience("Consultant Jan 2018 to Mar 2020", today=TODAY)
    assert report.ranges[0].start == date(2018, 1, 1)
    assert report.ranges[0].end == date(2020, 3, 1)


def test_parses_a_bare_year_range_generously_at_both_ends():
    report = calculate_experience("Education 2011-2015 BSc", today=TODAY)
    assert report.ranges[0].start == date(2011, 1, 1)
    assert report.ranges[0].end == date(2015, 12, 1)


def test_total_years_counts_overlapping_roles_once():
    text = "Lead 01/2018to01/2022 Contractor 01/2020to01/2023"
    report = calculate_experience(text, today=TODAY)
    assert len(report.ranges) == 2
    assert report.overlaps_merged == 1
    assert report.total_years == 5.0  # not 4 + 3


def test_total_years_excludes_a_career_gap():
    text = "Analyst 01/2010to01/2012 Engineer 01/2016to01/2018"
    report = calculate_experience(text, today=TODAY)
    assert report.overlaps_merged == 0
    assert report.total_years == 4.0


def test_evidence_offsets_slice_the_original_text():
    text = "Data Analyst II12/2011to01/2015Reports"
    report = calculate_experience(text, today=TODAY)
    source = report.ranges[0].source
    assert text[source.start : source.end] == source.quote
    assert "12/2011" in source.quote


def test_drops_an_implausibly_long_range():
    # 01/1920-06/2017 appears verbatim in the dev split and is a typo.
    report = calculate_experience("Role 01/1920-06/2017", today=TODAY)
    assert report.ranges == []
    assert report.total_years == 0.0


def test_drops_a_range_starting_before_the_earliest_plausible_year():
    report = calculate_experience("Role 01/1940-01/1950", today=TODAY)
    assert report.ranges == []


def test_drops_a_range_that_starts_in_the_future():
    report = calculate_experience("Role 01/2030to01/2032", today=TODAY)
    assert report.ranges == []


def test_drops_a_reversed_range():
    report = calculate_experience("Role 01/2020to01/2015", today=TODAY)
    assert report.ranges == []


def test_reports_the_largest_self_declared_claim_without_using_it():
    text = "4 years analytic experience. 13 years of professional experience with Excel."
    report = calculate_experience(text, today=TODAY)
    assert report.self_declared_years == 13.0
    assert len(report.self_declared_evidence) == 2
    assert report.total_years == 0.0  # no date ranges, so nothing is computed


def test_self_declared_evidence_offsets_slice_the_original_text():
    text = "Summary13 years of experience"
    report = calculate_experience(text, today=TODAY)
    evidence = report.self_declared_evidence[0]
    assert text[evidence.start : evidence.end] == evidence.quote


def test_ignores_an_absurd_self_declared_claim():
    report = calculate_experience("I have 99 years of experience", today=TODAY)
    assert report.self_declared_years is None


def test_a_cv_with_no_dates_yields_an_empty_report():
    report = calculate_experience("Skilled communicator and team player.", today=TODAY)
    assert report == ExperienceReport(total_years=0.0)


def test_is_deterministic():
    text = "A 01/2018to01/2022 B 01/2020toPresent 7 years of experience"
    assert calculate_experience(text, today=TODAY) == calculate_experience(text, today=TODAY)


def test_ranges_are_returned_in_text_order():
    text = "Recent 01/2020to01/2022 Older 01/2010to01/2012"
    report = calculate_experience(text, today=TODAY)
    assert [r.source.start for r in report.ranges] == sorted(
        r.source.start for r in report.ranges
    )
