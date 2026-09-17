import json

import pytest

from eval.confidence_probe import (
    ConfidenceStats,
    load_extractions,
    render_confidence,
    summarise_confidence,
)


def cached(key, confidence, years):
    return {
        "key": key,
        "value": {
            "model": "gpt-4o-mini",
            "content": json.dumps(
                {
                    "skills": [],
                    "work_periods": [],
                    "degrees": [],
                    "certifications": [],
                    "total_experience_years": years,
                    "extraction_confidence": confidence,
                }
            ),
            "prompt_tokens": 1,
            "completion_tokens": 1,
        },
    }


def write_cache(tmp_path, entries):
    path = tmp_path / "cache.jsonl"
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    return path


def test_a_repeated_key_is_counted_once_with_the_last_line_winning(tmp_path):
    """The cache holds ~707 duplicated keys from a run that went out twice.

    Counting lines instead of keys would weight those rows double and quietly
    move every percentage in the report.
    """
    path = write_cache(
        tmp_path,
        [cached("a", 0.0, None), cached("a", 0.9, 4.0), cached("b", 0.9, 4.0)],
    )

    rows = load_extractions(path)

    assert len(rows) == 2
    assert all(row["extraction_confidence"] == 0.9 for row in rows)


def test_lines_that_are_not_extractions_are_skipped(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text(
        json.dumps({"key": "x", "value": {"content": json.dumps({"label": "Good Fit"})}})
        + "\n"
        + json.dumps({"key": "y", "value": {"content": "not json at all"}})
        + "\n",
        encoding="utf-8",
    )

    assert load_extractions(path) == []


def test_the_summary_measures_the_flag_against_the_field_it_stands_in_for():
    rows = [
        {"extraction_confidence": 0.0, "total_experience_years": None},
        {"extraction_confidence": 0.0, "total_experience_years": None},
        {"extraction_confidence": 0.0, "total_experience_years": 3.0},
        {"extraction_confidence": 0.9, "total_experience_years": 4.0},
        {"extraction_confidence": 0.9, "total_experience_years": None},
    ]

    stats = summarise_confidence(rows)

    assert stats.total == 5
    assert stats.values[0.0] == 3
    assert stats.missing_years_when_zero == 2
    assert stats.precision == pytest.approx(2 / 3)
    assert stats.recall == pytest.approx(2 / 3)


def test_a_cache_with_no_zero_confidence_rows_does_not_divide_by_zero():
    rows = [{"extraction_confidence": 0.9, "total_experience_years": 4.0}]

    stats = summarise_confidence(rows)

    assert stats.precision == 0.0
    assert stats.recall == 0.0


def test_the_report_says_the_field_is_redundant_rather_than_uninformative():
    """The claim the old docstring got wrong, now stated from the numbers.

    "Never drops below 0.90" and "uninformative" were both false. The report has
    to carry the correction, not just drop the sentence.
    """
    stats = ConfidenceStats(
        total=1030,
        values={0.0: 245, 0.9: 689, 0.95: 77, 1.0: 19},
        zeros=245,
        missing_years_when_zero=221,
        missing_years_when_nonzero=125,
        precision=0.902,
        recall=0.639,
    )

    text = render_confidence(stats)

    assert "redundant" in text
    assert "0.902" in text
    assert "23.8%" in text
