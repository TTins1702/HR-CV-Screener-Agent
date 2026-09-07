import pandas as pd
import pytest

from src.data.demo_assets import (
    JD_COLUMNS,
    kaggle_credentials_available,
    load_job_descriptions,
    select_it_jds,
    select_it_resumes,
)


def _jd_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company_name": ["A", "B", "C", "D", "E"],
            "job_description": ["jd a", "jd b", "jd c", "jd d", "jd e"],
            "position_title": [
                "Sales Specialist",
                "Web Developer",
                "Frontend Web Developer",
                "Licensing Coordinator",
                "Backend Engineer",
            ],
            "description_length": [10, 20, 30, 40, 50],
            "model_response": ["", "", "", "", ""],
        }
    )


def _resume_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Resume_str": ["r1", "r2", "r3", "r4"],
            "Resume_html": ["", "", "", ""],
            "Category": [
                "INFORMATION-TECHNOLOGY",
                "HR",
                "Information-Technology",
                "DESIGNER",
            ],
        }
    )


def test_select_it_jds_keeps_only_technical_titles():
    selected = select_it_jds(_jd_frame(), n=5)

    assert set(selected["position_title"]) == {
        "Web Developer",
        "Frontend Web Developer",
        "Backend Engineer",
    }


def test_select_it_jds_respects_n_and_is_deterministic():
    first = select_it_jds(_jd_frame(), n=2)
    second = select_it_jds(_jd_frame(), n=2)

    assert len(first) == 2
    pd.testing.assert_frame_equal(first, second)


def test_select_it_jds_keeps_the_expected_columns():
    selected = select_it_jds(_jd_frame(), n=2)

    assert list(selected.columns) == JD_COLUMNS


def test_select_it_resumes_matches_the_category_case_insensitively():
    selected = select_it_resumes(_resume_frame(), n=10)

    assert len(selected) == 2
    assert set(selected["ID"]) == {1, 3}


def test_select_it_resumes_raises_when_the_category_column_is_absent():
    with pytest.raises(ValueError, match="Category"):
        select_it_resumes(pd.DataFrame({"ID": [1]}), n=1)


def test_kaggle_credentials_available_returns_a_bool():
    assert isinstance(kaggle_credentials_available(), bool)


@pytest.mark.network
def test_load_job_descriptions_downloads_the_real_file():
    frame = load_job_descriptions()

    assert len(frame) == 853
    assert list(frame.columns) == JD_COLUMNS
    assert len(select_it_jds(frame, n=5)) == 5
