import pandas as pd
import pytest

from src.data.fit_dataset import (
    FIT_COLUMNS,
    VALID_LABELS,
    load_fit_split,
    validate_fit_frame,
)


def _frame(**overrides) -> pd.DataFrame:
    data = {
        "resume_text": ["a resume", "another resume"],
        "job_description_text": ["a jd", "another jd"],
        "label": ["Good Fit", "No Fit"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_valid_labels_are_the_three_dataset_strings():
    assert VALID_LABELS == {"Good Fit", "Potential Fit", "No Fit"}


def test_validate_accepts_a_well_formed_frame():
    result = validate_fit_frame(_frame())

    assert list(result.columns) == FIT_COLUMNS
    assert len(result) == 2


def test_validate_narrows_extra_columns_away():
    frame = _frame()
    frame["extra"] = [1, 2]

    result = validate_fit_frame(frame)

    assert list(result.columns) == FIT_COLUMNS


def test_validate_rejects_a_missing_column():
    frame = _frame().drop(columns=["label"])

    with pytest.raises(ValueError, match="missing columns"):
        validate_fit_frame(frame)


def test_validate_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="unexpected labels"):
        validate_fit_frame(_frame(label=["Good Fit", "Perfect Fit"]))


def test_validate_rejects_nulls_in_required_columns():
    with pytest.raises(ValueError, match="null values"):
        validate_fit_frame(_frame(resume_text=["a resume", None]))


def test_load_fit_split_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="split must be"):
        load_fit_split("validation")


@pytest.mark.network
def test_load_fit_split_downloads_the_real_test_split():
    frame = load_fit_split("test")

    assert len(frame) == 1759
    assert list(frame.columns) == FIT_COLUMNS
    assert set(frame["label"].unique()) == VALID_LABELS
