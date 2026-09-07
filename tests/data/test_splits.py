import json

import pandas as pd
import pytest

from src.data.splits import stratified_sample, write_jsonl


def _population() -> pd.DataFrame:
    labels = ["No Fit"] * 50 + ["Good Fit"] * 30 + ["Potential Fit"] * 20
    return pd.DataFrame(
        {
            "resume_text": [f"resume {i}" for i in range(100)],
            "job_description_text": [f"jd {i}" for i in range(100)],
            "label": labels,
        }
    )


def test_stratified_sample_preserves_label_proportions():
    sample = stratified_sample(_population(), n=10, seed=42)

    counts = sample["label"].value_counts().to_dict()
    assert counts == {"No Fit": 5, "Good Fit": 3, "Potential Fit": 2}


def test_stratified_sample_returns_exactly_n_rows_when_proportions_are_uneven():
    sample = stratified_sample(_population(), n=7, seed=42)

    assert len(sample) == 7
    assert set(sample["label"]) <= {"No Fit", "Good Fit", "Potential Fit"}


def test_stratified_sample_is_deterministic_for_a_given_seed():
    first = stratified_sample(_population(), n=10, seed=42)
    second = stratified_sample(_population(), n=10, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_stratified_sample_differs_across_seeds():
    first = stratified_sample(_population(), n=10, seed=42)
    second = stratified_sample(_population(), n=10, seed=7)

    assert not first.equals(second)


def test_stratified_sample_rejects_n_larger_than_the_population():
    with pytest.raises(ValueError, match="cannot sample"):
        stratified_sample(_population(), n=1000, seed=42)


def test_stratified_sample_rejects_non_positive_n():
    with pytest.raises(ValueError, match="must be positive"):
        stratified_sample(_population(), n=0, seed=42)


def test_write_jsonl_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "out.jsonl"

    write_jsonl(_population().head(3), path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert set(first) == {"resume_text", "job_description_text", "label"}
