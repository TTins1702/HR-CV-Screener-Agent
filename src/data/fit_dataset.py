"""Load the labelled resume/JD fit dataset from HuggingFace.

The dataset ships as two flat CSVs and needs no authentication. Its text has
whitespace stripped between sentences, so downstream components must not assume
word boundaries around punctuation.
"""

from __future__ import annotations

import pandas as pd
from huggingface_hub import hf_hub_download

HF_FIT_REPO = "cnamuangtoun/resume-job-description-fit"
FIT_COLUMNS = ["resume_text", "job_description_text", "label"]
VALID_LABELS = {"Good Fit", "Potential Fit", "No Fit"}
SPLITS = ("train", "test")


def validate_fit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check the schema and return the frame narrowed to the three columns."""
    missing = [column for column in FIT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    narrowed = df[FIT_COLUMNS]

    unexpected = set(narrowed["label"].unique()) - VALID_LABELS
    if unexpected:
        raise ValueError(f"unexpected labels: {sorted(unexpected)}")

    if narrowed.isna().any().any():
        raise ValueError("null values in required columns")

    return narrowed


def load_fit_split(split: str) -> pd.DataFrame:
    """Download and validate one split. Cached by huggingface_hub after first call."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    path = hf_hub_download(HF_FIT_REPO, f"{split}.csv", repo_type="dataset")
    return validate_fit_frame(pd.read_csv(path))
