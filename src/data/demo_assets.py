"""Assets used only by the live demo, never by the measured evaluation.

Job descriptions come from HuggingFace and need no credentials. The PDF resumes
live on Kaggle, which requires an API token; `kaggle_credentials_available`
lets the caller fail with instructions instead of a stack trace.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

HF_JD_REPO = "jacob-hugging-face/job-descriptions"
HF_JD_FILE = "training_data.csv"
JD_COLUMNS = [
    "company_name",
    "job_description",
    "position_title",
    "description_length",
    "model_response",
]

KAGGLE_RESUME_DATASET = "snehaanbhawal/resume-dataset"
IT_CATEGORY = "information-technology"

IT_TITLE_KEYWORDS = (
    "developer",
    "engineer",
    "programmer",
    "software",
    "web",
    "frontend",
    "front end",
    "backend",
    "back end",
    "full stack",
    "data scientist",
    "devops",
    "qa",
)


def load_job_descriptions() -> pd.DataFrame:
    """Download the 853-row job description table from HuggingFace."""
    path = hf_hub_download(HF_JD_REPO, HF_JD_FILE, repo_type="dataset")
    frame = pd.read_csv(path)

    missing = [column for column in JD_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    return frame[JD_COLUMNS]


def select_it_jds(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Pick `n` technical job descriptions, ordered deterministically by title."""
    if "position_title" not in df.columns:
        raise ValueError("frame has no position_title column")

    titles = df["position_title"].fillna("").str.lower()
    is_technical = titles.apply(
        lambda title: any(keyword in title for keyword in IT_TITLE_KEYWORDS)
    )

    return (
        df[is_technical]
        .sort_values(["position_title", "company_name"], kind="mergesort")
        .head(n)
        .reset_index(drop=True)[JD_COLUMNS]
    )


def select_it_resumes(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Pick `n` IT resumes from the Kaggle resume table, case-insensitively."""
    if "Category" not in df.columns:
        raise ValueError("frame has no Category column")

    is_it = df["Category"].fillna("").str.strip().str.lower() == IT_CATEGORY
    return df[is_it].head(n).reset_index(drop=True)


def kaggle_credentials_available() -> bool:
    """True when any credential form kagglehub accepts is present.

    Kaggle's current format is a single `KGAT_...` bearer token read from
    `~/.kaggle/access_token`. The `kaggle.json` file and the
    `KAGGLE_USERNAME`/`KAGGLE_KEY` pair are the legacy forms; kagglehub 1.x
    resolves the bearer token first and falls back to those.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    kaggle_dir = Path.home() / ".kaggle"
    if (kaggle_dir / "access_token").exists() or (kaggle_dir / "access_token.txt").exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (kaggle_dir / "kaggle.json").exists()
