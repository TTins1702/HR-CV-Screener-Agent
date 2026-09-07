"""Fetch the demo assets.

Job descriptions always succeed. Resumes need a Kaggle API token; without one
the script prints instructions and exits with status 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.data.demo_assets import (
    KAGGLE_RESUME_DATASET,
    kaggle_credentials_available,
    load_job_descriptions,
    select_it_jds,
    select_it_resumes,
)
from src.data.splits import write_jsonl

DEMO_DIR = Path("data/demo")

KAGGLE_INSTRUCTIONS = f"""
Kaggle credentials not found, so the demo resumes were not downloaded.
The job descriptions were written successfully; only the PDF resumes are missing.

To fix this:
  1. Sign in at https://www.kaggle.com and open https://www.kaggle.com/settings
  2. Under "API", click "Create New Token" and copy the KGAT_... string.
  3. Write it, with no trailing newline, to: {Path.home() / ".kaggle" / "access_token"}
  4. Re-run: python -m scripts.fetch_demo_assets

Alternative without an API token: download the dataset by hand from
https://www.kaggle.com/datasets/{KAGGLE_RESUME_DATASET}
and unzip it so that Resume/Resume.csv and data/ sit under data/raw/kaggle_resumes/.
""".strip()


def fetch_job_descriptions(n: int = 5) -> Path:
    selected = select_it_jds(load_job_descriptions(), n=n)
    path = write_jsonl(selected, DEMO_DIR / "jds.jsonl")
    print(f"job descriptions: {path} ({len(selected)} rows)")
    for title in selected["position_title"]:
        print(f"  - {title}")
    return path


def fetch_resumes(n: int = 25) -> Path:
    import kagglehub

    root = Path(kagglehub.dataset_download(KAGGLE_RESUME_DATASET))
    csv_path = root / "Resume" / "Resume.csv"
    if not csv_path.exists():
        matches = list(root.rglob("Resume.csv"))
        if not matches:
            raise FileNotFoundError(f"Resume.csv not found under {root}")
        csv_path = matches[0]

    selected = select_it_resumes(pd.read_csv(csv_path), n=n)
    path = write_jsonl(selected, DEMO_DIR / "resumes.jsonl")
    print(f"resumes: {path} ({len(selected)} rows) from {csv_path}")
    return path


def main() -> int:
    fetch_job_descriptions()

    if not kaggle_credentials_available():
        print(KAGGLE_INSTRUCTIONS, file=sys.stderr)
        return 2

    fetch_resumes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
