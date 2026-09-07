"""Read and write rubrics as YAML so a recruiter can edit them without code."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.contracts.rubric import JDRubric


def load_rubric(path: str | Path) -> JDRubric:
    """Load a rubric from a YAML file, validating it against the contract."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"rubric not found: {file_path}")

    data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"rubric YAML must be a mapping, got {type(data).__name__}")

    return JDRubric.model_validate(data)


def save_rubric(rubric: JDRubric, path: str | Path) -> None:
    """Write a rubric to YAML, preserving field order for readable diffs."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        yaml.safe_dump(
            rubric.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
