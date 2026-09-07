"""Deterministic stratified sampling for the dev and test evaluation sets.

The dev set is drawn from the dataset's train split and is where thresholds and
prompts get tuned. The test set is drawn from the dataset's test split and is
run exactly once, at the end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fit_dataset import FIT_COLUMNS, load_fit_split

DEFAULT_SEED = 42
DEFAULT_OUT_DIR = Path("data/samples")


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int = DEFAULT_SEED,
    label_col: str = "label",
) -> pd.DataFrame:
    """Sample `n` rows keeping each label's share, using largest-remainder allocation."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n > len(df):
        raise ValueError(f"cannot sample {n} rows from a population of {len(df)}")

    counts = df[label_col].value_counts().sort_index()
    exact = counts / len(df) * n
    allocation = np.floor(exact).astype(int)

    shortfall = n - int(allocation.sum())
    if shortfall > 0:
        remainders = (exact - allocation).sort_values(ascending=False, kind="mergesort")
        for label in list(remainders.index)[:shortfall]:
            allocation[label] += 1

    parts = [
        df[df[label_col] == label].sample(n=int(size), random_state=seed)
        for label, size in allocation.items()
        if size > 0
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_jsonl(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the frame as one JSON object per line, UTF-8, no index."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(file_path, orient="records", lines=True, force_ascii=False)
    return file_path


def build_splits(
    dev_size: int = 300,
    test_size: int = 500,
    seed: int = DEFAULT_SEED,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Path]:
    """Materialize the dev and test evaluation sets on disk."""
    dev = stratified_sample(load_fit_split("train")[FIT_COLUMNS], dev_size, seed)
    test = stratified_sample(load_fit_split("test")[FIT_COLUMNS], test_size, seed)

    return {
        "dev": write_jsonl(dev, Path(out_dir) / f"dev_{dev_size}.jsonl"),
        "test": write_jsonl(test, Path(out_dir) / f"test_{test_size}.jsonl"),
    }
