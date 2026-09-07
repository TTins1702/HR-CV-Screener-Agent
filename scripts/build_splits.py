"""Materialize data/samples/dev_300.jsonl and data/samples/test_500.jsonl.

Usage:
    python -m scripts.build_splits
"""

from __future__ import annotations

from src.data.splits import build_splits


def main() -> None:
    paths = build_splits()
    for name, path in paths.items():
        line_count = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"{name}: {path} ({line_count} rows)")


if __name__ == "__main__":
    main()
