"""Run the agent over a split and write one record per pair.

This module deliberately knows nothing about macro-F1. It runs the graph and
records what happened; `eval/report.py` does the arithmetic. Keeping the two
apart is what makes a 300-row run a one-off cost rather than a prerequisite for
every question.

`today` is pinned rather than taken from the clock. `calculate_experience`
measures every work period against it, so a wall-clock date would mean the same
CV scores differently next week and every comparison with the Day 3 numbers
quietly stops being a comparison.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from eval.records import RowRecord, record_from_result, write_records
from src.contracts.ablations import Ablations
from src.graph.build import screen
from src.graph.rubric_nodes import DERIVED_RUBRIC_DIR
from src.llm.cache import DEFAULT_CACHE_PATH, JSONLCache
from src.llm.client import StructuredLLM

# Must equal `scripts.measure_branch_traffic.DEFAULT_TODAY`; a test pins the pair.
EVAL_TODAY = date(2026, 9, 7)

DEFAULT_SPLIT = Path("data/samples/dev_300.jsonl")
DEFAULT_OUT_DIR = Path("data/eval")

ABLATION_CHOICES: dict[str, Ablations] = {
    "shipped": Ablations(),
    "no_must_have_gate": Ablations(must_have_gate=False),
    "no_guard": Ablations(guard=False),
    "no_gray_zone": Ablations(gray_zone=False),
}

OnRow = Callable[[int, int, RowRecord], None]


def load_split(path: Path | str, limit: int = 0) -> list[dict[str, Any]]:
    """Read a split as dicts. `limit=0` means the whole thing."""
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit] if limit else rows


def records_path(out_dir: Path | str, system: str, config: str) -> Path:
    """Where one run's records live. The name carries both halves of its identity."""
    return Path(out_dir) / f"{system}__{config}.jsonl"


def run_split(
    rows: Sequence[dict[str, Any]],
    *,
    llm: Any,
    ablations: Ablations,
    today: date = EVAL_TODAY,
    system: str = "agent",
    derived_dir: Path | str = DERIVED_RUBRIC_DIR,
    on_row: OnRow | None = None,
) -> list[RowRecord]:
    """Screen every row, recording one `RowRecord` each.

    A row that raises is recorded with its traceback and the run continues. Three
    hundred rows is too expensive to throw away because one resume broke the
    extractor, and a row silently missing from the output is worse than a row
    marked broken.

    `derived_dir` is a parameter rather than a constant so a test can point it at
    `tmp_path`. Without that, running the suite writes YAML into the same directory
    the real measurements read from, and a later run silently reuses a rubric that
    a stub invented.
    """
    records: list[RowRecord] = []
    for index, row in enumerate(rows):
        try:
            result = screen(
                row["resume_text"],
                row["job_description_text"],
                llm=llm,
                today=today,
                derived_dir=derived_dir,
                ablations=ablations,
            )
            record = record_from_result(
                index,
                row["label"],
                result,
                system=system,
                config=ablations.label,
            )
        except Exception as error:  # noqa: BLE001 - the traceback is the record
            record = RowRecord(
                row_index=index,
                system=system,
                config=ablations.label,
                true_label=row["label"],
                predicted_label=None,
                error=f"{type(error).__name__}: {error}",
            )
            traceback.print_exc()
        records.append(record)
        if on_row is not None:
            on_row(index + 1, len(rows), record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=0, help="0 means the whole split")
    parser.add_argument("--ablations", choices=sorted(ABLATION_CHOICES), default="shipped")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    ablations = ABLATION_CHOICES[args.ablations]
    rows = load_split(args.split, args.limit)
    llm = StructuredLLM(cache=JSONLCache(DEFAULT_CACHE_PATH))

    def progress(done: int, total: int, _record: RowRecord) -> None:
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total} (cache hits {llm.hits}, misses {llm.misses})")

    print(f"running {len(rows)} rows, config={ablations.label}")
    records = run_split(
        rows, llm=llm, ablations=ablations, system="agent", on_row=progress
    )

    path = records_path(args.out_dir, "agent", ablations.label)
    write_records(records, path)
    failed = sum(1 for record in records if record.error is not None)
    print(f"wrote {path} ({len(records)} rows, {failed} errored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
