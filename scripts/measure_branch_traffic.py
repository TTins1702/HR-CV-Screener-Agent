"""Measure how much traffic each conditional edge actually carries.

Spec section 4: a branch with no measured traffic is decoration. This is the command
that produces the numbers -- and the one that is allowed to report 0%, because 0% is
a finding, not a failure. `guard` is expected to be exactly that on clean data.

`summarise` is a pure function over finished results so the arithmetic behind every
percentage on the slide is unit-tested offline; `main` only runs the graph and
prints what `summarise` returns.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from src.contracts.screening import ScreeningResult
from src.graph.build import screen
from src.llm.cache import DEFAULT_CACHE_PATH, JSONLCache
from src.llm.client import StructuredLLM

DEFAULT_SPLIT = Path("data/samples/dev_300.jsonl")
DEFAULT_TODAY = date(2026, 9, 7)

# Every conditional edge, so one that never fires is reported as 0% rather than
# quietly missing from the table.
BRANCHES: tuple[tuple[str, str], ...] = (
    ("guard", "quarantine"),
    ("guard", "extract"),
    ("extract", "repair"),
    ("must_have_check", "reject_fast"),
    ("must_have_check", "score_criteria"),
    ("aggregate", "deep_review"),
    ("aggregate", "decide"),
)

_CORRECTION_RE = re.compile(r"correction=([0-9]+\.[0-9]+)")


def _took(path: list[str], source: str, target: str) -> bool:
    """Did this run follow `source` immediately by `target`?"""
    return any(
        path[index] == source and path[index + 1] == target
        for index in range(len(path) - 1)
    )


def summarise(results: list[ScreeningResult]) -> dict[str, Any]:
    """Branch traffic, label mix, cost, and the size of the experience correction."""
    rows = len(results)
    share = (lambda count: round(100.0 * count / rows, 1)) if rows else (lambda _: 0.0)

    branches: dict[str, dict[str, Any]] = {}
    for source, target in BRANCHES:
        count = sum(1 for result in results if _took(result.path_taken, source, target))
        branches[f"{source} -> {target}"] = {"count": count, "pct": share(count)}

    corrections = [
        float(match.group(1))
        for result in results
        for trace in result.node_traces
        for match in [_CORRECTION_RE.search(trace.note)]
        if match
    ]

    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)
    return {
        "rows": rows,
        "branches": branches,
        "labels": dict(Counter(result.label.value for result in results)),
        "tokens": {
            "total": total_tokens,
            "per_row": round(total_tokens / rows, 1) if rows else 0.0,
            "llm_calls": sum(result.llm_calls for result in results),
            "cached_calls": sum(result.cached_calls for result in results),
        },
        "experience_correction": {
            "compared": len(corrections),
            "median": round(statistics.median(corrections), 2) if corrections else 0.0,
            "max": round(max(corrections), 2) if corrections else 0.0,
            "over_1y": sum(1 for value in corrections if value >= 1.0),
        },
    }


def _render(summary: dict[str, Any]) -> str:
    lines = [
        f"# Branch traffic over {summary['rows']} real pairs",
        "",
        "| Branch | Runs | Traffic |",
        "|---|---:|---:|",
    ]
    for name, stats in summary["branches"].items():
        lines.append(f"| `{name}` | {stats['count']} | {stats['pct']}% |")
    correction = summary["experience_correction"]
    lines += [
        "",
        f"Labels: {summary['labels']}",
        f"Tokens: {summary['tokens']['total']} total, "
        f"{summary['tokens']['per_row']} per row, "
        f"{summary['tokens']['llm_calls']} live calls, "
        f"{summary['tokens']['cached_calls']} cache hits",
        "",
        f"calculate_experience corrected the model on {correction['compared']} rows: "
        f"median {correction['median']}y, max {correction['max']}y, "
        f"{correction['over_1y']} corrections of a year or more",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=0, help="0 means the whole split")
    parser.add_argument("--out", type=Path, default=Path("docs/measurements/branch_traffic.md"))
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.split.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    llm = StructuredLLM(cache=JSONLCache(DEFAULT_CACHE_PATH))
    results: list[ScreeningResult] = []
    for index, row in enumerate(rows, start=1):
        results.append(
            screen(
                row["resume_text"],
                row["job_description_text"],
                llm=llm,
                today=DEFAULT_TODAY,
            )
        )
        if index % 25 == 0:
            print(f"  {index}/{len(rows)} (cache hits {llm.hits}, misses {llm.misses})")

    report = _render(summarise(results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print()
    print(report)
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
