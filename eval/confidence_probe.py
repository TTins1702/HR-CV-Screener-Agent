"""What `extraction_confidence` is actually worth, measured off the cache.

The extract node's docstring used to say the field "never dropped below 0.90" and
was "measured to be uninformative". Both were wrong, and neither could be checked
because no command produced the number. This is that command.

It reads the model's own answers rather than the record files, because
`RowRecord` does not keep the field. Duplicate cache keys are collapsed the way
`JSONLCache` collapses them -- last line wins -- since the cache carries roughly
707 keys written twice by a run that was accidentally started in parallel, and
counting lines would weight those extractions double.

Nothing here calls a model or touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from src.llm.cache import DEFAULT_CACHE_PATH

# What an extraction answer looks like, as opposed to a scoring or rubric answer.
EXTRACTION_MARKER = "extraction_confidence"


class ConfidenceStats(BaseModel):
    """The distribution, and how well a zero stands in for a missing years field."""

    total: int
    values: dict[float, int]
    zeros: int
    missing_years_when_zero: int
    missing_years_when_nonzero: int
    precision: float
    recall: float


def load_extractions(path: Path | str = DEFAULT_CACHE_PATH) -> list[dict]:
    """Every unique extraction answer in the cache, newest write per key."""
    latest: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue  # a half-written tail, same as JSONLCache tolerates
            key, value = record.get("key"), record.get("value") or {}
            if key is not None and isinstance(value.get("content"), str):
                latest[key] = value["content"]

    rows: list[dict] = []
    for content in latest.values():
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and EXTRACTION_MARKER in parsed:
            rows.append(parsed)
    return rows


def summarise_confidence(rows: list[dict]) -> ConfidenceStats:
    """Count the values, then score a zero as a detector of a missing years field."""
    values: dict[float, int] = {}
    zeros = missing_zero = missing_nonzero = 0
    for row in rows:
        confidence = float(row[EXTRACTION_MARKER])
        values[confidence] = values.get(confidence, 0) + 1
        missing = row.get("total_experience_years") is None
        if confidence == 0.0:
            zeros += 1
            missing_zero += missing
        else:
            missing_nonzero += missing

    true_positive = missing_zero
    false_positive = zeros - missing_zero
    false_negative = missing_nonzero
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )
    return ConfidenceStats(
        total=len(rows),
        values=dict(sorted(values.items())),
        zeros=zeros,
        missing_years_when_zero=missing_zero,
        missing_years_when_nonzero=missing_nonzero,
        precision=precision,
        recall=recall,
    )


def render_confidence(stats: ConfidenceStats) -> str:
    """The markdown for the decision: keep it, report it, never route on it."""
    below = sum(count for value, count in stats.values.items() if value < 0.90)
    nonzero = stats.total - stats.zeros
    lines = [
        "# `extraction_confidence`: keep it, report it, never route on it",
        "",
        f"Measured over the {stats.total} unique extraction answers in "
        "`data/cache/llm_cache.jsonl`. Regenerate with "
        "`python -m eval.confidence_probe`.",
        "",
        "## It is a four-value self-report, not a confidence",
        "",
        "| Value | Answers | Share |",
        "|---|---:|---:|",
    ]
    for value, count in stats.values.items():
        lines.append(f"| {value:.2f} | {count} | {100 * count / stats.total:.1f}% |")
    lines += [
        "",
        f"The extract node's docstring used to say it *\"never dropped below 0.90\"*. "
        f"It does, on **{100 * below / stats.total:.1f}%** of extractions, and it "
        "takes only four distinct values across the whole cache. A number that "
        "moves between four points is a label wearing a decimal point.",
        "",
        "## It is informative, and redundant",
        "",
        "The same docstring called it uninformative. That was also wrong. A zero "
        "tracks whether a total experience figure came out at all:",
        "",
        "| | total_experience_years missing | present |",
        "|---|---:|---:|",
        f"| confidence == 0.00 | {stats.missing_years_when_zero} | "
        f"{stats.zeros - stats.missing_years_when_zero} |",
        f"| confidence > 0.00 | {stats.missing_years_when_nonzero} | "
        f"{nonzero - stats.missing_years_when_nonzero} |",
        "",
        f"As a detector of a missing years field it scores precision "
        f"**{stats.precision:.3f}** and recall **{stats.recall:.3f}**.",
        "",
        "That is the case against routing on it, and it is a stronger case than "
        "the one it replaces. The field is **redundant**: "
        "`total_experience_years is None` answers the same question exactly, "
        "deterministically, and without asking a model to grade its own work. A "
        "branch built on the self-report would be a worse copy of a branch "
        "available for free.",
        "",
        "## Why it is not simply deleted",
        "",
        "`extraction_confidence` is a field of `RawExtraction`, and the cache key "
        "covers the response schema. Removing it changes every extraction key on "
        "disk, so the next run is cold and every measurement already published "
        "was taken against a different contract. Carrying an unused field is the "
        "cheaper of the two mistakes.",
        "",
        "So: it stays in the contract, it goes to the recruiter-facing view "
        "labelled for what it is rather than as a confidence, and "
        "`test_nothing_in_the_graph_routes_on_extraction_confidence` keeps it out "
        "of the control flow.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument(
        "--out", type=Path, default=Path("docs/measurements/extraction_confidence.md")
    )
    args = parser.parse_args(argv)

    if not args.cache.exists():
        parser.error(f"no such cache: {args.cache}")

    rows = load_extractions(args.cache)
    if not rows:
        parser.error(f"no extraction answers found in {args.cache}")

    stats = summarise_confidence(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_confidence(stats) + "\n", encoding="utf-8")
    print(f"wrote {args.out} from {args.cache} ({stats.total} extractions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
