"""Records in, markdown out. No model calls, no graph, no network.

Everything spec section 7 asks to see is computed here as a pure function of the
record files, which means the report can be regenerated, corrected and re-cut for
a slide long after the run that produced it -- including from the single
`test_500` run that spec section 3 allows.

Branch traffic is recomputed from `RowRecord.path_taken` rather than imported
from `scripts/measure_branch_traffic.py`: that module's `summarise` takes
`ScreeningResult` objects, which records deliberately are not. The arithmetic is
the same and both are tested against hand-checked fixtures.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

from eval.metrics import LABELS, confusion_matrix, macro_f1, per_class
from eval.records import RowRecord, scored_pairs

BRANCHES: tuple[tuple[str, str], ...] = (
    ("guard", "quarantine"),
    ("guard", "extract"),
    ("extract", "repair"),
    ("must_have_check", "reject_fast"),
    ("must_have_check", "score_criteria"),
    ("aggregate", "deep_review"),
    ("aggregate", "decide"),
)


class BranchCount(BaseModel):
    count: int
    pct: float


class RunSummary(BaseModel):
    """Everything one run's records say, arithmetic already done."""

    system: str
    config: str
    rows: int
    scored: int
    errors: int
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]
    predicted_mix: dict[str, int]
    truth_mix: dict[str, int]
    branches: dict[str, BranchCount]
    tokens: int
    tokens_per_row: float
    llm_calls: int
    cached_calls: int
    evidence_coverage: float


class GateDiagnosis(BaseModel):
    """What the must-have gate did, judged against the ground truth."""

    rejected: int
    true_labels: dict[str, int] = Field(default_factory=dict)
    wrongly_rejected: int
    precision: float
    blocking_counts: dict[str, int] = Field(default_factory=dict)


def _branch_counts(
    records: Sequence[RowRecord], share: Callable[[int], float]
) -> dict[str, BranchCount]:
    """How many rows took each conditional edge, and what share that is."""
    counts: dict[str, BranchCount] = {}
    for source, target in BRANCHES:
        count = sum(1 for record in records if record.took(source, target))
        counts[f"{source} -> {target}"] = BranchCount(count=count, pct=share(count))
    return counts


def summarise_records(records: Sequence[RowRecord]) -> RunSummary:
    """Fold one run's records into every number the report needs."""
    pairs = scored_pairs(records)
    rows = len(records)
    share = (lambda count: round(100.0 * count / rows, 1)) if rows else (lambda _: 0.0)

    scores = per_class(pairs)
    tokens = sum(record.tokens for record in records)
    return RunSummary(
        system=records[0].system if records else "unknown",
        config=records[0].config if records else "unknown",
        rows=rows,
        scored=len(pairs),
        errors=sum(1 for record in records if record.error is not None),
        macro_f1=macro_f1(pairs),
        per_class={
            label: {
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.f1,
                "support": float(score.support),
                "predicted": float(score.predicted),
            }
            for label, score in scores.items()
        },
        confusion=confusion_matrix(pairs),
        predicted_mix=dict(Counter(predicted for _, predicted in pairs)),
        truth_mix=dict(Counter(truth for truth, _ in pairs)),
        branches=_branch_counts(records, share),
        tokens=tokens,
        tokens_per_row=round(tokens / rows, 1) if rows else 0.0,
        llm_calls=sum(record.llm_calls for record in records),
        cached_calls=sum(record.cached_calls for record in records),
        evidence_coverage=(
            round(
                sum(record.criteria_with_evidence for record in records)
                / max(sum(record.criteria_scored for record in records), 1),
                4,
            )
        ),
    )


def gate_diagnosis(records: Sequence[RowRecord]) -> GateDiagnosis:
    """Judge the must-have gate against the labels.

    `precision` is the share of rejections that were truly `No Fit`. It is the
    number that decides whether the gate is a shortcut or a bug: a gate that
    rejects true `Good Fit` candidates is not saving tokens, it is buying a wrong
    answer with them.
    """
    rejected = [record for record in records if record.took("must_have_check", "reject_fast")]
    truths = Counter(record.true_label for record in rejected)
    correct = truths.get(LABELS[2], 0)  # "No Fit"
    blocking = Counter(
        criterion for record in rejected for criterion in record.blocking_must_haves
    )
    return GateDiagnosis(
        rejected=len(rejected),
        true_labels=dict(truths),
        wrongly_rejected=len(rejected) - correct,
        precision=(correct / len(rejected)) if rejected else 0.0,
        blocking_counts=dict(blocking),
    )


COLLAPSE_SHARE = 0.95


def collapse_warning(summary: RunSummary) -> str | None:
    """Flag a system that answered almost every row with the same label.

    A collapsed system is not a weak system, it is a broken one, and comparing
    against it flatters whatever it is compared against. Spec section 7 wants the
    rubric-in-prompt baseline to be genuinely hard to beat, so a collapse has to be
    visible in the report rather than showing up as a low F1 that reads like a
    result. Measured 2026-09-17: the first draft of that baseline's prompt told the
    model that a missing must-have ruled out `Good Fit`, and it answered `No Fit` on
    297 of 300 rows.
    """
    if not summary.scored:
        return None
    label, count = max(summary.predicted_mix.items(), key=lambda item: item[1])
    share = count / summary.scored
    if share < COLLAPSE_SHARE:
        return None
    return (
        f"**Collapsed: {count} of {summary.scored} predictions ({share:.0%}) are "
        f"`{label}`.** Treat this run as a broken system rather than a weak one; "
        f"comparing against it flatters the other systems."
    )


def render_run(summary: RunSummary) -> str:
    """The markdown for one run: headline, per class, confusion, branches, cost."""
    lines = [
        f"# {summary.system} / {summary.config} over {summary.rows} rows",
        "",
        f"**macro-F1: {summary.macro_f1:.4f}** "
        f"({summary.scored} scored, {summary.errors} errored)",
        "",
        "| Label | Precision | Recall | F1 | Support | Predicted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        score = summary.per_class[label]
        lines.append(
            f"| {label} | {score['precision']:.3f} | {score['recall']:.3f} | "
            f"{score['f1']:.3f} | {int(score['support'])} | {int(score['predicted'])} |"
        )

    lines += ["", "| Truth \\ Predicted | " + " | ".join(LABELS) + " |",
              "|---" * (len(LABELS) + 1) + "|"]
    for truth in LABELS:
        row = summary.confusion[truth]
        lines.append(f"| {truth} | " + " | ".join(str(row[p]) for p in LABELS) + " |")

    lines += ["", "| Branch | Runs | Traffic |", "|---|---:|---:|"]
    for name, branch in summary.branches.items():
        lines.append(f"| `{name}` | {branch.count} | {branch.pct}% |")

    lines += [
        "",
        f"Cost: {summary.tokens} tokens, {summary.tokens_per_row} per row, "
        f"{summary.llm_calls} live calls, {summary.cached_calls} cache hits",
        "",
        f"Evidence coverage (E1): **{summary.evidence_coverage:.3f}** of scored "
        f"criteria carry a verbatim CV span.",
    ]
    warning = collapse_warning(summary)
    if warning is not None:
        lines += ["", warning]
    return "\n".join(lines)


def render_gate(diagnosis: GateDiagnosis) -> str:
    """The markdown for the must-have gate's report card."""
    lines = [
        "## The must-have gate, judged against the labels",
        "",
        f"Rejected **{diagnosis.rejected}** rows before scoring. "
        f"**{diagnosis.wrongly_rejected}** of them were not truly `No Fit`, "
        f"so the gate's precision is **{diagnosis.precision:.3f}**.",
        "",
        "| True label of a rejected row | Rows |",
        "|---|---:|",
    ]
    for label, count in sorted(diagnosis.true_labels.items(), key=lambda item: -item[1]):
        lines.append(f"| {label} | {count} |")

    lines += ["", "| Blocking criterion | Rows |", "|---|---:|"]
    for criterion, count in sorted(
        diagnosis.blocking_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"| `{criterion}` | {count} |")
    return "\n".join(lines)


def compare(shipped: Sequence[RowRecord], ablated: Sequence[RowRecord]) -> str:
    """Two configurations of the same rows, side by side.

    Requires the same row indices in both, because the whole point is a paired
    comparison: what did this branch change, on these rows.
    """
    left, right = {r.row_index for r in shipped}, {r.row_index for r in ablated}
    if left != right:
        raise ValueError("both runs must cover the same rows to be compared")

    a, b = summarise_records(shipped), summarise_records(ablated)
    flipped = sum(
        1
        for x, y in zip(
            sorted(shipped, key=lambda r: r.row_index),
            sorted(ablated, key=lambda r: r.row_index),
        )
        if x.predicted_label != y.predicted_label
    )
    return "\n".join(
        [
            f"## `{a.config}` against `{b.config}` over {a.rows} rows",
            "",
            "| | " + f"{a.config} | {b.config} | delta |",
            "|---|---:|---:|---:|",
            f"| macro-F1 | {a.macro_f1:.4f} | {b.macro_f1:.4f} | "
            f"{b.macro_f1 - a.macro_f1:+.4f} |",
            f"| tokens | {a.tokens} | {b.tokens} | {b.tokens - a.tokens:+d} |",
            f"| errors | {a.errors} | {b.errors} | {b.errors - a.errors:+d} |",
            "",
            f"Predictions that changed: **{flipped}** of {a.rows}.",
        ]
    )
