"""One row of one evaluation run, and the JSONL it lives in.

The expensive thing in this project is running a system over a split. The cheap
thing is asking a question about what happened. Separating them is what this
module is for: a run writes records once, and every number in spec section 7 --
macro-F1, the confusion matrix, branch traffic, the gate diagnosis, the cost
comparison -- is computed afterwards from the file, offline and for free.

That property is also the insurance policy on spec section 3's "test runs once":
a question nobody thought to ask on the day of the run can still be answered
from the records months later, without a second run.

The agent and all three baselines emit this same shape, so comparing them is one
function over two files rather than two code paths that drift apart.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from src.contracts.screening import ScreeningResult

# `must_have_check` writes `must_haves=<n> blocking=<ids|none>`; this reads it back.
# Parsing the trace note rather than widening `ScreeningResult` follows the idiom
# `scripts/measure_branch_traffic.py` already uses for `correction=`.
_BLOCKING_RE = re.compile(r"blocking=(\S+)")


class RowRecord(BaseModel):
    """What one system did to one CV-JD pair."""

    row_index: int = Field(ge=0)
    system: str = Field(min_length=1)
    config: str = Field(min_length=1)
    true_label: str = Field(min_length=1)
    predicted_label: str | None = None

    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    path_taken: list[str] = Field(default_factory=list)
    blocking_must_haves: list[str] = Field(default_factory=list)
    in_gray_zone: bool = False

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    llm_calls: int = Field(default=0, ge=0)
    cached_calls: int = Field(default=0, ge=0)

    criteria_scored: int = Field(default=0, ge=0)
    criteria_with_evidence: int = Field(default=0, ge=0)

    rejected_reason: str | None = None
    error: str | None = None

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def evidence_coverage(self) -> float:
        """Share of scored criteria carrying at least one verbatim CV span.

        This is E1 in one number. Spec section 7 says that if the rubric-in-prompt
        baseline ties the agent on macro-F1, the agent's case rests on evidence
        linkage, robustness and cost -- so the evidence has to be counted, not
        asserted. A baseline that answers with a bare label scores 0.0 here, and
        that gap is the argument.
        """
        if not self.criteria_scored:
            return 0.0
        return self.criteria_with_evidence / self.criteria_scored

    def took(self, source: str, target: str) -> bool:
        """Did this row follow `source` immediately by `target`?"""
        return any(
            self.path_taken[index] == source and self.path_taken[index + 1] == target
            for index in range(len(self.path_taken) - 1)
        )


def blocking_from_traces(result: ScreeningResult) -> list[str]:
    """The must-have criteria the gate found, whether or not it acted on them.

    With the gate ablated the run never visits `reject_fast`, but `must_have_check`
    still writes what it found -- which is exactly what pairs a counterfactual row
    with the row it is the counterfactual of.
    """
    for trace in result.node_traces:
        if trace.node != "must_have_check":
            continue
        match = _BLOCKING_RE.search(trace.note)
        if match is None or match.group(1) == "none":
            return []
        return [part for part in match.group(1).split(",") if part]
    return []


def record_from_result(
    row_index: int,
    true_label: str,
    result: ScreeningResult,
    *,
    system: str,
    config: str,
    in_gray_zone: bool = False,
) -> RowRecord:
    """Flatten one finished `ScreeningResult` into one record."""
    return RowRecord(
        row_index=row_index,
        system=system,
        config=config,
        true_label=true_label,
        predicted_label=result.label.value,
        overall_score=result.overall_score,
        path_taken=list(result.path_taken),
        blocking_must_haves=blocking_from_traces(result),
        in_gray_zone=in_gray_zone,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
        llm_calls=result.llm_calls,
        cached_calls=result.cached_calls,
        criteria_scored=len(result.criterion_scores),
        criteria_with_evidence=sum(
            1 for score in result.criterion_scores if score.evidence
        ),
        rejected_reason=result.rejected_reason,
    )


def write_records(records: Iterable[RowRecord], path: Path | str) -> None:
    """One JSON object per line, sorted by row index for a stable diff."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: record.row_index)
    with target.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(record.model_dump_json() + "\n")


def read_records(path: Path | str) -> list[RowRecord]:
    """Read back what `write_records` wrote."""
    with Path(path).open(encoding="utf-8") as handle:
        return [RowRecord.model_validate(json.loads(line)) for line in handle if line.strip()]


def scored_pairs(records: Sequence[RowRecord]) -> list[tuple[str, str]]:
    """The `(true, predicted)` pairs the metrics may see.

    Rows that errored are left out rather than scored as anything. Inventing a
    prediction a system never made is the one thing a measurement must not do; the
    report states how many rows were excluded.
    """
    return [
        (record.true_label, record.predicted_label)
        for record in records
        if record.predicted_label is not None and record.error is None
    ]
