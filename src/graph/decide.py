"""decide and rank: the run becomes a result, and the result becomes readable.

`rank` orders criteria by weighted contribution rather than ranking candidates
against each other, because the graph screens one pair at a time. What a recruiter
needs from a single screening is the criteria in the order they moved the decision:
the top of that list is the "why", the bottom is the gap.
`Scorecard.weighted_contributions` already holds those numbers, computed
deterministically by `aggregate_scorecard`.
"""

from __future__ import annotations

import time

from src.contracts.screening import ScreeningResult
from src.contracts.state import ScreeningState
from src.contracts.trace import NodeTrace, trace_totals


def decide(state: ScreeningState) -> dict:
    """Turn the scorecard and the traces into the run's `ScreeningResult`."""
    started = time.perf_counter()
    card = state.scorecard
    if card is None:
        raise ValueError("decide reached without a scorecard")

    reason = None
    if card.missing_must_haves:
        reason = "must-have criteria scored below threshold: " + ", ".join(
            card.missing_must_haves
        )

    result = ScreeningResult(
        overall_score=card.overall_score,
        label=card.label,
        criterion_scores=list(state.criterion_scores),
        rejected_reason=reason,
        path_taken=[*state.path_taken, "decide"],
        node_traces=list(state.node_traces),
        **trace_totals(state.node_traces),
    )
    return {
        "path_taken": ["decide"],
        "result": result,
        "node_traces": [
            NodeTrace.of(
                "decide",
                started,
                note=f"label={card.label.value} overall={card.overall_score:.4f}",
            )
        ],
    }


def rank(state: ScreeningState) -> dict:
    """Order the result's criteria by how much each one moved the overall score."""
    started = time.perf_counter()
    if state.result is None or state.scorecard is None:
        return {
            "path_taken": ["rank"],
            "node_traces": [
                NodeTrace.of("rank", started, note="skipped: no result to rank")
            ],
        }

    contributions = state.scorecard.weighted_contributions
    ordered = sorted(
        state.result.criterion_scores,
        key=lambda score: (-contributions.get(score.criterion_id, 0.0), score.criterion_id),
    )
    result = state.result.model_copy(
        update={
            "criterion_scores": ordered,
            "path_taken": [*state.result.path_taken, "rank"],
        }
    )
    top = ordered[0].criterion_id if ordered else "none"
    return {
        "path_taken": ["rank"],
        "result": result,
        "node_traces": [NodeTrace.of("rank", started, note=f"top_criterion={top}")],
    }
