"""The first three nodes: accept the pair, screen it for injection, or refuse it.

`ingest` deliberately does not touch the text. Every `Evidence.start` and `.end` in
the whole system is an index into `cv_text` exactly as it arrived, so normalising
here would break E1 everywhere at once. The tools normalise against a copy and keep
an index map back (`src/tools/text_norm.py`). PDF-to-text is an app-layer concern
and does not belong in the graph.
"""

from __future__ import annotations

import time

from src.contracts.screening import FitLabel, ScreeningResult
from src.contracts.state import ScreeningState
from src.contracts.tools import InjectionSeverity
from src.contracts.trace import NodeTrace, trace_totals
from src.tools.injection import scan_injection

EMPTY_DOCUMENT_FLAG = "empty_document"


def ingest(state: ScreeningState) -> dict:
    """Accept the raw pair. A blank CV or JD is quarantined, not crashed on."""
    started = time.perf_counter()
    update: dict = {
        "path_taken": ["ingest"],
        "node_traces": [
            NodeTrace.of(
                "ingest",
                started,
                note=f"cv_chars={len(state.cv_text)} jd_chars={len(state.jd_text)}",
            )
        ],
    }
    if not state.cv_text.strip() or not state.jd_text.strip():
        update["quarantined"] = True
        update["injection_flags"] = [EMPTY_DOCUMENT_FLAG]
    return update


def guard(state: ScreeningState) -> dict:
    """Scan the untrusted CV for hidden instructions before it reaches the model.

    Only HIGH severity quarantines. The two LOW rules fire on phrasings that occur
    in honest reference letters, so they are recorded and shown to the recruiter
    rather than used to refuse the candidate.
    """
    started = time.perf_counter()
    report = scan_injection(state.cv_text)
    return {
        "path_taken": ["guard"],
        "quarantined": state.quarantined or report.severity is InjectionSeverity.HIGH,
        "injection_flags": [*state.injection_flags, *report.flags],
        "node_traces": [
            NodeTrace.of(
                "guard",
                started,
                note=f"severity={report.severity.value} findings={len(report.findings)}",
            )
        ],
    }


def quarantine(state: ScreeningState) -> dict:
    """Terminal node for a document the graph refuses to score.

    `label` is NO_FIT because the contract has no fourth label; `rejected_reason`
    carries the real story. Anything that reads the label must read the reason too.
    """
    started = time.perf_counter()
    reason = "quarantined: " + (", ".join(state.injection_flags) or "unknown")
    return {
        "path_taken": ["quarantine"],
        "result": ScreeningResult(
            overall_score=0.0,
            label=FitLabel.NO_FIT,
            rejected_reason=reason,
            path_taken=[*state.path_taken, "quarantine"],
            node_traces=list(state.node_traces),
            **trace_totals(state.node_traces),
        ),
        "node_traces": [NodeTrace.of("quarantine", started, note=reason)],
    }
