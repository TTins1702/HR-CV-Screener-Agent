"""Per-node cost and latency accounting.

Spec section 8 asks for token and latency numbers *per branch*, not per run. The
graph gets that by having every node append one `NodeTrace`; the reducer on
`ScreeningState.node_traces` keeps them in execution order, so the eval can group
by node name and the Streamlit view can show the run as a timeline.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from pydantic import BaseModel, Field


class LLMUsage(BaseModel):
    """What one call to the model cost. `cached` means it never left the process."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    cached: bool = False


class NodeTrace(BaseModel):
    """One node's contribution to the run's cost, latency and story.

    `note` is free text the node writes for the slide -- the size of an experience
    correction, the severity of an injection finding, the number of quotes dropped.
    """

    node: str = Field(min_length=1)
    latency_ms: float = Field(default=0.0, ge=0.0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    cached_calls: int = Field(default=0, ge=0)
    note: str = ""

    @classmethod
    def of(
        cls,
        node: str,
        started: float,
        usages: Sequence[LLMUsage] = (),
        note: str = "",
    ) -> "NodeTrace":
        """Build a trace for a node that started at `started`, a `perf_counter` value."""
        return cls(
            node=node,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            prompt_tokens=sum(usage.prompt_tokens for usage in usages),
            completion_tokens=sum(usage.completion_tokens for usage in usages),
            llm_calls=sum(1 for usage in usages if not usage.cached),
            cached_calls=sum(1 for usage in usages if usage.cached),
            note=note,
        )


def trace_totals(traces: Sequence[NodeTrace]) -> dict[str, float | int]:
    """Roll a run's traces up into the totals `ScreeningResult` carries.

    Every terminal node uses this, not just `decide`. A run that stopped at
    `reject_fast` or `quarantine` still cost real tokens, and spec section 7 argues
    the agent is cheaper *because* of those shortcuts -- an argument that needs the
    short-cut rows to report what they spent rather than zero.
    """
    return {
        "prompt_tokens": sum(trace.prompt_tokens for trace in traces),
        "completion_tokens": sum(trace.completion_tokens for trace in traces),
        "latency_ms": sum(trace.latency_ms for trace in traces),
        "llm_calls": sum(trace.llm_calls for trace in traces),
        "cached_calls": sum(trace.cached_calls for trace in traces),
    }
