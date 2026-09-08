import operator
import time
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.contracts.trace import LLMUsage, NodeTrace


def test_usage_defaults_to_free_and_uncached():
    usage = LLMUsage()

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.cached is False


def test_node_trace_of_sums_the_usages_it_is_given():
    trace = NodeTrace.of(
        "extract",
        time.perf_counter(),
        [
            LLMUsage(prompt_tokens=100, completion_tokens=20),
            LLMUsage(prompt_tokens=50, completion_tokens=10, cached=True),
        ],
        note="corrected 3.2y",
    )

    assert trace.node == "extract"
    assert trace.prompt_tokens == 150
    assert trace.completion_tokens == 30
    assert trace.llm_calls == 1
    assert trace.cached_calls == 1
    assert trace.note == "corrected 3.2y"


def test_node_trace_of_measures_elapsed_time():
    started = time.perf_counter()
    trace = NodeTrace.of("guard", started)

    assert trace.latency_ms >= 0.0
    assert trace.llm_calls == 0


def test_node_traces_accumulate_through_a_langgraph_reducer():
    class Probe(BaseModel):
        traces: Annotated[list[NodeTrace], operator.add] = Field(default_factory=list)

    def first(state: Probe) -> dict:
        return {"traces": [NodeTrace(node="first")]}

    def second(state: Probe) -> dict:
        return {"traces": [NodeTrace(node="second")]}

    builder = StateGraph(Probe)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)

    final = Probe.model_validate(builder.compile().invoke(Probe()))

    assert [trace.node for trace in final.traces] == ["first", "second"]
