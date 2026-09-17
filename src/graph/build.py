"""Assemble the thirteen nodes and four conditional edges into one graph.

The node names here are the strings that appear in `ScreeningResult.path_taken`, in
the Mermaid diagram on slide 1, and in the branch-traffic table. They must stay
identical to the names in spec section 4; renaming one silently invalidates every
measurement taken before the rename.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.contracts.ablations import Ablations
from src.contracts.rubric import JDRubric
from src.contracts.screening import ScreeningResult
from src.contracts.state import ScreeningState
from src.graph.decide import decide, rank
from src.graph.extract import make_extract_node, make_repair_node
from src.graph.ingest import guard, ingest, quarantine
from src.graph.routes import (
    route_gray_zone,
    route_guard,
    route_must_have,
    route_repair,
)
from src.graph.rubric_nodes import (
    DERIVED_RUBRIC_DIR,
    make_load_rubric_node,
    must_have_check,
    reject_fast,
)
from src.graph.scoring import aggregate, make_deep_review_node, make_score_criteria_node

NODE_NAMES: tuple[str, ...] = (
    "ingest",
    "guard",
    "quarantine",
    "extract",
    "repair",
    "load_rubric",
    "must_have_check",
    "reject_fast",
    "score_criteria",
    "aggregate",
    "deep_review",
    "decide",
    "rank",
)


def build_graph(
    llm: Any | None = None,
    *,
    today: date | None = None,
    derived_dir: Path | str = DERIVED_RUBRIC_DIR,
) -> Any:
    """Wire and compile the screening graph.

    `llm` is anything with `parse(*, system, user, schema)`. It is left optional so a
    caller can build the real client lazily; passing a stub is how every offline test
    exercises the whole graph.
    """
    if llm is None:
        from src.llm.client import StructuredLLM

        llm = StructuredLLM()

    builder = StateGraph(ScreeningState)
    builder.add_node("ingest", ingest)
    builder.add_node("guard", guard)
    builder.add_node("quarantine", quarantine)
    builder.add_node("extract", make_extract_node(llm, today=today))
    builder.add_node("repair", make_repair_node(llm, today=today))
    builder.add_node("load_rubric", make_load_rubric_node(llm, derived_dir=derived_dir))
    builder.add_node("must_have_check", must_have_check)
    builder.add_node("reject_fast", reject_fast)
    builder.add_node("score_criteria", make_score_criteria_node(llm))
    builder.add_node("aggregate", aggregate)
    builder.add_node("deep_review", make_deep_review_node(llm))
    builder.add_node("decide", decide)
    builder.add_node("rank", rank)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "guard")

    # Conditional edge 1 of 4.
    builder.add_conditional_edges(
        "guard", route_guard, {"quarantine": "quarantine", "extract": "extract"}
    )
    builder.add_edge("quarantine", END)

    # Conditional edge 2 of 4 -- the loop, capped by `max_repair_attempts`.
    repair_targets = {"repair": "repair", "load_rubric": "load_rubric"}
    builder.add_conditional_edges("extract", route_repair, repair_targets)
    builder.add_conditional_edges("repair", route_repair, repair_targets)

    builder.add_edge("load_rubric", "must_have_check")

    # Conditional edge 3 of 4.
    builder.add_conditional_edges(
        "must_have_check",
        route_must_have,
        {"reject_fast": "reject_fast", "score_criteria": "score_criteria"},
    )
    builder.add_edge("reject_fast", END)

    builder.add_edge("score_criteria", "aggregate")

    # Conditional edge 4 of 4.
    builder.add_conditional_edges(
        "aggregate", route_gray_zone, {"deep_review": "deep_review", "decide": "decide"}
    )
    builder.add_edge("deep_review", "decide")
    builder.add_edge("decide", "rank")
    builder.add_edge("rank", END)

    return builder.compile()


def screen(
    cv_text: str,
    jd_text: str,
    *,
    llm: Any | None = None,
    rubric: JDRubric | None = None,
    today: date | None = None,
    derived_dir: Path | str = DERIVED_RUBRIC_DIR,
    ablations: Ablations = Ablations(),
) -> ScreeningResult:
    """Screen one CV against one job description.

    `invoke` returns a plain dict, not a `ScreeningState`, so the output is
    re-validated before anything reads a field off it.

    `build_graph` is deliberately not given the ablations: the switches arrive with
    the state at `invoke`, so one compiled graph serves every configuration.
    """
    graph = build_graph(llm, today=today, derived_dir=derived_dir)
    output = graph.invoke(
        ScreeningState(
            cv_text=cv_text, jd_text=jd_text, rubric=rubric, ablations=ablations
        )
    )
    final = ScreeningState.model_validate(output)
    if final.result is None:
        raise RuntimeError(f"the graph ended without a result; path={final.path_taken}")
    return final.result


def graph_mermaid() -> str:
    """Mermaid source for slide 1. Conditional edges come out dotted (`-.->`)."""

    class _Unused:
        def parse(self, **_kwargs):  # pragma: no cover - never called for a diagram
            raise AssertionError("drawing the graph must not call the model")

    return build_graph(_Unused()).get_graph().draw_mermaid()
