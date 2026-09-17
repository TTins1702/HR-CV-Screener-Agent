from langgraph.graph import END, START, StateGraph

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.state import ScreeningState


def test_state_starts_empty_apart_from_the_inputs():
    state = ScreeningState(cv_text="cv", jd_text="jd")

    assert state.rubric is None
    assert state.profile is None
    assert state.result is None
    assert state.criterion_scores == []
    assert state.repair_attempts == 0
    assert state.max_repair_attempts == 2
    assert state.quarantined is False
    assert state.injection_flags == []
    assert state.path_taken == []


def test_state_carries_a_rubric():
    rubric = JDRubric(
        job_title="Backend Engineer",
        criteria=[Criterion(id="python", description="Python", weight=1.0)],
    )
    state = ScreeningState(cv_text="cv", jd_text="jd", rubric=rubric)

    assert state.rubric is not None
    assert state.rubric.criteria[0].id == "python"


def test_path_taken_accumulates_across_nodes_without_the_caller_rebuilding_it():
    def first(state: ScreeningState) -> dict:
        return {"path_taken": ["first"]}

    def second(state: ScreeningState) -> dict:
        return {"path_taken": ["second"]}

    builder = StateGraph(ScreeningState)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)

    output = builder.compile().invoke(ScreeningState(cv_text="cv", jd_text="jd"))
    final = ScreeningState.model_validate(output)

    assert final.path_taken == ["first", "second"]
    assert final.cv_text == "cv"


def test_invoke_returns_a_plain_dict_so_callers_must_revalidate():
    builder = StateGraph(ScreeningState)
    builder.add_node("noop", lambda state: {"path_taken": ["noop"]})
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)

    output = builder.compile().invoke(ScreeningState(cv_text="cv", jd_text="jd"))

    assert isinstance(output, dict)
    assert not isinstance(output, ScreeningState)


def test_state_carries_the_new_graph_fields():
    state = ScreeningState(cv_text="cv", jd_text="jd")

    assert state.scorecard is None
    assert state.blocking_must_haves == []
    assert state.node_traces == []
    assert not hasattr(state, "visit")
