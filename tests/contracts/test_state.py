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


def test_visit_records_the_node_in_order():
    state = ScreeningState(cv_text="cv", jd_text="jd")

    state.visit("guard").visit("extract")

    assert state.path_taken == ["guard", "extract"]


def test_state_carries_a_rubric():
    rubric = JDRubric(
        job_title="Backend Engineer",
        criteria=[Criterion(id="python", description="Python", weight=1.0)],
    )
    state = ScreeningState(cv_text="cv", jd_text="jd", rubric=rubric)

    assert state.rubric is not None
    assert state.rubric.criteria[0].id == "python"


def test_state_works_as_a_langgraph_state_schema():
    def probe(state: ScreeningState) -> dict:
        return {"path_taken": [*state.path_taken, "probe"]}

    builder = StateGraph(ScreeningState)
    builder.add_node("probe", probe)
    builder.add_edge(START, "probe")
    builder.add_edge("probe", END)
    graph = builder.compile()

    output = graph.invoke(ScreeningState(cv_text="cv", jd_text="jd"))
    final = (
        output
        if isinstance(output, ScreeningState)
        else ScreeningState.model_validate(output)
    )

    assert final.path_taken == ["probe"]
    assert final.cv_text == "cv"
