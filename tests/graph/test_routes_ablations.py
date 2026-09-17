from src.contracts.ablations import Ablations
from src.contracts.rubric import Criterion, JDRubric
from src.contracts.screening import FitLabel
from src.contracts.state import ScreeningState
from src.contracts.tools import Scorecard
from src.graph.routes import route_gray_zone, route_guard, route_must_have


def state(**kwargs) -> ScreeningState:
    return ScreeningState(cv_text="cv", jd_text="jd", **kwargs)


def test_guard_still_quarantines_when_the_defence_is_on():
    assert route_guard(state(quarantined=True)) == "quarantine"


def test_guard_lets_a_poisoned_cv_through_when_the_defence_is_off():
    off = Ablations(guard=False)

    assert route_guard(state(quarantined=True, ablations=off)) == "extract"


def test_the_must_have_gate_blocks_when_it_is_on():
    assert route_must_have(state(blocking_must_haves=["c1"])) == "reject_fast"


def test_the_must_have_gate_scores_everyone_when_it_is_off():
    off = Ablations(must_have_gate=False)

    assert (
        route_must_have(state(blocking_must_haves=["c1"], ablations=off))
        == "score_criteria"
    )


def test_the_gray_zone_branch_can_be_switched_off_without_touching_the_rubric():
    card = Scorecard(overall_score=0.69, label=FitLabel.POTENTIAL_FIT, in_gray_zone=True)
    off = Ablations(gray_zone=False)

    assert route_gray_zone(state(scorecard=card)) == "deep_review"
    assert route_gray_zone(state(scorecard=card, ablations=off)) == "decide"


def test_switching_the_gray_zone_off_still_records_that_the_row_was_in_it():
    """The scorecard keeps the truth; only the routing ignores it.

    This is what lets the report count how many rows *would* have had a second
    look without paying for one.
    """
    card = Scorecard(overall_score=0.69, label=FitLabel.POTENTIAL_FIT, in_gray_zone=True)
    off = Ablations(gray_zone=False)

    routed = state(scorecard=card, ablations=off)

    assert routed.scorecard.in_gray_zone is True
    assert route_gray_zone(routed) == "decide"


def test_a_rubric_still_drives_the_gate_when_ablations_are_default():
    rubric = JDRubric(
        job_title="Backend Engineer",
        criteria=[Criterion(id="c1", description="Python", weight=1.0, must_have=True)],
    )

    assert route_must_have(state(rubric=rubric, blocking_must_haves=[])) == "score_criteria"
