from src.contracts.rubric import Criterion, JDRubric
from src.contracts.screening import CandidateProfile
from src.contracts.state import ScreeningState
from src.graph.scoring import (
    RawScore,
    RawScores,
    deterministic_skill_evidence,
    make_score_criteria_node,
)
from tests.graph.fixtures.poisoned import CLEAN_CV
from tests.graph.stub import StubLLM

RUBRIC = JDRubric(
    job_title="Backend Engineer",
    criteria=[
        Criterion(
            id="lang", description="Python in production", weight=0.4, kind="skill",
            skill_terms=["Python"],
        ),
        Criterion(
            id="seniority", description="At least 7 years of experience", weight=0.3,
            kind="experience_years",
        ),
        Criterion(id="domain", description="Payment systems", weight=0.3, kind="domain"),
    ],
)


def state(rubric: JDRubric = RUBRIC, years: float | None = 3.5) -> ScreeningState:
    before = ScreeningState(cv_text=CLEAN_CV, jd_text="Backend Engineer wanted.")
    before.rubric = rubric
    before.profile = CandidateProfile(
        raw_text=CLEAN_CV,
        skills=["Python"],
        total_experience_years=years,
        llm_declared_years=2.0,
        extraction_confidence=0.9,
    )
    return before


def scores(*items: RawScore) -> RawScores:
    return RawScores(scores=list(items))


def test_skill_terms_are_found_in_the_cv_before_the_model_is_asked():
    hits = deterministic_skill_evidence(CLEAN_CV, RUBRIC)

    assert "lang" in hits
    assert hits["lang"][0].quote.lower().startswith("python")
    assert CLEAN_CV[hits["lang"][0].start : hits["lang"][0].end] == hits["lang"][0].quote


def test_a_skill_absent_from_the_cv_produces_no_hit():
    rubric = JDRubric(
        job_title="x",
        criteria=[
            Criterion(id="infra", description="k8s", weight=1.0, kind="skill",
                      skill_terms=["Kubernetes"])
        ],
    )

    assert deterministic_skill_evidence(CLEAN_CV, rubric) == {}


def test_non_skill_criteria_are_not_searched_for():
    assert "domain" not in deterministic_skill_evidence(CLEAN_CV, RUBRIC)


def test_every_stored_quote_slices_back_out_of_the_cv():
    node = make_score_criteria_node(
        StubLLM([scores(
            RawScore(criterion_id="lang", score=0.9,
                     quotes=["Built payment APIs in Python and PostgreSQL"], reasoning="r"),
            RawScore(criterion_id="domain", score=0.8,
                     quotes=["payment APIs. Built in Python"], reasoning="r"),
        )])
    )

    result = node(state())

    stored = [e for s in result["criterion_scores"] for e in s.evidence]
    assert stored
    for evidence in stored:
        assert CLEAN_CV[evidence.start : evidence.end] == evidence.quote


def test_a_quote_the_model_invented_is_dropped_not_stored():
    node = make_score_criteria_node(
        StubLLM([scores(
            RawScore(criterion_id="domain", score=0.5,
                     quotes=["led a team of forty engineers in Antarctica"], reasoning="r"),
        )])
    )

    update = node(state())

    domain = next(s for s in update["criterion_scores"] if s.criterion_id == "domain")
    assert domain.evidence == []
    assert "quotes_dropped=1" in update["node_traces"][0].note


def test_a_fuzzy_quote_survives_the_glued_text():
    node = make_score_criteria_node(
        StubLLM([scores(
            RawScore(criterion_id="domain", score=0.7,
                     quotes=["payment APIs. Built in Python"], reasoning="r"),
        )])
    )

    update = node(state())

    domain = next(s for s in update["criterion_scores"] if s.criterion_id == "domain")
    assert len(domain.evidence) == 1
    assert domain.evidence[0].score < 1.0


def test_the_experience_score_comes_from_the_tool_not_the_model():
    node = make_score_criteria_node(
        StubLLM([scores(
            RawScore(criterion_id="seniority", score=1.0, quotes=[], reasoning="looks senior"),
        )])
    )

    update = node(state(years=3.5))

    seniority = next(s for s in update["criterion_scores"] if s.criterion_id == "seniority")
    assert seniority.score == 0.5  # 3.5 years against a stated requirement of 7
    assert seniority.tool_used == "calculate_experience"
    assert "3.50 years" in seniority.reasoning


def test_the_experience_score_is_capped_at_one():
    node = make_score_criteria_node(
        StubLLM([scores(RawScore(criterion_id="seniority", score=0.1, quotes=[], reasoning="r"))])
    )

    update = node(state(years=30.0))

    seniority = next(s for s in update["criterion_scores"] if s.criterion_id == "seniority")
    assert seniority.score == 1.0


def test_a_criterion_the_model_invented_is_ignored():
    node = make_score_criteria_node(
        StubLLM([scores(RawScore(criterion_id="not_in_rubric", score=1.0, quotes=[], reasoning="r"))])
    )

    update = node(state())

    assert all(s.criterion_id != "not_in_rubric" for s in update["criterion_scores"])


def test_scores_come_back_in_rubric_order_whatever_order_the_model_used():
    node = make_score_criteria_node(
        StubLLM([scores(
            RawScore(criterion_id="domain", score=0.5, quotes=[], reasoning="r"),
            RawScore(criterion_id="lang", score=0.9, quotes=[], reasoning="r"),
        )])
    )

    update = node(state())

    assert [s.criterion_id for s in update["criterion_scores"]] == ["lang", "seniority", "domain"]


def test_an_out_of_range_score_is_clamped():
    node = make_score_criteria_node(
        StubLLM([scores(RawScore(criterion_id="lang", score=1.7, quotes=[], reasoning="r"))])
    )

    update = node(state())

    assert next(s for s in update["criterion_scores"] if s.criterion_id == "lang").score == 1.0


def test_the_model_is_shown_the_criteria_the_weights_and_the_tool_hits():
    stub = StubLLM([scores(RawScore(criterion_id="lang", score=1.0, quotes=[], reasoning="r"))])

    make_score_criteria_node(stub)(state())

    prompt = stub.calls[0]["user"]
    assert "lang (weight 0.40" in prompt
    assert "already found in the resume" in prompt
    assert CLEAN_CV in prompt


def test_evidence_is_not_duplicated_when_the_model_quotes_what_the_tool_found():
    stub = StubLLM([scores(RawScore(criterion_id="lang", score=1.0, quotes=["Python"], reasoning="r"))])

    update = make_score_criteria_node(stub)(state())

    lang = next(s for s in update["criterion_scores"] if s.criterion_id == "lang")
    spans = {(e.start, e.end) for e in lang.evidence}
    assert len(spans) == len(lang.evidence)


from src.contracts.screening import CriterionScore, Evidence, FitLabel
from src.graph.routes import route_gray_zone
from src.graph.scoring import (
    RawRevision,
    RawRevisions,
    aggregate,
    make_deep_review_node,
    needs_second_look,
)

GRAY_RUBRIC = JDRubric(
    job_title="Backend Engineer",
    criteria=[
        Criterion(id="a", description="A", weight=0.5),
        Criterion(id="b", description="B", weight=0.5),
    ],
    good_fit_threshold=0.70,
    potential_fit_threshold=0.40,
    gray_zone_margin=0.05,
)


def scored(**by_id: float) -> ScreeningState:
    before = state(rubric=GRAY_RUBRIC)
    before.criterion_scores = [
        CriterionScore(criterion_id=key, score=value, reasoning="r")
        for key, value in by_id.items()
    ]
    return before


def test_aggregate_delegates_to_the_deterministic_tool():
    update = aggregate(scored(a=1.0, b=0.0))

    assert update["path_taken"] == ["aggregate"]
    assert update["scorecard"].overall_score == 0.5
    assert update["scorecard"].label is FitLabel.POTENTIAL_FIT


def test_aggregate_records_the_branch_signals_in_its_trace():
    update = aggregate(scored(a=1.0, b=0.44))

    assert "gray=True" in update["node_traces"][0].note


def test_a_score_far_from_a_threshold_is_not_in_the_gray_zone():
    assert aggregate(scored(a=1.0, b=1.0))["scorecard"].in_gray_zone is False


def test_route_gray_zone_sends_a_borderline_candidate_to_deep_review():
    before = scored(a=1.0, b=0.44)
    before.scorecard = aggregate(before)["scorecard"]

    assert route_gray_zone(before) == "deep_review"


def test_route_gray_zone_sends_a_clear_candidate_straight_to_decide():
    before = scored(a=1.0, b=1.0)
    before.scorecard = aggregate(before)["scorecard"]

    assert route_gray_zone(before) == "decide"


def test_route_gray_zone_is_off_when_the_margin_is_zero():
    rubric = GRAY_RUBRIC.model_copy(update={"gray_zone_margin": 0.0})
    before = state(rubric=rubric)
    before.criterion_scores = [
        CriterionScore(criterion_id="a", score=1.0, reasoning="r"),
        CriterionScore(criterion_id="b", score=0.44, reasoning="r"),
    ]
    before.scorecard = aggregate(before)["scorecard"]

    assert route_gray_zone(before) == "decide"


def test_a_criterion_with_no_evidence_needs_a_second_look():
    thin = CriterionScore(criterion_id="a", score=0.9, reasoning="r")

    assert [s.criterion_id for s in needs_second_look([thin])] == ["a"]


def test_a_middling_criterion_needs_a_second_look():
    middling = CriterionScore(
        criterion_id="a", score=0.5, reasoning="r",
        evidence=[Evidence(quote="Python", start=0, end=6)],
    )

    assert [s.criterion_id for s in needs_second_look([middling])] == ["a"]


def test_a_confident_well_evidenced_criterion_is_left_alone():
    settled = CriterionScore(
        criterion_id="a", score=0.95, reasoning="r",
        evidence=[Evidence(quote="Python", start=0, end=6)],
    )

    assert needs_second_look([settled]) == []


def test_deep_review_replaces_only_the_scores_it_revised():
    before = scored(a=0.5, b=0.44)
    before.scorecard = aggregate(before)["scorecard"]
    stub = StubLLM([RawRevisions(revisions=[RawRevision(criterion_id="a", score=0.9, reasoning="second look")])])

    update = make_deep_review_node(stub)(before)

    revised = {s.criterion_id: s for s in update["criterion_scores"]}
    assert revised["a"].score == 0.9
    assert revised["a"].tool_used == "deep_review"
    assert revised["b"].score == 0.44


def test_deep_review_re_aggregates_so_decide_reads_a_fresh_scorecard():
    before = scored(a=0.5, b=0.44)
    before.scorecard = aggregate(before)["scorecard"]
    stub = StubLLM([RawRevisions(revisions=[RawRevision(criterion_id="a", score=1.0, reasoning="r")])])

    update = make_deep_review_node(stub)(before)

    assert update["scorecard"].overall_score == 0.72
    assert update["path_taken"] == ["deep_review"]


def test_deep_review_ignores_a_revision_for_a_criterion_that_does_not_exist():
    before = scored(a=0.5, b=0.44)
    before.scorecard = aggregate(before)["scorecard"]
    stub = StubLLM([RawRevisions(revisions=[RawRevision(criterion_id="ghost", score=1.0, reasoning="r")])])

    update = make_deep_review_node(stub)(before)

    assert {s.criterion_id for s in update["criterion_scores"]} == {"a", "b"}


def test_deep_review_does_not_call_the_model_when_nothing_is_thin():
    before = state(rubric=GRAY_RUBRIC)
    before.criterion_scores = [
        CriterionScore(criterion_id="a", score=0.95, reasoning="r",
                       evidence=[Evidence(quote="Python", start=0, end=6)]),
        CriterionScore(criterion_id="b", score=0.02, reasoning="r",
                       evidence=[Evidence(quote="Python", start=0, end=6)]),
    ]
    before.scorecard = aggregate(before)["scorecard"]
    stub = StubLLM([])

    update = make_deep_review_node(stub)(before)

    assert stub.calls == []
    assert "skipped" in update["node_traces"][0].note
