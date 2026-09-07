import pytest

from src.tools.evidence import search_evidence

CV = (
    "Professional Summary4 years analytic experience in the cost accounting "
    "department.Extracts, manipulates, validates and submits data from numerous "
    "sources. 13 years of professional experience with Excel for varied purposes."
)


def test_finds_an_exact_quote_and_scores_it_one():
    hits = search_evidence(CV, "13 years of professional experience with Excel")
    assert len(hits) == 1
    assert hits[0].score == 1.0
    assert hits[0].quote == "13 years of professional experience with Excel"


def test_returned_offsets_slice_the_original_text():
    hits = search_evidence(CV, "cost accounting department")
    assert CV[hits[0].start : hits[0].end] == hits[0].quote


def test_finds_a_quote_that_crosses_a_glued_sentence_boundary():
    # An LLM quoting the CV writes the space the dataset dropped.
    query = "cost accounting department. Extracts"
    assert CV.lower().find(query.lower()) == -1  # the naive approach fails
    hits = search_evidence(CV, query)
    assert hits, "the normalizer must recover a span across the glued boundary"
    assert hits[0].score == 1.0
    assert hits[0].quote == "cost accounting department.Extracts"


def test_matching_ignores_case_and_extra_whitespace_in_the_query():
    hits = search_evidence(CV, "  PROFESSIONAL   experience   with   excel  ")
    assert hits
    assert hits[0].score == 1.0


def test_falls_back_to_a_fuzzy_match_for_a_paraphrased_quote():
    hits = search_evidence(CV, "thirteen years of professional experience using Excel")
    assert hits
    assert 0.75 <= hits[0].score < 1.0
    assert "professional experience with Excel" in hits[0].quote


def test_returns_nothing_when_the_quote_is_absent():
    assert search_evidence(CV, "Kubernetes cluster administration at scale") == []


def test_finds_every_occurrence_up_to_max_results():
    text = "Python here. Python there. Python everywhere."
    hits = search_evidence(text, "Python", max_results=2)
    assert len(hits) == 2
    assert [h.start for h in hits] == sorted(h.start for h in hits)
    assert hits[0].start != hits[1].start


def test_results_never_overlap_each_other():
    hits = search_evidence(CV, "years of professional experience", max_results=3)
    spans = [(h.start, h.end) for h in hits]
    for (a_start, a_end), (b_start, b_end) in zip(spans, spans[1:]):
        assert a_end <= b_start or b_end <= a_start


def test_a_higher_min_score_rejects_a_weak_fuzzy_match():
    query = "thirteen years of professional experience using Excel"
    assert search_evidence(CV, query, min_score=0.99) == []


def test_is_deterministic():
    first = search_evidence(CV, "manipulates validates and submits data")
    second = search_evidence(CV, "manipulates validates and submits data")
    assert [h.model_dump() for h in first] == [h.model_dump() for h in second]


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_an_empty_query_finds_nothing(query):
    assert search_evidence(CV, query) == []


def test_an_empty_text_finds_nothing():
    assert search_evidence("", "python") == []


def test_max_results_must_be_positive():
    with pytest.raises(ValueError):
        search_evidence(CV, "Excel", max_results=0)
