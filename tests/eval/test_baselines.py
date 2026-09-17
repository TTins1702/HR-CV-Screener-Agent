import pytest

from eval.baselines import NaiveVerdict, naive_baseline, rubric_baseline, tfidf_baseline
from src.contracts.screening import FitLabel

ROWS = [
    {
        "resume_text": "Six years of Python, Django and PostgreSQL in production.",
        "job_description_text": "Backend engineer. Python, Django, PostgreSQL.",
        "label": "Good Fit",
    },
    {
        "resume_text": "Fifteen years of restaurant floor management and catering.",
        "job_description_text": "Backend engineer. Python, Django, PostgreSQL.",
        "label": "No Fit",
    },
    {
        "resume_text": "Three years of Python scripting, some SQL, no web frameworks.",
        "job_description_text": "Backend engineer. Python, Django, PostgreSQL.",
        "label": "Potential Fit",
    },
]


class FixedLLM:
    """Answers verdicts with the same label, rubrics with a fixed rubric.

    The rubric baseline asks for two different schemas -- `RawRubric` to derive the
    rubric and `NaiveVerdict` to judge with it -- so a stub that only knows how to
    build a verdict cannot drive it.
    """

    def __init__(self, label=FitLabel.GOOD_FIT):
        self.label = label
        self.calls = 0

    def parse(self, *, system, user, schema):
        from src.contracts.trace import LLMUsage
        from src.graph.rubric_nodes import RawCriterion, RawRubric

        self.calls += 1
        usage = LLMUsage(prompt_tokens=400, completion_tokens=8)
        if schema is RawRubric:
            return (
                RawRubric(
                    job_title="Backend Engineer",
                    criteria=[
                        RawCriterion(id="lang", description="Python in production",
                                     weight=0.6, must_have=True, kind="skill",
                                     skill_terms=["Python"]),
                        RawCriterion(id="db", description="Relational databases",
                                     weight=0.4, must_have=False, kind="skill",
                                     skill_terms=["PostgreSQL"]),
                    ],
                ),
                usage,
            )
        return schema(label=self.label), usage


def test_the_naive_baseline_spends_exactly_one_call_per_row():
    llm = FixedLLM()

    record = naive_baseline(ROWS[0], 0, llm=llm)

    assert llm.calls == 1
    assert record.system == "baseline_naive"
    assert record.config == "single_call"
    assert record.true_label == "Good Fit"
    assert record.predicted_label == "Good Fit"
    assert record.prompt_tokens == 400
    assert record.path_taken == []


def test_the_naive_verdict_schema_only_admits_the_three_dataset_labels():
    assert NaiveVerdict(label=FitLabel.NO_FIT).label is FitLabel.NO_FIT

    with pytest.raises(Exception):
        NaiveVerdict(label="Maybe")


def test_the_rubric_baseline_is_also_one_call_but_a_bigger_one(tmp_path):
    llm = FixedLLM(FitLabel.POTENTIAL_FIT)

    record = rubric_baseline(ROWS[0], 0, llm=llm, derived_dir=tmp_path)

    # One call to derive the rubric, one to judge with it.
    assert llm.calls == 2
    assert record.system == "baseline_rubric"
    assert record.predicted_label == "Potential Fit"


def test_the_rubric_baseline_reuses_a_derived_rubric_rather_than_paying_twice(tmp_path):
    llm = FixedLLM(FitLabel.POTENTIAL_FIT)

    rubric_baseline(ROWS[0], 0, llm=llm, derived_dir=tmp_path)
    calls_after_first = llm.calls
    rubric_baseline(ROWS[0], 1, llm=llm, derived_dir=tmp_path)

    assert llm.calls == calls_after_first + 1


def test_tfidf_costs_nothing_and_still_answers_every_row():
    records = tfidf_baseline(ROWS)

    assert len(records) == 3
    assert all(record.system == "baseline_tfidf" for record in records)
    assert all(record.llm_calls == 0 and record.tokens == 0 for record in records)
    assert all(record.predicted_label is not None for record in records)


def test_tfidf_ranks_the_matching_resume_above_the_unrelated_one():
    """The thresholds may be crude, but the ordering must not be nonsense.

    If cosine similarity cannot separate a Django CV from a catering CV against a
    Django job description, the baseline is broken rather than merely weak.
    """
    records = tfidf_baseline(ROWS)

    assert records[0].overall_score > records[1].overall_score


def test_every_baseline_emits_the_same_record_shape_the_agent_does():
    from eval.records import RowRecord

    records = [
        naive_baseline(ROWS[0], 0, llm=FixedLLM()),
        *tfidf_baseline(ROWS[:1]),
    ]

    assert all(isinstance(record, RowRecord) for record in records)
    assert len({record.row_index for record in records}) == 1
