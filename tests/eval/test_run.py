import json
from datetime import date

import pytest

from eval.records import read_records
from eval.run import EVAL_TODAY, load_split, records_path, run_split
from src.contracts.ablations import Ablations
from src.contracts.trace import LLMUsage
from src.graph.extract import RawExtraction, RawPeriod
from src.graph.rubric_nodes import RawCriterion, RawRubric
from src.graph.scoring import RawRevision, RawRevisions, RawScore, RawScores


class SchemaStub:
    """Answers by schema rather than by position, so it survives a loop.

    `parse` is the whole surface the graph uses; anything else reaching for an
    attribute here is a design regression and will fail loudly.
    """

    def __init__(self) -> None:
        self.calls = 0

    def parse(self, *, system, user, schema):
        self.calls += 1
        usage = LLMUsage(prompt_tokens=100, completion_tokens=20)
        if schema is RawExtraction:
            return (
                RawExtraction(
                    skills=["Python", "Django"],
                    work_periods=[
                        RawPeriod(title="Backend Engineer", company="Acme",
                                  start="2019-06", end="2022-12")
                    ],
                    degrees=["B.S. Computer Science"],
                    certifications=[],
                    total_experience_years=3.5,
                    extraction_confidence=0.95,
                ),
                usage,
            )
        if schema is RawRubric:
            return (
                RawRubric(
                    job_title="Backend Engineer",
                    criteria=[
                        RawCriterion(id="lang", description="Python in production",
                                     weight=0.5, must_have=True, kind="skill",
                                     skill_terms=["Python"]),
                        RawCriterion(id="seniority",
                                     description="At least 3 years of experience",
                                     weight=0.5, must_have=False,
                                     kind="experience_years", skill_terms=[]),
                    ],
                ),
                usage,
            )
        if schema is RawScores:
            return (
                RawScores(scores=[
                    RawScore(criterion_id="lang", score=0.9,
                             quotes=["Python"], reasoning="r"),
                    RawScore(criterion_id="seniority", score=0.5,
                             quotes=[], reasoning="r"),
                ]),
                usage,
            )
        if schema is RawRevisions:
            # These scores land at 0.70 overall, which is inside the default gray
            # zone, so `deep_review` runs and asks for this one too.
            return (
                RawRevisions(revisions=[
                    RawRevision(criterion_id="seniority", score=0.55, reasoning="r"),
                ]),
                usage,
            )
        raise AssertionError(f"SchemaStub has no answer for {schema.__name__}")


ROWS = [
    {
        "resume_text": "Professional Summary 6 years of Python and Django experience.",
        "job_description_text": "Backend engineer. 3 years Python required.",
        "label": "Good Fit",
    },
    {
        "resume_text": "Professional Summary Retail floor manager, 8 years.",
        "job_description_text": "Backend engineer. 3 years Python required.",
        "label": "No Fit",
    },
]


def test_the_eval_date_is_pinned_to_the_one_the_day_3_numbers_were_taken_on():
    """Wall-clock dates silently invalidate every comparison with Day 3.

    `calculate_experience` measures against `today`; let it drift and the same CV
    scores differently next week for no reason anybody recorded.
    """
    from scripts.measure_branch_traffic import DEFAULT_TODAY

    assert EVAL_TODAY == date(2026, 9, 7)
    assert EVAL_TODAY == DEFAULT_TODAY


def test_a_split_is_loaded_as_dicts_and_can_be_truncated(tmp_path):
    path = tmp_path / "split.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")

    assert len(load_split(path)) == 2
    assert len(load_split(path, limit=1)) == 1
    assert load_split(path)[0]["label"] == "Good Fit"


def test_one_record_per_row_carrying_the_ground_truth_label(tmp_path):
    records = run_split(
        ROWS, llm=SchemaStub(), ablations=Ablations(),
        system="agent", derived_dir=tmp_path,
    )

    assert [record.row_index for record in records] == [0, 1]
    assert [record.true_label for record in records] == ["Good Fit", "No Fit"]
    assert all(record.system == "agent" for record in records)
    assert all(record.config == "shipped" for record in records)
    assert all(record.predicted_label is not None for record in records)


def test_the_configuration_label_travels_with_the_records(tmp_path):
    records = run_split(
        ROWS, llm=SchemaStub(), ablations=Ablations(must_have_gate=False),
        system="agent", derived_dir=tmp_path,
    )

    assert all(record.config == "no_must_have_gate" for record in records)


def test_a_row_that_raises_is_recorded_as_an_error_and_the_run_continues(tmp_path):
    class Exploding(SchemaStub):
        def parse(self, *, system, user, schema):
            if "Retail floor manager" in user:
                raise RuntimeError("boom")
            return super().parse(system=system, user=user, schema=schema)

    records = run_split(
        ROWS, llm=Exploding(), ablations=Ablations(),
        system="agent", derived_dir=tmp_path,
    )

    assert len(records) == 2
    assert records[0].error is None
    assert records[1].error is not None
    assert "boom" in records[1].error
    assert records[1].predicted_label is None


def test_progress_is_reported_through_the_callback_not_through_print(tmp_path):
    seen = []

    run_split(
        ROWS,
        llm=SchemaStub(),
        ablations=Ablations(),
        system="agent",
        derived_dir=tmp_path,
        on_row=lambda index, total, record: seen.append((index, total)),
    )

    assert seen == [(1, 2), (2, 2)]


def test_the_records_path_names_the_system_and_the_configuration(tmp_path):
    path = records_path(tmp_path, "agent", "no_must_have_gate")

    assert path.name == "agent__no_must_have_gate.jsonl"
    assert path.parent == tmp_path


def test_a_run_never_writes_into_the_real_derived_rubric_directory(tmp_path):
    """`derived_dir` is a parameter so a test cannot pollute `data/rubrics/derived/`.

    Both rows share a job description, so the second row must reuse the YAML the
    first one wrote instead of paying to derive it again.
    """
    stub = SchemaStub()

    run_split(ROWS, llm=stub, ablations=Ablations(), system="agent", derived_dir=tmp_path)

    written = list(tmp_path.glob("*.yaml"))
    assert len(written) == 1


def test_records_written_by_a_run_can_be_read_straight_back(tmp_path):
    from eval.records import write_records

    records = run_split(
        ROWS, llm=SchemaStub(), ablations=Ablations(),
        system="agent", derived_dir=tmp_path / "rubrics",
    )
    path = records_path(tmp_path, "agent", "shipped")
    write_records(records, path)

    assert read_records(path) == records
