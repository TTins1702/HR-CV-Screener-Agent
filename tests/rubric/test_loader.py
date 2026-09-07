import pytest

from src.contracts.rubric import Criterion, JDRubric
from src.rubric.loader import load_rubric, save_rubric

VALID_YAML = """
job_title: Backend Engineer
good_fit_threshold: 0.72
potential_fit_threshold: 0.45
criteria:
  - id: python
    description: Production Python experience
    weight: 0.6
    must_have: true
    kind: skill
  - id: years
    description: At least 3 years of backend experience
    weight: 0.4
    kind: experience_years
"""


def test_load_rubric_parses_a_valid_file(tmp_path):
    path = tmp_path / "rubric.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")

    rubric = load_rubric(path)

    assert rubric.job_title == "Backend Engineer"
    assert rubric.good_fit_threshold == 0.72
    assert [c.id for c in rubric.criteria] == ["python", "years"]
    assert [c.id for c in rubric.must_haves()] == ["python"]
    assert rubric.criteria[1].kind == "experience_years"


def test_load_rubric_raises_on_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rubric(tmp_path / "does_not_exist.yaml")


def test_load_rubric_rejects_yaml_that_is_not_a_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_rubric(path)


def test_load_rubric_propagates_contract_violations(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "job_title: X\ncriteria:\n  - id: a\n    description: a\n    weight: 0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sum to 1.0"):
        load_rubric(path)


def test_save_then_load_round_trips(tmp_path):
    original = JDRubric(
        job_title="Frontend Engineer",
        criteria=[
            Criterion(id="react", description="React", weight=0.7, kind="skill"),
            Criterion(id="css", description="CSS", weight=0.3, kind="skill"),
        ],
    )
    path = tmp_path / "out.yaml"

    save_rubric(original, path)
    reloaded = load_rubric(path)

    assert reloaded == original


def test_the_committed_backend_rubric_loads():
    rubric = load_rubric("data/rubrics/backend_engineer.yaml")

    assert rubric.job_title
    assert len(rubric.criteria) >= 4
    assert rubric.must_haves()
