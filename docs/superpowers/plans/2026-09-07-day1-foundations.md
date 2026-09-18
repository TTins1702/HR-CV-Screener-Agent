# HR CV Screener Agent — Day 1: Contracts, Rubric & Data Foundations

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Execution mode is **inline** — run the tasks sequentially in the current session with a checkpoint after each task; do not dispatch subagents. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Stand up the repository, the pydantic contracts every later node/tool/eval depends on, the YAML rubric format, and reproducible dev/test/demo data splits.

**Architecture:** Everything downstream (LangGraph nodes, the 5 deterministic tools, the eval harness, the Streamlit app) reads and writes a single set of pydantic models. Day 1 freezes those models and the data on disk, so Days 2-5 never renegotiate a field name. Data acquisition is split by source: the scored dataset and the demo JDs come from HuggingFace (public, no credentials); the demo PDF resumes come from Kaggle and are gated behind a credential check that fails loudly today rather than on Day 4.

**Tech Stack:** Python 3.13.12 (miniconda base), pydantic 2.12.4, LangGraph 1.2.1, pandas 3.0.3, huggingface_hub 1.12.0, PyYAML 6.0.3, pytest 9.0.3. `pdfplumber` and `kagglehub` get installed in Task 1.

**Spec:** `docs/spec/2026-09-07-hr-cv-screener-spec.md`

## Global Constraints

- Python 3.13.12 is the interpreter on this machine. Run everything as `python -m ...`. `uv` is NOT installed; use `python -m pip`.
- All code, identifiers, docstrings and Streamlit copy are **English**. Only `docs/spec/*.md` is Vietnamese.
- The three fit labels are the exact strings `"Good Fit"`, `"Potential Fit"`, `"No Fit"` — they must match the dataset byte-for-byte or every eval number is wrong.
- The scored dataset is HF `cnamuangtoun/resume-job-description-fit`, files `train.csv` and `test.csv`, columns `resume_text`, `job_description_text`, `label`.
- Demo JDs are HF `jacob-hugging-face/job-descriptions`, file `training_data.csv`, columns `company_name`, `job_description`, `position_title`, `description_length`, `model_response`.
- Dataset text has whitespace stripped between sentences (`"consulting projects.Proven ability"`). No component may assume word boundaries around punctuation.
- Random seed is `42` everywhere. Sampling must be deterministic.
- Never write raw datasets or generated splits into git. Only code, fixtures and rubrics are committed.
- Tests must run offline by default. Any test that touches the network is marked `@pytest.mark.network` and excluded by the default pytest options.
- The working directory `d:\CN12_2024_2028\NCKH\Agent_Lab\Background\Agent` is NOT yet a git repository. Task 1 initializes it.
- Kaggle credentials are already installed: a `KGAT_...` bearer token at `~/.kaggle/access_token`, verified against `kagglehub.whoami()` (account `ttins1702`). This is Kaggle's current format and replaces `kaggle.json`. Treat the file as a secret: never read it, print it, copy it into `.env`, or commit it.

---

### Task 1: Repository scaffold and environment smoke test

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`, `src/contracts/__init__.py`, `src/rubric/__init__.py`, `src/data/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_environment.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `src` package rooted at the repo root; pytest configured with `pythonpath = ["."]` and a `network` marker that is deselected by default.

- [x] **Step 1: Initialize the git repository**

```bash
git init
git config user.name "TTins1702"
git config user.email "nonametg1702@gmail.com"
```

- [x] **Step 2: Install the two missing packages**

```bash
python -m pip install pdfplumber kagglehub
```

Expected: both report as already satisfied — they were installed ahead of time (pdfplumber 0.11.10, kagglehub 1.0.2). Everything else in the stack is already present in the miniconda base environment. Run the command anyway so the step is verified rather than assumed.

- [x] **Step 3: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.env
.venv/
data/raw/
data/samples/
data/demo/
eval/cache/
outputs/
*.egg-info/
.ipynb_checkpoints/
```

- [x] **Step 4: Create `pyproject.toml`**

```toml
[project]
name = "hr-cv-screener"
version = "0.1.0"
description = "HR CV Screener Agent - LangGraph demo"
requires-python = ">=3.12"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-m 'not network' -q"
markers = [
    "network: test requires internet access (deselected by default)",
]
```

- [x] **Step 5: Create `requirements.txt`**

```text
pydantic>=2.12
langgraph>=1.2
openai>=2.0
pandas>=3.0
huggingface_hub>=1.12
PyYAML>=6.0
python-dotenv>=1.2
tenacity>=9.0
scikit-learn>=1.8
numpy>=2.4
rank-bm25>=0.2
pdfplumber>=0.11
kagglehub>=0.3
streamlit>=1.61
matplotlib>=3.10
pytest>=9.0
```

- [x] **Step 6: Create `.env.example`**

```text
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
SCREENER_SEED=42
```

- [x] **Step 7: Create the empty package markers**

Create these five files, each containing a single docstring line:

`src/__init__.py`:
```python
"""HR CV Screener Agent."""
```

`src/contracts/__init__.py`:
```python
"""Pydantic contracts shared by every component."""
```

`src/rubric/__init__.py`:
```python
"""Rubric loading and serialization."""
```

`src/data/__init__.py`:
```python
"""Dataset acquisition, validation and sampling."""
```

`tests/__init__.py`:
```python
"""Test suite."""
```

- [x] **Step 8: Write the environment test**

`tests/test_environment.py`:
```python
import sys


def test_python_is_at_least_3_12():
    assert sys.version_info >= (3, 12)


def test_core_dependencies_import():
    import langgraph.graph
    import pandas
    import pydantic
    import yaml

    assert pydantic.VERSION.startswith("2.")
    assert hasattr(langgraph.graph, "StateGraph")
    assert hasattr(yaml, "safe_load")
    assert pandas.__version__


def test_src_package_is_importable():
    import src

    assert src.__doc__ == "HR CV Screener Agent."
```

- [x] **Step 9: Run the test to verify it passes**

Run: `python -m pytest tests/test_environment.py -v`
Expected: 3 passed. If `test_src_package_is_importable` fails with `ModuleNotFoundError`, the `pythonpath = ["."]` line in `pyproject.toml` is missing or misspelled.

- [x] **Step 10: Commit**

```bash
git add .gitignore pyproject.toml requirements.txt .env.example src tests
git commit -m "chore: scaffold repo, pytest config and environment smoke test"
```

---

### Task 2: Screening contracts

**Files:**
- Create: `src/contracts/screening.py`
- Create: `tests/contracts/__init__.py`
- Test: `tests/contracts/test_screening.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `FitLabel` — str enum with members `GOOD_FIT = "Good Fit"`, `POTENTIAL_FIT = "Potential Fit"`, `NO_FIT = "No Fit"`.
  - `Evidence(quote: str, start: int, end: int)`
  - `WorkPeriod(title: str, company: str | None, start: date | None, end: date | None)`
  - `CandidateProfile(raw_text, skills, work_periods, degrees, certifications, total_experience_years, extraction_confidence, missing_fields)`
  - `CriterionScore(criterion_id: str, score: float, evidence: list[Evidence], reasoning: str, tool_used: str | None)`
  - `ScreeningResult(overall_score, label, criterion_scores, rejected_reason, path_taken, prompt_tokens, completion_tokens, latency_ms)`

- [x] **Step 1: Write the failing test**

`tests/contracts/__init__.py`:
```python
"""Contract tests."""
```

`tests/contracts/test_screening.py`:
```python
import pytest
from pydantic import ValidationError

from src.contracts.screening import (
    CandidateProfile,
    CriterionScore,
    Evidence,
    FitLabel,
    ScreeningResult,
    WorkPeriod,
)


def test_fit_label_values_match_the_dataset_strings_exactly():
    assert FitLabel.GOOD_FIT.value == "Good Fit"
    assert FitLabel.POTENTIAL_FIT.value == "Potential Fit"
    assert FitLabel.NO_FIT.value == "No Fit"
    assert {label.value for label in FitLabel} == {"Good Fit", "Potential Fit", "No Fit"}


def test_fit_label_parses_from_the_raw_dataset_string():
    assert FitLabel("Potential Fit") is FitLabel.POTENTIAL_FIT


def test_evidence_accepts_a_forward_span():
    evidence = Evidence(quote="7+ years", start=7, end=15)
    assert evidence.end > evidence.start


def test_evidence_rejects_an_empty_quote():
    with pytest.raises(ValidationError):
        Evidence(quote="", start=0, end=5)


def test_evidence_rejects_a_non_forward_span():
    with pytest.raises(ValidationError):
        Evidence(quote="x", start=10, end=10)


def test_candidate_profile_defaults_to_empty_collections():
    profile = CandidateProfile(raw_text="cv", extraction_confidence=0.5)
    assert profile.skills == []
    assert profile.work_periods == []
    assert profile.degrees == []
    assert profile.certifications == []
    assert profile.missing_fields == []
    assert profile.total_experience_years is None


def test_candidate_profile_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        CandidateProfile(raw_text="cv", extraction_confidence=1.5)


def test_work_period_allows_an_open_ended_current_job():
    period = WorkPeriod(title="Backend Engineer", company="Base", start=None, end=None)
    assert period.end is None


def test_criterion_score_rejects_a_score_above_one():
    with pytest.raises(ValidationError):
        CriterionScore(criterion_id="python", score=1.5)


def test_screening_result_carries_the_label_and_defaults_usage_to_zero():
    result = ScreeningResult(overall_score=0.81, label=FitLabel.GOOD_FIT)
    assert result.label is FitLabel.GOOD_FIT
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.latency_ms == 0.0
    assert result.rejected_reason is None
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/contracts/test_screening.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.contracts.screening'`.

- [x] **Step 3: Write the implementation**

`src/contracts/screening.py`:
```python
"""Models describing a candidate, the scoring of one criterion, and the outcome."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class FitLabel(str, Enum):
    """The three outcome classes. Values match the dataset labels byte-for-byte."""

    GOOD_FIT = "Good Fit"
    POTENTIAL_FIT = "Potential Fit"
    NO_FIT = "No Fit"


class Evidence(BaseModel):
    """A verbatim span of the CV that supports a score."""

    quote: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _span_must_be_forward(self) -> "Evidence":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class WorkPeriod(BaseModel):
    """One employment entry. `end is None` means the role is current."""

    title: str
    company: str | None = None
    start: date | None = None
    end: date | None = None


class CandidateProfile(BaseModel):
    """Structured view of one CV, produced by the extract node."""

    raw_text: str
    skills: list[str] = Field(default_factory=list)
    work_periods: list[WorkPeriod] = Field(default_factory=list)
    degrees: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    total_experience_years: float | None = Field(default=None, ge=0.0)
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)


class CriterionScore(BaseModel):
    """The score for a single rubric criterion, with the evidence behind it."""

    criterion_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    reasoning: str = ""
    tool_used: str | None = None


class ScreeningResult(BaseModel):
    """The outcome for one CV-JD pair, including the path the graph took."""

    overall_score: float = Field(ge=0.0, le=1.0)
    label: FitLabel
    criterion_scores: list[CriterionScore] = Field(default_factory=list)
    rejected_reason: str | None = None
    path_taken: list[str] = Field(default_factory=list)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
```

- [x] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/contracts/test_screening.py -v`
Expected: 10 passed.

- [x] **Step 5: Commit**

```bash
git add src/contracts/screening.py tests/contracts
git commit -m "feat: add screening contracts (FitLabel, Evidence, CandidateProfile, CriterionScore, ScreeningResult)"
```

---

### Task 3: Rubric contract, YAML loader, and a real IT rubric

**Files:**
- Create: `src/contracts/rubric.py`
- Create: `src/rubric/loader.py`
- Create: `data/rubrics/backend_engineer.yaml`
- Create: `tests/rubric/__init__.py`
- Test: `tests/contracts/test_rubric.py`
- Test: `tests/rubric/test_loader.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `CriterionKind = Literal["skill", "experience_years", "education", "domain", "other"]`
  - `Criterion(id: str, description: str, weight: float, must_have: bool, kind: CriterionKind)`
  - `JDRubric(job_title: str, criteria: list[Criterion], good_fit_threshold: float, potential_fit_threshold: float)` with `must_haves() -> list[Criterion]`
  - `load_rubric(path: str | Path) -> JDRubric` and `save_rubric(rubric: JDRubric, path: str | Path) -> None`
  - A committed rubric at `data/rubrics/backend_engineer.yaml`, used by the Streamlit demo and by the strong single-call baseline.

- [x] **Step 1: Write the failing contract test**

`tests/contracts/test_rubric.py`:
```python
import pytest
from pydantic import ValidationError

from src.contracts.rubric import Criterion, JDRubric


def _criteria(*weights: float) -> list[Criterion]:
    return [
        Criterion(id=f"c{i}", description=f"criterion {i}", weight=w)
        for i, w in enumerate(weights)
    ]


def test_rubric_accepts_weights_that_sum_to_one():
    rubric = JDRubric(job_title="Backend Engineer", criteria=_criteria(0.5, 0.3, 0.2))
    assert len(rubric.criteria) == 3


def test_rubric_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1.0"):
        JDRubric(job_title="Backend Engineer", criteria=_criteria(0.5, 0.3))


def test_rubric_tolerates_float_rounding_in_the_weight_sum():
    rubric = JDRubric(
        job_title="Backend Engineer",
        criteria=_criteria(1 / 3, 1 / 3, 1 / 3),
    )
    assert len(rubric.criteria) == 3


def test_rubric_rejects_duplicate_criterion_ids():
    duplicated = [
        Criterion(id="python", description="Python", weight=0.5),
        Criterion(id="python", description="Python again", weight=0.5),
    ]
    with pytest.raises(ValidationError, match="unique"):
        JDRubric(job_title="Backend Engineer", criteria=duplicated)


def test_rubric_rejects_thresholds_that_are_not_ordered():
    with pytest.raises(ValidationError, match="greater than"):
        JDRubric(
            job_title="Backend Engineer",
            criteria=_criteria(1.0),
            good_fit_threshold=0.3,
            potential_fit_threshold=0.6,
        )


def test_rubric_requires_at_least_one_criterion():
    with pytest.raises(ValidationError):
        JDRubric(job_title="Backend Engineer", criteria=[])


def test_must_haves_returns_only_the_flagged_criteria():
    rubric = JDRubric(
        job_title="Backend Engineer",
        criteria=[
            Criterion(id="python", description="Python", weight=0.6, must_have=True),
            Criterion(id="k8s", description="Kubernetes", weight=0.4),
        ],
    )
    assert [c.id for c in rubric.must_haves()] == ["python"]


def test_default_thresholds_are_ordered():
    rubric = JDRubric(job_title="Backend Engineer", criteria=_criteria(1.0))
    assert rubric.good_fit_threshold > rubric.potential_fit_threshold
```

- [x] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/contracts/test_rubric.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.contracts.rubric'`.

- [x] **Step 3: Write the rubric contract**

`src/contracts/rubric.py`:
```python
"""The rubric: a JD turned into structured, weighted criteria."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

CriterionKind = Literal["skill", "experience_years", "education", "domain", "other"]

WEIGHT_SUM_TOLERANCE = 1e-6


class Criterion(BaseModel):
    """One requirement lifted out of a job description."""

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)
    must_have: bool = False
    kind: CriterionKind = "other"


class JDRubric(BaseModel):
    """A full scoring rubric for one job description."""

    job_title: str = Field(min_length=1)
    criteria: list[Criterion] = Field(min_length=1)
    good_fit_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    potential_fit_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_invariants(self) -> "JDRubric":
        ids = [criterion.id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion ids must be unique")

        total = sum(criterion.weight for criterion in self.criteria)
        if not math.isclose(total, 1.0, abs_tol=WEIGHT_SUM_TOLERANCE):
            raise ValueError(f"criterion weights must sum to 1.0, got {total}")

        if self.good_fit_threshold <= self.potential_fit_threshold:
            raise ValueError(
                "good_fit_threshold must be greater than potential_fit_threshold"
            )
        return self

    def must_haves(self) -> list[Criterion]:
        """The criteria whose absence sends the candidate down `reject_fast`."""
        return [criterion for criterion in self.criteria if criterion.must_have]
```

- [x] **Step 4: Run the contract test to verify it passes**

Run: `python -m pytest tests/contracts/test_rubric.py -v`
Expected: 8 passed.

- [x] **Step 5: Write the failing loader test**

`tests/rubric/__init__.py`:
```python
"""Rubric loader tests."""
```

`tests/rubric/test_loader.py`:
```python
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
```

- [x] **Step 6: Run it to verify it fails**

Run: `python -m pytest tests/rubric/test_loader.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.rubric.loader'`.

- [x] **Step 7: Write the loader**

`src/rubric/loader.py`:
```python
"""Read and write rubrics as YAML so a recruiter can edit them without code."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.contracts.rubric import JDRubric


def load_rubric(path: str | Path) -> JDRubric:
    """Load a rubric from a YAML file, validating it against the contract."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"rubric not found: {file_path}")

    data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"rubric YAML must be a mapping, got {type(data).__name__}")

    return JDRubric.model_validate(data)


def save_rubric(rubric: JDRubric, path: str | Path) -> None:
    """Write a rubric to YAML, preserving field order for readable diffs."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        yaml.safe_dump(
            rubric.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
```

- [x] **Step 8: Write the committed IT rubric**

`data/rubrics/backend_engineer.yaml`:
```yaml
job_title: Backend Engineer
good_fit_threshold: 0.70
potential_fit_threshold: 0.40
criteria:
  - id: backend_language
    description: Production experience with a backend language (Python, Java, Go or Node.js)
    weight: 0.30
    must_have: true
    kind: skill
  - id: years_experience
    description: At least 3 years of professional software engineering experience
    weight: 0.25
    must_have: true
    kind: experience_years
  - id: databases
    description: Hands-on experience with relational databases and SQL
    weight: 0.15
    kind: skill
  - id: cloud_devops
    description: Experience with cloud platforms, containers or CI/CD pipelines
    weight: 0.15
    kind: skill
  - id: education
    description: Bachelor's degree in Computer Science or a related field
    weight: 0.10
    kind: education
  - id: domain
    description: Experience building APIs or distributed backend services
    weight: 0.05
    kind: domain
```

Note: the weights sum to exactly 1.00 (0.30 + 0.25 + 0.15 + 0.15 + 0.10 + 0.05). If you edit them, keep the sum at 1.0 or `load_rubric` will refuse the file.

- [x] **Step 9: Run the loader tests to verify they pass**

Run: `python -m pytest tests/rubric -v`
Expected: 6 passed.

- [x] **Step 10: Commit**

```bash
git add src/contracts/rubric.py src/rubric/loader.py data/rubrics/backend_engineer.yaml tests/contracts/test_rubric.py tests/rubric
git commit -m "feat: add JDRubric contract, YAML loader and the backend engineer rubric"
```

Note: `data/rubrics/` is tracked normally — `.gitignore` excludes only `data/raw/`, `data/samples/` and `data/demo/`.

---

### Task 4: Graph state contract, proven against LangGraph

**Files:**
- Create: `src/contracts/state.py`
- Test: `tests/contracts/test_state.py`

**Interfaces:**
- Consumes: `JDRubric` from Task 3; `CandidateProfile`, `CriterionScore`, `ScreeningResult` from Task 2.
- Produces: `ScreeningState` — the single object that flows through every LangGraph node, with `visit(node: str) -> ScreeningState` appending to `path_taken`.

This task exists to de-risk Day 2: it proves on Day 1 that a pydantic model actually works as a LangGraph state schema on LangGraph 1.2.1, rather than discovering otherwise while building nodes.

- [x] **Step 1: Write the failing test**

`tests/contracts/test_state.py`:
```python
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
```

- [x] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/contracts/test_state.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.contracts.state'`.

- [x] **Step 3: Write the implementation**

`src/contracts/state.py`:
```python
"""The single state object that flows through the screening graph."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.contracts.rubric import JDRubric
from src.contracts.screening import CandidateProfile, CriterionScore, ScreeningResult


class ScreeningState(BaseModel):
    """Everything the graph knows about one CV-JD pair at a point in time."""

    cv_text: str
    jd_text: str
    rubric: JDRubric | None = None
    profile: CandidateProfile | None = None
    criterion_scores: list[CriterionScore] = Field(default_factory=list)
    result: ScreeningResult | None = None

    repair_attempts: int = Field(default=0, ge=0)
    max_repair_attempts: int = Field(default=2, ge=0)
    quarantined: bool = False
    injection_flags: list[str] = Field(default_factory=list)
    path_taken: list[str] = Field(default_factory=list)

    def visit(self, node: str) -> "ScreeningState":
        """Record that `node` ran. Returns self so calls can be chained."""
        self.path_taken.append(node)
        return self
```

- [x] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/contracts/test_state.py -v`
Expected: 4 passed.

If `test_state_works_as_a_langgraph_state_schema` fails, do NOT weaken the assertion — read the actual error. A pydantic validation error there means LangGraph 1.2.1 wants different node return semantics, and Day 2's node design must account for it. Write down what you find at the bottom of this plan file before continuing.

- [x] **Step 5: Commit**

```bash
git add src/contracts/state.py tests/contracts/test_state.py
git commit -m "feat: add ScreeningState and prove it works as a LangGraph state schema"
```

---

### Task 5: Fit dataset loader and validation

**Files:**
- Create: `src/data/fit_dataset.py`
- Create: `tests/data/__init__.py`
- Test: `tests/data/test_fit_dataset.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `HF_FIT_REPO = "cnamuangtoun/resume-job-description-fit"`, `FIT_COLUMNS = ["resume_text", "job_description_text", "label"]`, `VALID_LABELS = {"Good Fit", "Potential Fit", "No Fit"}`, `SPLITS = ("train", "test")`
  - `validate_fit_frame(df: pd.DataFrame) -> pd.DataFrame` — returns the frame narrowed to `FIT_COLUMNS`, raising `ValueError` on schema drift.
  - `load_fit_split(split: str) -> pd.DataFrame` — downloads and validates `train.csv` or `test.csv`.

- [x] **Step 1: Write the failing test**

`tests/data/__init__.py`:
```python
"""Data layer tests."""
```

`tests/data/test_fit_dataset.py`:
```python
import pandas as pd
import pytest

from src.data.fit_dataset import (
    FIT_COLUMNS,
    VALID_LABELS,
    load_fit_split,
    validate_fit_frame,
)


def _frame(**overrides) -> pd.DataFrame:
    data = {
        "resume_text": ["a resume", "another resume"],
        "job_description_text": ["a jd", "another jd"],
        "label": ["Good Fit", "No Fit"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_valid_labels_are_the_three_dataset_strings():
    assert VALID_LABELS == {"Good Fit", "Potential Fit", "No Fit"}


def test_validate_accepts_a_well_formed_frame():
    result = validate_fit_frame(_frame())

    assert list(result.columns) == FIT_COLUMNS
    assert len(result) == 2


def test_validate_narrows_extra_columns_away():
    frame = _frame()
    frame["extra"] = [1, 2]

    result = validate_fit_frame(frame)

    assert list(result.columns) == FIT_COLUMNS


def test_validate_rejects_a_missing_column():
    frame = _frame().drop(columns=["label"])

    with pytest.raises(ValueError, match="missing columns"):
        validate_fit_frame(frame)


def test_validate_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="unexpected labels"):
        validate_fit_frame(_frame(label=["Good Fit", "Perfect Fit"]))


def test_validate_rejects_nulls_in_required_columns():
    with pytest.raises(ValueError, match="null values"):
        validate_fit_frame(_frame(resume_text=["a resume", None]))


def test_load_fit_split_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="split must be"):
        load_fit_split("validation")


@pytest.mark.network
def test_load_fit_split_downloads_the_real_test_split():
    frame = load_fit_split("test")

    assert len(frame) == 1759
    assert list(frame.columns) == FIT_COLUMNS
    assert set(frame["label"].unique()) == VALID_LABELS
```

- [x] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/data/test_fit_dataset.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.data.fit_dataset'`.

- [x] **Step 3: Write the implementation**

`src/data/fit_dataset.py`:
```python
"""Load the labelled resume/JD fit dataset from HuggingFace.

The dataset ships as two flat CSVs and needs no authentication. Its text has
whitespace stripped between sentences, so downstream components must not assume
word boundaries around punctuation.
"""

from __future__ import annotations

import pandas as pd
from huggingface_hub import hf_hub_download

HF_FIT_REPO = "cnamuangtoun/resume-job-description-fit"
FIT_COLUMNS = ["resume_text", "job_description_text", "label"]
VALID_LABELS = {"Good Fit", "Potential Fit", "No Fit"}
SPLITS = ("train", "test")


def validate_fit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check the schema and return the frame narrowed to the three columns."""
    missing = [column for column in FIT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    narrowed = df[FIT_COLUMNS]

    unexpected = set(narrowed["label"].unique()) - VALID_LABELS
    if unexpected:
        raise ValueError(f"unexpected labels: {sorted(unexpected)}")

    if narrowed.isna().any().any():
        raise ValueError("null values in required columns")

    return narrowed


def load_fit_split(split: str) -> pd.DataFrame:
    """Download and validate one split. Cached by huggingface_hub after first call."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    path = hf_hub_download(HF_FIT_REPO, f"{split}.csv", repo_type="dataset")
    return validate_fit_frame(pd.read_csv(path))
```

- [x] **Step 4: Run the offline tests to verify they pass**

Run: `python -m pytest tests/data/test_fit_dataset.py -v`
Expected: 7 passed, 1 deselected (the network test).

- [x] **Step 5: Run the network test once, deliberately**

Run: `python -m pytest tests/data/test_fit_dataset.py -m network -v`
Expected: 1 passed. This downloads roughly 30 MB on the first run and confirms the row count is still 1759. If the row count assertion fails, the upstream dataset changed — stop and update the spec before continuing.

- [x] **Step 6: Commit**

```bash
git add src/data/fit_dataset.py tests/data
git commit -m "feat: add HuggingFace fit dataset loader with schema validation"
```

---

### Task 6: Deterministic stratified splits

**Files:**
- Create: `src/data/splits.py`
- Create: `scripts/__init__.py`
- Create: `scripts/build_splits.py`
- Test: `tests/data/test_splits.py`

**Interfaces:**
- Consumes: `load_fit_split`, `FIT_COLUMNS` from Task 5.
- Produces:
  - `stratified_sample(df: pd.DataFrame, n: int, seed: int = 42, label_col: str = "label") -> pd.DataFrame`
  - `write_jsonl(df: pd.DataFrame, path: str | Path) -> Path`
  - `build_splits(dev_size: int = 300, test_size: int = 500, seed: int = 42, out_dir: Path = Path("data/samples")) -> dict[str, Path]`
  - `python -m scripts.build_splits` as the single command that materializes both files.

- [x] **Step 1: Write the failing test**

`tests/data/test_splits.py`:
```python
import json

import pandas as pd
import pytest

from src.data.splits import stratified_sample, write_jsonl


def _population() -> pd.DataFrame:
    labels = ["No Fit"] * 50 + ["Good Fit"] * 30 + ["Potential Fit"] * 20
    return pd.DataFrame(
        {
            "resume_text": [f"resume {i}" for i in range(100)],
            "job_description_text": [f"jd {i}" for i in range(100)],
            "label": labels,
        }
    )


def test_stratified_sample_preserves_label_proportions():
    sample = stratified_sample(_population(), n=10, seed=42)

    counts = sample["label"].value_counts().to_dict()
    assert counts == {"No Fit": 5, "Good Fit": 3, "Potential Fit": 2}


def test_stratified_sample_returns_exactly_n_rows_when_proportions_are_uneven():
    sample = stratified_sample(_population(), n=7, seed=42)

    assert len(sample) == 7
    assert set(sample["label"]) <= {"No Fit", "Good Fit", "Potential Fit"}


def test_stratified_sample_is_deterministic_for_a_given_seed():
    first = stratified_sample(_population(), n=10, seed=42)
    second = stratified_sample(_population(), n=10, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_stratified_sample_differs_across_seeds():
    first = stratified_sample(_population(), n=10, seed=42)
    second = stratified_sample(_population(), n=10, seed=7)

    assert not first.equals(second)


def test_stratified_sample_rejects_n_larger_than_the_population():
    with pytest.raises(ValueError, match="cannot sample"):
        stratified_sample(_population(), n=1000, seed=42)


def test_stratified_sample_rejects_non_positive_n():
    with pytest.raises(ValueError, match="must be positive"):
        stratified_sample(_population(), n=0, seed=42)


def test_write_jsonl_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "out.jsonl"

    write_jsonl(_population().head(3), path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert set(first) == {"resume_text", "job_description_text", "label"}
```

- [x] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/data/test_splits.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.data.splits'`.

- [x] **Step 3: Write the implementation**

`src/data/splits.py`:
```python
"""Deterministic stratified sampling for the dev and test evaluation sets.

The dev set is drawn from the dataset's train split and is where thresholds and
prompts get tuned. The test set is drawn from the dataset's test split and is
run exactly once, at the end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fit_dataset import FIT_COLUMNS, load_fit_split

DEFAULT_SEED = 42
DEFAULT_OUT_DIR = Path("data/samples")


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int = DEFAULT_SEED,
    label_col: str = "label",
) -> pd.DataFrame:
    """Sample `n` rows keeping each label's share, using largest-remainder allocation."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n > len(df):
        raise ValueError(f"cannot sample {n} rows from a population of {len(df)}")

    counts = df[label_col].value_counts().sort_index()
    exact = counts / len(df) * n
    allocation = np.floor(exact).astype(int)

    shortfall = n - int(allocation.sum())
    if shortfall > 0:
        remainders = (exact - allocation).sort_values(ascending=False, kind="mergesort")
        for label in list(remainders.index)[:shortfall]:
            allocation[label] += 1

    parts = [
        df[df[label_col] == label].sample(n=int(size), random_state=seed)
        for label, size in allocation.items()
        if size > 0
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_jsonl(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the frame as one JSON object per line, UTF-8, no index."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(file_path, orient="records", lines=True, force_ascii=False)
    return file_path


def build_splits(
    dev_size: int = 300,
    test_size: int = 500,
    seed: int = DEFAULT_SEED,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Path]:
    """Materialize the dev and test evaluation sets on disk."""
    dev = stratified_sample(load_fit_split("train")[FIT_COLUMNS], dev_size, seed)
    test = stratified_sample(load_fit_split("test")[FIT_COLUMNS], test_size, seed)

    return {
        "dev": write_jsonl(dev, Path(out_dir) / f"dev_{dev_size}.jsonl"),
        "test": write_jsonl(test, Path(out_dir) / f"test_{test_size}.jsonl"),
    }
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/data/test_splits.py -v`
Expected: 7 passed.

- [x] **Step 5: Write the build script**

`scripts/__init__.py`:
```python
"""Operational scripts."""
```

`scripts/build_splits.py`:
```python
"""Materialize data/samples/dev_300.jsonl and data/samples/test_500.jsonl.

Usage:
    python -m scripts.build_splits
"""

from __future__ import annotations

from src.data.splits import build_splits


def main() -> None:
    paths = build_splits()
    for name, path in paths.items():
        line_count = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"{name}: {path} ({line_count} rows)")


if __name__ == "__main__":
    main()
```

- [x] **Step 6: Run the script for real**

Run: `python -m scripts.build_splits`

Expected output:
```
dev: data\samples\dev_300.jsonl (300 rows)
test: data\samples\test_500.jsonl (500 rows)
```

This downloads the train split (roughly 90 MB) on the first run.

- [x] **Step 7: Verify the split label balance by hand**

Run:
```bash
python -c "import pandas as pd; d=pd.read_json('data/samples/dev_300.jsonl',lines=True); print(d['label'].value_counts().to_dict())"
```

Expected: all three labels present, `No Fit` the largest class, totalling 300. Put the exact counts in the commit message.

- [x] **Step 8: Commit**

```bash
git add src/data/splits.py scripts/__init__.py scripts/build_splits.py tests/data/test_splits.py
git commit -m "feat: add deterministic stratified dev/test splits and the build script"
```

The generated `.jsonl` files stay gitignored on purpose — they are reproducible from seed 42.

---

### Task 7: Demo assets — HF job descriptions and the gated Kaggle resumes

**Files:**
- Create: `src/data/demo_assets.py`
- Create: `scripts/fetch_demo_assets.py`
- Test: `tests/data/test_demo_assets.py`

**Interfaces:**
- Consumes: `write_jsonl` from Task 6.
- Produces:
  - `HF_JD_REPO`, `HF_JD_FILE`, `JD_COLUMNS`, `IT_TITLE_KEYWORDS`, `KAGGLE_RESUME_DATASET`, `IT_CATEGORY`
  - `load_job_descriptions() -> pd.DataFrame`
  - `select_it_jds(df: pd.DataFrame, n: int = 5) -> pd.DataFrame`
  - `select_it_resumes(df: pd.DataFrame, n: int = 25) -> pd.DataFrame`
  - `kaggle_credentials_available() -> bool`
  - `python -m scripts.fetch_demo_assets` — writes `data/demo/jds.jsonl`, then either fetches the Kaggle resumes or prints exact manual instructions and exits 2.

- [x] **Step 1: Write the failing test**

`tests/data/test_demo_assets.py`:
```python
import pandas as pd
import pytest

from src.data.demo_assets import (
    JD_COLUMNS,
    kaggle_credentials_available,
    load_job_descriptions,
    select_it_jds,
    select_it_resumes,
)


def _jd_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company_name": ["A", "B", "C", "D", "E"],
            "job_description": ["jd a", "jd b", "jd c", "jd d", "jd e"],
            "position_title": [
                "Sales Specialist",
                "Web Developer",
                "Frontend Web Developer",
                "Licensing Coordinator",
                "Backend Engineer",
            ],
            "description_length": [10, 20, 30, 40, 50],
            "model_response": ["", "", "", "", ""],
        }
    )


def _resume_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Resume_str": ["r1", "r2", "r3", "r4"],
            "Resume_html": ["", "", "", ""],
            "Category": [
                "INFORMATION-TECHNOLOGY",
                "HR",
                "Information-Technology",
                "DESIGNER",
            ],
        }
    )


def test_select_it_jds_keeps_only_technical_titles():
    selected = select_it_jds(_jd_frame(), n=5)

    assert set(selected["position_title"]) == {
        "Web Developer",
        "Frontend Web Developer",
        "Backend Engineer",
    }


def test_select_it_jds_respects_n_and_is_deterministic():
    first = select_it_jds(_jd_frame(), n=2)
    second = select_it_jds(_jd_frame(), n=2)

    assert len(first) == 2
    pd.testing.assert_frame_equal(first, second)


def test_select_it_jds_keeps_the_expected_columns():
    selected = select_it_jds(_jd_frame(), n=2)

    assert list(selected.columns) == JD_COLUMNS


def test_select_it_resumes_matches_the_category_case_insensitively():
    selected = select_it_resumes(_resume_frame(), n=10)

    assert len(selected) == 2
    assert set(selected["ID"]) == {1, 3}


def test_select_it_resumes_raises_when_the_category_column_is_absent():
    with pytest.raises(ValueError, match="Category"):
        select_it_resumes(pd.DataFrame({"ID": [1]}), n=1)


def test_kaggle_credentials_available_returns_a_bool():
    assert isinstance(kaggle_credentials_available(), bool)


@pytest.mark.network
def test_load_job_descriptions_downloads_the_real_file():
    frame = load_job_descriptions()

    assert len(frame) == 853
    assert list(frame.columns) == JD_COLUMNS
    assert len(select_it_jds(frame, n=5)) == 5
```

- [x] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/data/test_demo_assets.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.data.demo_assets'`.

- [x] **Step 3: Write the implementation**

`src/data/demo_assets.py`:
```python
"""Assets used only by the live demo, never by the measured evaluation.

Job descriptions come from HuggingFace and need no credentials. The PDF resumes
live on Kaggle, which requires an API token; `kaggle_credentials_available`
lets the caller fail with instructions instead of a stack trace.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

HF_JD_REPO = "jacob-hugging-face/job-descriptions"
HF_JD_FILE = "training_data.csv"
JD_COLUMNS = [
    "company_name",
    "job_description",
    "position_title",
    "description_length",
    "model_response",
]

KAGGLE_RESUME_DATASET = "snehaanbhawal/resume-dataset"
IT_CATEGORY = "information-technology"

IT_TITLE_KEYWORDS = (
    "developer",
    "engineer",
    "programmer",
    "software",
    "web",
    "frontend",
    "front end",
    "backend",
    "back end",
    "full stack",
    "data scientist",
    "devops",
    "qa",
)


def load_job_descriptions() -> pd.DataFrame:
    """Download the 853-row job description table from HuggingFace."""
    path = hf_hub_download(HF_JD_REPO, HF_JD_FILE, repo_type="dataset")
    frame = pd.read_csv(path)

    missing = [column for column in JD_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    return frame[JD_COLUMNS]


def select_it_jds(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Pick `n` technical job descriptions, ordered deterministically by title."""
    if "position_title" not in df.columns:
        raise ValueError("frame has no position_title column")

    titles = df["position_title"].fillna("").str.lower()
    is_technical = titles.apply(
        lambda title: any(keyword in title for keyword in IT_TITLE_KEYWORDS)
    )

    return (
        df[is_technical]
        .sort_values(["position_title", "company_name"], kind="mergesort")
        .head(n)
        .reset_index(drop=True)[JD_COLUMNS]
    )


def select_it_resumes(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Pick `n` IT resumes from the Kaggle resume table, case-insensitively."""
    if "Category" not in df.columns:
        raise ValueError("frame has no Category column")

    is_it = df["Category"].fillna("").str.strip().str.lower() == IT_CATEGORY
    return df[is_it].head(n).reset_index(drop=True)


def kaggle_credentials_available() -> bool:
    """True when any credential form kagglehub accepts is present.

    Kaggle's current format is a single `KGAT_...` bearer token read from
    `~/.kaggle/access_token`. The `kaggle.json` file and the
    `KAGGLE_USERNAME`/`KAGGLE_KEY` pair are the legacy forms; kagglehub 1.x
    resolves the bearer token first and falls back to those.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    kaggle_dir = Path.home() / ".kaggle"
    if (kaggle_dir / "access_token").exists() or (kaggle_dir / "access_token.txt").exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (kaggle_dir / "kaggle.json").exists()
```

- [x] **Step 4: Run the offline tests to verify they pass**

Run: `python -m pytest tests/data/test_demo_assets.py -v`
Expected: 6 passed, 1 deselected.

- [x] **Step 5: Write the fetch script with the credential gate**

`scripts/fetch_demo_assets.py`:
```python
"""Fetch the demo assets.

Usage:
    python -m scripts.fetch_demo_assets

Job descriptions always succeed. Resumes need a Kaggle API token; without one
the script prints instructions and exits with status 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.data.demo_assets import (
    KAGGLE_RESUME_DATASET,
    kaggle_credentials_available,
    load_job_descriptions,
    select_it_jds,
    select_it_resumes,
)
from src.data.splits import write_jsonl

DEMO_DIR = Path("data/demo")

KAGGLE_INSTRUCTIONS = f"""
Kaggle credentials not found, so the demo resumes were not downloaded.
The job descriptions were written successfully; only the PDF resumes are missing.

To fix this:
  1. Sign in at https://www.kaggle.com and open https://www.kaggle.com/settings
  2. Under "API", click "Create New Token" and copy the KGAT_... string.
  3. Write it, with no trailing newline, to: {Path.home() / ".kaggle" / "access_token"}
  4. Re-run: python -m scripts.fetch_demo_assets

Alternative without an API token: download the dataset by hand from
https://www.kaggle.com/datasets/{KAGGLE_RESUME_DATASET}
and unzip it so that Resume/Resume.csv and data/ sit under data/raw/kaggle_resumes/.
""".strip()


def fetch_job_descriptions(n: int = 5) -> Path:
    selected = select_it_jds(load_job_descriptions(), n=n)
    path = write_jsonl(selected, DEMO_DIR / "jds.jsonl")
    print(f"job descriptions: {path} ({len(selected)} rows)")
    for title in selected["position_title"]:
        print(f"  - {title}")
    return path


def fetch_resumes(n: int = 25) -> Path:
    import kagglehub

    root = Path(kagglehub.dataset_download(KAGGLE_RESUME_DATASET))
    csv_path = root / "Resume" / "Resume.csv"
    if not csv_path.exists():
        matches = list(root.rglob("Resume.csv"))
        if not matches:
            raise FileNotFoundError(f"Resume.csv not found under {root}")
        csv_path = matches[0]

    selected = select_it_resumes(pd.read_csv(csv_path), n=n)
    path = write_jsonl(selected, DEMO_DIR / "resumes.jsonl")
    print(f"resumes: {path} ({len(selected)} rows) from {csv_path}")
    return path


def main() -> int:
    fetch_job_descriptions()

    if not kaggle_credentials_available():
        print(KAGGLE_INSTRUCTIONS, file=sys.stderr)
        return 2

    fetch_resumes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 6: Run the script and observe the credential gate**

Run:
```bash
python -m scripts.fetch_demo_assets; echo "exit=$?"
```

Expected on this machine: `data/demo/jds.jsonl` is written, five IT titles print, then the Kaggle resume dataset downloads (`snehaanbhawal/resume-dataset`, a 62.5 MB archive on first run, cached under `~/.cache/kagglehub`), `data/demo/resumes.jsonl` is written with 25 `INFORMATION-TECHNOLOGY` rows, and the last line is `exit=0`.

The credential gate is a branch to verify, not to trip: this machine has a valid `~/.kaggle/access_token` (verified — `kagglehub.whoami()` returns `ttins1702`). To exercise the gate, override the home directory, **not** `KAGGLE_CONFIG_DIR` — `kaggle_credentials_available` reads `Path.home()` directly, so `KAGGLE_CONFIG_DIR` leaves it returning `True`:

```bash
USERPROFILE="$(mktemp -d)" HOME="$(mktemp -d)" python -c \n  "from src.data.demo_assets import kaggle_credentials_available as k; print(k())"
```

Expected: `False`. Verified on 2026-09-07.

- [x] **Step 7: Confirm the job descriptions landed**

Run:
```bash
python -c "import pandas as pd; d=pd.read_json('data/demo/jds.jsonl',lines=True); print(len(d)); print(d['position_title'].tolist())"
```

Expected: `5` and a list of five technical job titles.

- [x] **Step 8: Commit**

```bash
git add src/data/demo_assets.py scripts/fetch_demo_assets.py tests/data/test_demo_assets.py
git commit -m "feat: add demo asset loaders with a Kaggle credential gate"
```

- [x] **Step 9: Run the whole suite**

Run: `python -m pytest -v`
Expected: every test passes, with the `network`-marked tests deselected. Record the total count.

---

## Day 1 Definition of Done

- [x] `python -m pytest` is green with zero failures.
- [x] `python -m scripts.build_splits` produces `data/samples/dev_300.jsonl` (300 rows) and `data/samples/test_500.jsonl` (500 rows) with preserved label proportions.
- [x] `python -m scripts.fetch_demo_assets` writes `data/demo/jds.jsonl` with 5 IT job descriptions.
- [x] The Kaggle credential situation is resolved: a `KGAT_...` bearer token is installed at `~/.kaggle/access_token` and authenticates as `ttins1702`. The PDF demo is happening; the Day 2 fallback (drop PDF parsing) is no longer credential-driven.
- [x] Seven commits exist, one per task.
- [x] `ScreeningState` is proven to work as a LangGraph state schema, so Day 2 starts on nodes rather than on plumbing.

## Carried into Day 2

The five deterministic tools: `calculate_experience`, `normalize_skill`, `search_evidence`, `scan_injection`, `aggregate_scorecard`.

Note for `search_evidence`: the dataset text has whitespace stripped between sentences, so exact-substring matching against LLM-generated quotes will fail often. Plan on normalizing whitespace and casing on both sides before matching, and on returning character offsets into the original string so `Evidence.start`/`Evidence.end` stay meaningful.
