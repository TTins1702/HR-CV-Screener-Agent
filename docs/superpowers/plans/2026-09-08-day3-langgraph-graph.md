# HR CV Screener Agent — Day 3: The LangGraph Graph

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Execution mode is **inline** — run the tasks sequentially in the current session with a checkpoint after each task; do not dispatch subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the thirteen-node LangGraph screening graph and the `src/llm/` layer under it, so that one `(cv_text, jd_text)` pair goes in and one evidence-linked `ScreeningResult` comes out, over four conditional edges that each carry measured traffic.

**Architecture:** One pydantic `ScreeningState` flows through the graph; every node returns a *partial update dict*, never a mutated state object. The five Day 2 tools stay pure and deterministic; the LLM is confined to `src/llm/` and is reached only through `StructuredLLM.parse`, which is `temperature=0`, seeded, retried, and backed by a `(model, messages, schema)`-hash JSONL cache. Where the LLM and a tool compute the same quantity, **the LLM answers first and the tool overrides it**, and the size of the override is recorded — that recorded difference is the E2 evidence.

**Tech Stack:** Python 3.13.12 (miniconda base), langgraph 1.2.1, langchain-core 1.6.0, openai 2.38.0, pydantic 2.12.4, tenacity 9.1.4, python-dotenv 1.2.1, PyYAML 6.0.3, pytest 9.0.3. **No new dependencies.**

**Spec:** `docs/spec/2026-09-07-hr-cv-screener-spec.md` — §4 (graph shape), §6 (rubric as data), §7 (measurement), §8 (reproducibility), §10 (repo layout).

**Predecessors:** `docs/superpowers/plans/2026-09-07-day1-foundations.md` (contracts, rubric loader, splits) and `docs/superpowers/plans/2026-09-07-day2-deterministic-tools.md` (the five tools). Both complete; `main` is at `2388979` with **210 tests green**, 2 network-marked deselected.

## Global Constraints

- Python 3.13.12. Run everything as `python -m ...`. `uv` is NOT installed; use `python -m pip`. **No new packages today** — every import below is already in the base env.
- All code, identifiers, docstrings and comments are **English**. Only `docs/spec/*.md` and `docs/superpowers/plans/*.md` prose may be Vietnamese.
- The three fit labels are the exact strings `"Good Fit"`, `"Potential Fit"`, `"No Fit"`.
- **Every node returns a partial update dict.** Mutating `state` inside a node does nothing — LangGraph discards the mutation. `ScreeningState.visit()` is deleted in Task 1 precisely because it invites that mistake.
- **`graph.invoke()` returns a `dict`, not a `ScreeningState`.** Always finish with `ScreeningState.model_validate(output)`. (Measured — see the facts table.)
- `temperature=0` and `seed=42` are set on every call, but **they do not make the model deterministic** (measured: the same 30 resumes produced 13 malformed-date resumes in one run and 12 in the next, same prompt). The JSONL cache is what makes a *run* reproducible. Never claim determinism from `temperature=0` alone.
- Tests must run offline. Every test that touches the network is `@pytest.mark.network` and deselected by default. Node tests use a stub LLM, never a live client.
- `data/samples/*.jsonl` is **gitignored** and absent in a fresh clone. Tests reading it must `pytest.mark.skipif` on the file's absence.
- Never print or commit `OPENAI_API_KEY`. `.env` is gitignored and stays that way. `data/cache/` and `data/rubrics/derived/` are generated and must be gitignored.
- **Commits carry the user's name only.** No `Co-Authored-By` trailer, no Claude/AI attribution anywhere in a commit message.
- `docs/` is deliberately untracked and must not be committed. The user added `docs/` to `.gitignore` themselves; **leave that uncommitted `.gitignore` line alone.**
- One commit per task. Tests first, always: write the failing test, watch it fail, implement, watch it pass, commit.
- `git add` prints `LF will be replaced by CRLF`. Expected noise; ignore.

> **Running the tests.** `pyproject.toml` already sets `addopts = "-m 'not network' -q"`.
> Adding another `-q` makes it `-qq`, which hides the `N passed` summary line these
> steps tell you to read. Run `python -m pytest <path>` with no verbosity flag.

## The decision that was open, and is now closed

The Day 2 plan left one design question: `extract` fills `CandidateProfile.work_periods`, and `calculate_experience` parses the same dates independently — does `extract` just call the tool, or does the LLM guess and the tool correct it?

**The user chose option 2 on 2026-09-07: the LLM guesses, the tool corrects, and the correction is recorded.** Do not revisit this. Concretely:

- `CandidateProfile` gains `llm_declared_years` — what the model said.
- `total_experience_years` holds what `calculate_experience` computed, whenever the tool found at least one parseable range.
- The `extract` node writes the delta into its `NodeTrace.note`, so `scripts/measure_branch_traffic.py` can report it over the whole dev split.

The measurement below is why this was the right call: over 24 real pairs the two numbers **never** agreed, and the median gap was **4.38 years**.

## Measured facts this plan is built on

Established by prototyping against `data/samples/dev_300.jsonl` and live `gpt-4o-mini` calls on 2026-09-07, **before** this plan was written. Reproduction commands are given with each task that depends on a number.

### Library behaviour

| Fact | Value |
|---|---|
| `Annotated[list[str], operator.add]` as a reducer **inside a pydantic state model** | Works on LangGraph 1.2.1 |
| What a node receives | The `ScreeningState` instance |
| What `graph.invoke()` returns | A **`dict`** — must be re-validated |
| How `draw_mermaid()` renders a conditional edge | Dotted `-.->` (solid `-->` for a static edge) — slide 1's colouring is free |
| Structured-output entry point in openai 2.38.0 | `client.chat.completions.parse` is **stable**, not `.beta` |
| Bare `load_dotenv()` | Resolves relative to the **calling module's file**, not the cwd. Always pass an explicit path. |

### `extract` — the LLM against the tool (E2)

| Fact | Value |
|---|---|
| LLM returned a `total_experience_years` at all | 17 / 30 resumes |
| `calculate_experience` found ≥ 1 parseable range | 29 / 30 resumes |
| Comparable pairs where the two agreed to within 0.05 y | **0 / 16** |
| \|LLM − tool\| over 24 pairs | median **4.38 y**, max **11.50 y** |
| Direction of the correction | tool is larger in 16 / 24 |
| Worst single case seen | LLM 5.0 y, tool 12.92 y |
| Resumes where the LLM emitted an unparseable date token | 13 / 30 (22 tokens) |
| The exact failure mode | the literal strings `"null"`, `"present"`, `"N/A"` in a nullable date field |
| LLM's own `missing_fields` list | non-empty on **30 / 30** — useless as a routing signal |
| LLM's own `extraction_confidence` | min **0.90** — also useless as a routing signal |

The last two rows are why the `repair` edge is driven by **deterministic date validation**, not by anything the model says about itself. Routing on `missing_fields` would send 100% of traffic to `repair`, which is the definition of a decorative branch.

### `repair` — does the branch earn its place

> **Superseded on execution (2026-09-08).** The numbers below were measured against
> the *prototype* prompt. `EXTRACT_SYSTEM` as shipped names the three bad strings
> explicitly, and that fixed the failure at the source: over the first 25 real
> resumes, 26 of 182 date fields came back as genuine JSON nulls and **not one
> non-null value failed to parse** — `extract -> repair` measured **0 %**, not 40 %.
> Diagnosing that (as Task 12 Step 7 instructs) turned up a different failure mode:
> `0001-01` was the single most common date token the model wrote, 8 of 182 fields,
> a sentinel for "the text does not say" that parsed cleanly as year 1 and entered
> `WorkPeriod` as a real date. `parse_month` now carries Day 2's
> `EARLIEST_PLAUSIBLE_YEAR` floor, and the branch measures **4 %** on those 25 rows.
> Keep the table below as the record of what the weaker prompt did; trust
> `docs/measurements/branch_traffic.md` for what the shipped system does.

| Fact | Value |
|---|---|
| Fires (≥ 1 unparseable date after extraction) | **12 / 30 = 40 %** |
| Fully repaired within the 2-attempt cap | **12 / 12** |
| Repaired on the **first** attempt | **12 / 12** — the second attempt never fired on real data |
| Bad date fields before → after | 21 → **0** |
| Tokens per resume, no repair | median 1562 |
| Tokens per resume, with repair | median 3421 (2.2×) |

### `score_criteria` — evidence linkage (E1) with LLM-written quotes

Day 2 proved E1 for *tool-generated* evidence (3179 / 3179). This is the harder case: quotes the model writes itself.

| Fact | Value |
|---|---|
| Quotes written by the LLM across 20 pairs | 122 |
| Resolved by `search_evidence`, exact | 95 |
| Resolved fuzzily (naive `str.find` would have failed) | 24 |
| Unresolvable — must be dropped | **3 / 122 = 2.5 %** |
| `cv[e.start:e.end] == e.quote` on every resolved quote | **True, 119 / 119** |
| Criteria the LLM failed to return a score for | 0 |

**22 % of LLM quotes (27 / 122) are not findable with `str.find`.** That is the number that justifies `search_evidence` being a tool.

### `load_rubric` — deriving a rubric from a real JD

| Fact | Value |
|---|---|
| Criteria produced per JD | min 6, median 8, max 8 |
| LLM weights already summed to 1.0 | **16 / 20** (worst sum 1.100) → renormalization is mandatory |
| Duplicate criterion ids | 0 / 20 |
| Invalid `kind` values | 0 / 20 |
| `must_have` criteria per rubric | 2–5, median 3; **zero rubrics had none** |
| Distinct JDs in `dev_300.jsonl` | **159 / 300** |
| Distinct JDs in `test_500.jsonl` | **69 / 500** |

The last two rows are why derived rubrics are cached to YAML keyed by a hash of the JD text: the full test run costs **69** derivations, not 500.

### `must_have_check` → `reject_fast` — the pre-scoring gate

This gate runs *before* any criterion is scored, so it cannot use scores. It can only use the tools against the extracted profile.

| Fact | Value |
|---|---|
| `must_have` criteria that are **pre-checkable** (skill or experience_years) | median 2 of 3 |
| Rubrics with no pre-checkable must-have (gate cannot fire) | 3 / 24 |
| Gate fires, profile skills only | 13 / 24 = **54 %** |
| Gate fires, **with** the `expand_skill` + `search_evidence` fallback | 10 / 24 = **42 %** |
| Criteria rescued by that fallback | 8 |
| True `Good Fit` rows rejected | **1 / 4** — and it survives the fallback |

That one false reject is real and stays in the report: the JD required Excel, the CV never mentions it, and no surface form of it appears anywhere in the text. The gate is behaving correctly against the rubric; the dataset label disagrees. **Report it, do not tune it away on Day 3.** `must_have_min_score` and the ANY/ALL semantics of `skill_terms` are Day 4 knobs.

### `aggregate` → `deep_review` — the gray zone

| Fact | Value |
|---|---|
| Rows in the gray zone at the default `gray_zone_margin = 0.05` | 2 / 20 = **10 %** |
| Overall score distribution | min 0.00, median 0.17, max 0.90 |

Setting `gray_zone_margin = 0.0` drives this to 0 % and disables the branch — that is the ablation spec §7 asks for.

### `guard` → `quarantine`

Day 2 measured **0 / 300** real dev resumes tripping `scan_injection`. So on clean data this branch carries **0 %** traffic, and it is honest to say so. It is exercised by a poisoned-CV fixture set built in Task 4, and by the guard on/off ablation in spec §7. A branch whose only traffic is synthetic must be *labelled* as such on the slide.

### Cost and latency

| Fact | Value |
|---|---|
| `extract` call latency | median 2933 ms, p95 4024 ms |
| Tokens per pair, rubric derivation + scoring | median 2860 |
| Prototyping spend for this plan | ~234 k `gpt-4o-mini` tokens total |
| Projected full `test_500` run (69 rubrics + 500 extracts + 40 % repair + 500 scorings + ~10 % deep review) | ≈ **2.8 M tokens**, well under **$1** |

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/contracts/trace.py` | **new** — `LLMUsage`, `NodeTrace`. Per-node token/latency accounting (spec §8). No imports from `screening.py`, so no cycle. |
| `src/contracts/state.py` | **modify** — reducers on `path_taken` and `node_traces`; add `scorecard`, `blocking_must_haves`; **delete `visit()`**. |
| `src/contracts/screening.py` | **modify** — `CandidateProfile.llm_declared_years`; `ScreeningResult.llm_calls`, `.cached_calls`, `.node_traces`. |
| `src/contracts/rubric.py` | **modify** — `Criterion.skill_terms: list[str] = []`, the concrete skill names a skill criterion requires. Additive; `backend_engineer.yaml` stays valid. |
| `src/llm/cache.py` | **new** — `cache_key()` and `JSONLCache`. The reproducibility layer from spec §8. |
| `src/llm/client.py` | **new** — `StructuredLLM`: `temperature=0`, seed, tenacity retry, cache, usage accounting. **The only module in the repo that imports `openai`.** |
| `src/graph/ingest.py` | **new** — `ingest`, `guard`, `quarantine`. Deterministic, no LLM. |
| `src/graph/extract.py` | **new** — `make_extract_node`, `make_repair_node`, the shared profile builder, and the tool correction (E2). |
| `src/graph/rubric_nodes.py` | **new** — `make_load_rubric_node`, `must_have_check`, `reject_fast`. |
| `src/graph/scoring.py` | **new** — `make_score_criteria_node`, `aggregate`, `make_deep_review_node`. |
| `src/graph/decide.py` | **new** — `decide`, `rank`. |
| `src/graph/routes.py` | **new** — the four conditional-edge functions, and nothing else. One file so a reviewer can read the entire control flow at once. |
| `src/graph/build.py` | **new** — `build_graph`, `screen`, `graph_mermaid`. |
| `scripts/export_graph_diagram.py` | **new** — writes the Mermaid source for slide 1. |
| `scripts/measure_branch_traffic.py` | **new** — the spec §4 requirement: real traffic per conditional edge over `dev_300`. |
| `tests/llm/`, `tests/graph/` | **new** — one test module per source module, plus `tests/graph/test_end_to_end.py`. |
| `tests/graph/fixtures/poisoned.py` | **new** — the synthetic injected CVs that give `guard` its only traffic. |

Nodes are grouped by **stage**, not one file per node: `ingest`/`guard`/`quarantine` change together, and so do `extract`/`repair`. `routes.py` is split out on purpose — the four conditional edges are the thing the assignment is graded on, and they should be readable in one screen.

---
### Task 1: Trace contracts and state rewiring

**Files:**
- Create: `src/contracts/trace.py`
- Modify: `src/contracts/state.py` (reducers, new fields, delete `visit()`)
- Modify: `src/contracts/screening.py` (`CandidateProfile.llm_declared_years`; `ScreeningResult.llm_calls`, `.cached_calls`, `.node_traces`)
- Modify: `src/contracts/rubric.py` (`Criterion.skill_terms`)
- Create: `tests/contracts/test_trace.py`
- Modify: `tests/contracts/test_state.py`

**Interfaces:**
- Consumes: `JDRubric`, `CandidateProfile`, `CriterionScore`, `ScreeningResult`, `Scorecard`.
- Produces: `LLMUsage(prompt_tokens, completion_tokens, latency_ms, cached)`; `NodeTrace(node, latency_ms, prompt_tokens, completion_tokens, llm_calls, cached_calls, note)` with the classmethod `NodeTrace.of(node: str, started: float, usages: Sequence[LLMUsage] = (), note: str = "") -> NodeTrace`. `ScreeningState` gains `scorecard: Scorecard | None`, `blocking_must_haves: list[str]`, and `operator.add` reducers on `path_taken` and `node_traces`. `Criterion` gains `skill_terms: list[str]`. **Every node in Tasks 4–11 returns `{"path_taken": ["<node>"], "node_traces": [NodeTrace.of(...)], ...}`.**

All four contract changes are **additive with defaults** except the deletion of `visit()`, which is dead code (grep confirms it is referenced only by its own test) and an active trap: mutating state inside a LangGraph node has no effect.

- [x] **Step 1: Write the failing tests**

Create `tests/contracts/test_trace.py`:

```python
import operator
import time
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.contracts.trace import LLMUsage, NodeTrace


def test_usage_defaults_to_free_and_uncached():
    usage = LLMUsage()

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.cached is False


def test_node_trace_of_sums_the_usages_it_is_given():
    trace = NodeTrace.of(
        "extract",
        time.perf_counter(),
        [
            LLMUsage(prompt_tokens=100, completion_tokens=20),
            LLMUsage(prompt_tokens=50, completion_tokens=10, cached=True),
        ],
        note="corrected 3.2y",
    )

    assert trace.node == "extract"
    assert trace.prompt_tokens == 150
    assert trace.completion_tokens == 30
    assert trace.llm_calls == 1
    assert trace.cached_calls == 1
    assert trace.note == "corrected 3.2y"


def test_node_trace_of_measures_elapsed_time():
    started = time.perf_counter()
    trace = NodeTrace.of("guard", started)

    assert trace.latency_ms >= 0.0
    assert trace.llm_calls == 0


def test_node_traces_accumulate_through_a_langgraph_reducer():
    class Probe(BaseModel):
        traces: Annotated[list[NodeTrace], operator.add] = Field(default_factory=list)

    def first(state: Probe) -> dict:
        return {"traces": [NodeTrace(node="first")]}

    def second(state: Probe) -> dict:
        return {"traces": [NodeTrace(node="second")]}

    builder = StateGraph(Probe)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)

    final = Probe.model_validate(builder.compile().invoke(Probe()))

    assert [trace.node for trace in final.traces] == ["first", "second"]
```

Then rewrite the two affected tests in `tests/contracts/test_state.py`. **Delete** `test_visit_records_the_node_in_order` entirely and **replace** `test_state_works_as_a_langgraph_state_schema` with the two tests below, keeping the rest of the file untouched:

```python
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
```

Add to `tests/contracts/test_state.py` (imports at the top of the file already cover `StateGraph`, `START`, `END`):

```python
def test_state_carries_the_new_graph_fields():
    state = ScreeningState(cv_text="cv", jd_text="jd")

    assert state.scorecard is None
    assert state.blocking_must_haves == []
    assert state.node_traces == []
    assert not hasattr(state, "visit")
```

Add to `tests/contracts/test_screening.py`:

```python
def test_profile_records_what_the_llm_claimed_alongside_the_corrected_total():
    from src.contracts.screening import CandidateProfile

    profile = CandidateProfile(
        raw_text="cv",
        extraction_confidence=0.9,
        llm_declared_years=5.0,
        total_experience_years=12.92,
    )

    assert profile.llm_declared_years == 5.0
    assert profile.total_experience_years == 12.92


def test_result_counts_live_and_cached_model_calls():
    from src.contracts.screening import FitLabel, ScreeningResult

    result = ScreeningResult(
        overall_score=0.5, label=FitLabel.POTENTIAL_FIT, llm_calls=2, cached_calls=1
    )

    assert result.llm_calls == 2
    assert result.cached_calls == 1
    assert result.node_traces == []
```

Add to `tests/contracts/test_rubric.py`:

```python
def test_criterion_carries_the_concrete_skill_terms_the_gate_checks():
    from src.contracts.rubric import Criterion

    plain = Criterion(id="db", description="SQL", weight=1.0)
    typed = Criterion(
        id="db", description="SQL", weight=1.0, kind="skill", skill_terms=["postgresql", "mysql"]
    )

    assert plain.skill_terms == []
    assert typed.skill_terms == ["postgresql", "mysql"]
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/contracts`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.contracts.trace'`, plus assertion failures on the new fields.

- [x] **Step 3: Create `src/contracts/trace.py`**

```python
"""Per-node cost and latency accounting.

Spec section 8 asks for token and latency numbers *per branch*, not per run. The
graph gets that by having every node append one `NodeTrace`; the reducer on
`ScreeningState.node_traces` keeps them in execution order, so the eval can group
by node name and the Streamlit view can show the run as a timeline.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from pydantic import BaseModel, Field


class LLMUsage(BaseModel):
    """What one call to the model cost. `cached` means it never left the process."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    cached: bool = False


class NodeTrace(BaseModel):
    """One node's contribution to the run's cost, latency and story.

    `note` is free text the node writes for the slide -- the size of an experience
    correction, the severity of an injection finding, the number of quotes dropped.
    """

    node: str = Field(min_length=1)
    latency_ms: float = Field(default=0.0, ge=0.0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    cached_calls: int = Field(default=0, ge=0)
    note: str = ""

    @classmethod
    def of(
        cls,
        node: str,
        started: float,
        usages: Sequence[LLMUsage] = (),
        note: str = "",
    ) -> "NodeTrace":
        """Build a trace for a node that started at `started`, a `perf_counter` value."""
        return cls(
            node=node,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            prompt_tokens=sum(usage.prompt_tokens for usage in usages),
            completion_tokens=sum(usage.completion_tokens for usage in usages),
            llm_calls=sum(1 for usage in usages if not usage.cached),
            cached_calls=sum(1 for usage in usages if usage.cached),
            note=note,
        )
```

- [x] **Step 4: Rewrite `src/contracts/state.py`**

Replace the whole file:

```python
"""The single state object that flows through the screening graph."""

from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from src.contracts.rubric import JDRubric
from src.contracts.screening import CandidateProfile, CriterionScore, ScreeningResult
from src.contracts.tools import Scorecard
from src.contracts.trace import NodeTrace


class ScreeningState(BaseModel):
    """Everything the graph knows about one CV-JD pair at a point in time.

    Nodes never mutate this object -- LangGraph discards in-place changes. A node
    returns a partial update dict instead. `path_taken` and `node_traces` carry
    `operator.add` reducers so those updates append; every other field is
    last-write-wins, which is why a node that wants to extend `injection_flags`
    must write `[*state.injection_flags, ...]` explicitly.
    """

    cv_text: str
    jd_text: str
    rubric: JDRubric | None = None
    profile: CandidateProfile | None = None
    criterion_scores: list[CriterionScore] = Field(default_factory=list)
    scorecard: Scorecard | None = None
    result: ScreeningResult | None = None

    repair_attempts: int = Field(default=0, ge=0)
    max_repair_attempts: int = Field(default=2, ge=0)
    quarantined: bool = False
    injection_flags: list[str] = Field(default_factory=list)
    blocking_must_haves: list[str] = Field(default_factory=list)

    path_taken: Annotated[list[str], operator.add] = Field(default_factory=list)
    node_traces: Annotated[list[NodeTrace], operator.add] = Field(default_factory=list)
```

- [x] **Step 5: Extend `src/contracts/screening.py`**

Add the import `from src.contracts.trace import NodeTrace` at the top, then add one field to `CandidateProfile`, immediately after `total_experience_years`:

```python
    llm_declared_years: float | None = Field(default=None, ge=0.0)
```

and update its docstring to:

```python
    """Structured view of one CV, produced by the extract node.

    `llm_declared_years` is what the model claimed; `total_experience_years` is what
    `calculate_experience` computed and is the number the graph scores against. Over
    24 real pairs the two never agreed, median gap 4.38 years -- keeping both is what
    makes the tool's contribution measurable rather than assumed.
    """
```

Add three fields to `ScreeningResult`, after `latency_ms`:

```python
    llm_calls: int = Field(default=0, ge=0)
    cached_calls: int = Field(default=0, ge=0)
    node_traces: list[NodeTrace] = Field(default_factory=list)
```

- [x] **Step 6: Extend `src/contracts/rubric.py`**

Add one field to `Criterion`, after `kind`:

```python
    skill_terms: list[str] = Field(default_factory=list)
```

and extend its docstring:

```python
    """One requirement lifted out of a job description.

    `skill_terms` names the concrete skills a `kind="skill"` criterion requires, so
    `must_have_check` can test for them deterministically before any scoring happens.
    Empty for every other kind.
    """
```

- [x] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/contracts`
Expected: PASS.

Then the whole suite, because `Evidence`, `JDRubric` and `ScreeningState` are used everywhere:

Run: `python -m pytest`
Expected: PASS. The count drops by exactly 1 from 210 (the deleted `visit` test) and rises by the 8 tests added here — **218 passed** (measured on execution; this plan first predicted 217). If any Day 1 or Day 2 test fails, a change was not additive; fix the change, not the old test.

- [x] **Step 8: Confirm the committed rubric still loads**

Run: `python -c "from src.rubric.loader import load_rubric; r = load_rubric('data/rubrics/backend_engineer.yaml'); print(r.job_title, len(r.criteria), [c.skill_terms for c in r.criteria])"`
Expected: `Backend Engineer 6 [[], [], [], [], [], []]`

- [x] **Step 9: Commit**

```bash
git add src/contracts tests/contracts
git commit -m "feat: add node tracing contracts and wire ScreeningState for LangGraph reducers"
```

---

### Task 2: The reproducibility cache

**Files:**
- Create: `src/llm/__init__.py` (empty)
- Create: `src/llm/cache.py`
- Create: `tests/llm/__init__.py` (empty)
- Test: `tests/llm/test_cache.py`
- Modify: `.gitignore` — **append only**, see the warning below

**Interfaces:**
- Consumes: nothing from this repo. Stdlib only.
- Produces: `DEFAULT_CACHE_PATH: Path`; `cache_key(*, model: str, messages: list[dict[str, str]], schema_name: str, schema: dict, temperature: float, seed: int | None) -> str`; `JSONLCache(path=DEFAULT_CACHE_PATH)` with `.get(key) -> dict | None`, `.put(key, value: dict) -> None`, `.path`, `__len__`. Task 3 is the only consumer.

> **`.gitignore` warning.** That file has an **uncommitted** user-owned modification adding `docs/`, and the standing instruction is to leave it alone. So the two new lines are appended to the working copy and **`.gitignore` is never staged** — committing it would commit the user's `docs/` line with it. It stays modified-but-uncommitted alongside their change, and a fresh clone therefore needs all three lines re-added by hand.
>
> The file also has **no trailing newline** after `docs/` (measured on execution: a naive `>>` append produced the line `docs/data/cache/` and silently corrupted the user's entry). Append with a script that normalises the newline, then read `git diff .gitignore` and confirm `docs/` is still its own line.

- [x] **Step 1: Write the failing test**

Create `tests/llm/__init__.py` (empty), then `tests/llm/test_cache.py`:

```python
import json

from src.llm.cache import JSONLCache, cache_key

MESSAGES = [{"role": "system", "content": "extract"}, {"role": "user", "content": "cv"}]
SCHEMA = {"type": "object", "properties": {"skills": {"type": "array"}}}


def key(**overrides) -> str:
    base = dict(
        model="gpt-4o-mini",
        messages=MESSAGES,
        schema_name="Extraction",
        schema=SCHEMA,
        temperature=0.0,
        seed=42,
    )
    base.update(overrides)
    return cache_key(**base)


def test_the_key_is_stable_across_calls():
    assert key() == key()


def test_the_key_is_a_sha256_hex_digest():
    assert len(key()) == 64
    assert set(key()) <= set("0123456789abcdef")


def test_every_input_that_can_change_the_answer_changes_the_key():
    baseline = key()

    assert key(model="gpt-4o") != baseline
    assert key(messages=[{"role": "user", "content": "other"}]) != baseline
    assert key(schema_name="Other") != baseline
    assert key(schema={"type": "string"}) != baseline
    assert key(temperature=0.7) != baseline
    assert key(seed=7) != baseline


def test_a_miss_returns_none(tmp_path):
    cache = JSONLCache(tmp_path / "c.jsonl")

    assert cache.get("nope") is None
    assert len(cache) == 0


def test_put_then_get_round_trips(tmp_path):
    cache = JSONLCache(tmp_path / "c.jsonl")

    cache.put("k", {"content": '{"skills": []}', "prompt_tokens": 10, "completion_tokens": 2})

    assert cache.get("k")["prompt_tokens"] == 10
    assert len(cache) == 1


def test_entries_survive_a_new_process(tmp_path):
    path = tmp_path / "c.jsonl"
    JSONLCache(path).put("k", {"content": "{}", "prompt_tokens": 1, "completion_tokens": 1})

    assert JSONLCache(path).get("k") == {"content": "{}", "prompt_tokens": 1, "completion_tokens": 1}


def test_the_last_write_for_a_key_wins(tmp_path):
    path = tmp_path / "c.jsonl"
    cache = JSONLCache(path)
    cache.put("k", {"content": "1", "prompt_tokens": 1, "completion_tokens": 1})
    cache.put("k", {"content": "2", "prompt_tokens": 1, "completion_tokens": 1})

    assert JSONLCache(path).get("k")["content"] == "2"


def test_a_truncated_tail_from_an_interrupted_run_does_not_poison_the_cache(tmp_path):
    path = tmp_path / "c.jsonl"
    JSONLCache(path).put("k", {"content": "{}", "prompt_tokens": 1, "completion_tokens": 1})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "half", "val')

    cache = JSONLCache(path)

    assert cache.get("k") is not None
    assert len(cache) == 1


def test_the_file_is_one_json_object_per_line(tmp_path):
    path = tmp_path / "c.jsonl"
    cache = JSONLCache(path)
    cache.put("a", {"content": "1", "prompt_tokens": 1, "completion_tokens": 1})
    cache.put("b", {"content": "2", "prompt_tokens": 1, "completion_tokens": 1})

    lines = path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 2
    assert {json.loads(line)["key"] for line in lines} == {"a", "b"}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/llm/test_cache.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm'`

- [x] **Step 3: Implement `src/llm/cache.py`**

Create `src/llm/__init__.py` empty, then:

```python
"""A content-addressed JSONL cache for model calls.

Spec section 8 asks that every number on a slide come from one command. That needs
the model to be replayable, and `temperature=0` does not give you that: measured on
2026-09-07, the same 30 resumes at temperature 0 with the same prompt produced 13
malformed-date extractions in one run and 12 in the next. The cache does give you
that -- a second run of the eval reads every answer back off disk.

The key covers everything that can change an answer: model id, the exact messages,
the response schema (name and JSON schema), temperature and seed. Change a prompt
and you get a clean miss rather than a stale hit.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

DEFAULT_CACHE_PATH = Path("data/cache/llm_cache.jsonl")


def cache_key(
    *,
    model: str,
    messages: list[dict[str, str]],
    schema_name: str,
    schema: dict[str, Any],
    temperature: float,
    seed: int | None,
) -> str:
    """A stable sha256 over every input that can change the model's answer."""
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "schema_name": schema_name,
            "schema": schema,
            "temperature": temperature,
            "seed": seed,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JSONLCache:
    """Append-only JSONL, read into memory once and appended to thereafter.

    Append-only rather than rewrite-in-place so an interrupted run loses at most the
    last line, and so the file stays a readable audit trail of what the model was
    asked. A repeated key is legal: the last line for a key wins.
    """

    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue  # a half-written tail from an interrupted run
                key = record.get("key")
                if key is not None:
                    self._entries[key] = record["value"]

    def get(self, key: str) -> dict[str, Any] | None:
        """The cached payload for `key`, or None."""
        return self._entries.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Record `value` under `key`, in memory and on disk."""
        with self._lock:
            self._entries[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n"
                )

    def __len__(self) -> int:
        return len(self._entries)
```

- [x] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/llm/test_cache.py`
Expected: `9 passed`

- [x] **Step 5: Ignore the generated artefacts, without staging the file**

Append these two lines to the end of `.gitignore`, repairing the missing trailing newline:

```bash
python -c "
import pathlib
p = pathlib.Path('.gitignore')
s = p.read_text(encoding='utf-8')
if not s.endswith('
'):
    s += '
'
p.write_text(s + 'data/cache/
data/rubrics/derived/
', encoding='utf-8')
"
```

Run: `git diff .gitignore`
Expected: exactly three added lines — `docs/` (the user's, still its own line), `data/cache/`, `data/rubrics/derived/`. If `docs/data/cache/` appears, the trailing newline was missing and the user's line has been corrupted; repair it before going on.

- [x] **Step 6: Commit**

```bash
git add src/llm/__init__.py src/llm/cache.py tests/llm
git commit -m "feat: add a content-addressed JSONL cache for model calls"
```

`.gitignore` is deliberately **not** staged — see the warning at the top of this task.

---

### Task 3: The structured-output client

**Files:**
- Create: `src/llm/client.py`
- Test: `tests/llm/test_client.py`

**Interfaces:**
- Consumes: `LLMUsage` from `src/contracts/trace.py`; `JSONLCache`, `cache_key`, `DEFAULT_CACHE_PATH` from `src/llm/cache.py`.
- Produces: `LLMRefusal(RuntimeError)`; `load_environment() -> None`; `StructuredLLM(*, model=None, seed=None, temperature=0.0, cache=None, client=None)` with the single method **`parse(*, system: str, user: str, schema: type[T]) -> tuple[T, LLMUsage]`** and the counters `.hits` / `.misses`. Every node in Tasks 5–10 calls exactly this method and nothing else, which is what makes the stub in Task 4 a one-method duck type.

`src/llm/client.py` is the **only** module in the repo allowed to import `openai`. Task 12's end-to-end test asserts that.

- [x] **Step 1: Write the failing test**

Create `tests/llm/test_client.py`:

```python
import json
import os
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.llm.cache import JSONLCache
from src.llm.client import LLMRefusal, StructuredLLM


class Answer(BaseModel):
    value: int


def fake_response(content: str, refusal: str | None = None, prompt=100, completion=20):
    message = SimpleNamespace(content=content, refusal=refusal, parsed=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


class FakeOpenAI:
    """Records every request and replays scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeOpenAI ran out of scripted responses")
        return self.responses.pop(0)


def llm(tmp_path, responses, **kwargs) -> tuple[StructuredLLM, FakeOpenAI]:
    fake = FakeOpenAI(responses)
    return (
        StructuredLLM(
            model="gpt-4o-mini",
            seed=42,
            cache=JSONLCache(tmp_path / "c.jsonl"),
            client=fake,
            **kwargs,
        ),
        fake,
    )


def test_parse_returns_the_validated_schema_and_the_usage(tmp_path):
    client, _ = llm(tmp_path, [fake_response('{"value": 7}')])

    answer, usage = client.parse(system="s", user="u", schema=Answer)

    assert answer.value == 7
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 20
    assert usage.cached is False


def test_every_request_pins_temperature_zero_and_the_seed(tmp_path):
    client, fake = llm(tmp_path, [fake_response('{"value": 1}')])

    client.parse(system="s", user="u", schema=Answer)

    request = fake.requests[0]
    assert request["temperature"] == 0.0
    assert request["seed"] == 42
    assert request["model"] == "gpt-4o-mini"
    assert request["response_format"] is Answer
    assert request["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


def test_the_second_identical_call_is_served_from_cache(tmp_path):
    client, fake = llm(tmp_path, [fake_response('{"value": 7}')])

    client.parse(system="s", user="u", schema=Answer)
    answer, usage = client.parse(system="s", user="u", schema=Answer)

    assert answer.value == 7
    assert usage.cached is True
    assert usage.prompt_tokens == 100  # the cost it *would* have been, replayed
    assert len(fake.requests) == 1
    assert (client.hits, client.misses) == (1, 1)


def test_a_different_prompt_is_a_miss_not_a_stale_hit(tmp_path):
    client, fake = llm(tmp_path, [fake_response('{"value": 1}'), fake_response('{"value": 2}')])

    first, _ = client.parse(system="s", user="u", schema=Answer)
    second, _ = client.parse(system="s", user="different", schema=Answer)

    assert (first.value, second.value) == (1, 2)
    assert len(fake.requests) == 2


def test_the_cache_survives_a_new_client_on_the_same_file(tmp_path):
    first, _ = llm(tmp_path, [fake_response('{"value": 7}')])
    first.parse(system="s", user="u", schema=Answer)

    second, fake = llm(tmp_path, [])
    answer, usage = second.parse(system="s", user="u", schema=Answer)

    assert answer.value == 7
    assert usage.cached is True
    assert fake.requests == []


def test_a_refusal_raises_rather_than_returning_an_empty_answer(tmp_path):
    client, _ = llm(tmp_path, [fake_response("", refusal="I cannot help with that")])

    with pytest.raises(LLMRefusal, match="cannot help"):
        client.parse(system="s", user="u", schema=Answer)


def test_a_refusal_is_not_cached(tmp_path):
    client, fake = llm(
        tmp_path, [fake_response("", refusal="no"), fake_response('{"value": 3}')]
    )
    with pytest.raises(LLMRefusal):
        client.parse(system="s", user="u", schema=Answer)

    answer, _ = client.parse(system="s", user="u", schema=Answer)

    assert answer.value == 3
    assert len(fake.requests) == 2


def test_model_and_seed_fall_back_to_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("SCREENER_SEED", "7")

    client = StructuredLLM(cache=JSONLCache(tmp_path / "c.jsonl"), client=FakeOpenAI([]))

    assert client.model == "gpt-4o"
    assert client.seed == 7


def test_no_api_key_is_needed_until_a_real_call_is_made(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = StructuredLLM(cache=JSONLCache(tmp_path / "c.jsonl"))

    assert client.model  # constructing it did not touch openai


@pytest.mark.network
def test_a_live_call_returns_a_parsed_answer_and_real_usage(tmp_path):
    client = StructuredLLM(cache=JSONLCache(tmp_path / "c.jsonl"))

    answer, usage = client.parse(
        system="Answer with the number the user names.",
        user="The number is seven.",
        schema=Answer,
    )

    assert answer.value == 7
    assert usage.prompt_tokens > 0
    assert usage.cached is False
```

Note `test_model_and_seed_fall_back_to_the_environment`: `monkeypatch.setenv` must beat `.env`, so `load_environment()` must **not** override already-set variables. `python-dotenv`'s `load_dotenv` defaults to `override=False`, which is the behaviour we want — do not pass `override=True`.

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/llm/test_client.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.client'`

- [x] **Step 3: Implement `src/llm/client.py`**

```python
"""The only module in this repo that talks to OpenAI.

Confining the SDK here is what lets every graph node be tested offline against a
one-method stub, and what keeps the retry, cache and usage-accounting policy in a
single place instead of scattered across thirteen nodes.

`temperature=0` and a fixed seed are pinned on every request, but they are a
best-effort hint, not a guarantee: the same 30 resumes at temperature 0 produced 13
malformed-date extractions in one run and 12 in the next (measured 2026-09-07).
Reproducibility comes from `src/llm/cache.py`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.contracts.trace import LLMUsage
from src.llm.cache import DEFAULT_CACHE_PATH, JSONLCache, cache_key

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SEED = 42

# Transient only. A 4xx means the request itself is wrong; retrying it just burns
# money and hides the bug.
RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

T = TypeVar("T", bound=BaseModel)


class LLMRefusal(RuntimeError):
    """The model declined to answer. Never cached -- a refusal is not a result."""


def load_environment() -> None:
    """Load `.env` from the repo root, without overriding what is already set.

    A bare `load_dotenv()` resolves relative to the *calling module's* file rather
    than the working directory, so a script run from elsewhere silently gets nothing.
    Be explicit about the path.
    """
    load_dotenv(REPO_ROOT / ".env")


class StructuredLLM:
    """A temperature-0, seeded, cached, retried structured-output client."""

    def __init__(
        self,
        *,
        model: str | None = None,
        seed: int | None = None,
        temperature: float = 0.0,
        cache: JSONLCache | None = None,
        client: Any | None = None,
    ) -> None:
        load_environment()
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.seed = (
            seed
            if seed is not None
            else int(os.environ.get("SCREENER_SEED", DEFAULT_SEED))
        )
        self.temperature = temperature
        self.cache = cache if cache is not None else JSONLCache(DEFAULT_CACHE_PATH)
        self._client = client
        self.hits = 0
        self.misses = 0

    @property
    def client(self) -> Any:
        """The OpenAI client, built on first use so offline tests never need a key."""
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def parse(self, *, system: str, user: str, schema: type[T]) -> tuple[T, LLMUsage]:
        """Ask the model for one `schema`-shaped answer, from cache when possible."""
        started = time.perf_counter()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        key = cache_key(
            model=self.model,
            messages=messages,
            schema_name=schema.__name__,
            schema=schema.model_json_schema(),
            temperature=self.temperature,
            seed=self.seed,
        )

        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return schema.model_validate_json(cached["content"]), LLMUsage(
                prompt_tokens=cached["prompt_tokens"],
                completion_tokens=cached["completion_tokens"],
                latency_ms=(time.perf_counter() - started) * 1000.0,
                cached=True,
            )

        self.misses += 1
        response = self._request(messages, schema)
        message = response.choices[0].message
        if message.refusal:
            raise LLMRefusal(f"model refused: {message.refusal}")

        content = message.content or ""
        value = schema.model_validate_json(content)
        self.cache.put(
            key,
            {
                "model": self.model,
                "content": content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        )
        return value, LLMUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            cached=False,
        )

    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _request(self, messages: list[dict[str, str]], schema: type[T]) -> Any:
        """One HTTP round trip, retried on transient failures only."""
        return self.client.chat.completions.parse(
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
            messages=messages,
            response_format=schema,
        )
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/llm`
Expected: `18 passed, 1 deselected` (the network test). Measured on execution; this plan first predicted 19.

- [x] **Step 5: Prove the live path still works**

Run: `python -m pytest tests/llm -m network`
Expected: `1 passed, 18 deselected`. This costs a few tokens and is the only live call in the suite. If it fails with an auth error, stop and check `.env` — do not print the key.

- [x] **Step 6: Commit**

```bash
git add src/llm/client.py tests/llm/test_client.py
git commit -m "feat: add the temperature-0 cached structured-output client"
```

---

### Task 4: ingest, guard, quarantine, and the guard edge

**Files:**
- Create: `src/graph/__init__.py` (empty)
- Create: `src/graph/ingest.py`
- Create: `src/graph/routes.py`
- Create: `tests/graph/__init__.py` (empty)
- Create: `tests/graph/stub.py`
- Create: `tests/graph/fixtures/__init__.py` (empty)
- Create: `tests/graph/fixtures/poisoned.py`
- Test: `tests/graph/test_ingest.py`

**Interfaces:**
- Consumes: `ScreeningState`, `NodeTrace`, `FitLabel`, `ScreeningResult`, `InjectionSeverity`, `scan_injection`.
- Produces: `EMPTY_DOCUMENT_FLAG: str`; the nodes `ingest(state) -> dict`, `guard(state) -> dict`, `quarantine(state) -> dict`; the route `route_guard(state) -> str` returning `"quarantine"` or `"extract"`. Also the test helper `StubLLM` (used by Tasks 5–11) and `poisoned_cvs() -> list[tuple[str, str]]`.

Three design points, each of which a reviewer should be able to challenge:

1. **`ingest` does not touch the text.** Every `Evidence.start`/`.end` is an index into `cv_text` exactly as it arrived. Normalising here would silently break E1 for the whole graph. Normalisation happens inside the tools, against a copy, with an index map back — that is what `src/tools/text_norm.py` is for. PDF-to-text is an app-layer concern and is **not** in this graph.
2. **`LOW` severity does not quarantine.** The two LOW rules (`must_hire`, `hidden_directive`) fire on phrasings that occur in honest reference letters. They are recorded in `injection_flags` and shown to the recruiter; only `HIGH` stops the run. Day 2 measured zero false positives across 300 real resumes, so this is a deliberate margin, not a fudge.
3. **A quarantined document gets `label=NO_FIT`.** `ScreeningResult.label` is required and there is no fourth label; `rejected_reason` carries the real story (`"quarantined: instruction_override"`). Anything reading the label alone must also read `rejected_reason` — the Streamlit view and the eval both do.

- [x] **Step 1: Write the failing test**

Create `tests/graph/__init__.py` and `tests/graph/fixtures/__init__.py` as empty files.

Create `tests/graph/stub.py` — the one-method duck type every node test uses:

```python
"""A scripted stand-in for `StructuredLLM`, so node tests never touch the network."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.contracts.trace import LLMUsage


class StubLLM:
    """Returns scripted answers in order and records what it was asked.

    It implements `parse` and nothing else, because `parse` is the entire surface
    the graph nodes use. If a node ever reaches for another attribute, that is a
    design regression and this stub will fail loudly with AttributeError.
    """

    def __init__(
        self, responses: list[BaseModel], usage: LLMUsage | None = None
    ) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.usage = usage or LLMUsage(prompt_tokens=100, completion_tokens=20)

    def parse(self, *, system: str, user: str, schema: type[BaseModel]):
        if not self.responses:
            raise AssertionError(
                f"StubLLM ran out of scripted responses; asked for {schema.__name__}"
            )
        self.calls.append({"system": system, "user": user, "schema": schema.__name__})
        return self.responses.pop(0), self.usage
```

Create `tests/graph/fixtures/poisoned.py`:

```python
"""Synthetic CVs carrying prompt injections.

The `guard` branch carries zero traffic on real data -- Day 2 measured 0 of 300 dev
resumes tripping any rule -- so its traffic has to come from a fixture set, and the
slide must label it as synthetic. Building the fixtures from the rule table's own
`example` strings means a rule added later automatically gets a fixture.
"""

from __future__ import annotations

from src.contracts.tools import InjectionSeverity
from src.tools.injection import INJECTION_RULES

CLEAN_CV = (
    "Jordan Blake, Backend Engineer.06/2019to12/2022 Backend Engineer at Acme Corp."
    "Built payment APIs in Python and PostgreSQL.Skills: Python, Django, PostgreSQL, "
    "Docker, AWS.B.S. Computer Science, State University, 2019."
)


def poisoned_cvs(severity: InjectionSeverity | None = None) -> list[tuple[str, str]]:
    """`(rule_id, cv_text)` for every rule, or only those of `severity`."""
    return [
        (rule.rule_id, f"{CLEAN_CV} {rule.example}")
        for rule in INJECTION_RULES
        if severity is None or rule.severity is severity
    ]
```

Create `tests/graph/test_ingest.py`:

```python
import pytest

from src.contracts.state import ScreeningState
from src.contracts.tools import InjectionSeverity
from src.graph.ingest import EMPTY_DOCUMENT_FLAG, guard, ingest, quarantine
from src.graph.routes import route_guard
from tests.graph.fixtures.poisoned import CLEAN_CV, poisoned_cvs


def state(cv: str = CLEAN_CV, jd: str = "Backend Engineer wanted.") -> ScreeningState:
    return ScreeningState(cv_text=cv, jd_text=jd)


def test_ingest_records_itself_and_leaves_the_text_byte_identical():
    update = ingest(state())

    assert update["path_taken"] == ["ingest"]
    assert "cv_text" not in update  # the text is never rewritten
    assert update["node_traces"][0].node == "ingest"
    assert "cv_chars=" in update["node_traces"][0].note


def test_ingest_quarantines_a_blank_document_instead_of_crashing():
    update = ingest(state(cv="   \n  "))

    assert update["quarantined"] is True
    assert update["injection_flags"] == [EMPTY_DOCUMENT_FLAG]


def test_ingest_quarantines_a_blank_job_description_too():
    assert ingest(state(jd=""))["quarantined"] is True


def test_ingest_passes_a_normal_pair_through():
    update = ingest(state())

    assert update.get("quarantined") is None


def test_guard_leaves_a_clean_cv_alone():
    update = guard(state())

    assert update["quarantined"] is False
    assert update["injection_flags"] == []
    assert "severity=none" in update["node_traces"][0].note


@pytest.mark.parametrize("rule_id,cv", poisoned_cvs(InjectionSeverity.HIGH))
def test_guard_quarantines_every_high_severity_rule(rule_id, cv):
    update = guard(state(cv=cv))

    assert update["quarantined"] is True
    assert rule_id in update["injection_flags"]


@pytest.mark.parametrize("rule_id,cv", poisoned_cvs(InjectionSeverity.LOW))
def test_guard_records_but_does_not_quarantine_a_low_severity_rule(rule_id, cv):
    update = guard(state(cv=cv))

    assert update["quarantined"] is False
    assert rule_id in update["injection_flags"]


def test_guard_keeps_the_flag_ingest_already_set():
    before = state(cv="")
    before.quarantined = True
    before.injection_flags = [EMPTY_DOCUMENT_FLAG]

    update = guard(before)

    assert update["quarantined"] is True
    assert update["injection_flags"] == [EMPTY_DOCUMENT_FLAG]


def test_quarantine_produces_a_terminal_result_naming_the_reason():
    before = state()
    before.injection_flags = ["instruction_override"]
    before.path_taken = ["ingest", "guard"]

    result = quarantine(before)["result"]

    assert result.overall_score == 0.0
    assert result.label.value == "No Fit"
    assert "instruction_override" in result.rejected_reason
    assert result.rejected_reason.startswith("quarantined:")
    assert result.path_taken == ["ingest", "guard", "quarantine"]


def test_route_guard_sends_a_quarantined_document_to_quarantine():
    before = state()
    before.quarantined = True

    assert route_guard(before) == "quarantine"


def test_route_guard_sends_a_clean_document_to_extract():
    assert route_guard(state()) == "extract"
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/graph/test_ingest.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.graph'`

- [x] **Step 3: Implement `src/graph/ingest.py`**

Create `src/graph/__init__.py` empty, then:

```python
"""The first three nodes: accept the pair, screen it for injection, or refuse it.

`ingest` deliberately does not touch the text. Every `Evidence.start` and `.end` in
the whole system is an index into `cv_text` exactly as it arrived, so normalising
here would break E1 everywhere at once. The tools normalise against a copy and keep
an index map back (`src/tools/text_norm.py`). PDF-to-text is an app-layer concern
and does not belong in the graph.
"""

from __future__ import annotations

import time

from src.contracts.screening import FitLabel, ScreeningResult
from src.contracts.state import ScreeningState
from src.contracts.tools import InjectionSeverity
from src.contracts.trace import NodeTrace
from src.tools.injection import scan_injection

EMPTY_DOCUMENT_FLAG = "empty_document"


def ingest(state: ScreeningState) -> dict:
    """Accept the raw pair. A blank CV or JD is quarantined, not crashed on."""
    started = time.perf_counter()
    update: dict = {
        "path_taken": ["ingest"],
        "node_traces": [
            NodeTrace.of(
                "ingest",
                started,
                note=f"cv_chars={len(state.cv_text)} jd_chars={len(state.jd_text)}",
            )
        ],
    }
    if not state.cv_text.strip() or not state.jd_text.strip():
        update["quarantined"] = True
        update["injection_flags"] = [EMPTY_DOCUMENT_FLAG]
    return update


def guard(state: ScreeningState) -> dict:
    """Scan the untrusted CV for hidden instructions before it reaches the model.

    Only HIGH severity quarantines. The two LOW rules fire on phrasings that occur
    in honest reference letters, so they are recorded and shown to the recruiter
    rather than used to refuse the candidate.
    """
    started = time.perf_counter()
    report = scan_injection(state.cv_text)
    return {
        "path_taken": ["guard"],
        "quarantined": state.quarantined or report.severity is InjectionSeverity.HIGH,
        "injection_flags": [*state.injection_flags, *report.flags],
        "node_traces": [
            NodeTrace.of(
                "guard",
                started,
                note=f"severity={report.severity.value} findings={len(report.findings)}",
            )
        ],
    }


def quarantine(state: ScreeningState) -> dict:
    """Terminal node for a document the graph refuses to score.

    `label` is NO_FIT because the contract has no fourth label; `rejected_reason`
    carries the real story. Anything that reads the label must read the reason too.
    """
    started = time.perf_counter()
    reason = "quarantined: " + (", ".join(state.injection_flags) or "unknown")
    return {
        "path_taken": ["quarantine"],
        "result": ScreeningResult(
            overall_score=0.0,
            label=FitLabel.NO_FIT,
            rejected_reason=reason,
            path_taken=[*state.path_taken, "quarantine"],
        ),
        "node_traces": [NodeTrace.of("quarantine", started, note=reason)],
    }
```

- [x] **Step 4: Create `src/graph/routes.py` with the first of the four edges**

```python
"""The four conditional edges, and nothing else.

These functions are the control flow the assignment is graded on, so they live in
one file a reviewer can read in a single screen. Each one is a pure function of the
state: no tools, no model calls, no side effects. The names they return are the keys
of the mapping passed to `add_conditional_edges` in `src/graph/build.py`.
"""

from __future__ import annotations

from src.contracts.state import ScreeningState


def route_guard(state: ScreeningState) -> str:
    """Unsafe or empty documents stop here; everything else goes on to `extract`."""
    return "quarantine" if state.quarantined else "extract"
```

- [x] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/graph/test_ingest.py`
Expected: `17 passed` — 9 plain tests plus 8 parametrised cases (6 HIGH rules, 2 LOW rules). Measured on execution.

- [x] **Step 6: Confirm the poisoned fixtures actually fire**

Run:
```bash
python -c "
from src.tools.injection import scan_injection
from tests.graph.fixtures.poisoned import CLEAN_CV, poisoned_cvs
print('clean:', scan_injection(CLEAN_CV).severity.value, scan_injection(CLEAN_CV).flags)
for rule_id, cv in poisoned_cvs():
    report = scan_injection(cv)
    print(f'{rule_id:22s} {report.severity.value:5s} {report.flags}')
"
```
Expected: `clean: none []`, then one line per rule whose `flags` contains its own `rule_id`. If a fixture does not fire its own rule, the rule's `example` string no longer matches its own pattern — that is a Day 2 bug worth fixing there, not papering over here.

- [x] **Step 7: Commit**

```bash
git add src/graph tests/graph
git commit -m "feat: add ingest, guard and quarantine nodes with the guard edge"
```

---
### Task 5: extract, repair, and the tool correction (E2)

**Files:**
- Create: `src/graph/extract.py`
- Modify: `src/graph/routes.py` (add `route_repair`)
- Test: `tests/graph/test_extract.py`

**Interfaces:**
- Consumes: `StubLLM`-compatible `.parse`; `CandidateProfile`, `WorkPeriod`, `NodeTrace`, `calculate_experience`.
- Produces: `EXTRACT_SYSTEM: str`, `REPAIR_TEMPLATE: str`, `UNUSABLE_DATE_TOKENS: frozenset[str]`; `RawPeriod`, `RawExtraction` (the response schemas); `parse_month(raw: str | None) -> date | None`; `unusable_date_fields(extraction: RawExtraction) -> list[str]`; `build_profile(state, extraction, *, today=None) -> tuple[CandidateProfile, float | None]`; `make_extract_node(llm, *, today=None) -> Callable`; `make_repair_node(llm, *, today=None) -> Callable`; `route_repair(state) -> str` returning `"repair"` or `"load_rubric"`.

**This is the task the user's decision lands in.** The model guesses `total_experience_years`, `calculate_experience` recomputes it from the raw text, `llm_declared_years` keeps what the model said, and the node writes the size of the correction into its trace. Measured over 24 real pairs: the two numbers agreed **zero** times, median gap **4.38 years**, worst case LLM 5.0 vs tool 12.92.

Three sub-decisions worth defending to a reviewer:

1. **`RawExtraction` has no `missing_fields` and no `extraction_confidence`-based routing.** The model's self-assessment was measured useless: `missing_fields` was non-empty on 30/30 real resumes and `extraction_confidence` never fell below 0.90. `extraction_confidence` is still collected because the UI shows it; nothing routes on it.
2. **`"present"` and `"current"` count as unusable date tokens.** The protocol says dates are `YYYY-MM` and null means current. Accepting a second spelling of "current" invites a third. (Measured on execution: with the shipped prompt the model almost never reaches for these strings anyway — the rule costs nothing and guards a real failure mode.)
3. **`repair` appends its instruction to the user message** rather than adding a third message, so `StructuredLLM.parse` keeps its two-message shape and one cache key per distinct prompt.

- [x] **Step 1: Write the failing test**

Create `tests/graph/test_extract.py`:

```python
from datetime import date

from src.contracts.state import ScreeningState
from src.graph.extract import (
    RawExtraction,
    RawPeriod,
    build_profile,
    make_extract_node,
    make_repair_node,
    parse_month,
    unusable_date_fields,
)
from src.graph.routes import route_repair
from tests.graph.fixtures.poisoned import CLEAN_CV
from tests.graph.stub import StubLLM

TODAY = date(2026, 9, 7)


def state() -> ScreeningState:
    return ScreeningState(cv_text=CLEAN_CV, jd_text="Backend Engineer wanted.")


def extraction(**overrides) -> RawExtraction:
    base = dict(
        skills=["Python", "Django", "PostgreSQL"],
        work_periods=[
            RawPeriod(title="Backend Engineer", company="Acme Corp", start="2019-06", end="2022-12")
        ],
        degrees=["B.S. Computer Science"],
        certifications=[],
        total_experience_years=2.0,
        extraction_confidence=0.95,
    )
    base.update(overrides)
    return RawExtraction(**base)


def test_parse_month_accepts_the_documented_formats():
    assert parse_month("2019-06") == date(2019, 6, 1)
    assert parse_month("2019") == date(2019, 1, 1)
    assert parse_month("2019-06-15") == date(2019, 6, 15)


def test_parse_month_rejects_the_tokens_the_model_actually_emits():
    for token in ("null", "None", "N/A", "present", "Current", "", "  ", "sometime"):
        assert parse_month(token) is None, token


def test_parse_month_passes_a_genuine_null_through():
    assert parse_month(None) is None


def test_unusable_date_fields_ignores_a_genuine_null():
    raw = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="2021-01", end=None)]
    )

    assert unusable_date_fields(raw) == []


def test_unusable_date_fields_names_the_field_the_value_and_the_role():
    raw = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="2021-01", end="null")]
    )

    bad = unusable_date_fields(raw)

    assert len(bad) == 1
    assert "work_periods[0].end" in bad[0]
    assert "'null'" in bad[0]
    assert "Consultant" in bad[0]


def test_the_tool_overrides_what_the_model_claimed():
    profile, delta = build_profile(state(), extraction(total_experience_years=2.0), today=TODAY)

    assert profile.llm_declared_years == 2.0
    assert profile.total_experience_years == 3.5  # 06/2019 to 12/2022, from the raw text
    assert delta == 1.5


def test_the_model_is_kept_when_the_tool_finds_no_dates():
    bare = ScreeningState(cv_text="No dates here at all.", jd_text="jd")

    profile, delta = build_profile(bare, extraction(total_experience_years=4.0), today=TODAY)

    assert profile.total_experience_years == 4.0
    assert profile.llm_declared_years == 4.0
    assert delta == 0.0


def test_unparseable_model_dates_are_dropped_from_the_profile_not_guessed():
    raw = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="present", end="null")]
    )

    profile, _ = build_profile(state(), raw, today=TODAY)

    assert profile.work_periods[0].start is None
    assert profile.work_periods[0].end is None
    assert len(profile.missing_fields) == 2


def test_extract_records_the_correction_in_its_trace():
    node = make_extract_node(StubLLM([extraction()]), today=TODAY)

    update = node(state())

    assert update["path_taken"] == ["extract"]
    assert update["profile"].total_experience_years == 3.5
    note = update["node_traces"][0].note
    assert "years_llm=2.00" in note
    assert "years_tool=3.50" in note
    assert "correction=1.50" in note
    assert update["node_traces"][0].llm_calls == 1


def test_extract_shows_the_model_the_raw_cv():
    stub = StubLLM([extraction()])

    make_extract_node(stub, today=TODAY)(state())

    assert stub.calls[0]["user"] == CLEAN_CV
    assert stub.calls[0]["schema"] == "RawExtraction"


def test_repair_names_the_broken_fields_in_its_prompt_and_counts_the_attempt():
    broken = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="2021-01", end="null")]
    )
    before = state()
    before.profile = build_profile(before, broken, today=TODAY)[0]
    stub = StubLLM([extraction()])

    update = make_repair_node(stub, today=TODAY)(before)

    assert update["path_taken"] == ["repair"]
    assert update["repair_attempts"] == 1
    assert "work_periods[0].end" in stub.calls[0]["user"]
    assert CLEAN_CV in stub.calls[0]["user"]
    assert update["profile"].missing_fields == []
    assert "before=1 after=0" in update["node_traces"][0].note


def test_route_repair_sends_a_broken_profile_to_repair():
    broken = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="x", end=None)]
    )
    before = state()
    before.profile = build_profile(before, broken, today=TODAY)[0]

    assert route_repair(before) == "repair"


def test_route_repair_gives_up_at_the_cap():
    broken = extraction(
        work_periods=[RawPeriod(title="Consultant", company=None, start="x", end=None)]
    )
    before = state()
    before.profile = build_profile(before, broken, today=TODAY)[0]
    before.repair_attempts = 2

    assert route_repair(before) == "load_rubric"


def test_route_repair_moves_on_when_the_profile_is_clean():
    before = state()
    before.profile = build_profile(before, extraction(), today=TODAY)[0]

    assert route_repair(before) == "load_rubric"


def test_route_repair_moves_on_when_there_is_no_profile_at_all():
    assert route_repair(state()) == "load_rubric"
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/graph/test_extract.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.graph.extract'`

- [x] **Step 3: Implement `src/graph/extract.py`**

```python
"""The extract and repair nodes, and the tool correction that makes E2 measurable.

The user chose on 2026-09-07 that the model guesses the years and
`calculate_experience` corrects it, rather than the node calling the tool directly.
The reason is that a correction can be counted: over 24 real (resume, JD) pairs the
two numbers agreed zero times, the median gap was 4.38 years, and the worst case was
the model saying 5.0 against the tool's 12.92. `llm_declared_years` keeps the claim
so the difference survives into the eval and onto the slide.

Nothing here routes on what the model says about its own output. Measured on 30 real
resumes, the model's `missing_fields` was non-empty 30 times out of 30 and its
`extraction_confidence` never dropped below 0.90 -- routing on either would send 100%
of traffic down `repair`, which is the definition of a decorative branch. The repair
edge is driven by parsing the dates the model wrote, which fires on 40%.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from src.contracts.screening import CandidateProfile, WorkPeriod
from src.contracts.state import ScreeningState
from src.contracts.trace import NodeTrace
from src.tools.experience import calculate_experience

EXTRACT_SYSTEM = (
    "You extract structured facts from a resume. The resume text often has spaces "
    "missing between sentences and around dates, for example "
    "'consulting projects.Proven ability' and '12/2011toPresent'. Report only what "
    "the text states. Every date is the literal format YYYY-MM. If the text gives no "
    'date, or the role is current, set the field to JSON null - never the string '
    '"null", "present" or "N/A".'
)

REPAIR_TEMPLATE = (
    "Your previous extraction of this resume had unusable date values in these "
    "fields:\n{fields}\n"
    "Re-read the resume and return the whole extraction again. Every date must be the "
    "literal format YYYY-MM taken from the text. If the text truly gives no date for a "
    'field, or the role is current, set it to JSON null - never the string "null", '
    '"present" or "N/A".'
)

# Everything the model writes instead of a date. "present" and "current" are here on
# purpose: the protocol says null means current, and accepting a second spelling of
# it invites a third. The 40% repair rate was measured under exactly this definition.
UNUSABLE_DATE_TOKENS = frozenset(
    {
        "",
        "null",
        "none",
        "n/a",
        "na",
        "unknown",
        "present",
        "current",
        "ongoing",
        "to date",
        "till date",
    }
)


class RawPeriod(BaseModel):
    """One employment entry as the model writes it, before any parsing."""

    title: str
    company: str | None
    start: str | None = Field(description="YYYY-MM, or null if the text does not say")
    end: str | None = Field(
        description="YYYY-MM, or null if the role is current or the text does not say"
    )


class RawExtraction(BaseModel):
    """The extract node's response schema.

    No `missing_fields` and nothing routes on `extraction_confidence`: both were
    measured to be uninformative. `extraction_confidence` is kept only because the
    recruiter-facing view shows it.
    """

    skills: list[str]
    work_periods: list[RawPeriod]
    degrees: list[str]
    certifications: list[str]
    total_experience_years: float | None
    extraction_confidence: float


def parse_month(raw: str | None) -> date | None:
    """`YYYY-MM`, `YYYY` or `YYYY-MM-DD` to a date; None for anything unusable."""
    if raw is None:
        return None
    token = raw.strip().lower()
    if token in UNUSABLE_DATE_TOKENS:
        return None
    try:
        if len(token) == 4 and token.isdigit():
            return date(int(token), 1, 1)
        return date.fromisoformat(token if len(token) > 7 else f"{token}-01")
    except ValueError:
        return None


def unusable_date_fields(extraction: RawExtraction) -> list[str]:
    """Which date fields the model wrote that cannot be parsed.

    A genuine JSON null is not a failure -- the prompt asks for it. Only a non-null
    value that will not parse counts, which is what makes this a signal rather than
    a constant.
    """
    bad: list[str] = []
    for index, period in enumerate(extraction.work_periods):
        for field_name, raw in (("start", period.start), ("end", period.end)):
            if raw is not None and parse_month(raw) is None:
                bad.append(
                    f"work_periods[{index}].{field_name} = {raw!r} ({period.title})"
                )
    return bad


def build_profile(
    state: ScreeningState, extraction: RawExtraction, *, today: date | None = None
) -> tuple[CandidateProfile, float | None]:
    """Turn a raw extraction into a profile, letting the tool correct the total.

    Returns the profile and the size of the correction, or None when there is
    nothing to compare against.
    """
    report = calculate_experience(state.cv_text, today=today)
    declared = extraction.total_experience_years
    corrected = report.total_years if report.ranges else declared
    delta = None if declared is None or corrected is None else abs(corrected - declared)

    profile = CandidateProfile(
        raw_text=state.cv_text,
        skills=extraction.skills,
        work_periods=[
            WorkPeriod(
                title=period.title,
                company=period.company,
                start=parse_month(period.start),
                end=parse_month(period.end),
            )
            for period in extraction.work_periods
        ],
        degrees=extraction.degrees,
        certifications=extraction.certifications,
        total_experience_years=corrected,
        llm_declared_years=declared,
        extraction_confidence=min(max(extraction.extraction_confidence, 0.0), 1.0),
        missing_fields=unusable_date_fields(extraction),
    )
    return profile, delta


def _correction_note(profile: CandidateProfile, delta: float | None) -> str:
    """The trace line that carries the E2 number into the eval."""
    parts = [
        f"skills={len(profile.skills)}",
        f"periods={len(profile.work_periods)}",
        f"bad_dates={len(profile.missing_fields)}",
    ]
    if delta is None:
        parts.append(
            f"years_llm={profile.llm_declared_years} "
            f"years_tool={profile.total_experience_years}"
        )
    else:
        parts.append(
            f"years_llm={profile.llm_declared_years:.2f} "
            f"years_tool={profile.total_experience_years:.2f} "
            f"correction={delta:.2f}"
        )
    return " ".join(parts)


def make_extract_node(
    llm: Any, *, today: date | None = None
) -> Callable[[ScreeningState], dict]:
    """Build the `extract` node bound to a model client."""

    def extract(state: ScreeningState) -> dict:
        started = time.perf_counter()
        extraction, usage = llm.parse(
            system=EXTRACT_SYSTEM, user=state.cv_text, schema=RawExtraction
        )
        profile, delta = build_profile(state, extraction, today=today)
        return {
            "path_taken": ["extract"],
            "profile": profile,
            "node_traces": [
                NodeTrace.of(
                    "extract", started, [usage], note=_correction_note(profile, delta)
                )
            ],
        }

    return extract


def make_repair_node(
    llm: Any, *, today: date | None = None
) -> Callable[[ScreeningState], dict]:
    """Build the `repair` node: re-extract, naming the fields that failed to parse."""

    def repair(state: ScreeningState) -> dict:
        started = time.perf_counter()
        broken = state.profile.missing_fields if state.profile else []
        instruction = REPAIR_TEMPLATE.format(
            fields="\n".join(f"  - {field}" for field in broken)
        )
        extraction, usage = llm.parse(
            system=EXTRACT_SYSTEM,
            user=f"{state.cv_text}\n\n{instruction}",
            schema=RawExtraction,
        )
        profile, delta = build_profile(state, extraction, today=today)
        attempt = state.repair_attempts + 1
        return {
            "path_taken": ["repair"],
            "profile": profile,
            "repair_attempts": attempt,
            "node_traces": [
                NodeTrace.of(
                    "repair",
                    started,
                    [usage],
                    note=(
                        f"attempt={attempt} before={len(broken)} "
                        f"after={len(profile.missing_fields)} "
                        f"{_correction_note(profile, delta)}"
                    ),
                )
            ],
        }

    return repair
```

- [x] **Step 4: Add `route_repair` to `src/graph/routes.py`**

Append to the file:

```python
def route_repair(state: ScreeningState) -> str:
    """Back to `repair` while a date the model wrote will not parse, up to the cap.

    The cap is the loop guard the spec asks for. Measured on real data it never
    fires: all 12 of 12 resumes that needed repair were fixed on the first attempt,
    so the second attempt is a safety net, not a workhorse.
    """
    profile = state.profile
    if profile is None:
        return "load_rubric"
    if profile.missing_fields and state.repair_attempts < state.max_repair_attempts:
        return "repair"
    return "load_rubric"
```

- [x] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/graph/test_extract.py`
Expected: `15 passed`

- [x] **Step 6: Commit**

```bash
git add src/graph/extract.py src/graph/routes.py tests/graph/test_extract.py
git commit -m "feat: add extract and repair nodes with deterministic experience correction"
```

---

### Task 6: load_rubric — a job description becomes data

**Files:**
- Create: `src/graph/rubric_nodes.py`
- Test: `tests/graph/test_rubric_nodes.py`

**Interfaces:**
- Consumes: `StubLLM`-compatible `.parse`; `JDRubric`, `Criterion`; `load_rubric`/`save_rubric` from `src/rubric/loader.py` (imported as `read_rubric` / `write_rubric` to leave the name `load_rubric` free for the node).
- Produces: `DERIVED_RUBRIC_DIR: Path`, `RUBRIC_SYSTEM: str`, `VALID_KINDS: frozenset[str]`; `RawCriterion`, `RawRubric`; `jd_fingerprint(jd_text: str) -> str`; `repair_rubric(raw: RawRubric) -> JDRubric`; `make_load_rubric_node(llm, *, derived_dir=DERIVED_RUBRIC_DIR) -> Callable`. Task 7 extends this same module.

Spec §6 says the rubric is data a recruiter edits, not a prompt. That is exactly right for the demo, where an HR user fills in a form. It leaves a hole for the eval, where 500 test rows carry 500 job descriptions nobody is going to hand-author rubrics for. This node closes the hole with a three-level lookup, cheapest first:

1. `state.rubric` already set → the recruiter supplied it. No model call.
2. `data/rubrics/derived/<sha256(jd)[:16]>.yaml` exists → read it. No model call.
3. Otherwise derive one, repair it deterministically, and **write it to that YAML path**, where a recruiter can open and edit it exactly like a hand-authored one.

Keying on a hash of the JD text is what makes this cheap: `test_500.jsonl` has **69 distinct JDs across its 500 rows**, and `dev_300.jsonl` has 159 across 300. The full test run costs 69 derivations, not 500.

`repair_rubric` is not optional politeness — `JDRubric` rejects weights that do not sum to 1.0 within 1e-6, and the model produced a valid sum on only **16 of 20** real JDs, worst case 1.100. Without renormalisation the eval dies partway through with a `ValidationError`.

- [x] **Step 1: Write the failing test**

Create `tests/graph/test_rubric_nodes.py`:

```python
import pytest

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.state import ScreeningState
from src.graph.rubric_nodes import (
    RawCriterion,
    RawRubric,
    jd_fingerprint,
    make_load_rubric_node,
    repair_rubric,
)
from tests.graph.fixtures.poisoned import CLEAN_CV
from tests.graph.stub import StubLLM

JD = "We need a Backend Engineer with 3 years of Python and PostgreSQL."


def state(jd: str = JD) -> ScreeningState:
    return ScreeningState(cv_text=CLEAN_CV, jd_text=jd)


def raw(*criteria: RawCriterion, job_title: str = "Backend Engineer") -> RawRubric:
    return RawRubric(job_title=job_title, criteria=list(criteria))


def criterion(identifier: str, weight: float, **overrides) -> RawCriterion:
    base = dict(
        id=identifier,
        description=f"{identifier} requirement",
        weight=weight,
        must_have=False,
        kind="skill",
        skill_terms=[],
    )
    base.update(overrides)
    return RawCriterion(**base)


def test_the_fingerprint_is_stable_and_jd_specific():
    assert jd_fingerprint(JD) == jd_fingerprint(JD)
    assert jd_fingerprint(JD) != jd_fingerprint(JD + " Remote.")
    assert len(jd_fingerprint(JD)) == 16


def test_weights_that_do_not_sum_to_one_are_renormalised():
    rubric = repair_rubric(raw(criterion("a", 0.6), criterion("b", 0.5)))

    assert sum(c.weight for c in rubric.criteria) == pytest.approx(1.0, abs=1e-9)
    assert rubric.criteria[0].weight == pytest.approx(0.6 / 1.1, abs=1e-6)


def test_weights_that_already_sum_to_one_are_left_alone():
    rubric = repair_rubric(raw(criterion("a", 0.7), criterion("b", 0.3)))

    assert rubric.criteria[0].weight == pytest.approx(0.7, abs=1e-9)
    assert rubric.criteria[1].weight == pytest.approx(0.3, abs=1e-9)


def test_all_zero_weights_still_produce_a_valid_rubric():
    rubric = repair_rubric(raw(criterion("a", 0.0), criterion("b", 0.0)))

    assert sum(c.weight for c in rubric.criteria) == pytest.approx(1.0, abs=1e-9)


def test_duplicate_ids_are_disambiguated_rather_than_rejected():
    rubric = repair_rubric(raw(criterion("python", 0.5), criterion("python", 0.5)))

    assert [c.id for c in rubric.criteria] == ["python", "python_2"]


def test_an_unknown_kind_falls_back_to_other():
    rubric = repair_rubric(raw(criterion("a", 1.0, kind="hard_skill")))

    assert rubric.criteria[0].kind == "other"


def test_blank_skill_terms_are_dropped():
    rubric = repair_rubric(raw(criterion("a", 1.0, skill_terms=["Python", "  ", ""])))

    assert rubric.criteria[0].skill_terms == ["Python"]


def test_an_empty_criteria_list_becomes_a_single_overall_criterion():
    rubric = repair_rubric(raw())

    assert len(rubric.criteria) == 1
    assert rubric.criteria[0].weight == pytest.approx(1.0, abs=1e-9)


def test_a_recruiter_supplied_rubric_skips_the_model_entirely(tmp_path):
    supplied = JDRubric(
        job_title="Supplied",
        criteria=[Criterion(id="only", description="only", weight=1.0)],
    )
    before = state()
    before.rubric = supplied
    stub = StubLLM([])

    update = make_load_rubric_node(stub, derived_dir=tmp_path)(before)

    assert update["path_taken"] == ["load_rubric"]
    assert "rubric" not in update
    assert stub.calls == []
    assert "supplied" in update["node_traces"][0].note


def test_a_derived_rubric_is_written_to_yaml_for_the_recruiter(tmp_path):
    stub = StubLLM([raw(criterion("python", 0.6, must_have=True, skill_terms=["Python"]),
                        criterion("sql", 0.4, skill_terms=["PostgreSQL"]))])

    update = make_load_rubric_node(stub, derived_dir=tmp_path)(state())

    written = tmp_path / f"{jd_fingerprint(JD)}.yaml"
    assert written.exists()
    assert update["rubric"].job_title == "Backend Engineer"
    assert len(update["rubric"].criteria) == 2
    assert update["node_traces"][0].llm_calls == 1


def test_a_second_row_with_the_same_jd_reads_the_yaml_instead_of_calling_the_model(tmp_path):
    first = StubLLM([raw(criterion("python", 1.0, skill_terms=["Python"]))])
    make_load_rubric_node(first, derived_dir=tmp_path)(state())

    second = StubLLM([])
    update = make_load_rubric_node(second, derived_dir=tmp_path)(state())

    assert second.calls == []
    assert update["rubric"].criteria[0].id == "python"
    assert update["rubric"].criteria[0].skill_terms == ["Python"]
    assert "reused" in update["node_traces"][0].note
    assert update["node_traces"][0].llm_calls == 0


def test_a_different_jd_does_not_reuse_another_jds_rubric(tmp_path):
    make_load_rubric_node(
        StubLLM([raw(criterion("python", 1.0))]), derived_dir=tmp_path
    )(state())

    stub = StubLLM([raw(criterion("java", 1.0), job_title="Java Dev")])
    update = make_load_rubric_node(stub, derived_dir=tmp_path)(state(jd="Java role."))

    assert update["rubric"].job_title == "Java Dev"
    assert len(stub.calls) == 1


def test_the_model_sees_the_job_description(tmp_path):
    stub = StubLLM([raw(criterion("python", 1.0))])

    make_load_rubric_node(stub, derived_dir=tmp_path)(state())

    assert stub.calls[0]["user"] == JD
    assert stub.calls[0]["schema"] == "RawRubric"
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/graph/test_rubric_nodes.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.graph.rubric_nodes'`

- [x] **Step 3: Implement `src/graph/rubric_nodes.py`**

```python
"""The job-description half of the graph: load_rubric, must_have_check, reject_fast.

Spec section 6 wants the rubric to be data a recruiter edits, not a prompt. That is
right for the demo and leaves a hole for the eval, where 500 test rows carry job
descriptions nobody will hand-author rubrics for. The hole is closed by deriving a
rubric once per distinct job description and writing it to YAML, where a recruiter
can open and edit it exactly like a hand-authored one.

Keying the YAML on a hash of the job description text is what makes that affordable:
`test_500.jsonl` holds 500 rows but only 69 distinct job descriptions, and
`dev_300.jsonl` holds 159 across 300.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.state import ScreeningState
from src.contracts.trace import NodeTrace
from src.rubric.loader import load_rubric as read_rubric
from src.rubric.loader import save_rubric as write_rubric

DERIVED_RUBRIC_DIR = Path("data/rubrics/derived")

VALID_KINDS = frozenset({"skill", "experience_years", "education", "domain", "other"})

RUBRIC_SYSTEM = (
    "Turn the job description into 4 to 8 weighted screening criteria. Mark a "
    "criterion must_have only if the job description states it as a hard requirement. "
    "For kind=skill criteria, list the concrete skill names in skill_terms; leave "
    "skill_terms empty for every other kind. Weights are relative importance and "
    "should sum to about 1.0."
)


class RawCriterion(BaseModel):
    """One criterion as the model writes it, before validation."""

    id: str = Field(description="snake_case identifier, unique within the rubric")
    description: str
    weight: float = Field(description="relative importance, 0 to 1")
    must_have: bool
    kind: str = Field(
        description="one of: skill, experience_years, education, domain, other"
    )
    skill_terms: list[str] = Field(
        description="concrete skill names for kind=skill; empty for other kinds"
    )


class RawRubric(BaseModel):
    """The load_rubric node's response schema."""

    job_title: str
    criteria: list[RawCriterion]


def jd_fingerprint(jd_text: str) -> str:
    """A stable 16-hex-character id for one job description."""
    return hashlib.sha256(jd_text.encode("utf-8")).hexdigest()[:16]


def repair_rubric(raw: RawRubric) -> JDRubric:
    """Force a model-written rubric into a valid `JDRubric`.

    `JDRubric` requires weights summing to 1.0 within 1e-6, and the model managed
    that unaided on only 16 of 20 real job descriptions (worst sum 1.100), so
    renormalising is what stops the eval dying halfway through with a
    ValidationError. Ids and kinds are coerced for the same reason: neither went
    wrong in the sample, but neither is guaranteed, and a coerced value beats a
    crashed run.
    """
    incoming = raw.criteria or [
        RawCriterion(
            id="overall_fit",
            description="Overall fit for the role",
            weight=1.0,
            must_have=False,
            kind="other",
            skill_terms=[],
        )
    ]
    total = sum(max(item.weight, 0.0) for item in incoming) or float(len(incoming))

    seen: set[str] = set()
    criteria: list[Criterion] = []
    for item in incoming:
        identifier = item.id.strip() or "criterion"
        suffix = 2
        while identifier in seen:
            identifier = f"{item.id.strip() or 'criterion'}_{suffix}"
            suffix += 1
        seen.add(identifier)
        criteria.append(
            Criterion(
                id=identifier,
                description=item.description.strip() or identifier,
                weight=max(item.weight, 0.0) / total,
                must_have=item.must_have,
                kind=item.kind if item.kind in VALID_KINDS else "other",
                skill_terms=[term for term in item.skill_terms if term.strip()],
            )
        )

    # Absorb float drift (and the all-zero-weights case) into the first criterion.
    drift = 1.0 - sum(item.weight for item in criteria)
    criteria[0] = criteria[0].model_copy(
        update={"weight": min(1.0, max(0.0, criteria[0].weight + drift))}
    )
    return JDRubric(
        job_title=raw.job_title.strip() or "Unspecified role", criteria=criteria
    )


def make_load_rubric_node(
    llm: Any, *, derived_dir: Path | str = DERIVED_RUBRIC_DIR
) -> Callable[[ScreeningState], dict]:
    """Build the `load_rubric` node: supplied, then cached on disk, then derived."""

    def load_rubric(state: ScreeningState) -> dict:
        started = time.perf_counter()

        if state.rubric is not None:
            return {
                "path_taken": ["load_rubric"],
                "node_traces": [
                    NodeTrace.of("load_rubric", started, note="supplied by caller")
                ],
            }

        path = Path(derived_dir) / f"{jd_fingerprint(state.jd_text)}.yaml"
        if path.exists():
            rubric = read_rubric(path)
            return {
                "path_taken": ["load_rubric"],
                "rubric": rubric,
                "node_traces": [
                    NodeTrace.of("load_rubric", started, note=f"reused {path.name}")
                ],
            }

        raw, usage = llm.parse(
            system=RUBRIC_SYSTEM, user=state.jd_text, schema=RawRubric
        )
        rubric = repair_rubric(raw)
        write_rubric(rubric, path)
        return {
            "path_taken": ["load_rubric"],
            "rubric": rubric,
            "node_traces": [
                NodeTrace.of(
                    "load_rubric",
                    started,
                    [usage],
                    note=(
                        f"derived {len(rubric.criteria)} criteria, "
                        f"{len(rubric.must_haves())} must-have"
                    ),
                )
            ],
        }

    return load_rubric
```

- [x] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/graph/test_rubric_nodes.py`
Expected: `13 passed`

- [x] **Step 5: Prove the derived YAML is recruiter-readable**

Run:
```bash
python -c "
from pathlib import Path
from src.graph.rubric_nodes import RawCriterion, RawRubric, repair_rubric
from src.rubric.loader import save_rubric
raw = RawRubric(job_title='Backend Engineer', criteria=[
    RawCriterion(id='python', description='3 years of Python', weight=0.6, must_have=True, kind='skill', skill_terms=['Python']),
    RawCriterion(id='sql', description='PostgreSQL', weight=0.5, must_have=False, kind='skill', skill_terms=['PostgreSQL']),
])
rubric = repair_rubric(raw)
out = Path('data/rubrics/derived/_smoke.yaml')
save_rubric(rubric, out)
print(out.read_text(encoding='utf-8'))
print('weights sum:', sum(c.weight for c in rubric.criteria))
out.unlink()
"
```
Expected: readable YAML with `skill_terms` present on each criterion, and `weights sum: 1.0`. The raw 1.1 has been renormalised to 0.5454.../0.4545....

- [x] **Step 6: Commit**

```bash
git add src/graph/rubric_nodes.py tests/graph/test_rubric_nodes.py
git commit -m "feat: derive a weighted rubric from a job description and cache it as YAML"
```

---

### Task 7: must_have_check, reject_fast, and the must-have edge

**Files:**
- Modify: `src/graph/rubric_nodes.py` (append)
- Modify: `src/graph/routes.py` (add `route_must_have`)
- Modify: `tests/graph/test_rubric_nodes.py` (append)

**Interfaces:**
- Consumes: `skills_match`, `expand_skill` from `src/tools/skills.py`; `search_evidence`; `CandidateProfile`.
- Produces: `required_years(description: str) -> float | None` (Task 8 imports this); `blocking_must_haves(state) -> list[str]`; the nodes `must_have_check(state) -> dict` and `reject_fast(state) -> dict`; the route `route_must_have(state) -> str` returning `"reject_fast"` or `"score_criteria"`.

This gate runs **before** any criterion is scored, so it cannot use scores — a fact that is easy to get wrong, because `Scorecard.missing_must_haves` exists and looks like the right thing. It is not: that field is computed *after* scoring and feeds `decide`, not this edge.

The measured behaviour, on 24 real pairs:

| | |
|---|---|
| must-haves per rubric, of which pre-checkable | median 3, median 2 checkable |
| rubrics where the gate cannot fire at all | 3 / 24 |
| gate fires using extracted skills only | 13 / 24 = 54 % |
| gate fires with the `expand_skill` + `search_evidence` fallback | 10 / 24 = **42 %** |
| criteria rescued by the fallback | 8 |
| true `Good Fit` rows rejected | **1 / 4** |

The fallback exists because the extractor drops skills that are plainly present in the CV text; asking the tools directly recovers them. The one false reject is a JD requiring Excel against a CV that never mentions Excel in any surface form — the gate is right about the rubric and the dataset label disagrees. **Leave it. Report it.** Tuning `must_have_min_score` or making `skill_terms` ALL-instead-of-ANY is Day 4 work with its own measurement.

- [x] **Step 1: Write the failing test**

Append to `tests/graph/test_rubric_nodes.py`:

```python
from src.contracts.screening import CandidateProfile, FitLabel
from src.graph.rubric_nodes import (
    blocking_must_haves,
    must_have_check,
    reject_fast,
    required_years,
)
from src.graph.routes import route_must_have


def profile(skills: list[str], years: float | None = 5.0) -> CandidateProfile:
    return CandidateProfile(
        raw_text=CLEAN_CV,
        skills=skills,
        total_experience_years=years,
        extraction_confidence=0.9,
    )


def gated(rubric: JDRubric, skills: list[str], years: float | None = 5.0) -> ScreeningState:
    before = state()
    before.rubric = rubric
    before.profile = profile(skills, years)
    return before


def rubric_of(*criteria: Criterion) -> JDRubric:
    return JDRubric(job_title="Backend Engineer", criteria=list(criteria))


def skill_must_have(identifier: str, *terms: str, weight: float = 1.0) -> Criterion:
    return Criterion(
        id=identifier,
        description=f"Experience with {', '.join(terms)}",
        weight=weight,
        must_have=True,
        kind="skill",
        skill_terms=list(terms),
    )


def test_required_years_reads_the_number_out_of_a_description():
    assert required_years("At least 3 years of professional experience") == 3.0
    assert required_years("5+ yrs backend") == 5.0
    assert required_years("Strong backend experience") is None


def test_a_present_skill_does_not_block():
    rubric = rubric_of(skill_must_have("lang", "Python"))

    assert blocking_must_haves(gated(rubric, ["Python", "Django"])) == []


def test_an_alias_of_a_present_skill_does_not_block():
    rubric = rubric_of(skill_must_have("frontend", "React"))

    assert blocking_must_haves(gated(rubric, ["ReactJS"])) == []


def test_a_skill_the_extractor_missed_is_rescued_from_the_raw_text():
    # Django is in CLEAN_CV but deliberately absent from the extracted skills.
    rubric = rubric_of(skill_must_have("web", "Django"))

    assert blocking_must_haves(gated(rubric, ["Python"])) == []


def test_a_skill_that_is_nowhere_in_the_cv_blocks():
    rubric = rubric_of(skill_must_have("infra", "Kubernetes"))

    assert blocking_must_haves(gated(rubric, ["Python"])) == ["infra"]


def test_any_one_of_several_skill_terms_is_enough():
    rubric = rubric_of(skill_must_have("lang", "Go", "Python"))

    assert blocking_must_haves(gated(rubric, ["Python"])) == []


def test_too_few_years_blocks():
    rubric = rubric_of(
        Criterion(
            id="seniority",
            description="At least 8 years of experience",
            weight=1.0,
            must_have=True,
            kind="experience_years",
        )
    )

    assert blocking_must_haves(gated(rubric, ["Python"], years=3.0)) == ["seniority"]
    assert blocking_must_haves(gated(rubric, ["Python"], years=8.0)) == []


def test_a_must_have_with_nothing_checkable_is_left_to_the_scorer():
    rubric = rubric_of(
        Criterion(
            id="culture",
            description="Strong communication skills",
            weight=1.0,
            must_have=True,
            kind="other",
        )
    )

    assert blocking_must_haves(gated(rubric, [])) == []


def test_a_skill_must_have_with_no_skill_terms_is_left_to_the_scorer():
    rubric = rubric_of(
        Criterion(id="vague", description="Backend skills", weight=1.0, must_have=True, kind="skill")
    )

    assert blocking_must_haves(gated(rubric, [])) == []


def test_a_criterion_that_is_not_a_must_have_never_blocks():
    rubric = rubric_of(
        Criterion(
            id="nice", description="Kubernetes", weight=1.0, kind="skill", skill_terms=["Kubernetes"]
        )
    )

    assert blocking_must_haves(gated(rubric, ["Python"])) == []


def test_must_have_check_writes_the_blocking_ids_onto_the_state():
    rubric = rubric_of(skill_must_have("infra", "Kubernetes"))

    update = must_have_check(gated(rubric, ["Python"]))

    assert update["path_taken"] == ["must_have_check"]
    assert update["blocking_must_haves"] == ["infra"]
    assert "blocking=infra" in update["node_traces"][0].note


def test_reject_fast_produces_a_terminal_result_naming_the_criteria():
    before = gated(rubric_of(skill_must_have("infra", "Kubernetes")), ["Python"])
    before.blocking_must_haves = ["infra"]
    before.path_taken = ["ingest", "guard", "extract", "load_rubric", "must_have_check"]

    result = reject_fast(before)["result"]

    assert result.label is FitLabel.NO_FIT
    assert result.overall_score == 0.0
    assert "infra" in result.rejected_reason
    assert result.path_taken[-1] == "reject_fast"


def test_route_must_have_rejects_when_something_blocks():
    before = state()
    before.blocking_must_haves = ["infra"]

    assert route_must_have(before) == "reject_fast"


def test_route_must_have_scores_when_nothing_blocks():
    assert route_must_have(state()) == "score_criteria"
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/graph/test_rubric_nodes.py`
Expected: FAIL — `ImportError: cannot import name 'blocking_must_haves'`

- [x] **Step 3: Append to `src/graph/rubric_nodes.py`**

Add these imports at the top of the file:

```python
import re

from src.contracts.screening import CandidateProfile, FitLabel, ScreeningResult
from src.tools.evidence import search_evidence
from src.tools.skills import expand_skill, skills_match
```

then append:

```python
_YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years|yrs)")


def required_years(description: str) -> float | None:
    """The number of years a criterion description asks for, if it names one."""
    match = _YEARS_RE.search(description.lower())
    return float(match.group(1)) if match else None


def _has_skill(terms: list[str], profile: CandidateProfile, cv_text: str) -> bool:
    """Does the candidate have any of `terms`? Extracted skills first, then the text.

    The text fallback is worth 12 percentage points: on 24 real pairs the gate fired
    on 54% using extracted skills alone and 42% with the fallback, rescuing 8
    criteria. The extractor drops skills that are plainly in the CV, and asking
    `expand_skill` plus `search_evidence` directly finds them again.
    """
    for term in terms:
        for held in profile.skills:
            if skills_match(term, held):
                return True
    for term in terms:
        for surface in expand_skill(term):
            if search_evidence(cv_text, surface, max_results=1):
                return True
    return False


def blocking_must_haves(state: ScreeningState) -> list[str]:
    """Must-have criteria the candidate demonstrably fails, before any scoring.

    Only `skill` criteria that name `skill_terms` and `experience_years` criteria
    that name a number can be judged without a score. Everything else is left to
    `score_criteria` -- a gate that guesses is worse than a gate that abstains.

    This is NOT `Scorecard.missing_must_haves`: that one is computed after scoring
    and feeds `decide`. Confusing the two puts the whole rubric behind a gate that
    has no scores to read.
    """
    rubric, profile = state.rubric, state.profile
    if rubric is None or profile is None:
        return []

    blocking: list[str] = []
    for criterion in rubric.must_haves():
        if criterion.kind == "skill" and criterion.skill_terms:
            if not _has_skill(criterion.skill_terms, profile, state.cv_text):
                blocking.append(criterion.id)
        elif criterion.kind == "experience_years":
            required = required_years(criterion.description)
            if required is None:
                continue
            have = max(
                profile.total_experience_years or 0.0, profile.llm_declared_years or 0.0
            )
            if have + 1e-9 < required:
                blocking.append(criterion.id)
    return blocking


def must_have_check(state: ScreeningState) -> dict:
    """Run the pre-scoring gate and record what it found."""
    started = time.perf_counter()
    blocking = blocking_must_haves(state)
    checked = len(state.rubric.must_haves()) if state.rubric else 0
    return {
        "path_taken": ["must_have_check"],
        "blocking_must_haves": blocking,
        "node_traces": [
            NodeTrace.of(
                "must_have_check",
                started,
                note=f"must_haves={checked} blocking={','.join(blocking) or 'none'}",
            )
        ],
    }


def reject_fast(state: ScreeningState) -> dict:
    """Terminal node for a candidate missing a hard requirement.

    The shortcut is the point: no scoring call is made, which is where the token
    saving in spec section 7's cost argument comes from.
    """
    started = time.perf_counter()
    reason = "missing must-have criteria: " + ", ".join(state.blocking_must_haves)
    return {
        "path_taken": ["reject_fast"],
        "result": ScreeningResult(
            overall_score=0.0,
            label=FitLabel.NO_FIT,
            rejected_reason=reason,
            path_taken=[*state.path_taken, "reject_fast"],
        ),
        "node_traces": [NodeTrace.of("reject_fast", started, note=reason)],
    }
```

- [x] **Step 4: Add `route_must_have` to `src/graph/routes.py`**

```python
def route_must_have(state: ScreeningState) -> str:
    """A candidate failing a hard requirement skips scoring entirely."""
    return "reject_fast" if state.blocking_must_haves else "score_criteria"
```

- [x] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/graph/test_rubric_nodes.py`
Expected: `27 passed` (13 from Task 6 plus 14 here)

- [x] **Step 6: Commit**

```bash
git add src/graph/rubric_nodes.py src/graph/routes.py tests/graph/test_rubric_nodes.py
git commit -m "feat: add the pre-scoring must-have gate and the reject_fast shortcut"
```

---
### Task 8: score_criteria — three tools and a model, per criterion

**Files:**
- Create: `src/graph/scoring.py`
- Test: `tests/graph/test_scoring.py`

**Interfaces:**
- Consumes: `StubLLM`-compatible `.parse`; `required_years` from `src/graph/rubric_nodes.py`; `search_evidence`, `expand_skill`; `CriterionScore`, `Evidence`.
- Produces: `SCORE_SYSTEM: str`; `RawScore`, `RawScores`; `deterministic_skill_evidence(cv_text, rubric) -> dict[str, list[Evidence]]`; `make_score_criteria_node(llm) -> Callable`. Task 9 extends this same module.

This is the node the spec's tool table is about — three of the five tools run here:

- **`normalize_skill` / `expand_skill`** finds each skill criterion's terms in the CV *before* the model sees it, trying surface forms **longest first** so a React Native CV is not scored as plain React.
- **`search_evidence`** resolves every quote the model writes back to character offsets. Measured on 122 real LLM-written quotes: 95 exact, 24 fuzzy, **3 unresolvable and dropped**. Naive `str.find` would have failed on **27 of 122 (22 %)**.
- **`calculate_experience`** does not merely correct the profile — it *sets the score* for every `experience_years` criterion. That is what carries the tool's contribution into the final number instead of leaving it in a field the model can ignore.

An unresolvable quote is **dropped, not stored**. Storing it would put a `start`/`end` in the record that does not slice back to the quote, and E1 would stop being an invariant the moment the first one landed.

- [x] **Step 1: Write the failing test**

Create `tests/graph/test_scoring.py`:

```python
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
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/graph/test_scoring.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.graph.scoring'`

- [x] **Step 3: Implement `src/graph/scoring.py`**

```python
"""score_criteria: where three of the five tools meet the model.

`expand_skill` finds each skill criterion's surface forms in the CV before the model
is asked anything, longest form first so a React Native CV is not scored as plain
React. `search_evidence` resolves every quote the model writes back to character
offsets -- measured on 122 real LLM-written quotes, 95 resolved exactly, 24 only
fuzzily, and 3 not at all; a naive `str.find` would have failed on 27 of them.
`calculate_experience` does not just correct the profile, it sets the score for
every experience_years criterion, which is how the tool's contribution reaches the
final number instead of stopping at a field the model can ignore.

A quote that will not resolve is dropped. Storing it would put offsets in the record
that do not slice back to the quote, and E1 would stop being an invariant the moment
the first one landed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from src.contracts.rubric import JDRubric
from src.contracts.screening import CriterionScore, Evidence
from src.contracts.state import ScreeningState
from src.contracts.trace import NodeTrace
from src.graph.rubric_nodes import required_years
from src.tools.evidence import search_evidence
from src.tools.skills import expand_skill

SCORE_SYSTEM = (
    "Score the resume against each criterion from 0.0 to 1.0. For every criterion, "
    "copy 1 to 3 quotes VERBATIM from the resume that justify the score. Copy the "
    "characters exactly as they appear, including missing spaces between words. If "
    "nothing in the resume supports the criterion, score 0.0 and return no quotes."
)


class RawScore(BaseModel):
    """One criterion's score as the model writes it."""

    criterion_id: str
    score: float = Field(description="0.0 to 1.0")
    quotes: list[str] = Field(description="verbatim spans copied from the resume")
    reasoning: str


class RawScores(BaseModel):
    """The score_criteria node's response schema."""

    scores: list[RawScore]


def deterministic_skill_evidence(
    cv_text: str, rubric: JDRubric
) -> dict[str, list[Evidence]]:
    """Find each skill criterion's terms in the CV, without asking the model.

    `expand_skill` returns surface forms longest first, so "react native" is tried
    before "react" and the first hit wins. A criterion with no hit is absent from
    the mapping rather than present with an empty list, so callers can tell
    "searched and found nothing" from "not a skill criterion".
    """
    hits: dict[str, list[Evidence]] = {}
    for criterion in rubric.criteria:
        if criterion.kind != "skill" or not criterion.skill_terms:
            continue
        found: list[Evidence] = []
        for term in criterion.skill_terms:
            for surface in expand_skill(term):
                spans = search_evidence(cv_text, surface, max_results=1)
                if spans:
                    found.append(spans[0])
                    break
        if found:
            hits[criterion.id] = found
    return hits


def _dedupe(spans: list[Evidence]) -> list[Evidence]:
    """Drop repeated spans, keeping first-seen order."""
    seen: set[tuple[int, int]] = set()
    unique: list[Evidence] = []
    for span in spans:
        key = (span.start, span.end)
        if key not in seen:
            seen.add(key)
            unique.append(span)
    return unique


def _score_prompt(
    state: ScreeningState, rubric: JDRubric, tool_hits: dict[str, list[Evidence]]
) -> str:
    """The criteria, their weights, what the tools already found, then the resume."""
    lines: list[str] = []
    for criterion in rubric.criteria:
        marker = ", MUST HAVE" if criterion.must_have else ""
        lines.append(
            f"- {criterion.id} (weight {criterion.weight:.2f}{marker}): "
            f"{criterion.description}"
        )
        for span in tool_hits.get(criterion.id, []):
            lines.append(f"    already found in the resume: {span.quote!r}")
    return "CRITERIA:\n" + "\n".join(lines) + f"\n\nRESUME:\n{state.cv_text}"


def _build_scores(
    state: ScreeningState,
    rubric: JDRubric,
    raw: RawScores,
    tool_hits: dict[str, list[Evidence]],
) -> tuple[dict[str, CriterionScore], int, int]:
    """Resolve the model's quotes and keep only criteria the rubric actually has."""
    known = {criterion.id for criterion in rubric.criteria}
    by_id: dict[str, CriterionScore] = {}
    resolved = dropped = 0

    for item in raw.scores:
        if item.criterion_id not in known:
            continue
        evidence = list(tool_hits.get(item.criterion_id, []))
        for quote in item.quotes:
            spans = search_evidence(state.cv_text, quote, max_results=1)
            if not spans:
                dropped += 1
                continue
            resolved += 1
            evidence.append(spans[0])
        by_id[item.criterion_id] = CriterionScore(
            criterion_id=item.criterion_id,
            score=min(max(item.score, 0.0), 1.0),
            evidence=_dedupe(evidence),
            reasoning=item.reasoning,
            tool_used=(
                "normalize_skill+search_evidence"
                if item.criterion_id in tool_hits
                else "search_evidence"
            ),
        )
    return by_id, resolved, dropped


def _override_experience_scores(
    by_id: dict[str, CriterionScore], state: ScreeningState, rubric: JDRubric
) -> int:
    """Let `calculate_experience` set the score for every experience_years criterion."""
    profile = state.profile
    if profile is None or profile.total_experience_years is None:
        return 0

    overridden = 0
    for criterion in rubric.criteria:
        if criterion.kind != "experience_years":
            continue
        required = required_years(criterion.description)
        if required is None or required <= 0:
            continue
        existing = by_id.get(criterion.id)
        by_id[criterion.id] = CriterionScore(
            criterion_id=criterion.id,
            score=round(min(1.0, profile.total_experience_years / required), 4),
            evidence=existing.evidence if existing else [],
            reasoning=(
                f"calculate_experience found {profile.total_experience_years:.2f} "
                f"years against a stated requirement of {required:.0f}"
            ),
            tool_used="calculate_experience",
        )
        overridden += 1
    return overridden


def make_score_criteria_node(llm: Any) -> Callable[[ScreeningState], dict]:
    """Build the `score_criteria` node bound to a model client."""

    def score_criteria(state: ScreeningState) -> dict:
        started = time.perf_counter()
        rubric = state.rubric
        if rubric is None:
            raise ValueError("score_criteria reached without a rubric")

        tool_hits = deterministic_skill_evidence(state.cv_text, rubric)
        raw, usage = llm.parse(
            system=SCORE_SYSTEM,
            user=_score_prompt(state, rubric, tool_hits),
            schema=RawScores,
        )
        by_id, resolved, dropped = _build_scores(state, rubric, raw, tool_hits)
        overridden = _override_experience_scores(by_id, state, rubric)

        return {
            "path_taken": ["score_criteria"],
            # Rubric order, so a run is comparable with any other run of the same rubric.
            "criterion_scores": [
                by_id[criterion.id] for criterion in rubric.criteria if criterion.id in by_id
            ],
            "node_traces": [
                NodeTrace.of(
                    "score_criteria",
                    started,
                    [usage],
                    note=(
                        f"scored={len(by_id)} quotes_resolved={resolved} "
                        f"quotes_dropped={dropped} "
                        f"tool_hits={sum(len(v) for v in tool_hits.values())} "
                        f"experience_overrides={overridden}"
                    ),
                )
            ],
        }

    return score_criteria
```

- [x] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/graph/test_scoring.py`
Expected: `13 passed`

- [x] **Step 5: Commit**

```bash
git add src/graph/scoring.py tests/graph/test_scoring.py
git commit -m "feat: add score_criteria with tool-found evidence and tool-set experience scores"
```

---

### Task 9: aggregate, deep_review, and the gray-zone edge

**Files:**
- Modify: `src/graph/scoring.py` (append)
- Modify: `src/graph/routes.py` (add `route_gray_zone`)
- Modify: `tests/graph/test_scoring.py` (append)

**Interfaces:**
- Consumes: `aggregate_scorecard` from `src/tools/scorecard.py`; `Scorecard`.
- Produces: `DEEP_REVIEW_SYSTEM: str`; `RawRevision`, `RawRevisions`; `aggregate(state) -> dict`; `needs_second_look(scores) -> list[CriterionScore]`; `make_deep_review_node(llm) -> Callable`; `route_gray_zone(state) -> str` returning `"deep_review"` or `"decide"`.

`aggregate` is a pure wrapper around the Day 2 tool — the weighted sum and both threshold cuts stay deterministic and auditable, which is the whole reason `aggregate_scorecard` is a tool rather than a paragraph of prompt.

`deep_review` runs **once and only once**, then edges unconditionally to `decide`. It re-aggregates internally rather than looping back to `aggregate`, so there is no cycle to guard and no second gray-zone test that could oscillate. Measured traffic at the default `gray_zone_margin = 0.05`: **2 / 20 = 10 %**. Setting the margin to `0.0` drives it to 0 % and turns the branch off — that is exactly the ablation spec §7 asks for, and it needs no code change.

Which criteria get a second look is deterministic: those with **no evidence at all**, or a score in the **middle band [0.3, 0.7]**. A gray-zone overall score is by definition a pile of middling criteria, and re-asking about a criterion scored 0.95 with three quotes behind it spends tokens to confirm what is already settled.

- [x] **Step 1: Write the failing test**

Append to `tests/graph/test_scoring.py`:

```python
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
```

`state()` from Task 8 needs one extra keyword. Change its signature at the top of the file to:

```python
def state(rubric: JDRubric = RUBRIC, years: float | None = 3.5) -> ScreeningState:
```

(it already reads that way — no edit needed; the new tests pass `rubric=` by keyword.)

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/graph/test_scoring.py`
Expected: FAIL — `ImportError: cannot import name 'aggregate'`

- [x] **Step 3: Append to `src/graph/scoring.py`**

Add these imports at the top of the file:

```python
from src.tools.scorecard import aggregate_scorecard
```

then append:

```python
# A criterion scored inside this band has not really been decided, so it is what a
# gray-zone second look should spend its tokens on.
SECOND_LOOK_BAND = (0.3, 0.7)

DEEP_REVIEW_SYSTEM = (
    "A first pass scored this resume against the criteria below and the overall score "
    "landed close to a decision threshold. Re-examine only the criteria listed, using "
    "the quotes already found and the resume text. Return a revised score from 0.0 to "
    "1.0 for each, and say briefly what changed your mind or confirmed the score."
)


class RawRevision(BaseModel):
    """One revised score from the second look."""

    criterion_id: str
    score: float = Field(description="0.0 to 1.0")
    reasoning: str


class RawRevisions(BaseModel):
    """The deep_review node's response schema."""

    revisions: list[RawRevision]


def aggregate(state: ScreeningState) -> dict:
    """Combine the criterion scores deterministically.

    A thin wrapper on purpose: the weighted sum and both threshold cuts decide the
    `reject_fast` and `deep_review` branches, so they have to be reproducible to the
    decimal and auditable outside the graph. That is why `aggregate_scorecard` is a
    tool and not a paragraph of prompt.
    """
    started = time.perf_counter()
    rubric = state.rubric
    if rubric is None:
        raise ValueError("aggregate reached without a rubric")

    card = aggregate_scorecard(state.criterion_scores, rubric)
    return {
        "path_taken": ["aggregate"],
        "scorecard": card,
        "node_traces": [
            NodeTrace.of(
                "aggregate",
                started,
                note=(
                    f"overall={card.overall_score:.4f} label={card.label.value} "
                    f"gray={card.in_gray_zone} "
                    f"missing_must_haves={len(card.missing_must_haves)} "
                    f"unscored={len(card.unscored_criteria)}"
                ),
            )
        ],
    }


def needs_second_look(scores: list[CriterionScore]) -> list[CriterionScore]:
    """Criteria worth re-asking about: no evidence, or a score in the middle band.

    Deterministic on purpose -- which criteria the second pass looks at must not
    itself depend on a model call, or the branch stops being reproducible.
    """
    low, high = SECOND_LOOK_BAND
    return [
        score
        for score in scores
        if not score.evidence or low <= score.score <= high
    ]


def _revision_prompt(state: ScreeningState, thin: list[CriterionScore]) -> str:
    descriptions = {
        criterion.id: criterion.description
        for criterion in (state.rubric.criteria if state.rubric else [])
    }
    lines: list[str] = []
    for score in thin:
        lines.append(
            f"- {score.criterion_id}: {descriptions.get(score.criterion_id, '')} "
            f"(current score {score.score:.2f})"
        )
        for span in score.evidence:
            lines.append(f"    quote already found: {span.quote!r}")
        if not score.evidence:
            lines.append("    no supporting quote was found")
    return (
        "CRITERIA TO RE-EXAMINE:\n"
        + "\n".join(lines)
        + f"\n\nRESUME:\n{state.cv_text}"
    )


def make_deep_review_node(llm: Any) -> Callable[[ScreeningState], dict]:
    """Build the `deep_review` node: one extra pass over the undecided criteria.

    It runs at most once per screening and then edges unconditionally to `decide`,
    re-aggregating internally rather than looping back to `aggregate`. That is why
    there is no cycle to guard here and no second gray-zone test to oscillate on.
    """

    def deep_review(state: ScreeningState) -> dict:
        started = time.perf_counter()
        rubric = state.rubric
        if rubric is None:
            raise ValueError("deep_review reached without a rubric")

        thin = needs_second_look(state.criterion_scores)
        if not thin:
            return {
                "path_taken": ["deep_review"],
                "node_traces": [
                    NodeTrace.of(
                        "deep_review", started, note="skipped: nothing thin to revise"
                    )
                ],
            }

        raw, usage = llm.parse(
            system=DEEP_REVIEW_SYSTEM,
            user=_revision_prompt(state, thin),
            schema=RawRevisions,
        )

        by_id = {score.criterion_id: score for score in state.criterion_scores}
        revised = 0
        for revision in raw.revisions:
            existing = by_id.get(revision.criterion_id)
            if existing is None:
                continue
            by_id[revision.criterion_id] = existing.model_copy(
                update={
                    "score": min(max(revision.score, 0.0), 1.0),
                    "reasoning": revision.reasoning,
                    "tool_used": "deep_review",
                }
            )
            revised += 1

        scores = [
            by_id[criterion.id] for criterion in rubric.criteria if criterion.id in by_id
        ]
        card = aggregate_scorecard(scores, rubric)
        return {
            "path_taken": ["deep_review"],
            "criterion_scores": scores,
            "scorecard": card,
            "node_traces": [
                NodeTrace.of(
                    "deep_review",
                    started,
                    [usage],
                    note=(
                        f"reviewed={len(thin)} revised={revised} "
                        f"overall={card.overall_score:.4f} label={card.label.value}"
                    ),
                )
            ],
        }

    return deep_review
```

- [x] **Step 4: Add `route_gray_zone` to `src/graph/routes.py`**

```python
def route_gray_zone(state: ScreeningState) -> str:
    """A score close to a threshold earns one more model pass; a clear one does not.

    `JDRubric.gray_zone_margin` is the knob: at the default 0.05 this fired on 2 of
    20 real pairs (10%), and at 0.0 it never fires, which is the ablation spec
    section 7 asks for.
    """
    card = state.scorecard
    return "deep_review" if card is not None and card.in_gray_zone else "decide"
```

- [x] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/graph/test_scoring.py`
Expected: `26 passed` (13 from Task 8 plus 13 here)

- [x] **Step 6: Commit**

```bash
git add src/graph/scoring.py src/graph/routes.py tests/graph/test_scoring.py
git commit -m "feat: add deterministic aggregation and the gray-zone deep review branch"
```

---

### Task 10: decide and rank

**Files:**
- Create: `src/graph/decide.py`
- Test: `tests/graph/test_decide.py`

**Interfaces:**
- Consumes: `Scorecard`, `ScreeningResult`, `NodeTrace`.
- Produces: `decide(state) -> dict`, `rank(state) -> dict`. Both are pure functions of the state; neither calls a model.

`decide` is where the run becomes a `ScreeningResult`: it copies the scorecard's verdict, sums every `NodeTrace` into the result's token and latency totals, and attaches the traces themselves so the Streamlit view and the eval read the same object.

`rank` orders `criterion_scores` by **weighted contribution, descending**. The graph screens one pair at a time, so there is no cohort to rank a candidate against; what a recruiter actually needs is the criteria in the order they moved the decision — the top of that list is the "why" line on the summary card, and the bottom is the gap. `Scorecard.weighted_contributions` already holds those numbers, computed deterministically by the Day 2 tool.

- [x] **Step 1: Write the failing test**

Create `tests/graph/test_decide.py`:

```python
import pytest

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.screening import CriterionScore, FitLabel
from src.contracts.state import ScreeningState
from src.contracts.trace import NodeTrace
from src.graph.decide import decide, rank
from src.graph.scoring import aggregate
from tests.graph.fixtures.poisoned import CLEAN_CV

RUBRIC = JDRubric(
    job_title="Backend Engineer",
    criteria=[
        Criterion(id="small", description="Nice to have", weight=0.2),
        Criterion(id="big", description="Core requirement", weight=0.8, must_have=True),
    ],
)


def decided(small: float = 1.0, big: float = 1.0) -> ScreeningState:
    before = ScreeningState(cv_text=CLEAN_CV, jd_text="jd")
    before.rubric = RUBRIC
    before.criterion_scores = [
        CriterionScore(criterion_id="small", score=small, reasoning="r"),
        CriterionScore(criterion_id="big", score=big, reasoning="r"),
    ]
    before.scorecard = aggregate(before)["scorecard"]
    before.path_taken = ["ingest", "guard", "extract", "load_rubric"]
    before.node_traces = [
        NodeTrace(node="extract", latency_ms=2000.0, prompt_tokens=1200,
                  completion_tokens=250, llm_calls=1),
        NodeTrace(node="score_criteria", latency_ms=3000.0, prompt_tokens=1600,
                  completion_tokens=300, cached_calls=1),
    ]
    return before


def test_decide_copies_the_scorecard_verdict():
    result = decide(decided())["result"]

    assert result.overall_score == 1.0
    assert result.label is FitLabel.GOOD_FIT
    assert result.rejected_reason is None


def test_decide_sums_the_traces_into_the_run_totals():
    result = decide(decided())["result"]

    assert result.prompt_tokens == 2800
    assert result.completion_tokens == 550
    assert result.latency_ms == pytest.approx(5000.0)
    assert result.llm_calls == 1
    assert result.cached_calls == 1
    assert [t.node for t in result.node_traces] == ["extract", "score_criteria"]


def test_decide_records_the_path_including_itself():
    result = decide(decided())["result"]

    assert result.path_taken[-1] == "decide"
    assert result.path_taken[0] == "ingest"


def test_decide_explains_a_must_have_that_scored_too_low():
    result = decide(decided(big=0.1))["result"]

    assert result.label is FitLabel.NO_FIT
    assert "big" in result.rejected_reason


def test_decide_needs_a_scorecard():
    bare = ScreeningState(cv_text="cv", jd_text="jd")

    with pytest.raises(ValueError, match="scorecard"):
        decide(bare)


def test_rank_orders_criteria_by_what_they_contributed():
    before = decided()
    before.result = decide(before)["result"]

    result = rank(before)["result"]

    assert [s.criterion_id for s in result.criterion_scores] == ["big", "small"]
    assert result.path_taken[-1] == "rank"


def test_rank_puts_a_high_weight_zero_score_last():
    before = decided(small=1.0, big=0.0)
    before.result = decide(before)["result"]

    result = rank(before)["result"]

    assert [s.criterion_id for s in result.criterion_scores] == ["small", "big"]


def test_rank_is_a_no_op_without_a_result():
    bare = ScreeningState(cv_text="cv", jd_text="jd")

    assert rank(bare)["path_taken"] == ["rank"]
    assert "result" not in rank(bare)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/graph/test_decide.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.graph.decide'`

- [x] **Step 3: Implement `src/graph/decide.py`**

```python
"""decide and rank: the run becomes a result, and the result becomes readable.

`rank` orders criteria by weighted contribution rather than ranking candidates
against each other, because the graph screens one pair at a time. What a recruiter
needs from a single screening is the criteria in the order they moved the decision:
the top of that list is the "why", the bottom is the gap.
`Scorecard.weighted_contributions` already holds those numbers, computed
deterministically by `aggregate_scorecard`.
"""

from __future__ import annotations

import time

from src.contracts.screening import ScreeningResult
from src.contracts.state import ScreeningState
from src.contracts.trace import NodeTrace


def decide(state: ScreeningState) -> dict:
    """Turn the scorecard and the traces into the run's `ScreeningResult`."""
    started = time.perf_counter()
    card = state.scorecard
    if card is None:
        raise ValueError("decide reached without a scorecard")

    reason = None
    if card.missing_must_haves:
        reason = "must-have criteria scored below threshold: " + ", ".join(
            card.missing_must_haves
        )

    result = ScreeningResult(
        overall_score=card.overall_score,
        label=card.label,
        criterion_scores=list(state.criterion_scores),
        rejected_reason=reason,
        path_taken=[*state.path_taken, "decide"],
        prompt_tokens=sum(trace.prompt_tokens for trace in state.node_traces),
        completion_tokens=sum(trace.completion_tokens for trace in state.node_traces),
        latency_ms=sum(trace.latency_ms for trace in state.node_traces),
        llm_calls=sum(trace.llm_calls for trace in state.node_traces),
        cached_calls=sum(trace.cached_calls for trace in state.node_traces),
        node_traces=list(state.node_traces),
    )
    return {
        "path_taken": ["decide"],
        "result": result,
        "node_traces": [
            NodeTrace.of(
                "decide",
                started,
                note=f"label={card.label.value} overall={card.overall_score:.4f}",
            )
        ],
    }


def rank(state: ScreeningState) -> dict:
    """Order the result's criteria by how much each one moved the overall score."""
    started = time.perf_counter()
    if state.result is None or state.scorecard is None:
        return {
            "path_taken": ["rank"],
            "node_traces": [
                NodeTrace.of("rank", started, note="skipped: no result to rank")
            ],
        }

    contributions = state.scorecard.weighted_contributions
    ordered = sorted(
        state.result.criterion_scores,
        key=lambda score: (-contributions.get(score.criterion_id, 0.0), score.criterion_id),
    )
    result = state.result.model_copy(
        update={
            "criterion_scores": ordered,
            "path_taken": [*state.result.path_taken, "rank"],
        }
    )
    top = ordered[0].criterion_id if ordered else "none"
    return {
        "path_taken": ["rank"],
        "result": result,
        "node_traces": [NodeTrace.of("rank", started, note=f"top_criterion={top}")],
    }
```

- [x] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/graph/test_decide.py`
Expected: `8 passed`

- [x] **Step 5: Commit**

```bash
git add src/graph/decide.py tests/graph/test_decide.py
git commit -m "feat: add decide and rank with per-node cost accounting"
```

---

### Task 11: assemble the graph

**Files:**
- Create: `src/graph/build.py`
- Test: `tests/graph/test_build.py`

**Interfaces:**
- Consumes: every node and route from Tasks 4–10; `StructuredLLM`.
- Produces: `NODE_NAMES: tuple[str, ...]`; `build_graph(llm=None, *, today=None, derived_dir=DERIVED_RUBRIC_DIR)`; `screen(cv_text, jd_text, *, llm=None, rubric=None, today=None, derived_dir=DERIVED_RUBRIC_DIR) -> ScreeningResult`; `graph_mermaid() -> str`. Task 12 and every later day consume `screen`.

The wiring, exactly as spec §4 draws it — thirteen nodes and **four** conditional edges:

| From | Kind | Route | Targets |
|---|---|---|---|
| `START` | static | — | `ingest` |
| `ingest` | static | — | `guard` |
| `guard` | **conditional** | `route_guard` | `quarantine`, `extract` |
| `quarantine` | static | — | `END` |
| `extract` | **conditional** | `route_repair` | `repair`, `load_rubric` |
| `repair` | **conditional** | `route_repair` | `repair`, `load_rubric` |
| `load_rubric` | static | — | `must_have_check` |
| `must_have_check` | **conditional** | `route_must_have` | `reject_fast`, `score_criteria` |
| `reject_fast` | static | — | `END` |
| `score_criteria` | static | — | `aggregate` |
| `aggregate` | **conditional** | `route_gray_zone` | `deep_review`, `decide` |
| `deep_review` | static | — | `decide` |
| `decide` | static | — | `rank` |
| `rank` | static | — | `END` |

`route_repair` is attached to two nodes but is one edge in the design: it is the loop, and the loop is what `max_repair_attempts` caps. The longest possible path is 13 node visits plus 2 repair re-entries, comfortably inside LangGraph's default recursion limit of 25 — no override needed, and Step 6 proves it rather than assuming it.

- [x] **Step 1: Write the failing test**

Create `tests/graph/test_build.py`:

```python
from datetime import date
from pathlib import Path

import pytest

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.screening import FitLabel
from src.graph.build import NODE_NAMES, build_graph, graph_mermaid, screen
from src.graph.extract import RawExtraction, RawPeriod
from src.graph.rubric_nodes import RawCriterion, RawRubric
from src.graph.scoring import RawScore, RawScores
from tests.graph.fixtures.poisoned import CLEAN_CV, poisoned_cvs
from tests.graph.stub import StubLLM

TODAY = date(2026, 9, 7)
JD = "Backend Engineer with 3 years of Python."


def extraction(**overrides) -> RawExtraction:
    base = dict(
        skills=["Python", "PostgreSQL"],
        work_periods=[
            RawPeriod(title="Backend Engineer", company="Acme Corp",
                      start="2019-06", end="2022-12")
        ],
        degrees=["B.S. Computer Science"],
        certifications=[],
        total_experience_years=2.0,
        extraction_confidence=0.95,
    )
    base.update(overrides)
    return RawExtraction(**base)


def rubric(*, must_have_skill: str = "Python") -> RawRubric:
    return RawRubric(
        job_title="Backend Engineer",
        criteria=[
            RawCriterion(id="lang", description="Python in production", weight=0.5,
                         must_have=True, kind="skill", skill_terms=[must_have_skill]),
            RawCriterion(id="seniority", description="At least 3 years of experience",
                         weight=0.5, must_have=False, kind="experience_years",
                         skill_terms=[]),
        ],
    )


def all_scores(lang: float = 0.9) -> RawScores:
    return RawScores(scores=[
        RawScore(criterion_id="lang", score=lang,
                 quotes=["Built payment APIs in Python and PostgreSQL"], reasoning="r"),
        RawScore(criterion_id="seniority", score=0.5, quotes=[], reasoning="r"),
    ])


def run(responses, cv: str = CLEAN_CV, jd: str = JD, tmp_path: Path | None = None,
        **kwargs):
    return screen(cv, jd, llm=StubLLM(responses), today=TODAY,
                  derived_dir=tmp_path or Path("data/rubrics/derived"), **kwargs)


def test_the_graph_has_the_thirteen_nodes_the_spec_names():
    assert NODE_NAMES == (
        "ingest", "guard", "quarantine", "extract", "repair", "load_rubric",
        "must_have_check", "reject_fast", "score_criteria", "aggregate",
        "deep_review", "decide", "rank",
    )
    drawn = set(build_graph(StubLLM([])).get_graph().nodes)
    assert set(NODE_NAMES) <= drawn


def test_the_happy_path_visits_the_nodes_in_order(tmp_path):
    result = run([extraction(), rubric(), all_scores()], tmp_path=tmp_path)

    assert result.path_taken == [
        "ingest", "guard", "extract", "load_rubric", "must_have_check",
        "score_criteria", "aggregate", "decide", "rank",
    ]
    assert result.label in set(FitLabel)


def test_the_happy_path_produces_verifiable_evidence(tmp_path):
    result = run([extraction(), rubric(), all_scores()], tmp_path=tmp_path)

    spans = [e for s in result.criterion_scores for e in s.evidence]
    assert spans
    for evidence in spans:
        assert CLEAN_CV[evidence.start : evidence.end] == evidence.quote


def test_the_experience_criterion_is_scored_by_the_tool(tmp_path):
    result = run([extraction(), rubric(), all_scores()], tmp_path=tmp_path)

    seniority = next(s for s in result.criterion_scores if s.criterion_id == "seniority")
    assert seniority.tool_used == "calculate_experience"
    assert seniority.score == 1.0  # 3.5 measured years against a stated 3


@pytest.mark.parametrize("rule_id,cv", poisoned_cvs())
def test_a_poisoned_cv_either_quarantines_or_is_flagged(rule_id, cv, tmp_path):
    result = run([extraction(), rubric(), all_scores()], cv=cv, tmp_path=tmp_path)

    quarantined = result.path_taken == ["ingest", "guard", "quarantine"]
    assert quarantined or result.path_taken[-1] == "rank"
    if quarantined:
        assert rule_id in result.rejected_reason


def test_an_empty_cv_is_quarantined_without_a_single_model_call(tmp_path):
    stub = StubLLM([])

    result = screen("   ", JD, llm=stub, today=TODAY, derived_dir=tmp_path)

    assert result.path_taken == ["ingest", "guard", "quarantine"]
    assert "empty_document" in result.rejected_reason
    assert stub.calls == []


def test_a_broken_date_sends_the_run_through_repair_once(tmp_path):
    broken = extraction(work_periods=[
        RawPeriod(title="Backend Engineer", company="Acme Corp", start="2019-06", end="null")
    ])

    result = run([broken, extraction(), rubric(), all_scores()], tmp_path=tmp_path)

    assert result.path_taken[:5] == [
        "ingest", "guard", "extract", "repair", "load_rubric",
    ]


def test_the_repair_loop_stops_at_the_cap(tmp_path):
    broken = extraction(work_periods=[
        RawPeriod(title="Backend Engineer", company="Acme Corp", start="2019-06", end="null")
    ])

    result = run([broken, broken, broken, rubric(), all_scores()], tmp_path=tmp_path)

    assert result.path_taken.count("repair") == 2
    assert "load_rubric" in result.path_taken


def test_a_missing_must_have_short_circuits_before_any_scoring(tmp_path):
    stub = StubLLM([extraction(skills=["COBOL"]), rubric(must_have_skill="Kubernetes")])

    result = screen(CLEAN_CV, JD, llm=stub, today=TODAY, derived_dir=tmp_path)

    assert result.path_taken == [
        "ingest", "guard", "extract", "load_rubric", "must_have_check", "reject_fast",
    ]
    assert result.label is FitLabel.NO_FIT
    assert "lang" in result.rejected_reason
    assert len(stub.calls) == 2  # extract and load_rubric only -- no scoring call


def test_a_recruiter_supplied_rubric_skips_derivation(tmp_path):
    supplied = JDRubric(
        job_title="Supplied",
        criteria=[Criterion(id="lang", description="Python", weight=1.0, kind="skill",
                            skill_terms=["Python"])],
    )
    stub = StubLLM([
        extraction(),
        RawScores(scores=[RawScore(criterion_id="lang", score=0.9, quotes=[], reasoning="r")]),
    ])

    result = screen(CLEAN_CV, JD, llm=stub, rubric=supplied, today=TODAY,
                    derived_dir=tmp_path)

    assert [call["schema"] for call in stub.calls] == ["RawExtraction", "RawScores"]
    assert result.path_taken[-1] == "rank"


def test_the_run_totals_add_up_to_the_traces(tmp_path):
    result = run([extraction(), rubric(), all_scores()], tmp_path=tmp_path)

    assert result.prompt_tokens == sum(t.prompt_tokens for t in result.node_traces)
    assert result.llm_calls == 3
    assert result.latency_ms > 0


def test_the_mermaid_source_marks_the_conditional_edges_as_dotted():
    diagram = graph_mermaid()

    assert "guard -.-> quarantine" in diagram
    assert "guard -.-> extract" in diagram
    assert "ingest --> guard" in diagram
    assert "repair -.-> repair" in diagram


def test_openai_is_imported_in_exactly_one_module():
    root = Path("src")
    importers = sorted(
        path.as_posix()
        for path in root.rglob("*.py")
        if "openai" in path.read_text(encoding="utf-8")
    )

    assert importers == ["src/llm/client.py"]
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/graph/test_build.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.graph.build'`

- [x] **Step 3: Implement `src/graph/build.py`**

```python
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
) -> ScreeningResult:
    """Screen one CV against one job description.

    `invoke` returns a plain dict, not a `ScreeningState`, so the output is
    re-validated before anything reads a field off it.
    """
    graph = build_graph(llm, today=today, derived_dir=derived_dir)
    output = graph.invoke(
        ScreeningState(cv_text=cv_text, jd_text=jd_text, rubric=rubric)
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
```

- [x] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/graph/test_build.py`
Expected: `20 passed` (12 plain plus 8 parametrised poisoned CVs)

- [x] **Step 5: Run the whole suite**

Run: `python -m pytest`
Expected: PASS, **3 deselected** (the two network tests from earlier days plus the new live-call test). Nothing from Days 1 and 2 may fail.

- [x] **Step 6: Prove the repair loop stays inside the recursion limit**

Run:
```bash
python -c "
from datetime import date
from pathlib import Path
import tempfile
from src.graph.build import screen
from src.graph.extract import RawExtraction, RawPeriod
from src.graph.rubric_nodes import RawCriterion, RawRubric
from src.graph.scoring import RawScore, RawScores
from tests.graph.stub import StubLLM
from tests.graph.fixtures.poisoned import CLEAN_CV

broken = RawExtraction(skills=['Python'], work_periods=[RawPeriod(title='Eng', company=None, start='nope', end='null')], degrees=[], certifications=[], total_experience_years=2.0, extraction_confidence=0.9)
rubric = RawRubric(job_title='Backend Engineer', criteria=[RawCriterion(id='lang', description='Python', weight=1.0, must_have=False, kind='skill', skill_terms=['Python'])])
scores = RawScores(scores=[RawScore(criterion_id='lang', score=0.9, quotes=['Python'], reasoning='r')])
with tempfile.TemporaryDirectory() as tmp:
    result = screen(CLEAN_CV, 'Backend Engineer with Python.', llm=StubLLM([broken, broken, broken, rubric, scores]), today=date(2026,9,7), derived_dir=Path(tmp))
print('nodes visited:', len(result.path_taken))
print('path:', ' -> '.join(result.path_taken))
"
```
Expected: `nodes visited: 11` and a path containing `repair -> repair` exactly once, ending in `rank`. No `GraphRecursionError`. If that error appears, the loop cap is not working — fix `route_repair`, do not raise the recursion limit.

- [x] **Step 7: Commit**

```bash
git add src/graph/build.py tests/graph/test_build.py
git commit -m "feat: assemble the screening graph with its four conditional edges"
```

---

### Task 12: measure the branches, draw the graph

**Files:**
- Create: `scripts/export_graph_diagram.py`
- Create: `scripts/measure_branch_traffic.py`
- Test: `tests/graph/test_scripts.py`

**Interfaces:**
- Consumes: `graph_mermaid`, `screen`, `NODE_NAMES`, `StructuredLLM`, `load_fit_split`-style JSONL reading.
- Produces: `export_graph_diagram.main(argv=None) -> int`; `measure_branch_traffic.summarise(results: list[ScreeningResult]) -> dict`, `measure_branch_traffic.main(argv=None) -> int`.

Spec §4 is blunt: *"Mỗi nhánh phải có % lưu lượng thật đo trên dataset — không có số thì nhánh đó là trang trí."* This task is what stops that sentence being an accusation. `summarise` is a pure function over finished results, so the arithmetic behind every percentage on the slide is unit-tested offline; `main` is the thin shell that runs the graph and prints it.

The expected shape of the answer, from the prototypes:

| Branch | Expected traffic | Where it came from |
|---|---|---|
| `guard` → `quarantine` | **0 %** on real data | Day 2: 0/300 resumes trip a rule. Synthetic fixtures only. |
| `extract` → `repair` | ~**40 %** | 12/30 resumes had an unparseable model date |
| `must_have_check` → `reject_fast` | ~**42 %** | 10/24 pairs failed a pre-checkable must-have |
| `aggregate` → `deep_review` | ~**10 %** | 2/20 pairs landed in the gray zone |

Those came from samples of 20–30 pairs, so the run over 300 will move them. **Record what the run says, not what this table says.** If a branch comes back at 0 % on real data, say so on the slide and label how it is exercised — that is the honest version of the spec's requirement, and it is the correct outcome for `guard`.

**Measured on execution, 2026-09-08, the FULL `dev_300.jsonl` (300 real pairs) — these are the numbers for the slide:**

| Branch | Runs | Traffic |
|---|---:|---:|
| `guard -> quarantine` | 0 | **0.0 %** — synthetic fixtures only; label it as such on the slide |
| `guard -> extract` | 300 | 100.0 % |
| `extract -> repair` | 9 | **3.0 %** |
| `must_have_check -> reject_fast` | 96 | **32.0 %** |
| `must_have_check -> score_criteria` | 204 | 68.0 % |
| `aggregate -> deep_review` | 51 | **17.0 %** |
| `aggregate -> decide` | 153 | 51.0 % |

Labels: 30 Good Fit, 79 Potential Fit, 191 No Fit. **1 203 546 tokens**, 4 012 per row, 567 live calls, 135 cache hits (the cache hits are the repeated job descriptions — 159 distinct JDs across 300 rows).

**The E2 headline, over 300 real pairs:** `calculate_experience` corrected the model on **208 of 300 rows**, median correction **3.75 years**, max **22.75 years**, and **150 of those corrections were a year or more**. That is the measured answer to "why is this a tool and not the LLM's job".

**Reproducibility (spec §8), measured 2026-09-08.** The split was run twice. The second run reported **567 cache hits and 0 misses**, and its branch-traffic table is **byte-identical** to the first. The spec §8 claim holds.

The two runs' *cost* lines differ, and the difference is by design rather than a cache failure — it was traced rather than assumed:

| | Run 1 (cold) | Run 2 (warm) |
|---|---:|---:|
| parse calls | 567 live + 135 cached = 702 | 0 live + 567 cached = 567 |
| tokens | 1 203 546 | 1 048 753 |

The 135 missing calls are **rubric derivations**. `load_rubric` has two cache layers, and they behave differently: the JSONL cache replays a call's recorded token usage, but the derived-rubric YAML layer skips the model *entirely*, so it contributes no tokens at all. `dev_300.jsonl` holds exactly **159 distinct JDs** and there are exactly **159 YAML files** on disk; 135 of them were written during run 1. 154 793 ÷ 135 = **1 147 tokens per derivation**, which is the right size for a ~2 400-character JD.

**Consequence for the slide: quote the cold-run cost.** A warm re-run understates the true cost by ~13 %, because the rubrics it reuses were paid for on a previous run.

Every conditional edge carries real traffic except `guard`, which is 0 % on clean data exactly as Day 2 predicted. Spec §4's rule is satisfied: no branch is undocumented decoration, and the one branch with no organic traffic is reported as such rather than dressed up.

**Measured on execution, 2026-09-08, first 25 rows only (kept as the smaller sample that caught the two defects):**

| Branch | Runs | Traffic |
|---|---:|---:|
| `guard -> quarantine` | 0 | **0.0 %** — as predicted; synthetic fixtures only |
| `guard -> extract` | 25 | 100.0 % |
| `extract -> repair` | 1 | **4.0 %** — the prediction of 40 % was wrong; see the superseded note above |
| `must_have_check -> reject_fast` | 8 | **32.0 %** — predicted 42 % |
| `must_have_check -> score_criteria` | 17 | 68.0 % |
| `aggregate -> deep_review` | 5 | **20.0 %** — predicted 10 % |
| `aggregate -> decide` | 12 | 48.0 % |

Labels: 2 Good Fit, 8 Potential Fit, 15 No Fit. 84 173 tokens, 3 367 per row.
`calculate_experience` corrected the model on **19 of 25 rows**, median **1.5 y**, max **15.08 y**, with **11** corrections of a year or more.

Step 7 also surfaced a second defect that this step's own output made visible: `reject_fast` and `quarantine` built their `ScreeningResult` without summing the traces, so the eight short-cut rows reported **zero tokens**. Spec §7's whole cost argument rests on those rows reporting what they actually spent. Fixed by `trace_totals` in `src/contracts/trace.py`, used by all three terminal nodes; total measured cost rose from 66 673 to 84 173 tokens over the same 25 rows.

- [x] **Step 1: Write the failing test**

Create `tests/graph/test_scripts.py`:

```python
from pathlib import Path

from src.contracts.screening import FitLabel, ScreeningResult
from src.contracts.trace import NodeTrace
from scripts.export_graph_diagram import main as export_main
from scripts.measure_branch_traffic import summarise


def result(path: list[str], label: FitLabel = FitLabel.NO_FIT, note: str = "") -> ScreeningResult:
    return ScreeningResult(
        overall_score=0.5,
        label=label,
        path_taken=path,
        prompt_tokens=1000,
        completion_tokens=200,
        llm_calls=2,
        node_traces=[NodeTrace(node="extract", note=note)] if note else [],
    )


HAPPY = ["ingest", "guard", "extract", "load_rubric", "must_have_check",
         "score_criteria", "aggregate", "decide", "rank"]


def test_a_branch_nothing_took_is_reported_as_zero_not_omitted():
    summary = summarise([result(HAPPY)])

    assert summary["branches"]["guard -> quarantine"] == {"count": 0, "pct": 0.0}
    assert summary["branches"]["aggregate -> deep_review"]["pct"] == 0.0


def test_traffic_is_counted_per_branch():
    summary = summarise([
        result(HAPPY),
        result(["ingest", "guard", "quarantine"]),
        result(["ingest", "guard", "extract", "repair", "load_rubric",
                "must_have_check", "reject_fast"]),
        result(HAPPY[:7] + ["deep_review", "decide", "rank"]),
    ])

    branches = summary["branches"]
    assert branches["guard -> quarantine"] == {"count": 1, "pct": 25.0}
    assert branches["extract -> repair"] == {"count": 1, "pct": 25.0}
    assert branches["must_have_check -> reject_fast"] == {"count": 1, "pct": 25.0}
    assert branches["aggregate -> deep_review"] == {"count": 1, "pct": 25.0}
    assert summary["rows"] == 4


def test_the_experience_correction_is_pulled_out_of_the_traces():
    summary = summarise([
        result(HAPPY, note="years_llm=2.00 years_tool=3.50 correction=1.50"),
        result(HAPPY, note="years_llm=5.00 years_tool=12.92 correction=7.92"),
        result(HAPPY, note="years_llm=None years_tool=None"),
    ])

    correction = summary["experience_correction"]
    assert correction["compared"] == 2
    assert correction["median"] == 4.71
    assert correction["max"] == 7.92


def test_labels_and_cost_are_summarised():
    summary = summarise([result(HAPPY, label=FitLabel.GOOD_FIT), result(HAPPY)])

    assert summary["labels"]["Good Fit"] == 1
    assert summary["labels"]["No Fit"] == 1
    assert summary["tokens"]["total"] == 2400
    assert summary["tokens"]["per_row"] == 1200.0


def test_summarising_nothing_does_not_divide_by_zero():
    summary = summarise([])

    assert summary["rows"] == 0
    assert summary["branches"]["guard -> quarantine"]["pct"] == 0.0


def test_the_diagram_script_writes_mermaid_without_calling_a_model(tmp_path):
    out = tmp_path / "graph.mmd"

    assert export_main(["--out", str(out)]) == 0

    text = out.read_text(encoding="utf-8")
    assert "graph TD" in text
    assert "guard -.-> quarantine" in text
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/graph/test_scripts.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.export_graph_diagram'`

- [x] **Step 3: Implement `scripts/export_graph_diagram.py`**

```python
"""Write the Mermaid source for slide 1, straight out of the compiled graph.

Spec section 9 asks that the diagram be generated from the graph rather than drawn
by hand, so it cannot drift from the code. Conditional edges come out dotted
(`-.->`), which is the distinction the slide colours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.graph.build import graph_mermaid

DEFAULT_OUT = Path("docs/diagrams/graph.mmd")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    diagram = graph_mermaid()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(diagram, encoding="utf-8")
    print(f"wrote {args.out} ({len(diagram.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 4: Implement `scripts/measure_branch_traffic.py`**

```python
"""Measure how much traffic each conditional edge actually carries.

Spec section 4: a branch with no measured traffic is decoration. This is the command
that produces the numbers -- and the one that is allowed to report 0%, because 0% is
a finding, not a failure. `guard` is expected to be exactly that on clean data.

`summarise` is a pure function over finished results so the arithmetic behind every
percentage on the slide is unit-tested offline; `main` only runs the graph and
prints what `summarise` returns.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from src.contracts.screening import ScreeningResult
from src.graph.build import screen
from src.llm.cache import DEFAULT_CACHE_PATH, JSONLCache
from src.llm.client import StructuredLLM

DEFAULT_SPLIT = Path("data/samples/dev_300.jsonl")
DEFAULT_TODAY = date(2026, 9, 7)

# Every conditional edge, so one that never fires is reported as 0% rather than
# quietly missing from the table.
BRANCHES: tuple[tuple[str, str], ...] = (
    ("guard", "quarantine"),
    ("guard", "extract"),
    ("extract", "repair"),
    ("must_have_check", "reject_fast"),
    ("must_have_check", "score_criteria"),
    ("aggregate", "deep_review"),
    ("aggregate", "decide"),
)

_CORRECTION_RE = re.compile(r"correction=([0-9]+\.[0-9]+)")


def _took(path: list[str], source: str, target: str) -> bool:
    """Did this run follow `source` immediately by `target`?"""
    return any(
        path[index] == source and path[index + 1] == target
        for index in range(len(path) - 1)
    )


def summarise(results: list[ScreeningResult]) -> dict[str, Any]:
    """Branch traffic, label mix, cost, and the size of the experience correction."""
    rows = len(results)
    share = (lambda count: round(100.0 * count / rows, 1)) if rows else (lambda _: 0.0)

    branches: dict[str, dict[str, Any]] = {}
    for source, target in BRANCHES:
        count = sum(1 for result in results if _took(result.path_taken, source, target))
        branches[f"{source} -> {target}"] = {"count": count, "pct": share(count)}

    corrections = [
        float(match.group(1))
        for result in results
        for trace in result.node_traces
        for match in [_CORRECTION_RE.search(trace.note)]
        if match
    ]

    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)
    return {
        "rows": rows,
        "branches": branches,
        "labels": dict(Counter(result.label.value for result in results)),
        "tokens": {
            "total": total_tokens,
            "per_row": round(total_tokens / rows, 1) if rows else 0.0,
            "llm_calls": sum(result.llm_calls for result in results),
            "cached_calls": sum(result.cached_calls for result in results),
        },
        "experience_correction": {
            "compared": len(corrections),
            "median": round(statistics.median(corrections), 2) if corrections else 0.0,
            "max": round(max(corrections), 2) if corrections else 0.0,
            "over_1y": sum(1 for value in corrections if value >= 1.0),
        },
    }


def _render(summary: dict[str, Any]) -> str:
    lines = [
        f"# Branch traffic over {summary['rows']} real pairs",
        "",
        "| Branch | Runs | Traffic |",
        "|---|---:|---:|",
    ]
    for name, stats in summary["branches"].items():
        lines.append(f"| `{name}` | {stats['count']} | {stats['pct']}% |")
    correction = summary["experience_correction"]
    lines += [
        "",
        f"Labels: {summary['labels']}",
        f"Tokens: {summary['tokens']['total']} total, "
        f"{summary['tokens']['per_row']} per row, "
        f"{summary['tokens']['llm_calls']} live calls, "
        f"{summary['tokens']['cached_calls']} cache hits",
        "",
        f"calculate_experience corrected the model on {correction['compared']} rows: "
        f"median {correction['median']}y, max {correction['max']}y, "
        f"{correction['over_1y']} corrections of a year or more",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=0, help="0 means the whole split")
    parser.add_argument("--out", type=Path, default=Path("docs/measurements/branch_traffic.md"))
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.split.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    llm = StructuredLLM(cache=JSONLCache(DEFAULT_CACHE_PATH))
    results: list[ScreeningResult] = []
    for index, row in enumerate(rows, start=1):
        results.append(
            screen(
                row["resume_text"],
                row["job_description_text"],
                llm=llm,
                today=DEFAULT_TODAY,
            )
        )
        if index % 25 == 0:
            print(f"  {index}/{len(rows)} (cache hits {llm.hits}, misses {llm.misses})")

    report = _render(summarise(results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print()
    print(report)
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/graph/test_scripts.py`
Expected: `6 passed`

- [x] **Step 6: Export the diagram**

Run: `python -m scripts.export_graph_diagram`
Expected: `wrote docs\diagrams\graph.mmd (...)`. Open it and confirm all four conditional edges are dotted and every one of the thirteen node names appears.

- [x] **Step 7: Measure the branches on a small slice first**

Run: `python -m scripts.measure_branch_traffic --limit 25`
Expected: a table with a non-zero number for `extract -> repair`, `must_have_check -> reject_fast` and `aggregate -> decide`, and `0.0%` for `guard -> quarantine`. Roughly 25 × 3 model calls; a couple of minutes and a few cents.

Read the numbers before continuing. If `extract -> repair` is 0 % or 100 %, the repair trigger has drifted from what was measured — stop and diagnose rather than running the full split.

- [x] **Step 8: Measure the full dev split**

Run: `python -m scripts.measure_branch_traffic`
Expected: the same table over 300 rows, written to `docs/measurements/branch_traffic.md`. Budget roughly 15–25 minutes and ~1.5 M tokens on the first run; a second run is served almost entirely from the cache.

Run it twice and confirm the second run reports mostly cache hits and an **identical** table. That is the spec §8 reproducibility claim, demonstrated rather than asserted.

- [x] **Step 9: Write the measured numbers into this plan**

Replace the "expected traffic" table in this task with what the run actually reported, and note the date. A future reader must be able to tell a measurement from a prediction.

- [x] **Step 10: Commit**

```bash
git add scripts/export_graph_diagram.py scripts/measure_branch_traffic.py tests/graph/test_scripts.py
git commit -m "feat: measure conditional edge traffic and export the graph diagram"
```

---

## Definition of Done

Day 3 is done when every one of these is true and each has been checked by running the command, not by remembering it:

**Every box below was re-verified by running its command on 2026-09-17**, on `day3-langgraph-graph` at `6fd8d49`, fourteen commits ahead of `main`. Recorded there: `pytest` 362 passed / 3 deselected; `pytest -m network` 3 passed in 9.76 s; `export_graph_diagram` rewrote all thirteen nodes with the four conditional edges dotted and left the file byte-identical; all seven branches in `measure_branch_traffic.BRANCHES` carry a percentage in `branch_traffic.md`; the key in `.env` appears in no tracked file, no artefact and nowhere in `git log --all -p`.

- [x] `python -m pytest` passes with 3 deselected. Measured on execution: **362 passed** (210 from Days 1-2, minus the deleted `visit` test, plus 153 new — including the 7 tests added by the two defects Task 12 Step 7 uncovered).
- [x] `python -m pytest -m network` passes (3 selected; 1 of them makes the live `gpt-4o-mini` call).
- [x] `python -m scripts.export_graph_diagram` writes a Mermaid diagram containing all thirteen node names, with all four conditional edges dotted.
- [x] `python -m scripts.measure_branch_traffic` has been run over the full `dev_300.jsonl`, and `docs/measurements/branch_traffic.md` holds a real percentage for every conditional edge.
- [x] Running it a second time reports **567 cache hits, 0 misses** and a **byte-identical branch table**. The cost line legitimately differs — the YAML rubric layer bypasses the model rather than replaying its tokens; see the reproducibility note in Task 12.
- [x] The measured numbers have been written back into Task 12 of this plan, dated.
- [x] `git status` shows a clean tree apart from the pre-existing uncommitted `.gitignore` change, which is still the user's and still untouched.
- [x] No commit message contains `Co-Authored-By` or any AI attribution.
- [x] `data/cache/` and `data/rubrics/derived/` are ignored by the working copy of `.gitignore` and untracked. `.gitignore` itself is still uncommitted, carrying the user's `docs/` line plus these two.
- [x] Nothing printed or committed contains the OpenAI key.

## Carried into Day 4

- **The eval harness** (`eval/run.py`, `baselines.py`, `bias.py`) over `test_500.jsonl`: macro-F1 and a confusion matrix, plus the three baselines from spec §7 — naive prompt, rubric-in-prompt single call, and TF-IDF cosine. `screen()` is the entry point; `StructuredLLM` with the shared cache keeps a re-run free.
- **The two ablations**, both now free of code changes: `gray_zone_margin = 0.0` disables `deep_review`, and skipping the `guard` node disables injection defence. Spec §7 wants the score to visibly collapse on poisoned CVs when the guard is off — `tests/graph/fixtures/poisoned.py` already holds the inputs.
- **Two numbers this plan deliberately left untuned**, both to be argued with evidence rather than adjusted on instinct:
  - `reject_fast` rejected **1 of 4** true `Good Fit` rows in the 24-pair sample. Confirm the rate over the full split before touching `must_have_min_score` or making `skill_terms` require ALL rather than ANY.
  - `must_have_check` cannot fire at all on **3 of 24** rubrics, because the model marked only unverifiable criteria as must-have. Worth reporting; possibly worth a prompt change.
- **`extraction_confidence` is collected and unused.** Either the Streamlit view shows it or it should be dropped from `RawExtraction`. Do not let it quietly become a routing signal — it was measured never to fall below 0.90.
- **PDF ingestion.** `pdfplumber` is still not installed and the graph deliberately takes text only. Spec §3 wants PDF resumes in the demo path; that belongs in `app/`, converting to text before `screen()` is called, so no `Evidence` offset ever depends on the parser.
- **Streamlit** (spec §9) and the three slide images. Slide 1 comes from `docs/diagrams/graph.mmd` plus the branch-traffic table; slide 2 is `TOOL_RATIONALE` next to the measured numbers in this plan; slide 3 is one candidate's `node_traces`, which `decide` already attaches to every result.
