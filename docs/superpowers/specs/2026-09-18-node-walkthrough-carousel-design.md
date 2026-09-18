# Node walkthrough carousel

A per-node walkthrough that opens from any trace card and explains, on the run's
own CV and JD, what that node did.

## Problem

The Trace tab says a node ran, how long it took and what its one-line note was.
It does not say what the node *did*. A reviewer watching the demo sees
`extract · 2523ms · 967 tok` and has to take on faith that the skills listed in
the Scorecard came out of this CV rather than out of the model's imagination.

The Evidence tab already answers that question for one node — `score_criteria` —
by highlighting quotes on the original CV. Nothing does it for the other twelve.

## Goal

Clicking **Chi tiết kỹ thuật** on a trace card opens a carousel of slides, one
per node the run actually executed, in path order. Each slide shows the
documents that node read, marks the exact spans it acted on, and lists what it
produced, with each output line tied to the marks behind it.

## Non-goals

- **No telemetry on the slides.** Latency, tokens and LLM-call counts stay on the
  trace card where they already are. A slide answers "what did it do", not "what
  did it cost".
- **No fixed example.** The slides are built from whichever run is on screen, so
  a quarantined run gets two slides and a full run gets eight. Nothing is
  authored against a particular demo case.
- **No editing.** The carousel is read-only.

## Design

### 1. Per-node snapshots, because the final state lies about the middle

`handleStreamEvent` currently keeps only the newest payload in
`latestScreeningState`. Building slides from it would attribute to `extract` a
profile that `repair` rewrote afterwards, and would leave the `repair` slide with
nothing to compare against.

The stream already delivers one payload per graph step (`stream_mode="values"`).
The client appends each to `nodeSnapshots` as `{node, payload}` when
`payload.current_node` changes. Every slide then reads the state as of the moment
its own node finished.

`nodeSnapshots` resets at the start of each run, alongside `latestScreeningState`.

### 2. Two fields the stream does not yet carry

The per-step SSE payload built in `app/server.py` gains:

- `rubric` — needed by the `load_rubric`, `must_have_check` and `reject_fast`
  slides. A run with `rubric_preset: null` derives its rubric inside the graph, so
  the client cannot reconstruct it.
- `criterion_scores` — `result.criterion_scores` only exists once the run
  finishes, and the `score_criteria` slide has to show the scores that node
  produced, before `deep_review` revised them.

Both are `model_dump(mode="json")` of the state fields, `None` when unset.

### 3. Span provenance: the marks are found by the project's own tools

`CandidateProfile` carries no offsets. `skills` is a list of strings; nothing in
it says where "Python" sits in the CV. The tempting shortcut — `indexOf` in
JavaScript — produces a highlight that the interface invented, and that is
exactly the claim a review panel should not accept.

So a new endpoint locates every mark server-side, using the functions the graph
itself uses:

| What is marked | Located by |
|---|---|
| injection findings (`guard`, `quarantine`) | already spans — `InjectionFinding.evidence` |
| criterion evidence (`score_criteria`, `deep_review`) | already spans — `CriterionScore.evidence` |
| skills (`extract`, `must_have_check`) | `search_evidence` over `load_skill_surface_forms` |
| work periods (`extract`) | `calculate_experience` → `ExperienceReport.ranges[].source` |
| degrees, certifications (`extract`) | `search_evidence` on the extracted string |
| rubric skill terms in the JD (`load_rubric`, `must_have_check`) | `search_evidence` against `jd_text` |

Every mark carries the `locator` that produced it, and the UI shows it as a small
chip on the slide. A span that came out of `criterion_scores` and a span the
walkthrough went looking for are different kinds of claim, and the slide says
which one it is showing.

**A value that cannot be located is reported, not dropped.** If `search_evidence`
returns nothing for an extracted skill — an alias that never appears verbatim —
the output row renders with a "không định vị được trên CV" note instead of
vanishing. Silently dropping it would make extraction look cleaner than it is.

### 4. `POST /api/walkthrough`

Request:

```json
{
  "cv_text": "...",
  "jd_text": "...",
  "snapshots": [{ "node": "extract", "profile": {...}, "rubric": {...}, "...": "..." }]
}
```

Response:

```json
{
  "slides": [
    {
      "node": "extract",
      "documents": ["cv"],
      "caption": "extract → load_rubric",
      "summary": "Đọc CV thô, trả về hồ sơ có cấu trúc…",
      "marks": [
        { "id": 1, "doc": "cv", "start": 812, "end": 818,
          "label": "skills", "locator": "search_evidence", "score": 1.0 }
      ],
      "outputs": [
        { "field": "skills", "value": "Python", "mark_ids": [1], "note": null }
      ]
    }
  ]
}
```

- `documents` names only the documents that node actually read, so `extract` shows
  the CV alone and `must_have_check` shows both.
- `caption` is the real edge out of this node, taken from `path_taken`, not from a
  hand-copied table.
- Marks never overlap within one document; the builder merges touching spans from
  the same label so a skill listed twice does not stack two boxes on one word.

The builder lives in `app/walkthrough.py`, not in `server.py` — thirteen node
cases would double that file. `server.py` keeps the route and the request model.

### 5. The slide frame

One frame, filled per node:

```
┌─ Chi tiết kỹ thuật ───────────────── ‹ 3/8 · extract ›  ✕ ─┐
│                                                            │
│  CV ứng viên                     │  extract sinh ra         │
│  ...Proficient in ╭────────╮     │  ① skills                │
│  ╭──────╮ and    │PostgreSQL│    │     Python, Go, PostgreSQL│
│  │Python│ ...    ╰────────╯      │  ② work_periods          │
│  ╰──────╯ ①            ③         │     03/2021 – nay        │
│  ...Senior Backend Engineer       │  ③ degrees              │
│  ╭───────────────╮ ②             │     B.Sc. Computer Science│
│  │03/2021 - Present│              │                          │
│  ╰───────────────╯                │  span định vị bởi:       │
│                                   │  ⚙ search_evidence       │
├────────────────────────────────────────────────────────────┤
│  extract → load_rubric                                      │
└────────────────────────────────────────────────────────────┘
```

- Marks are rounded outlines on the document text, each with a number badge.
- The output row carries the same numbers. Hovering or focusing either side
  raises the pair and draws a connector line between them.
- **Why numbers and a live connector rather than static arrows:** a real CV is
  long enough that its card always scrolls. An SVG arrow drawn once between two
  independently scrolling columns points at the wrong place the moment either one
  moves. The numbers survive scrolling; the connector is redrawn from the two
  elements' current positions and is hidden when either end scrolls out of view.
- Arrow keys and the `‹ ›` buttons move between slides. Escape closes.

### 6. What each node's slide shows

Only nodes present in `path_taken` get a slide.

| node | documents | marks | outputs |
|---|---|---|---|
| `ingest` | CV + JD | none — it deliberately does not touch the text | character counts, whether either document was empty |
| `guard` | CV | injection findings | severity, flags; a clean scan says "0 phát hiện" and shows the CV unmarked |
| `quarantine` | CV | the HIGH finding that triggered it | `rejected_reason` |
| `extract` | CV | skills, work periods, degrees, certifications | profile fields, `extraction_confidence`, `missing_fields` |
| `repair` | CV | the spans behind the fields that were rewritten | field-by-field before → after, `repair_attempts` |
| `load_rubric` | JD | rubric `skill_terms` found in the JD | criteria with weights and must-have flags |
| `must_have_check` | CV + JD | must-have terms on the JD, matching spans on the CV | one pass/fail row per must-have, with the reason it passed or blocked |
| `reject_fast` | CV + JD | the must-have that is missing | `blocking_must_haves`, `rejected_reason` |
| `score_criteria` | CV | `criterion_scores[].evidence` | per-criterion score, reasoning, `tool_used` |
| `aggregate` | none | none | weighted contributions summing to the overall score, gray-zone flag |
| `deep_review` | CV | evidence of the criteria whose scores changed | before → after per criterion |
| `decide` | none | none | thresholds against the overall score → label |
| `rank` | none | none | final label and the path the run took |

`must_have_check` reuses `blocking_must_haves` from `src/graph/rubric_nodes.py`
per criterion rather than re-deriving the rule, for the same reason
`predict_next_node` calls the graph's routers instead of copying them.

### 7. Delivery in two stages

**Status: both stages shipped.** Two things came out differently from what this
section anticipated, and the text below is kept as written for the record:

- `OutputRow` gained a `detail` field alongside `note`. `note` carries a problem
  the reader should see; `detail` carries the node's own explanation. Collapsing
  them made a failure to locate a value read like commentary.
- The bundled preset rubric names **no** `skill_terms` at all -- its criteria
  carry only a description -- so the JD marks fall back to locating the
  description, and any match below 1.0 says on the slide that it is approximate.
  For the same reason `must_have_check` reports three states, not two: a
  criterion the gate abstained on is not one it passed.


**Stage 1** — the frame and the backbone: snapshots, the two payload fields, the
endpoint, the modal carousel, marks and connectors, and the four slides that
carry the argument: `ingest`, `guard`, `extract`, `score_criteria`. Other
executed nodes get a slide with their summary and caption but no marks yet, so
the carousel is never missing a step.

**Stage 2** — the remaining nine: `quarantine`, `repair`, `load_rubric`,
`must_have_check`, `reject_fast`, `aggregate`, `deep_review`, `decide`, `rank`.

Stage 1 is reviewed on screen before stage 2 starts.

## Testing

`tests/test_walkthrough.py`:

- Every mark slices to a real substring of the document it names — `start`/`end`
  within bounds, `cv_text[start:end]` non-empty.
- Slide count and order match `path_taken`.
- A node with no profile or no rubric yields a slide with an explanatory output
  row, not an exception.
- `extract` marks carry `locator: "search_evidence"`; `score_criteria` marks carry
  `locator: "criterion_evidence"`. The two are not interchangeable.
- A skill absent from the CV produces an output row with the "not located" note
  rather than no row.
- `must_have_check` pass/fail agrees with `blocking_must_haves` for the same state.

Playwright, over all four demo presets: every executed node has a slide, arrow
keys walk the whole carousel, every rendered mark lies within the document text,
and no console errors.

## Risks

- **Long CVs.** The document card scrolls; the connector hides when an endpoint
  leaves the viewport. No virtualization — a CV is a few thousand characters.
- **Fuzzy locations.** `search_evidence` falls back to similarity matching. Marks
  below an exact match carry their score, and the slide shows it, so a fuzzy
  location is never presented as an exact one.
- **Derived rubrics.** With `rubric_preset: null` the rubric is LLM-derived per
  run. The `load_rubric` slide shows whatever the run produced, including a
  criterion whose `skill_terms` appear nowhere in the JD — which is worth seeing.
