"""FastAPI backend server for the HR CV Screener Agent web application."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any, AsyncGenerator

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from app.demo_cases import DEMO_CASES
from app.parsers import extract_text_from_file
from app.walkthrough import build_slides
from src.contracts.ablations import Ablations
from src.contracts.rubric import JDRubric
from src.contracts.state import ScreeningState
from src.graph.build import build_graph
from src.graph.routes import (
    route_gray_zone,
    route_guard,
    route_must_have,
    route_repair,
)
from src.llm.client import StructuredLLM
from src.rubric.loader import load_rubric

app = FastAPI(title="HR CV Screener Agent API", version="1.0.0")

# The UI is served from this same app, so cross-origin access only needs to cover
# a separately-served dev front-end. A wildcard origin with credentials on would let
# any page a reviewer opens spend the operator's model quota.
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = REPO_ROOT / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ScreenRequest(BaseModel):
    cv_text: str = Field(min_length=1)
    jd_text: str = Field(min_length=1)
    rubric_preset: str | None = None
    guard: bool = True
    must_have_gate: bool = True
    gray_zone: bool = True
    model_name: str = "gpt-4o-mini"
    api_key: str | None = None


@app.get("/")
async def get_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return {"message": "HR CV Screener API is running. index.html not yet created."}
    return FileResponse(
        index_file,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/presets")
async def get_presets():
    """Returns curated demo scenarios."""
    return [
        {
            "id": case.id,
            "title": case.title,
            "scenario_desc": case.scenario_desc,
            "expected_branch": case.expected_branch,
            "cv_text": case.cv_text,
            "jd_text": case.jd_text,
        }
        for case in DEMO_CASES
    ]


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Extract text from uploaded PDF or text file."""
    try:
        contents = await file.read()
        import io

        extracted = extract_text_from_file(io.BytesIO(contents), file.filename or "doc.txt")
        return {
            "filename": file.filename,
            "char_count": len(extracted),
            "text": extracted,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


RUBRIC_DIR = REPO_ROOT / "data" / "rubrics"


@app.get("/api/rubrics/{preset}")
async def get_rubric(preset: str):
    """Return one bundled rubric preset so the UI can show what it scores against."""
    # Servable presets are exactly the YAML files sitting directly in data/rubrics.
    # Matching against that listing rather than building a path from `preset` keeps
    # a caller from reaching either the LLM-derived rubrics under `derived/` or
    # anything outside the directory at all.
    available = {path.stem for path in RUBRIC_DIR.glob("*.yaml")}
    if preset not in available:
        raise HTTPException(status_code=404, detail=f"Unknown rubric preset: {preset!r}")

    rubric = load_rubric(RUBRIC_DIR / f"{preset}.yaml")
    return {
        "preset": preset,
        "job_title": rubric.job_title,
        "good_fit_threshold": rubric.good_fit_threshold,
        "potential_fit_threshold": rubric.potential_fit_threshold,
        "criteria": [
            {
                "id": criterion.id,
                "description": criterion.description,
                "weight": criterion.weight,
                "must_have": criterion.must_have,
                "kind": criterion.kind,
            }
            for criterion in rubric.criteria
        ],
    }


class WalkthroughRequest(BaseModel):
    """One completed run, as the client streamed it."""

    cv_text: str
    jd_text: str
    #: One entry per executed node, in order -- the step payloads the client kept.
    snapshots: list[dict[str, Any]] = Field(default_factory=list)


@app.post("/api/walkthrough")
async def walkthrough(req: WalkthroughRequest):
    """Build the per-node walkthrough slides for a run the client already made.

    Nothing is re-executed: the snapshots are the states the run passed through,
    and the locators only read them.
    """
    slides = build_slides(req.cv_text, req.jd_text, req.snapshots)
    return {"slides": [slide.model_dump(mode="json") for slide in slides]}


# The graph's four conditional edges, keyed by the node each one hangs off. The
# preview defers to `src/graph/routes.py` rather than restating the branch logic,
# because the restated copy had already drifted: it quarantined a run whose guard
# ablation was off, and it re-derived the gray zone from one threshold instead of
# both.
_CONDITIONAL_ROUTES = {
    "guard": route_guard,
    "extract": route_repair,
    "repair": route_repair,
    "must_have_check": route_must_have,
    "aggregate": route_gray_zone,
}

# The unconditional edges, mirroring `build_graph`.
_STATIC_EDGES = {
    "ingest": "guard",
    "load_rubric": "must_have_check",
    "score_criteria": "aggregate",
    "deep_review": "decide",
    "decide": "rank",
}

# The three nodes wired straight to END.
_TERMINAL_NODES = frozenset({"rank", "quarantine", "reject_fast"})


def predict_next_node(state: ScreeningState) -> str | None:
    """Predict the next active node, asking the graph's own routers where it goes."""
    if not state.path_taken:
        return "ingest"
    last = state.path_taken[-1]
    if last in _TERMINAL_NODES:
        return None
    route = _CONDITIONAL_ROUTES.get(last)
    if route is not None:
        return route(state)
    return _STATIC_EDGES.get(last)


@app.post("/api/screen/stream")
async def stream_screening(req: ScreenRequest):
    """Execute the multi-step LangGraph screening pipeline with live real-time SSE events."""
    # A per-request client, never `os.environ`: a key written to the process
    # environment outlives its request and is visible to every concurrent one.
    llm_client = OpenAI(api_key=req.api_key) if req.api_key else None

    rubric: JDRubric | None = None
    if req.rubric_preset:
        rubric_filename = (
            f"{req.rubric_preset}.yaml"
            if not req.rubric_preset.endswith(".yaml")
            else req.rubric_preset
        )
        # Falling back to `None` here would silently downgrade the preset the
        # caller chose to an LLM-derived rubric, and the run would look normal.
        try:
            rubric_path = REPO_ROOT / "data" / "rubrics" / rubric_filename
            if rubric_path.exists():
                rubric = load_rubric(rubric_path)
            else:
                rubric = load_rubric(rubric_filename)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not load rubric preset {req.rubric_preset!r}: {exc}",
            ) from exc

    ablations = Ablations(
        guard=req.guard,
        must_have_gate=req.must_have_gate,
        gray_zone=req.gray_zone,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            llm = StructuredLLM(
                model=req.model_name, use_cache=False, client=llm_client
            )
            graph = build_graph(llm)

            initial_state = ScreeningState(
                cv_text=req.cv_text,
                jd_text=req.jd_text,
                rubric=rubric,
                ablations=ablations,
            )

            loop = asyncio.get_running_loop()
            q: asyncio.Queue[Any] = asyncio.Queue()

            def run_sync_graph():
                try:
                    for chunk in graph.stream(initial_state, stream_mode="values"):
                        loop.call_soon_threadsafe(q.put_nowait, chunk)
                except Exception as exc:
                    loop.call_soon_threadsafe(q.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            loop.run_in_executor(None, run_sync_graph)

            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk

                current_st = (
                    chunk
                    if isinstance(chunk, ScreeningState)
                    else ScreeningState.model_validate(chunk)
                )
                current_node = (
                    current_st.path_taken[-1] if current_st.path_taken else None
                )
                next_node = predict_next_node(current_st)

                injection_findings = []
                if current_st.quarantined:
                    from src.tools.injection import scan_injection

                    report = scan_injection(current_st.cv_text)
                    injection_findings = [
                        {
                            "rule_id": f.rule_id,
                            "severity": f.severity.value,
                            "evidence": f.evidence.model_dump(mode="json"),
                        }
                        for f in report.findings
                    ]

                payload = {
                    "type": "step",
                    "current_node": current_node,
                    "next_node": next_node,
                    "path_taken": current_st.path_taken,
                    "quarantined": current_st.quarantined,
                    "injection_flags": current_st.injection_flags,
                    "injection_findings": injection_findings,
                    "blocking_must_haves": current_st.blocking_must_haves,
                    "repair_attempts": current_st.repair_attempts,
                    "node_traces": [
                        trace.model_dump(mode="json")
                        for trace in current_st.node_traces
                    ],
                    "profile": (
                        current_st.profile.model_dump(mode="json")
                        if current_st.profile
                        else None
                    ),
                    "rubric": (
                        current_st.rubric.model_dump(mode="json")
                        if current_st.rubric
                        else None
                    ),
                    "criterion_scores": [
                        score.model_dump(mode="json")
                        for score in current_st.criterion_scores
                    ],
                    "scorecard": (
                        current_st.scorecard.model_dump(mode="json")
                        if current_st.scorecard
                        else None
                    ),
                    "result": (
                        current_st.result.model_dump(mode="json")
                        if current_st.result
                        else None
                    ),
                }
                yield f"data: {json.dumps(payload, default=str)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            err_payload = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
