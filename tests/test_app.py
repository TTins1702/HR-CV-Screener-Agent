"""Tests for the app parsers and demo cases."""

from __future__ import annotations

import io
import pytest

from app.demo_cases import DEMO_CASES, get_demo_case
from app.parsers import extract_text_from_file
from src.contracts.screening import FitLabel
from src.contracts.state import ScreeningState
from src.graph.build import build_graph


class StubLLM:
    """Offline stub for graph testing."""
    def parse(self, *, system: str, user: str, schema: type):
        raise NotImplementedError("Offline stub")


def test_demo_cases():
    assert len(DEMO_CASES) == 4
    case = get_demo_case("good_fit")
    assert case is not None
    assert "Senior Backend Engineer" in case.title
    assert len(case.cv_text) > 50

    none_case = get_demo_case("non_existent_id")
    assert none_case is None


def test_parsers_text_file():
    txt_content = b"Candidate resume text content"
    f = io.BytesIO(txt_content)
    extracted = extract_text_from_file(f, "resume.txt")
    assert extracted == "Candidate resume text content"

    f2 = io.BytesIO(txt_content)
    with pytest.raises(ValueError, match="Unsupported file format"):
        extract_text_from_file(f2, "resume.docx")


def test_graph_streaming_quarantine_path():
    """Verify that graph.stream works cleanly with stream_mode='values' on an empty or malicious CV."""
    graph = build_graph(StubLLM())
    initial = ScreeningState(cv_text="", jd_text="")
    
    steps = []
    final_state = None
    for chunk in graph.stream(initial, stream_mode="values"):
        st = ScreeningState.model_validate(chunk)
        steps.append(st.path_taken[-1] if st.path_taken else "start")
        final_state = st

    assert "ingest" in steps
    assert "quarantine" in steps
    assert final_state is not None
    assert final_state.quarantined is True
    assert final_state.result is not None
    assert final_state.result.label == FitLabel.NO_FIT
