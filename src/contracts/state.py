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
