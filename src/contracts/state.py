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
