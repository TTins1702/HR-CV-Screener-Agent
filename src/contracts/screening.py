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
    """A verbatim span of the CV that supports a score.

    `quote` is sliced from the original CV text, so it may contain the dataset's
    glued sentence boundaries. `score` is 1.0 for an exact match and the
    similarity ratio for a fuzzy one.
    """

    quote: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    score: float = Field(default=1.0, ge=0.0, le=1.0)

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
