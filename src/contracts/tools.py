"""Outputs of the five deterministic tools.

Each tool returns one of these models rather than a bare number, so the graph
nodes, the Streamlit view and the eval harness all read the same field names,
and so every score carries the evidence that produced it.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from src.contracts.screening import Evidence, FitLabel


class DateRange(BaseModel):
    """One employment interval recovered from the CV text by regex."""

    start: date
    end: date
    is_current: bool = False
    source: Evidence

    @model_validator(mode="after")
    def _end_must_not_precede_start(self) -> "DateRange":
        if self.end < self.start:
            raise ValueError("end must not precede start")
        return self


class ExperienceReport(BaseModel):
    """What `calculate_experience` found.

    `total_years` is the union of the merged intervals, so overlapping roles are
    counted once. `self_declared_years` is the largest "N years" claim the CV
    makes about itself; it is reported for comparison, never used as the total.
    """

    total_years: float = Field(ge=0.0)
    ranges: list[DateRange] = Field(default_factory=list)
    overlaps_merged: int = Field(default=0, ge=0)
    self_declared_years: float | None = Field(default=None, ge=0.0)
    self_declared_evidence: list[Evidence] = Field(default_factory=list)


class SkillMatch(BaseModel):
    """The result of resolving one raw skill string to a canonical name."""

    raw: str = Field(min_length=1)
    canonical: str = Field(min_length=1)
    matched_alias: str | None = None
    known: bool = False


class InjectionSeverity(str, Enum):
    """How hard the guard edge should react."""

    NONE = "none"
    LOW = "low"
    HIGH = "high"


class InjectionFinding(BaseModel):
    """One rule that fired, and the CV span that tripped it."""

    rule_id: str = Field(min_length=1)
    severity: InjectionSeverity
    evidence: Evidence


class InjectionReport(BaseModel):
    """What `scan_injection` found. `severity` is the max over the findings."""

    is_suspicious: bool = False
    severity: InjectionSeverity = InjectionSeverity.NONE
    findings: list[InjectionFinding] = Field(default_factory=list)

    @property
    def flags(self) -> list[str]:
        """Rule ids in the order they were found, for `ScreeningState.injection_flags`."""
        return [finding.rule_id for finding in self.findings]


class Scorecard(BaseModel):
    """What `aggregate_scorecard` computed from the per-criterion scores."""

    overall_score: float = Field(ge=0.0, le=1.0)
    label: FitLabel
    in_gray_zone: bool = False
    missing_must_haves: list[str] = Field(default_factory=list)
    unscored_criteria: list[str] = Field(default_factory=list)
    weighted_contributions: dict[str, float] = Field(default_factory=dict)
