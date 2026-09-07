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
