"""The extract and repair nodes, and the tool correction that makes E2 measurable.

The user chose on 2026-09-07 that the model guesses the years and
`calculate_experience` corrects it, rather than the node calling the tool directly.
The reason is that a correction can be counted: over 24 real (resume, JD) pairs the
two numbers agreed zero times, the median gap was 4.38 years, and the worst case was
the model saying 5.0 against the tool's 12.92. `llm_declared_years` keeps the claim
so the difference survives into the eval and onto the slide.

Nothing here routes on what the model says about its own output. Measured on 30 real
resumes, the model's `missing_fields` was non-empty 30 times out of 30 -- routing on
it would send 100% of traffic down `repair`, which is the definition of a decorative
branch. The repair edge is driven instead by parsing the dates the model actually
wrote.

`extraction_confidence` is not decorative in that way, and an earlier version of this
note had it wrong. Over the 1030 unique extractions in the cache it takes four values
-- 0.0 (245), 0.90 (689), 0.95 (77) and 1.0 (19) -- so it does drop below 0.90, on
23.8% of extractions. It is not uninformative either: a 0.0 means the total years
figure is missing 90.2% of the time, against 15.9% everywhere else. What it is, is
redundant. `total_experience_years is None` answers the same question exactly rather
than at precision 0.902 and recall 0.639, and answers it without asking a model to
grade itself. That is why the field is reported and never routed on.

A note on how much traffic that carries, because the number moved during
development and the reason matters. A weaker prototype prompt produced unusable
date strings on 40% of resumes. `EXTRACT_SYSTEM` below names the three bad strings
explicitly, and on the first live run that fix held: across 25 real resumes, 26 of
182 date fields came back as genuine JSON nulls and not one non-null value failed to
parse. The prompt engineered the original failure away. What replaced it is subtler
-- a sentinel `0001-01` standing in for "the text does not say" -- which is why
`parse_month` carries a plausibility floor. The branch is therefore small, and the
honest reading is that it is now a safety net for a failure mode the prompt usually
prevents rather than a hot path. `scripts/measure_branch_traffic.py` prints the
current rate; trust it over any number written in prose.
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
from src.tools.experience import EARLIEST_PLAUSIBLE_YEAR, calculate_experience

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
# it invites a third. With the current prompt the model rarely reaches for any of
# these -- see the note in the module docstring.
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

    No `missing_fields`, and nothing routes on `extraction_confidence`. The first was
    measured to be uninformative; the second is informative but redundant, and the
    module docstring has the numbers.

    The field stays because removing it is not free: the cache key covers this
    schema, so dropping it invalidates every extraction answer on disk and turns the
    next run cold. It is reported to the recruiter-facing view instead, labelled for
    what it is rather than as a confidence.
    """

    skills: list[str]
    work_periods: list[RawPeriod]
    degrees: list[str]
    certifications: list[str]
    total_experience_years: float | None
    extraction_confidence: float


def parse_month(raw: str | None) -> date | None:
    """`YYYY-MM`, `YYYY` or `YYYY-MM-DD` to a date; None for anything unusable.

    A syntactically valid date is not automatically a usable one. Measured on the
    first live run over 25 real resumes, `0001-01` was the single most common date
    token the model wrote -- 8 of 182 date fields -- as a sentinel for "the text does
    not say". It parses cleanly as year 1, so without a plausibility floor it lands
    in a `WorkPeriod` looking like a real employment date. The floor is Day 2's
    `EARLIEST_PLAUSIBLE_YEAR`, reused rather than duplicated so the graph and
    `calculate_experience` cannot disagree about what a plausible career date is.
    """
    if raw is None:
        return None
    token = raw.strip().lower()
    if token in UNUSABLE_DATE_TOKENS:
        return None
    try:
        if len(token) == 4 and token.isdigit():
            parsed = date(int(token), 1, 1)
        else:
            parsed = date.fromisoformat(token if len(token) > 7 else f"{token}-01")
    except ValueError:
        return None
    return None if parsed.year < EARLIEST_PLAUSIBLE_YEAR else parsed


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
    # Parsed before the tool runs, because the tool uses them to tell an employment
    # range from a degree: the model classifies, `calculate_experience` counts.
    periods = [
        WorkPeriod(
            title=period.title,
            company=period.company,
            start=parse_month(period.start),
            end=parse_month(period.end),
        )
        for period in extraction.work_periods
    ]
    report = calculate_experience(state.cv_text, today=today, work_periods=periods)
    declared = extraction.total_experience_years
    # Counted ranges, not any range: roles that line up with nothing in the text
    # leave a total of zero, which would read as "never worked" rather than as the
    # extraction failure it is.
    counted = [item for item in report.ranges if item.is_employment]
    corrected = report.total_years if counted else declared
    delta = None if declared is None or corrected is None else abs(corrected - declared)

    profile = CandidateProfile(
        raw_text=state.cv_text,
        skills=extraction.skills,
        work_periods=periods,
        degrees=extraction.degrees,
        certifications=extraction.certifications,
        total_experience_years=corrected,
        excluded_years=report.excluded_years,
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
