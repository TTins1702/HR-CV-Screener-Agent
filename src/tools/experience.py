"""Compute years of experience from CV date ranges, deterministically.

Handed to a tool rather than the LLM because the arithmetic is where models fail:
198 of the 290 dev resumes that contain parseable dates have *overlapping*
employment ranges, so the naive sum of durations over-counts on two thirds of
real CVs. Gaps, open-ended "to Present" roles and typo'd years need the same
consistency.

Parsing runs on the normalized copy of the CV (see `text_norm`) because the
dataset glues dates to their surroundings: `"12/2011toPresentData Analyst"`.
Offsets are mapped back so every range carries verbatim evidence.

The regexes cannot tell an employment range from an education range on their
own, and on this dataset that is not a small effect: a degree runs before the
career rather than alongside it, so `merge_spans` unions the two and the degree
adds its whole length to the total. Pass `work_periods` -- the roles the extract
node already lifted out of the CV -- and a range counts only where one of them
corroborates it. That splits the work along the line each side is good at: the
model says which spans are jobs, the tool does the arithmetic. Leave the argument
out and the total is every dated range, as before.

`self_declared_years` is reported alongside the total so the scoring node can see
both. Never treat the self-declared number as the total -- it is the candidate's
own claim.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date

from src.contracts.screening import Evidence, WorkPeriod
from src.contracts.tools import DateRange, ExperienceReport
from src.tools.text_norm import normalize_with_map

MONTH_NUMBERS = {
    name: number
    for number, name in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}

#: A range starting before this is a typo, not a career. Drops 8 of 1149 raw
#: matches on the dev split, including the verbatim "01/1920-06/2017".
EARLIEST_PLAUSIBLE_YEAR = 1960
#: No single role runs longer than this.
MAX_RANGE_YEARS = 45.0
#: Above this, a self-declared "N years" claim is noise, not a claim.
MAX_SELF_DECLARED_YEARS = 50
#: How far a range's start may sit from a role's start and still be that role. A
#: bare "2018" in the text reads as January while the extractor may have taken the
#: month off the same line, so the two can differ by most of a year while naming
#: the same job.
MATCH_TOLERANCE_MONTHS = 12

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_CURRENT = r"(?:present|current|now|ongoing|to\s*date|till\s*date)"
_SEPARATOR = r"\s*(?:to|until|through|thru|-|–|—)\s*"
_ENDPOINT = (
    r"(?:(?:0?[1-9]|1[0-2])\s*[/.-]\s*(?:19|20)\d{2}"  # 01/2020, 1.2020, 01-2020
    rf"|{_MONTH}\s*,?\s*(?:19|20)\d{{2}}"  # Jan 2020, January, 2020
    r"|(?:19|20)\d{2})"  # 2020
)
# Matched against normalized (lowercased) text, so no re.I is needed.
_RANGE_RE = re.compile(rf"({_ENDPOINT}){_SEPARATOR}({_ENDPOINT}|{_CURRENT})")
_SELF_DECLARED_RE = re.compile(r"(?<!\d)(\d{1,2})\s*\+?\s*(?:years|yrs)\b")


def months_between(start: date, end: date) -> int:
    """Whole months from `start` to `end`, floored at zero. Days are ignored."""
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))


def merge_spans(spans: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Collapse overlapping and touching intervals into their union, in order."""
    merged: list[tuple[date, date]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def calculate_experience(
    text: str,
    *,
    today: date | None = None,
    work_periods: Sequence[WorkPeriod] | None = None,
) -> ExperienceReport:
    """Total dated experience in `text`, with overlaps merged and evidence attached.

    Pass `today` explicitly wherever the result is compared or stored -- an
    open-ended range is measured up to it, so leaving it to `date.today()` makes
    the output drift from one day to the next.

    Pass `work_periods` to count only the ranges one of those roles covers; see
    the module docstring. Omitting them counts every dated range.
    """
    reference_date = today or date.today()
    normalized, index_map = normalize_with_map(text)

    ranges = _find_ranges(text, normalized, index_map, reference_date)
    ranges = _mark_employment(ranges, work_periods, reference_date)

    spans = [(item.start, item.end) for item in ranges if item.is_employment]
    merged = merge_spans(spans)
    total_months = sum(months_between(start, end) for start, end in merged)
    excluded_months = sum(
        months_between(start, end)
        for start, end in merge_spans(
            [(item.start, item.end) for item in ranges if not item.is_employment]
        )
    )

    self_declared_evidence = _find_self_declared(text, normalized, index_map)
    claimed = [
        float(match.group(1))
        for match in _SELF_DECLARED_RE.finditer(normalized)
        if 0 < int(match.group(1)) <= MAX_SELF_DECLARED_YEARS
    ]

    return ExperienceReport(
        total_years=round(total_months / 12.0, 2),
        excluded_years=round(excluded_months / 12.0, 2),
        ranges=ranges,
        overlaps_merged=len(spans) - len(merged),
        self_declared_years=max(claimed) if claimed else None,
        self_declared_evidence=self_declared_evidence,
    )


def _mark_employment(
    ranges: list[DateRange], work_periods: Sequence[WorkPeriod] | None, today: date
) -> list[DateRange]:
    """Flag the ranges that are worked employment, leaving the rest reported uncounted.

    Two ways a range fails. It may not have happened yet: "2024 - 2028" on a second
    year student is an expected graduation, and counting it credits four years to
    somebody who has worked none of them. A current role is written `Present`, which
    resolves to today, so this never touches one.

    Or no extracted role covers it. The three cases are deliberately distinct:
    `None` means no roles were offered and nothing is classified; an empty list
    means the model read the CV and found no jobs, which is an answer; and roles
    that carry no dates classify nothing, because an extractor that missed the dates
    is not evidence that the candidate never worked.
    """
    if work_periods is None:
        roles: list[tuple[date, date]] | None = None
    elif not work_periods:
        roles = []
    else:
        dated = [
            (period.start, period.end or today)
            for period in work_periods
            if period.start is not None
        ]
        roles = dated or None

    marked: list[DateRange] = []
    for item in ranges:
        employment = item.end <= today
        if employment and roles is not None:
            employment = any(
                start <= item.end
                and item.start <= end
                and months_between(min(start, item.start), max(start, item.start))
                <= MATCH_TOLERANCE_MONTHS
                for start, end in roles
            )
        marked.append(item.model_copy(update={"is_employment": employment}))
    return marked


def _find_ranges(
    text: str, normalized: str, index_map: list[int], today: date
) -> list[DateRange]:
    """Every plausible date range in the normalized text, in text order."""
    ranges: list[DateRange] = []
    for match in _RANGE_RE.finditer(normalized):
        start, _ = _parse_endpoint(match.group(1), is_end=False, today=today)
        end, is_current = _parse_endpoint(match.group(2), is_end=True, today=today)
        if start is None or end is None:
            continue
        if end < start or start > today:
            continue
        if start.year < EARLIEST_PLAUSIBLE_YEAR:
            continue
        if months_between(start, end) / 12.0 > MAX_RANGE_YEARS:
            continue

        original_start = index_map[match.start()]
        original_end = index_map[match.end()]
        ranges.append(
            DateRange(
                start=start,
                end=end,
                is_current=is_current,
                source=Evidence(
                    quote=text[original_start:original_end],
                    start=original_start,
                    end=original_end,
                ),
            )
        )
    return ranges


def _parse_endpoint(raw: str, *, is_end: bool, today: date) -> tuple[date | None, bool]:
    """One end of a range as a date, plus whether it means "still there".

    A bare year is read generously: January if it starts a range, December if it
    ends one, so "2011-2015" is not silently shortened to four flat years.
    """
    candidate = raw.strip()

    if re.fullmatch(_CURRENT, candidate):
        return today, True

    match = re.fullmatch(r"(0?[1-9]|1[0-2])\s*[/.-]\s*((?:19|20)\d{2})", candidate)
    if match:
        return date(int(match.group(2)), int(match.group(1)), 1), False

    match = re.fullmatch(rf"({_MONTH})\s*,?\s*((?:19|20)\d{{2}})", candidate)
    if match:
        return date(int(match.group(2)), MONTH_NUMBERS[match.group(1)[:3]], 1), False

    if re.fullmatch(r"(?:19|20)\d{2}", candidate):
        return date(int(candidate), 12 if is_end else 1, 1), False

    return None, False


def _find_self_declared(text: str, normalized: str, index_map: list[int]) -> list[Evidence]:
    """Spans where the CV claims a number of years outright."""
    evidence: list[Evidence] = []
    for match in _SELF_DECLARED_RE.finditer(normalized):
        if not 0 < int(match.group(1)) <= MAX_SELF_DECLARED_YEARS:
            continue
        original_start = index_map[match.start()]
        original_end = index_map[match.end()]
        evidence.append(
            Evidence(
                quote=text[original_start:original_end],
                start=original_start,
                end=original_end,
            )
        )
    return evidence
