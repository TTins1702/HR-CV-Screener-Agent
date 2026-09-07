"""Locate the CV text behind a score, and report where it is.

This is the tool that makes evidence claim E1 checkable: a score is only
defensible if you can point at the characters that produced it. Matching happens
on the normalized copy of the CV (see `text_norm`) because the dataset drops
spaces between sentences, but the offsets and the quote come from the original.
"""

from __future__ import annotations

import difflib
import re

from src.contracts.screening import Evidence
from src.tools.text_norm import normalize_with_map

DEFAULT_MAX_RESULTS = 3
DEFAULT_MIN_SCORE = 0.75

_WORD_RE = re.compile(r"\S+")
# Window widths to try, as offsets from the query's word count. A paraphrase is
# usually within a word or two of the original either way.
_WINDOW_DELTAS = (0, -1, 1, 2)
# Skip a candidate window whose length is nowhere near the query's.
_LENGTH_SLACK_RATIO = 0.7
_LENGTH_SLACK_CHARS = 8


def search_evidence(
    text: str,
    query: str,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[Evidence]:
    """Find up to `max_results` spans of `text` matching `query`, best first.

    Exact matches on the normalized text score 1.0 and are returned alone. If
    there are none, a word-aligned sliding window scores candidates with
    `difflib.SequenceMatcher` and keeps those at or above `min_score`. Returned
    spans never overlap, and `Evidence.quote` is always a verbatim slice of
    `text`, glued sentence boundaries and all.
    """
    if max_results < 1:
        raise ValueError(f"max_results must be at least 1, got {max_results}")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be in [0, 1], got {min_score}")

    normalized_text, index_map = normalize_with_map(text)
    normalized_query = normalize_with_map(query)[0].strip()
    if not normalized_text or not normalized_query:
        return []

    exact = _find_exact(normalized_text, normalized_query, index_map, text, max_results)
    if exact:
        return exact
    return _find_fuzzy(
        normalized_text, normalized_query, index_map, text, max_results, min_score
    )


def _evidence(text: str, index_map: list[int], start: int, end: int, score: float) -> Evidence:
    """Build an Evidence for the normalized span [start, end)."""
    original_start = index_map[start]
    original_end = index_map[end]
    return Evidence(
        quote=text[original_start:original_end],
        start=original_start,
        end=original_end,
        score=score,
    )


def _find_exact(
    normalized_text: str,
    normalized_query: str,
    index_map: list[int],
    text: str,
    max_results: int,
) -> list[Evidence]:
    """Every non-overlapping literal occurrence of the normalized query."""
    hits: list[Evidence] = []
    cursor = 0
    while len(hits) < max_results:
        found = normalized_text.find(normalized_query, cursor)
        if found < 0:
            break
        end = found + len(normalized_query)
        hits.append(_evidence(text, index_map, found, end, 1.0))
        cursor = end
    return hits


def _find_fuzzy(
    normalized_text: str,
    normalized_query: str,
    index_map: list[int],
    text: str,
    max_results: int,
    min_score: float,
) -> list[Evidence]:
    """Best-scoring non-overlapping windows of the normalized text."""
    words = [(match.start(), match.end()) for match in _WORD_RE.finditer(normalized_text)]
    if not words:
        return []

    query_length = len(normalized_query)
    query_words = max(1, len(normalized_query.split()))
    widths = sorted({max(1, query_words + delta) for delta in _WINDOW_DELTAS})
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(normalized_query)

    scored: list[tuple[float, int, int]] = []
    for width in widths:
        for first in range(0, len(words) - width + 1):
            start = words[first][0]
            end = words[first + width - 1][1]
            if abs((end - start) - query_length) > (
                query_length * _LENGTH_SLACK_RATIO + _LENGTH_SLACK_CHARS
            ):
                continue
            matcher.set_seq1(normalized_text[start:end])
            if matcher.real_quick_ratio() < min_score or matcher.quick_ratio() < min_score:
                continue
            ratio = matcher.ratio()
            if ratio >= min_score:
                scored.append((round(ratio, 4), start, end))

    # Best score first; ties broken by position so the result is deterministic.
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))

    picked: list[Evidence] = []
    taken: list[tuple[int, int]] = []
    for score, start, end in scored:
        if any(start < other_end and end > other_start for other_start, other_end in taken):
            continue
        taken.append((start, end))
        picked.append(_evidence(text, index_map, start, end, score))
        if len(picked) >= max_results:
            break
    return picked
