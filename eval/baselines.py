"""The three baselines spec section 7 requires, emitting the agent's record shape.

The point of (b) is to be hard to beat. A single call carrying the whole derived
rubric is the honest strong baseline, and spec section 7 says what to do if it
ties the graph: stop claiming accuracy and argue evidence linkage, injection
defence, broken-CV handling and cost -- all of which this harness already
measures. Writing a weak (b) to make the agent look good would make the whole
evaluation worthless.

All three emit `RowRecord`, so `eval/report.py` scores them with the same
functions and no comparison depends on two code paths agreeing.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from eval.records import RowRecord, write_records
from eval.run import DEFAULT_OUT_DIR, DEFAULT_SPLIT, load_split, records_path
from src.contracts.screening import FitLabel
from src.graph.rubric_nodes import (
    DERIVED_RUBRIC_DIR,
    RUBRIC_SYSTEM,
    RawRubric,
    jd_fingerprint,
    repair_rubric,
)
from src.llm.cache import DEFAULT_CACHE_PATH, JSONLCache
from src.llm.client import StructuredLLM

# `src/rubric/loader.py` names these `load_rubric` and `save_rubric`;
# `src/graph/rubric_nodes.py` imports them under the read/write aliases. Use the
# real names here rather than adding a third spelling.
from src.rubric.loader import load_rubric, save_rubric

NAIVE_SYSTEM = (
    "You are screening a resume against a job description. Answer with exactly one "
    "label: 'Good Fit', 'Potential Fit', or 'No Fit'. Give no explanation."
)

RUBRIC_JUDGE_SYSTEM = (
    "You are screening a resume against a job description using a fixed rubric. "
    "Weigh every criterion by its weight. A candidate missing a criterion marked "
    "must-have cannot be a Good Fit. Answer with exactly one label: 'Good Fit', "
    "'Potential Fit', or 'No Fit'. Give no explanation."
)


class NaiveVerdict(BaseModel):
    """The whole of a baseline's answer: one of the three dataset labels."""

    label: FitLabel


def _prompt(row: dict[str, Any]) -> str:
    return (
        f"JOB DESCRIPTION:\n{row['job_description_text']}\n\n"
        f"RESUME:\n{row['resume_text']}"
    )


def _record(
    index: int,
    row: dict[str, Any],
    verdict: NaiveVerdict,
    usages: Sequence[Any],
    *,
    system: str,
    config: str,
    started: float,
) -> RowRecord:
    return RowRecord(
        row_index=index,
        system=system,
        config=config,
        true_label=row["label"],
        predicted_label=verdict.label.value,
        prompt_tokens=sum(usage.prompt_tokens for usage in usages),
        completion_tokens=sum(usage.completion_tokens for usage in usages),
        latency_ms=(time.perf_counter() - started) * 1000.0,
        llm_calls=sum(1 for usage in usages if not usage.cached),
        cached_calls=sum(1 for usage in usages if usage.cached),
    )


def naive_baseline(row: dict[str, Any], index: int, *, llm: Any) -> RowRecord:
    """Spec section 7 baseline (a): one call, no rubric, no tools."""
    started = time.perf_counter()
    verdict, usage = llm.parse(
        system=NAIVE_SYSTEM, user=_prompt(row), schema=NaiveVerdict
    )
    return _record(
        index, row, verdict, [usage],
        system="baseline_naive", config="single_call", started=started,
    )


def _rubric_for(jd_text: str, *, llm: Any, derived_dir: Path | str) -> tuple[Any, list[Any]]:
    """The same rubric the graph would derive, from the same cache on disk."""
    path = Path(derived_dir) / f"{jd_fingerprint(jd_text)}.yaml"
    if path.exists():
        return load_rubric(path), []
    raw, usage = llm.parse(system=RUBRIC_SYSTEM, user=jd_text, schema=RawRubric)
    rubric = repair_rubric(raw)
    save_rubric(rubric, path)
    return rubric, [usage]


def _rubric_block(rubric: Any) -> str:
    lines = [f"ROLE: {rubric.job_title}", "CRITERIA:"]
    for criterion in rubric.criteria:
        flag = " [MUST HAVE]" if criterion.must_have else ""
        lines.append(
            f"- {criterion.id} (weight {criterion.weight:.2f}){flag}: {criterion.description}"
        )
    lines.append(
        f"THRESHOLDS: Good Fit at {rubric.good_fit_threshold:.2f}, "
        f"Potential Fit at {rubric.potential_fit_threshold:.2f}"
    )
    return "\n".join(lines)


def rubric_baseline(
    row: dict[str, Any],
    index: int,
    *,
    llm: Any,
    derived_dir: Path | str = DERIVED_RUBRIC_DIR,
) -> RowRecord:
    """Spec section 7 baseline (b): one call, but carrying the full rubric.

    Uses the graph's own derived rubric so the comparison isolates the graph, not
    the rubric. If this ties the agent, spec section 7 says the agent's case moves
    to evidence, robustness and cost -- and this harness measures all three.
    """
    started = time.perf_counter()
    rubric, usages = _rubric_for(row["job_description_text"], llm=llm, derived_dir=derived_dir)
    verdict, usage = llm.parse(
        system=RUBRIC_JUDGE_SYSTEM,
        user=f"{_rubric_block(rubric)}\n\nRESUME:\n{row['resume_text']}",
        schema=NaiveVerdict,
    )
    return _record(
        index, row, verdict, [*usages, usage],
        system="baseline_rubric", config="single_call_with_rubric", started=started,
    )


def tfidf_baseline(rows: Sequence[dict[str, Any]]) -> list[RowRecord]:
    """Spec section 7 baseline (c): cosine similarity, no model at all.

    Thresholds are the tertiles of the similarity distribution over the rows given.
    Fitting them on the data being scored is generous to this baseline on purpose:
    a floor that flatters itself is a floor you can trust.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    resumes = [row["resume_text"] for row in rows]
    jds = [row["job_description_text"] for row in rows]
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(resumes + jds)
    resume_vectors, jd_vectors = matrix[: len(rows)], matrix[len(rows) :]

    similarities = [
        float(resume_vectors[index].multiply(jd_vectors[index]).sum())
        for index in range(len(rows))
    ]
    ordered = sorted(similarities)
    low = ordered[len(ordered) // 3] if ordered else 0.0
    high = ordered[2 * len(ordered) // 3] if ordered else 0.0

    records = []
    for index, (row, score) in enumerate(zip(rows, similarities)):
        if score >= high:
            label = FitLabel.GOOD_FIT
        elif score >= low:
            label = FitLabel.POTENTIAL_FIT
        else:
            label = FitLabel.NO_FIT
        records.append(
            RowRecord(
                row_index=index,
                system="baseline_tfidf",
                config="cosine_tertiles",
                true_label=row["label"],
                predicted_label=label.value,
                overall_score=min(max(score, 0.0), 1.0),
            )
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--only", choices=["naive", "rubric", "tfidf"], action="append", default=None
    )
    args = parser.parse_args(argv)

    rows = load_split(args.split, args.limit)
    wanted = set(args.only or ["naive", "rubric", "tfidf"])
    llm = StructuredLLM(cache=JSONLCache(DEFAULT_CACHE_PATH))

    if "tfidf" in wanted:
        records = tfidf_baseline(rows)
        write_records(records, records_path(args.out_dir, "baseline_tfidf", "cosine_tertiles"))
        print(f"tfidf: {len(records)} rows, 0 tokens")

    for name, runner, config in (
        ("naive", naive_baseline, "single_call"),
        ("rubric", rubric_baseline, "single_call_with_rubric"),
    ):
        if name not in wanted:
            continue
        records = []
        for index, row in enumerate(rows):
            records.append(runner(row, index, llm=llm))
            if (index + 1) % 25 == 0:
                print(f"  {name} {index + 1}/{len(rows)} (hits {llm.hits}, misses {llm.misses})")
        path = records_path(args.out_dir, f"baseline_{name}", config)
        write_records(records, path)
        print(f"wrote {path} ({sum(r.tokens for r in records)} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
