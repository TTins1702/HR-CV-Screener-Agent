"""Replay a split from cache and write one row per CV with its criterion scores.

`RowRecord` keeps counts -- `criteria_scored`, `criteria_with_evidence` -- but not
the scores themselves, so no published number could be traced back to the
per-criterion values that produced it. This writes those values out.

It is a replay, not a run. Every prompt is byte-identical to the shipped run's, so
every model call is a cache hit, and `--allow-live` has to be passed before this
module will make a single paid request. Without it a miss raises. That guard is
the point: the dump exists to be regenerated freely while a question is being
asked, and it would stop being free the moment a prompt drifted without anyone
noticing.

The rubric is read from its YAML on disk rather than off the result, because
`ScreeningResult` does not carry one, and the weights are needed to reproduce the
aggregate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.run import EVAL_TODAY, load_split
from src.contracts.ablations import Ablations
from src.graph.build import screen
from src.graph.rubric_nodes import DERIVED_RUBRIC_DIR, jd_fingerprint
from src.llm.cache import DEFAULT_CACHE_PATH, JSONLCache
from src.llm.client import StructuredLLM
from src.rubric.loader import load_rubric

DEFAULT_SPLIT = Path("data/samples/dev_300.jsonl")
DEFAULT_OUT = Path("data/eval/criterion_scores__shipped.jsonl")


class CacheOnlyLLM(StructuredLLM):
    """A client that will not spend money. A miss is a drifted prompt, not a purchase."""

    def _request(self, messages, schema):  # type: ignore[override]
        raise RuntimeError(
            "cache miss: a prompt has changed, so this is no longer a replay. "
            "Pass --allow-live to pay for it deliberately."
        )


def dump(
    split: Path,
    out: Path,
    *,
    llm: StructuredLLM,
    derived_dir: Path | str = DERIVED_RUBRIC_DIR,
    limit: int = 0,
    on_row=None,
) -> int:
    """Write one JSON line per row. Returns the number of rows written."""
    rows = load_split(split, limit)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            try:
                result = screen(
                    row["resume_text"],
                    row["job_description_text"],
                    llm=llm,
                    today=EVAL_TODAY,
                    derived_dir=derived_dir,
                    ablations=Ablations(),
                )
            except Exception as error:  # noqa: BLE001 - a broken row is recorded, not fatal
                handle.write(
                    json.dumps(
                        {"row": index, "error": f"{type(error).__name__}: {error}"}
                    )
                    + "\n"
                )
                continue

            fingerprint = jd_fingerprint(row["job_description_text"])
            rubric_path = Path(derived_dir) / f"{fingerprint}.yaml"
            rubric = load_rubric(rubric_path) if rubric_path.exists() else None
            by_id = {c.id: c for c in rubric.criteria} if rubric else {}

            handle.write(
                json.dumps(
                    {
                        "row": index,
                        "jd_fingerprint": fingerprint,
                        "true_label": row["label"],
                        "predicted_label": result.label.value,
                        "overall_score": result.overall_score,
                        "path_taken": list(result.path_taken),
                        "criteria": [
                            {
                                "id": score.criterion_id,
                                "score": score.score,
                                "has_evidence": bool(score.evidence),
                                "weight": getattr(
                                    by_id.get(score.criterion_id), "weight", None
                                ),
                                "kind": getattr(
                                    by_id.get(score.criterion_id), "kind", None
                                ),
                                "must_have": getattr(
                                    by_id.get(score.criterion_id), "must_have", None
                                ),
                                "tool_used": score.tool_used,
                            }
                            for score in result.criterion_scores
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if on_row is not None:
                on_row(index + 1, len(rows))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="0 means the whole split")
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="permit paid API calls on a cache miss instead of raising",
    )
    args = parser.parse_args(argv)

    cache = JSONLCache(DEFAULT_CACHE_PATH)
    llm = (StructuredLLM if args.allow_live else CacheOnlyLLM)(cache=cache)

    started = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total} (hits {llm.hits}, misses {llm.misses})", flush=True)

    written = dump(args.split, args.out, llm=llm, limit=args.limit, on_row=progress)
    print(
        f"wrote {args.out} ({written} rows) in {time.perf_counter() - started:.1f}s, "
        f"{llm.hits} cache hits, {llm.misses} misses"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
