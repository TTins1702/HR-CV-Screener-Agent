"""A content-addressed JSONL cache for model calls.

Spec section 8 asks that every number on a slide come from one command. That needs
the model to be replayable, and `temperature=0` does not give you that: measured on
2026-09-07, the same 30 resumes at temperature 0 with the same prompt produced 13
malformed-date extractions in one run and 12 in the next. The cache does give you
that -- a second run of the eval reads every answer back off disk.

The key covers everything that can change an answer: model id, the exact messages,
the response schema (name and JSON schema), temperature and seed. Change a prompt
and you get a clean miss rather than a stale hit.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

DEFAULT_CACHE_PATH = Path("data/cache/llm_cache.jsonl")


def cache_key(
    *,
    model: str,
    messages: list[dict[str, str]],
    schema_name: str,
    schema: dict[str, Any],
    temperature: float,
    seed: int | None,
) -> str:
    """A stable sha256 over every input that can change the model's answer."""
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "schema_name": schema_name,
            "schema": schema,
            "temperature": temperature,
            "seed": seed,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JSONLCache:
    """Append-only JSONL, read into memory once and appended to thereafter.

    Append-only rather than rewrite-in-place so an interrupted run loses at most the
    last line, and so the file stays a readable audit trail of what the model was
    asked. A repeated key is legal: the last line for a key wins.
    """

    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue  # a half-written tail from an interrupted run
                key = record.get("key")
                if key is not None:
                    self._entries[key] = record["value"]

    def get(self, key: str) -> dict[str, Any] | None:
        """The cached payload for `key`, or None."""
        return self._entries.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Record `value` under `key`, in memory and on disk."""
        with self._lock:
            self._entries[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n"
                )

    def __len__(self) -> int:
        return len(self._entries)
