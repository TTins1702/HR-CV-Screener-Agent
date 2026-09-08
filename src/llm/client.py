"""The only module in this repo that talks to OpenAI.

Confining the SDK here is what lets every graph node be tested offline against a
one-method stub, and what keeps the retry, cache and usage-accounting policy in a
single place instead of scattered across thirteen nodes.

`temperature=0` and a fixed seed are pinned on every request, but they are a
best-effort hint, not a guarantee: the same 30 resumes at temperature 0 produced 13
malformed-date extractions in one run and 12 in the next (measured 2026-09-07).
Reproducibility comes from `src/llm/cache.py`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.contracts.trace import LLMUsage
from src.llm.cache import DEFAULT_CACHE_PATH, JSONLCache, cache_key

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SEED = 42

# Transient only. A 4xx means the request itself is wrong; retrying it just burns
# money and hides the bug.
RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

T = TypeVar("T", bound=BaseModel)


class LLMRefusal(RuntimeError):
    """The model declined to answer. Never cached -- a refusal is not a result."""


def load_environment() -> None:
    """Load `.env` from the repo root, without overriding what is already set.

    A bare `load_dotenv()` resolves relative to the *calling module's* file rather
    than the working directory, so a script run from elsewhere silently gets nothing.
    Be explicit about the path.
    """
    load_dotenv(REPO_ROOT / ".env")


class StructuredLLM:
    """A temperature-0, seeded, cached, retried structured-output client."""

    def __init__(
        self,
        *,
        model: str | None = None,
        seed: int | None = None,
        temperature: float = 0.0,
        cache: JSONLCache | None = None,
        client: Any | None = None,
    ) -> None:
        load_environment()
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.seed = (
            seed
            if seed is not None
            else int(os.environ.get("SCREENER_SEED", DEFAULT_SEED))
        )
        self.temperature = temperature
        self.cache = cache if cache is not None else JSONLCache(DEFAULT_CACHE_PATH)
        self._client = client
        self.hits = 0
        self.misses = 0

    @property
    def client(self) -> Any:
        """The OpenAI client, built on first use so offline tests never need a key."""
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def parse(self, *, system: str, user: str, schema: type[T]) -> tuple[T, LLMUsage]:
        """Ask the model for one `schema`-shaped answer, from cache when possible."""
        started = time.perf_counter()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        key = cache_key(
            model=self.model,
            messages=messages,
            schema_name=schema.__name__,
            schema=schema.model_json_schema(),
            temperature=self.temperature,
            seed=self.seed,
        )

        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return schema.model_validate_json(cached["content"]), LLMUsage(
                prompt_tokens=cached["prompt_tokens"],
                completion_tokens=cached["completion_tokens"],
                latency_ms=(time.perf_counter() - started) * 1000.0,
                cached=True,
            )

        self.misses += 1
        response = self._request(messages, schema)
        message = response.choices[0].message
        if message.refusal:
            raise LLMRefusal(f"model refused: {message.refusal}")

        content = message.content or ""
        value = schema.model_validate_json(content)
        self.cache.put(
            key,
            {
                "model": self.model,
                "content": content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        )
        return value, LLMUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            cached=False,
        )

    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _request(self, messages: list[dict[str, str]], schema: type[T]) -> Any:
        """One HTTP round trip, retried on transient failures only."""
        return self.client.chat.completions.parse(
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
            messages=messages,
            response_format=schema,
        )
