"""A scripted stand-in for `StructuredLLM`, so node tests never touch the network."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.contracts.trace import LLMUsage


class StubLLM:
    """Returns scripted answers in order and records what it was asked.

    It implements `parse` and nothing else, because `parse` is the entire surface
    the graph nodes use. If a node ever reaches for another attribute, that is a
    design regression and this stub will fail loudly with AttributeError.
    """

    def __init__(
        self, responses: list[BaseModel], usage: LLMUsage | None = None
    ) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.usage = usage or LLMUsage(prompt_tokens=100, completion_tokens=20)

    def parse(self, *, system: str, user: str, schema: type[BaseModel]):
        if not self.responses:
            raise AssertionError(
                f"StubLLM ran out of scripted responses; asked for {schema.__name__}"
            )
        self.calls.append({"system": system, "user": user, "schema": schema.__name__})
        return self.responses.pop(0), self.usage
