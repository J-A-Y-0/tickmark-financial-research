"""Provider-neutral chat model interface."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from tickmark.models import Usage


class LLMError(RuntimeError):
    """A model provider failed (network, auth, rate limit, timeout or bad response)."""


@dataclass
class Completion:
    text: str
    usage: Usage


class StreamResult:
    """Iterate to receive text deltas; `usage` is filled once the stream is exhausted."""

    def __init__(self, deltas: Iterator[str]):
        self._deltas = deltas
        self.usage = Usage()

    def __iter__(self) -> Iterator[str]:
        return self._deltas


class ChatModel(Protocol):
    provider: str
    model: str

    def complete(self, system: str, user: str, *, max_tokens: int = 800, json_mode: bool = False) -> Completion: ...

    def stream(self, system: str, user: str, *, max_tokens: int = 1500) -> StreamResult: ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose.

    Raises ValueError when no JSON object can be recovered; callers decide the fallback.
    """
    cleaned = _FENCE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in model output") from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON in model output: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError("model output is JSON but not an object")
    return value
