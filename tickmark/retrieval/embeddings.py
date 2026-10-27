"""Text embeddings: a deterministic offline embedder and OpenAI embeddings."""

from __future__ import annotations

import hashlib
import os
import threading
from itertools import pairwise
from typing import Protocol

import numpy as np

from tickmark.llm.base import LLMError
from tickmark.llm.pricing import usage_for
from tickmark.models import Usage
from tickmark.text import content_words


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalised vectors."""
        ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing of words and word pairs.

    Lexical rather than semantic, but deterministic, free and offline. It lets the full
    pipeline (hybrid search, semantic cache) run without any API key; switch to
    EMBEDDING_PROVIDER=openai for real semantic similarity.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"
        self.usage = Usage()

    def _features(self, text: str) -> list[str]:
        tokens = content_words(text)
        return tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for feature in self._features(text):
                counts[feature] = counts.get(feature, 0) + 1
            for feature, count in counts.items():
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign * (1.0 + np.log(count))
        return _normalise(matrix)


class OpenAIEmbedder:
    """OpenAI embeddings, batched. Tracks token usage for cost reporting."""

    BATCH = 256

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        import openai

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set (needed for EMBEDDING_PROVIDER=openai)")
        self._openai = openai
        self.client = openai.OpenAI(api_key=key, timeout=60.0, max_retries=2)
        self.model = model
        self.name = f"openai:{model}"
        self.usage = Usage()
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH):
            batch = [t[:8000] or " " for t in texts[start : start + self.BATCH]]
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
            except self._openai.OpenAIError as exc:
                raise LLMError(f"OpenAI embeddings failed: {exc}") from exc
            rows.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
            with self._lock:
                self.usage = self.usage + usage_for(self.model, response.usage.prompt_tokens, 0)
        return _normalise(np.asarray(rows, dtype=np.float32))


def make_embedder(provider: str, model: str) -> Embedder:
    if provider == "openai":
        return OpenAIEmbedder(model)
    if provider == "hashing":
        return HashingEmbedder()
    raise ValueError(f"EMBEDDING_PROVIDER must be hashing or openai, got {provider!r}")
