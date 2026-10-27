"""Semantic answer cache, partitioned by permission scope.

Rules (enforced here and by the engine):
- Lookups happen only after routing and the access check, inside the caller's partition.
  A partition is (policy version, role, allowed routes); an answer cached for a portfolio
  manager can never be served to an analyst, even for the identical question.
- The partition also includes the index version, so a new upload or corpus rebuild
  makes older answers unreachable.
- Denials, failed or partial answers and time-sensitive questions are never stored.
- A hit must match the same depth and the same route plan as the new request.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np

from tickmark.db import Database
from tickmark.retrieval.embeddings import Embedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
    id INTEGER PRIMARY KEY,
    partition TEXT NOT NULL,
    signature TEXT NOT NULL,
    question TEXT NOT NULL,
    question_hash TEXT NOT NULL,
    embedding BLOB NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS answer_cache_partition ON answer_cache(partition, signature);
"""


def normalise(question: str) -> str:
    return " ".join(question.lower().replace("?", " ").split())


@dataclass
class CacheHit:
    payload: dict
    similarity: float
    cached_question: str
    age_s: float


class AnswerCache:
    def __init__(self, db: Database, embedder: Embedder, *, similarity: float = 0.92, ttl_hours: float = 24.0):
        self.db, self.embedder = db, embedder
        self.similarity, self.ttl_s = similarity, ttl_hours * 3600
        with db.tx() as conn:
            conn.executescript(SCHEMA)

    def lookup(self, partition: str, signature: str, question: str) -> CacheHit | None:
        now = time.time()
        rows = self.db.query(
            """SELECT question, question_hash, embedding, payload, created_at FROM answer_cache
               WHERE partition = ? AND signature = ? AND expires_at > ?""",
            (partition, signature, now),
        )
        if not rows:
            return None
        key = _hash(question)
        for row in rows:
            if row["question_hash"] == key:
                return CacheHit(json.loads(row["payload"]), 1.0, row["question"], now - row["created_at"])
        query = self.embedder.embed([normalise(question)])[0]
        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        scores = matrix @ query
        best = int(np.argmax(scores))
        if float(scores[best]) >= self.similarity:
            row = rows[best]
            return CacheHit(json.loads(row["payload"]), float(scores[best]), row["question"], now - row["created_at"])
        return None

    def store(self, partition: str, signature: str, question: str, payload: dict) -> None:
        vector = self.embedder.embed([normalise(question)])[0]
        now = time.time()
        with self.db.tx() as conn:
            conn.execute(
                """INSERT INTO answer_cache(partition, signature, question, question_hash, embedding,
                                            payload, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (partition, signature, question, _hash(question), vector.tobytes(), json.dumps(payload), now, now + self.ttl_s),
            )

    def clear(self) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM answer_cache")


def _hash(question: str) -> str:
    return hashlib.sha256(normalise(question).encode()).hexdigest()
