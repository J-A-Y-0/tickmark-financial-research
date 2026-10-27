from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tickmark.config import ROOT, Settings
from tickmark.db import Database
from tickmark.engine import Engine
from tickmark.llm.base import Completion, LLMError, StreamResult
from tickmark.models import Evidence, Route, Usage
from tickmark.retrieval.embeddings import HashingEmbedder
from tickmark.retrieval.ingest import build_corpus_index
from tickmark.retrieval.store import DocumentStore
from tickmark.sources import SourceResult


class FakeChat:
    """A scripted chat model. `replies` are returned in order by complete(); stream() writes `answer`."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, replies: list[str] | None = None, answer: str = "", fail: bool = False, fail_stream: bool = False):
        self.replies = list(replies or [])
        self.answer = answer
        self.fail = fail
        self.fail_stream = fail_stream
        self.complete_calls: list[tuple[str, str]] = []
        self.stream_calls: list[tuple[str, str]] = []

    def complete(self, system, user, *, max_tokens=800, json_mode=False):
        self.complete_calls.append((system, user))
        if self.fail:
            raise LLMError("provider down")
        text = self.replies.pop(0) if self.replies else "{}"
        return Completion(text, Usage(100, 20, 0.001, calls=1))

    def stream(self, system, user, *, max_tokens=1500):
        self.stream_calls.append((system, user))
        answer, fail = self.answer, self.fail_stream

        def deltas():
            if fail:
                raise LLMError("stream broke")
            yield from (answer[i : i + 10] for i in range(0, len(answer), 10))
            result.usage = Usage(500, 80, 0.002, calls=1)

        result = StreamResult(deltas())
        return result


class FakeWeb:
    route = Route.WEB

    def __init__(self, evidence: list[Evidence] | None = None, ok: bool = True):
        self.evidence, self.ok, self.calls = evidence or [], ok, 0
        self.providers = []

    def search(self, query, role):
        self.calls += 1
        if not self.ok:
            return SourceResult(Route.WEB, ok=False, error="provider timeout")
        return SourceResult(Route.WEB, [Evidence(**{**e.__dict__}) for e in self.evidence], provider="fake")


SECRET_ENV = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TYPESAFE_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY")


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    """Tests never see a developer's .env keys, so they stay offline and deterministic."""
    for name in SECRET_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(database_path=tmp_path / "test.sqlite3", data_dir=ROOT / "data")


@pytest.fixture(scope="session")
def corpus_db(tmp_path_factory) -> Path:
    """The bundled corpus indexed once; tests copy the file so each gets a clean database."""
    path = tmp_path_factory.mktemp("corpus") / "corpus.sqlite3"
    db = Database(path)
    build_corpus_index(DocumentStore(db, HashingEmbedder()), ROOT / "data")
    db.close()
    return path


@pytest.fixture
def db(corpus_db, tmp_path) -> Database:
    target = tmp_path / "db.sqlite3"
    target.write_bytes(corpus_db.read_bytes())
    return Database(target)


@pytest.fixture
def store(db) -> DocumentStore:
    return DocumentStore(db, HashingEmbedder())


@pytest.fixture
def make_engine(settings, db):
    """Build an engine on the indexed corpus. Pass chat=FakeChat(...) for the LLM paths."""

    def build(**kwargs) -> Engine:
        overrides = kwargs.pop("settings", {})
        kwargs.setdefault("web", FakeWeb())
        return Engine(replace(settings, **overrides), db=db, embedder=HashingEmbedder(), **kwargs)

    return build


def events_of(events, kind: str) -> list[dict]:
    return [e for e in events if e["type"] == kind]


def plan_json(*questions: str) -> str:
    return json.dumps({"sub_questions": [{"question": q, "reason": "test"} for q in questions]})
