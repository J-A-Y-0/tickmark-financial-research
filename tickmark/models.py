"""Data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum


class Route(StrEnum):
    """The three knowledge sources a sub-question can be sent to."""

    FILINGS = "filings"  # SEC 10-K filings
    INTERNAL = "internal"  # the firm's research notes, policies, deal memos, uploads
    WEB = "web"  # live web search

    @property
    def label(self) -> str:
        return ROUTE_LABELS[self]


ROUTE_LABELS = {
    Route.FILINGS: "SEC filings",
    Route.INTERNAL: "Internal research",
    Route.WEB: "Web search",
}


class Depth(StrEnum):
    QUICK = "quick"
    DEEP = "deep"


@dataclass
class Usage:
    """Tokens and dollars for one or more model calls. Calls to unpriced models add $0 and are counted."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    unpriced_calls: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
            self.calls + other.calls,
            self.unpriced_calls + other.unpriced_calls,
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "cost_usd": round(self.cost_usd, 6)}


@dataclass
class RouteDecision:
    route: Route
    reason: str
    router: str
    confidence: float | None = None
    latency_ms: float = 0.0
    usage: Usage = field(default_factory=Usage)
    fallback: str | None = None  # set when a router fell back to another

    def to_dict(self) -> dict:
        return {
            "route": self.route.value,
            "label": self.route.label,
            "reason": self.reason,
            "router": self.router,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 1),
            "usage": self.usage.to_dict(),
            "fallback": self.fallback,
        }


@dataclass
class SubQuestion:
    index: int  # 1-based
    text: str
    reason: str = ""


@dataclass
class Evidence:
    """One retrieved passage. `key` identifies it for de-duplication across sub-questions."""

    kind: Route
    title: str
    text: str
    url: str = ""
    locator: str = ""
    doc_id: str = ""
    score: float = 0.0
    sub_question: int = 0

    @property
    def key(self) -> str:
        if self.kind is Route.WEB:
            return f"web:{self.url}"
        return f"{self.kind.value}:{self.doc_id}:{self.locator}"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "title": self.title,
            "url": self.url,
            "locator": self.locator,
            "doc_id": self.doc_id,
            "snippet": self.text[:600],
            "passage": self.text[:2000],  # what the answer model read (it sees up to 1,800 chars)
            "sub_question": self.sub_question,
        }


@dataclass
class Source(Evidence):
    """Evidence with its final citation number in the answer."""

    number: int = 0
    sub_questions: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.pop("sub_question")
        return {"n": self.number, **data, "sub_questions": self.sub_questions}
