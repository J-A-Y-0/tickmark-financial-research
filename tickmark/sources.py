"""Source adapters: SEC filings, internal research, and live web search.

Every adapter returns a SourceResult instead of raising, so one failing source degrades
a single sub-question (visibly, in the trace) rather than the whole answer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from tickmark.db import Database
from tickmark.models import Evidence, Route
from tickmark.retrieval.store import DocumentStore
from tickmark.text import html_to_text

log = logging.getLogger(__name__)

WEB_TIMEOUT_S = 12.0
WEB_CACHE_TTL_S = 6 * 3600
FETCH_CHARS = 6000

TIME_SENSITIVE = re.compile(
    r"\b(today|tonight|yesterday|now|current(ly)?|latest|recent(ly)?|this (week|month|quarter|year)|"
    r"news|breaking|live|real[- ]time|stock price|share price|trading at|right now)\b",
    re.IGNORECASE,
)


def is_time_sensitive(text: str, current_year: int | None = None) -> bool:
    """Questions whose answer can change from one day to the next."""
    if TIME_SENSITIVE.search(text):
        return True
    year = current_year or time.gmtime().tm_year
    return any(int(y) >= year for y in re.findall(r"\b(20\d\d)\b", text))


@dataclass
class SourceResult:
    route: Route
    evidence: list[Evidence] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    cached: bool = False
    provider: str = ""


class Source(Protocol):
    route: Route

    def search(self, query: str, role: str) -> SourceResult: ...


class IndexedSource:
    """Filings and internal research share the hybrid index; only the collection differs."""

    def __init__(self, store: DocumentStore, route: Route, k: int = 6):
        self.store, self.route, self.k = store, route, k

    def search(self, query: str, role: str) -> SourceResult:
        started = time.perf_counter()
        try:
            evidence = self.store.search(query, self.route.value, role, k=self.k)
            return SourceResult(self.route, evidence, latency_ms=_ms(started), provider="hybrid-index")
        except Exception as exc:
            log.exception("Index search failed")
            return SourceResult(self.route, ok=False, error=f"index search failed: {exc}", latency_ms=_ms(started))


# ---- web search ---------------------------------------------------------------------


class SearchProviderError(RuntimeError):
    pass


class TavilySearch:
    name = "tavily"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self.api_key:
            raise SearchProviderError("TAVILY_API_KEY is not set")

    def search(self, query: str, client: httpx.Client) -> list[Evidence]:
        response = client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"query": query, "max_results": 5, "search_depth": "basic", "include_raw_content": "text"},
        )
        if response.status_code >= 400:
            raise SearchProviderError(f"Tavily HTTP {response.status_code}")
        results = response.json().get("results", [])
        return [
            Evidence(
                kind=Route.WEB,
                title=r.get("title") or r.get("url", ""),
                text=" ".join(((r.get("raw_content") or r.get("content") or "")[:FETCH_CHARS]).split()),
                url=r.get("url", ""),
                locator=_domain(r.get("url", "")),
            )
            for r in results
            if r.get("url")
        ]


class SerpApiSearch:
    """Google results via SerpApi; the top pages are fetched so answers use page text, not snippets."""

    name = "serpapi"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY", "")
        if not self.api_key:
            raise SearchProviderError("SERPAPI_API_KEY is not set")

    def search(self, query: str, client: httpx.Client) -> list[Evidence]:
        response = client.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "num": 5, "api_key": self.api_key},
        )
        if response.status_code >= 400:
            raise SearchProviderError(f"SerpApi HTTP {response.status_code}")
        evidence = []
        for result in response.json().get("organic_results", [])[:4]:
            url = result.get("link", "")
            if not url:
                continue
            text = result.get("snippet", "")
            try:
                page = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; TickmarkResearch/1.0)"})
                if page.status_code < 400 and "html" in page.headers.get("content-type", ""):
                    text = html_to_text(page.text)[:FETCH_CHARS] or text
            except httpx.HTTPError:
                pass  # keep the snippet; the trace still shows the source
            evidence.append(Evidence(kind=Route.WEB, title=result.get("title", url), text=text, url=url, locator=_domain(url)))
        return evidence


def make_search_provider(name: str):
    if name == "tavily":
        return TavilySearch()
    if name == "serpapi":
        return SerpApiSearch()
    return None


WEB_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS web_cache (
    key TEXT PRIMARY KEY, provider TEXT NOT NULL, results TEXT NOT NULL, expires_at REAL NOT NULL
);
"""


class WebSource:
    """Live web search with a TTL result cache and an optional fallback provider.

    Search results (not answers) are cached for six hours; time-sensitive questions
    bypass the cache. If the primary provider errors or times out, the fallback runs.
    """

    route = Route.WEB

    def __init__(self, db: Database, providers: list, http: httpx.Client | None = None):
        self.db = db
        self.providers = [p for p in providers if p is not None]
        self.http = http or httpx.Client(timeout=WEB_TIMEOUT_S, follow_redirects=True)
        with db.tx() as conn:
            conn.executescript(WEB_CACHE_SCHEMA)

    @property
    def configured(self) -> bool:
        return bool(self.providers)

    def _cache_key(self, query: str, provider: str) -> str:
        normalised = " ".join(query.lower().split())
        return hashlib.sha256(f"{provider}\n{normalised}".encode()).hexdigest()

    def search(self, query: str, role: str) -> SourceResult:
        started = time.perf_counter()
        if not self.providers:
            return SourceResult(
                Route.WEB,
                ok=False,
                error="web search is not configured (set SEARCH_PROVIDER and its API key)",
                latency_ms=_ms(started),
            )
        fresh = is_time_sensitive(query)
        errors = []
        for provider in self.providers:
            key = self._cache_key(query, provider.name)
            if not fresh:
                rows = self.db.query("SELECT results FROM web_cache WHERE key = ? AND expires_at > ?", (key, time.time()))
                if rows:
                    evidence = [Evidence(**{**e, "kind": Route.WEB}) for e in json.loads(rows[0]["results"])]
                    return SourceResult(Route.WEB, evidence, latency_ms=_ms(started), cached=True, provider=provider.name)
            try:
                evidence = provider.search(query, self.http)
            except (httpx.HTTPError, SearchProviderError, ValueError) as exc:
                errors.append(f"{provider.name}: {exc}")
                continue
            if not fresh:
                payload = json.dumps([{**e.__dict__, "kind": "web"} for e in evidence])
                with self.db.tx() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO web_cache(key, provider, results, expires_at) VALUES (?, ?, ?, ?)",
                        (key, provider.name, payload, time.time() + WEB_CACHE_TTL_S),
                    )
            return SourceResult(Route.WEB, evidence, latency_ms=_ms(started), provider=provider.name)
        return SourceResult(Route.WEB, ok=False, error="; ".join(errors), latency_ms=_ms(started))


def _domain(url: str) -> str:
    match = re.match(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else url


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
