"""The research pipeline, emitted as a stream of events.

    identify user -> spend gate -> plan -> route each sub-question -> access check
    -> answer cache (inside the caller's permission scope) -> retrieve in parallel
    -> merge citations -> stream answer -> verify -> cache -> audit

Every step is visible to the client as an event, in order: the plan arrives before any
retrieval, sources before the first token, and the verification after the last token.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from tickmark.access import AccessPolicy
from tickmark.audit import AuditLog
from tickmark.cache import AnswerCache
from tickmark.config import Settings
from tickmark.db import Database
from tickmark.llm.base import ChatModel, LLMError
from tickmark.llm.pricing import price_for
from tickmark.models import Depth, Route, Source, SubQuestion, Usage
from tickmark.planning import HeuristicPlanner, LLMPlanner, Plan
from tickmark.retrieval.embeddings import Embedder, make_embedder
from tickmark.retrieval.ingest import UploadWorker, build_corpus_index
from tickmark.retrieval.store import DocumentStore
from tickmark.routing import Router, make_router
from tickmark.sources import IndexedSource, SourceResult, WebSource, is_time_sensitive, make_search_provider
from tickmark.synthesis import ExtractiveSynthesizer, LLMSynthesizer, Part, merge_sources, no_evidence_answer
from tickmark.verify import JevSupportChecker, verify_answer

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1000
ROUTERS = ("keyword", "llm", "jev", "cascade")


@dataclass
class AskRequest:
    user_id: str
    question: str
    depth: Depth = Depth.QUICK
    router: str | None = None  # override the configured router (UI comparison, evaluation)
    use_cache: bool = True
    plan: list[str] | None = None  # fixed sub-questions: evaluation gives every router the same ones


@dataclass
class AnswerResult:
    """Everything a request produced, collected from its events (CLI and evaluation use this)."""

    events: list[dict] = field(default_factory=list)

    def _first(self, kind: str) -> dict:
        return next((e for e in self.events if e["type"] == kind), {})

    @property
    def answer(self) -> str:
        return "".join(e["text"] for e in self.events if e["type"] == "token")

    @property
    def done(self) -> dict:
        return self._first("done")

    @property
    def plan(self) -> dict:
        return self._first("plan")

    @property
    def routes(self) -> list[dict]:
        return [e for e in self.events if e["type"] == "route"]

    @property
    def sources(self) -> list[dict]:
        return self._first("sources").get("sources", [])

    @property
    def verification(self) -> dict:
        return self._first("verification")

    @property
    def status(self) -> str:
        return self.done.get("status", "error")


class Engine:
    def __init__(
        self,
        settings: Settings,
        *,
        db: Database | None = None,
        embedder: Embedder | None = None,
        chat: ChatModel | None = None,
        fast_chat: ChatModel | None = None,
        web: WebSource | None = None,
        routers: dict[str, Router] | None = None,
        policy: AccessPolicy | None = None,
        verifier=None,
        build_index: bool = True,
    ):
        self.settings = settings
        self.db = db or Database(settings.database_path)
        self.embedder = embedder or make_embedder(settings.embedding_provider, settings.embedding_model)
        self.store = DocumentStore(self.db, self.embedder)
        if build_index:
            build_corpus_index(self.store, settings.data_dir)
        self.uploads = UploadWorker(self.store)
        self.policy = policy or AccessPolicy.load(settings.data_dir / "access.json")
        self.audit = AuditLog(self.db)
        self.cache = AnswerCache(self.db, self.embedder, similarity=settings.cache_similarity, ttl_hours=settings.cache_ttl_hours)

        if chat is None and not settings.offline:
            from tickmark.llm.providers import make_chat_model

            chat = make_chat_model(settings.llm_provider, settings.answer_model)
            fast_chat = fast_chat or make_chat_model(settings.llm_provider, settings.fast_model)
        self.chat, self.fast_chat = chat, fast_chat or chat
        self.planner = LLMPlanner(self.fast_chat) if self.fast_chat else HeuristicPlanner()
        self.synthesizer = LLMSynthesizer(self.chat) if self.chat else ExtractiveSynthesizer()

        if web is None:
            providers = []
            for name in dict.fromkeys([settings.search_provider, settings.search_fallback]):
                try:
                    provider = make_search_provider(name)
                except Exception as exc:  # a missing key disables that provider, visibly
                    log.warning("Search provider %s disabled: %s", name, exc)
                    provider = None
                providers.append(provider)
            web = WebSource(self.db, providers)
        self.sources = {
            Route.FILINGS: IndexedSource(self.store, Route.FILINGS),
            Route.INTERNAL: IndexedSource(self.store, Route.INTERNAL),
            Route.WEB: web,
        }

        filings = [d for d in self.store.list_documents("filings")]
        self.catalog = "; ".join(f"{d.meta.get('company')} FY{d.meta.get('fiscal_year')}" for d in filings) or "none"
        self.companies = sorted(
            {d.meta.get("ticker", "") for d in filings} | {d.meta.get("company", " ").split()[0].rstrip(",") for d in filings}
        )
        self._routers: dict[str, Router] = dict(routers or {})
        self.verifier = verifier
        if verifier is None and settings.verifier == "jev" and os.environ.get("TYPESAFE_API_KEY"):
            self.verifier = JevSupportChecker(settings.jev_model)

    @classmethod
    def from_settings(cls, settings: Settings) -> Engine:
        return cls(settings)

    # ---- configuration --------------------------------------------------------------

    def available_routers(self) -> list[str]:
        names = ["keyword"]
        has_jev = bool(os.environ.get("TYPESAFE_API_KEY")) or "jev" in self._routers
        if self.fast_chat or "llm" in self._routers:
            names.append("llm")
        if has_jev:
            names.append("jev")
        if has_jev and (self.fast_chat or "cascade" in self._routers):
            names.append("cascade")
        return names

    def router(self, name: str | None = None) -> Router:
        name = name or self.settings.router
        if name not in ROUTERS:
            raise ValueError(f"unknown router {name!r}")
        if name not in self._routers:
            self._routers[name] = make_router(
                name,
                catalog=self.catalog,
                companies=self.companies,
                chat=self.fast_chat,
                jev_model=self.settings.jev_model,
                threshold=self.settings.cascade_threshold,
            )
        return self._routers[name]

    def describe(self) -> dict:
        return {
            "llm_provider": self.settings.llm_provider,
            "answer_model": self.chat.model if self.chat else "extractive (offline)",
            "fast_model": self.fast_chat.model if self.fast_chat else "heuristic (offline)",
            "router": self.settings.router,
            "routers": self.available_routers(),
            "embeddings": self.embedder.name,
            "vector_store": "sqlite-fts5+numpy (hybrid, RRF)",
            "web_search": [p.name for p in self.sources[Route.WEB].providers] or "not configured",
            "verifier": "lexical+jev" if self.verifier else "lexical",
            "documents": len(self.store.list_documents()),
            "chunks": self.store.chunk_count(),
        }

    # ---- cost estimate ----------------------------------------------------------------

    def estimate(self, user_id: str, depth: Depth) -> dict:
        """A rough pre-run cost estimate, shown before a user opts into deep research."""
        subs = 5 if depth is Depth.DEEP else 1
        sources = min(subs * 6, 24)
        plan_in, plan_out = 450, 80 * subs
        route_in, route_out = 450 * subs, 40 * subs
        answer_in, answer_out = 500 + sources * 450, 1500 if depth is Depth.DEEP else 400
        cost = 0.0
        priced = True
        for model, tokens_in, tokens_out in (
            (self.fast_chat.model if self.fast_chat else None, plan_in + route_in, plan_out + route_out),
            (self.chat.model if self.chat else None, answer_in, answer_out),
        ):
            if model is None:
                continue
            price = price_for(model)
            if price is None:
                priced = False
                continue
            cost += (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000
        user = self.policy.user(user_id)
        used = self.audit.deep_runs_today(user_id) if user else 0
        return {
            "depth": depth.value,
            "estimated_cost_usd": round(cost, 4) if priced else None,
            "deep_runs_left_today": max(0, self.settings.deep_daily_cap - used),
            "deep_daily_cap": self.settings.deep_daily_cap,
            "spend_today_usd": round(self.audit.spend_today(user_id), 4) if user else 0.0,
            "daily_budget_usd": self.settings.daily_budget_usd,
        }

    # ---- the pipeline -------------------------------------------------------------------

    def run(self, request: AskRequest) -> AnswerResult:
        return AnswerResult(list(self.ask(request)))

    def ask(self, request: AskRequest) -> Iterator[dict]:
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        timings: dict[str, float] = {}
        question = " ".join(request.question.split())
        user = self.policy.user(request.user_id)
        router_name = request.router or self.settings.router
        run = {
            "request_id": request_id,
            "user_id": request.user_id,
            "role": user.role if user else None,
            "question": question[:MAX_QUESTION_CHARS],
            "depth": request.depth.value,
            "router": router_name,
        }

        def finish(
            status: str,
            *,
            cache: str = "n/a",
            usage: Usage | None = None,
            terminated: str = "complete",
            routes: list | None = None,
            verification: dict | None = None,
            error: str | None = None,
        ) -> dict:
            usage = usage or Usage()
            timings["total"] = (time.perf_counter() - started) * 1000
            self.audit.record(
                {
                    **run,
                    "status": status,
                    "cache": cache,
                    "routes": routes or [],
                    "cost_usd": usage.cost_usd,
                    "latency_ms": timings["total"],
                    "verification": verification,
                    "detail": {"timings_ms": timings, "usage": usage.to_dict(), "terminated": terminated, "error": error},
                }
            )
            return {
                "type": "done",
                "request_id": request_id,
                "status": status,
                "cache": cache,
                "role": user.role if user else None,
                "terminated": terminated,
                "usage": usage.to_dict(),
                "cost_usd": round(usage.cost_usd, 6),
                "timings_ms": {k: round(v, 1) for k, v in timings.items()},
            }

        # 1. Identity: unknown users are rejected before any model call, search or cache lookup.
        if user is None:
            yield {"type": "error", "code": "unknown_user", "message": f"Unknown user {request.user_id!r}. Access denied."}
            yield finish("rejected")
            return
        if not question:
            yield {"type": "error", "code": "empty_question", "message": "Ask a question."}
            yield finish("rejected")
            return
        if len(question) > MAX_QUESTION_CHARS:
            yield {"type": "error", "code": "too_long", "message": f"Questions are limited to {MAX_QUESTION_CHARS} characters."}
            yield finish("rejected")
            return
        if router_name not in self.available_routers():
            yield {"type": "error", "code": "router_unavailable", "message": f"Router {router_name!r} is not configured."}
            yield finish("rejected")
            return

        # 2. Spend controls.
        if self.audit.spend_today(user.user_id) >= self.settings.daily_budget_usd:
            yield {
                "type": "error",
                "code": "budget_exceeded",
                "message": f"Daily budget of ${self.settings.daily_budget_usd:.2f} reached.",
            }
            yield finish("rejected")
            return
        if request.depth is Depth.DEEP and self.audit.deep_runs_today(user.user_id) >= self.settings.deep_daily_cap:
            yield {
                "type": "error",
                "code": "deep_cap",
                "message": f"Deep research is limited to {self.settings.deep_daily_cap} runs a day; try Quick.",
            }
            yield finish("rejected")
            return

        deadline = started + (self.settings.deep_timeout_s if request.depth is Depth.DEEP else self.settings.quick_timeout_s)
        yield {
            "type": "start",
            "request_id": request_id,
            "user": {"id": user.user_id, "name": user.name, "role": user.role, "role_label": user.role_label},
            "depth": request.depth.value,
            "router": router_name,
        }

        # 3. Plan.
        t = time.perf_counter()
        if request.plan:
            subs = [SubQuestion(i, text, "fixed by caller") for i, text in enumerate(request.plan, 1)]
            plan = Plan(subs, request.depth, "fixed")
        else:
            plan = self.planner.plan(question, request.depth)
        timings["plan"] = (time.perf_counter() - t) * 1000
        usage = plan.usage
        yield {"type": "plan", **plan.to_dict()}

        # 4. Route every sub-question (in parallel), then the access check.
        t = time.perf_counter()
        router = self.router(router_name)
        with ThreadPoolExecutor(max_workers=6) as pool:
            decisions = list(pool.map(lambda s: router.route(s.text), plan.sub_questions))
        timings["route"] = (time.perf_counter() - t) * 1000
        parts: list[Part] = []
        route_log = []
        for sub, decision in zip(plan.sub_questions, decisions, strict=True):
            usage = usage + decision.usage
            allowed = self.policy.can_use(user.role, decision.route)
            parts.append(Part(sub, decision, denied=not allowed))
            route_log.append({"sub_question": sub.index, "route": decision.route.value, "allowed": allowed})
            yield {"type": "route", "sub_question": sub.index, "allowed": allowed, **decision.to_dict()}
            if not allowed:
                yield {
                    "type": "denied",
                    "sub_question": sub.index,
                    "route": decision.route.value,
                    "message": f"{user.role_label}s cannot access {decision.route.label}.",
                }

        if all(p.denied for p in parts):
            text = no_evidence_answer(parts)
            yield {"type": "sources", "sources": []}
            yield from _tokens(text)
            yield finish("denied", usage=usage, routes=route_log)
            return

        # 5. Answer cache, inside the caller's permission scope.
        partition = f"{self.policy.scope_key(user.role)}:{self.store.index_version()}"
        signature = request.depth.value + ":" + ",".join(p.decision.route.value for p in parts)
        cache_ok = request.use_cache and not any(p.denied for p in parts) and not is_time_sensitive(question)
        if cache_ok:
            hit = self.cache.lookup(partition, signature, question)
            if hit:
                yield {
                    "type": "cache",
                    "status": "HIT",
                    "similarity": round(hit.similarity, 3),
                    "cached_question": hit.cached_question,
                    "age_s": round(hit.age_s),
                }
                yield {"type": "sources", "sources": hit.payload["sources"]}
                yield from _tokens(hit.payload["answer"])
                yield {"type": "verification", **hit.payload["verification"]}
                yield finish(
                    "answered", cache="HIT", usage=usage, routes=route_log, verification=hit.payload["verification"]["summary"]
                )
                return
            yield {"type": "cache", "status": "MISS"}
        else:
            reason = (
                "time-sensitive question"
                if is_time_sensitive(question)
                else ("partially denied request" if any(p.denied for p in parts) else "cache disabled for this request")
            )
            yield {"type": "cache", "status": "BYPASS", "reason": reason}

        # 6. Retrieve for each permitted sub-question in parallel; trace each search as it completes.
        t = time.perf_counter()
        evidence_by_part: dict[int, list] = {}
        work = [p for p in parts if not p.denied]
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(self.sources[p.decision.route].search, p.sub.text, user.role): p for p in work}
            for future in as_completed(futures):
                part = futures[future]
                result: SourceResult = future.result()
                for ev in result.evidence:
                    ev.sub_question = part.sub.index
                evidence_by_part[part.sub.index] = result.evidence
                if not result.ok:
                    part.error = result.error
                yield {
                    "type": "trace",
                    "step": "search",
                    "sub_question": part.sub.index,
                    "source": result.route.value,
                    "ok": result.ok,
                    "results": len(result.evidence),
                    "latency_ms": round(result.latency_ms, 1),
                    "cached": result.cached,
                    "provider": result.provider,
                    "error": result.error,
                }
        timings["retrieve"] = (time.perf_counter() - t) * 1000
        sources: list[Source] = merge_sources(parts, evidence_by_part)
        yield {"type": "sources", "sources": [s.to_dict() for s in sources]}

        # 7. Generate.
        t = time.perf_counter()
        terminated = "complete"
        answer = ""
        if not sources:
            answer = no_evidence_answer(parts)
            yield from _tokens(answer)
        elif time.perf_counter() > deadline:
            terminated = "timeout"
            answer = "_[Stopped: the time limit was reached before the answer could be written. Sources are listed.]_"
            yield from _tokens(answer)
        else:
            stream = self.synthesizer.stream(question, parts, sources, request.depth)
            try:
                for delta in stream:
                    answer += delta
                    yield {"type": "token", "text": delta}
                    if time.perf_counter() > deadline:
                        terminated = "timeout"
                        notice = "\n\n_[Stopped: time limit reached. This answer is incomplete.]_"
                        answer += notice
                        yield {"type": "token", "text": notice}
                        break
            except LLMError as exc:
                log.warning("Generation failed: %s", exc)
                timings["generate"] = (time.perf_counter() - t) * 1000
                yield {
                    "type": "error",
                    "code": "generation_failed",
                    "message": "The answer model failed. Sources above are still valid.",
                }
                yield finish("error", usage=usage + stream.usage, terminated="error", routes=route_log, error=str(exc)[:300])
                return
            usage = usage + stream.usage
        timings["generate"] = (time.perf_counter() - t) * 1000

        # 8. Verify citations and figures.
        t = time.perf_counter()
        verification = verify_answer(answer, sources, self.verifier)
        usage = usage + verification.usage
        timings["verify"] = (time.perf_counter() - t) * 1000
        report = verification.to_dict()
        yield {"type": "verification", **report}

        partial = terminated != "complete" or any(p.denied or p.error for p in parts)
        status = "partial" if partial else "answered"

        # 9. Cache only complete, permitted, non-time-sensitive answers without fabricated citations.
        if cache_ok and status == "answered" and sources and report["summary"]["fabricated_citations"] == 0:
            self.cache.store(
                partition,
                signature,
                question,
                {
                    "answer": answer,
                    "sources": [s.to_dict() for s in sources],
                    "verification": report,
                },
            )

        yield finish(
            status,
            cache="MISS" if cache_ok else "BYPASS",
            usage=usage,
            terminated=terminated,
            routes=route_log,
            verification=report["summary"],
        )


def _tokens(text: str) -> Iterator[dict]:
    """Replay stored text as token events (cache hits and canned answers)."""
    for i in range(0, len(text), 24):
        yield {"type": "token", "text": text[i : i + 24]}
