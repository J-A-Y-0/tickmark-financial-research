"""FastAPI app: the research UI, a streaming API, the document library and the audit log.

Identity is a demo sign-in (a cookie naming one of the policy's users, or an X-User-Id
header for API clients). It stands in for SSO; it is not authentication.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from tickmark import __version__
from tickmark.access import User
from tickmark.config import ROOT, load_settings
from tickmark.engine import AskRequest, Engine
from tickmark.models import Depth, Route

log = logging.getLogger("tickmark.web")
HERE = Path(__file__).resolve().parent
EVAL_RESULTS = ROOT / "evals" / "results"
COOKIE = "tickmark_user"
DEFAULT_USER = "analyst"

EXAMPLES = [
    {"label": "Reported figure", "q": "What was Lyft's net income in 2024?", "depth": "quick"},
    {
        "label": "Filing plus house view",
        "q": "What was Uber's revenue in 2021, and what is our price target on Uber?",
        "depth": "quick",
    },
    {"label": "Comparison", "q": "Compare Uber and Lyft Adjusted EBITDA in 2024", "depth": "quick"},
    {"label": "Restricted memo", "q": "What are the proposed terms of Project Atlas?", "depth": "quick"},
]
ROUTE_LABELS = {r.value: r.label for r in Route}
ROUTER_LABELS = {
    "keyword": "Keyword rules (offline)",
    "llm": "LLM router",
    "jev": "TypeSafe Jev",
    "cascade": "Cascade: Jev, then LLM if unsure",
}


def _asset_version() -> str:
    """Changes whenever a static file changes, so browsers never keep a stale stylesheet."""
    digest = hashlib.sha256()
    for path in sorted((HERE / "static").iterdir()):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


class AskBody(BaseModel):
    question: str = Field(..., max_length=2000)
    depth: Depth = Depth.QUICK
    router: str | None = None
    use_cache: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "engine"):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        app.state.engine = Engine.from_settings(load_settings())
    yield


def create_app(engine: Engine | None = None) -> FastAPI:
    app = FastAPI(title="Tickmark", version=__version__, lifespan=lifespan)
    if engine is not None:
        app.state.engine = engine
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["ts"] = lambda t: datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%d %H:%M:%S")
    templates.env.filters["clock"] = lambda t: datetime.fromtimestamp(t, UTC).strftime("%H:%M")
    asset_version = _asset_version()

    def eng(request: Request) -> Engine:
        return request.app.state.engine

    def identity(request: Request) -> str:
        """An API client's X-User-Id is taken as given (unknown ones are rejected later). A browser
        cookie that no longer names a policy identity, e.g. after the policy changed, falls back
        to the default access level instead of locking the visitor out."""
        if request.headers.get("x-user-id"):
            return request.headers["x-user-id"]
        cookie = request.cookies.get(COOKIE)
        return cookie if cookie and eng(request).policy.user(cookie) else DEFAULT_USER

    def current_user(request: Request) -> User | None:
        return eng(request).policy.user(identity(request))

    def require_user(request: Request) -> User:
        user = current_user(request)
        if user is None:
            raise HTTPException(401, "Unknown user")
        return user

    def page(request: Request, name: str, **context) -> HTMLResponse:
        engine = eng(request)
        user = current_user(request)
        return templates.TemplateResponse(
            request,
            name,
            {
                "user": user,
                "users": engine.policy.all_users(),
                "config": engine.describe(),
                "version": __version__,
                "asset_version": asset_version,
                "access_summary": {
                    role: ", ".join(ROUTE_LABELS[r.value] for r in engine.policy.allowed_routes(role))
                    for role in engine.policy.roles
                },
                "route_labels": ROUTE_LABELS,
                "router_labels": ROUTER_LABELS,
                "active": name.removesuffix(".html"),
                **context,
            },
        )

    # ---- pages ----------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        engine = eng(request)
        user = current_user(request)
        routes = [r.value for r in engine.policy.allowed_routes(user.role)] if user else []
        docs = engine.store.list_documents()
        corpus = {
            "filings": sum(d.collection == "filings" for d in docs),
            "internal": sum(d.collection == "internal" for d in docs),
            "chunks": engine.store.chunk_count(),
        }
        return page(request, "index.html", examples=EXAMPLES, allowed_routes=routes, corpus=corpus)

    @app.post("/session")
    def switch_user(request: Request, user_id: str = Form(...), next: str = Form("/")):
        if eng(request).policy.user(user_id) is None:
            raise HTTPException(400, "Unknown user")
        target = next if next.startswith("/") and not next.startswith("//") else "/"
        response = RedirectResponse(target, status_code=303)
        response.set_cookie(COOKIE, user_id, httponly=True, samesite="lax", max_age=30 * 86400)
        return response

    @app.get("/library", response_class=HTMLResponse)
    def library(request: Request):
        engine = eng(request)
        user = require_user(request)
        docs = [d for d in engine.store.list_documents() if _can_read(engine, user, d)]
        can_upload = engine.policy.can_use(user.role, Route.INTERNAL)
        roles = [r for r in engine.policy.roles if engine.policy.can_use(r, Route.INTERNAL)]
        return page(
            request, "library.html", docs=docs, can_upload=can_upload, upload_roles=roles, max_mb=engine.settings.max_upload_mb
        )

    @app.get("/documents/{doc_id}", response_class=HTMLResponse)
    def document(request: Request, doc_id: str):
        engine = eng(request)
        user = require_user(request)
        doc = engine.store.get_document(doc_id)
        if doc is None:
            raise HTTPException(404, "Document not found")
        if not _can_read(engine, user, doc):
            return page(request, "denied.html", doc_title="this document")
        return page(request, "document.html", doc=doc, chunks=engine.store.document_chunks(doc_id))

    @app.get("/activity", response_class=HTMLResponse)
    def activity(request: Request):
        engine = eng(request)
        user = require_user(request)
        sees_all = user.role == "pm"
        runs = engine.audit.recent(100, None if sees_all else user.user_id)
        return page(
            request,
            "activity.html",
            runs=runs,
            sees_all=sees_all,
            stats=engine.audit.stats(None if sees_all else user.user_id),
            role_labels={name: r["label"] for name, r in engine.policy.roles.items()},
        )

    @app.get("/evals", response_class=HTMLResponse)
    def evals(request: Request):
        report = _latest_report()
        return page(request, "evals.html", report=report)

    # ---- API ------------------------------------------------------------------------

    @app.post("/api/ask")
    def ask(request: Request, body: AskBody):
        engine = eng(request)
        user_id = identity(request)
        events = engine.ask(AskRequest(user_id, body.question, body.depth, body.router, body.use_cache))

        def stream():
            for event in events:
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/estimate")
    def estimate(request: Request, depth: Depth = Depth.DEEP):
        user = require_user(request)
        return eng(request).estimate(user.user_id, depth)

    @app.get("/api/config")
    def config(request: Request):
        engine = eng(request)
        user = current_user(request)
        return {
            **engine.describe(),
            "user": {"id": user.user_id, "name": user.name, "role": user.role} if user else None,
            "allowed_routes": [r.value for r in engine.policy.allowed_routes(user.role)] if user else [],
        }

    @app.get("/api/documents")
    def list_documents(request: Request):
        engine = eng(request)
        user = require_user(request)
        return [d.to_dict() for d in engine.store.list_documents() if _can_read(engine, user, d)]

    @app.post("/api/documents", status_code=202)
    async def upload(request: Request, file: UploadFile = File(...), visibility: str = Form("team")):
        engine = eng(request)
        user = require_user(request)
        if not engine.policy.can_use(user.role, Route.INTERNAL):
            raise HTTPException(403, "Your role cannot add internal documents")
        data = await file.read(engine.settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > engine.settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"Files are limited to {engine.settings.max_upload_mb} MB")
        internal_roles = [r for r in engine.policy.roles if engine.policy.can_use(r, Route.INTERNAL)]
        allowed = [user.role] if visibility == "private" else internal_roles
        if "pm" in internal_roles and "pm" not in allowed:
            allowed.append("pm")  # portfolio managers can read every internal document
        try:
            doc_id = engine.uploads.submit(file.filename or "upload.txt", data, allowed_roles=allowed, uploaded_by=user.user_id)
        except ValueError as exc:
            raise HTTPException(415, str(exc)) from exc
        return {"doc_id": doc_id, "status": "pending"}

    @app.get("/api/activity")
    def api_activity(request: Request, limit: int = 50):
        engine = eng(request)
        user = require_user(request)
        return engine.audit.recent(min(limit, 200), None if user.role == "pm" else user.user_id)

    @app.get("/evals/report.json")
    def report_json():
        report = _latest_report()
        if report is None:
            return JSONResponse({"error": "No evaluation report yet. Run: make eval"}, status_code=404)
        return report

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "version": __version__, **eng(request).describe()}

    return app


def _can_read(engine: Engine, user: User, doc) -> bool:
    return engine.policy.can_use(user.role, Route(doc.collection)) and doc.readable_by(user.role)


def _latest_report() -> dict | None:
    path = EVAL_RESULTS / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


app = create_app()
