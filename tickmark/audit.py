"""Audit log: one row per request. It also backs the per-user spend controls.

Each row records who asked what, which sources were chosen and which were denied, the
cache outcome, cost, latency and the verification summary. That trail is what lets a
compliance team answer "who saw what, and on what basis".
"""

from __future__ import annotations

import json
import time

from tickmark.db import Database

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    request_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT,
    question TEXT NOT NULL,
    depth TEXT NOT NULL,
    router TEXT,
    status TEXT NOT NULL,          -- answered | partial | denied | rejected | error
    cache TEXT NOT NULL,           -- HIT | MISS | BYPASS | n/a
    routes TEXT NOT NULL,          -- JSON list of {sub_question, route, allowed}
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    verification TEXT,             -- JSON summary
    detail TEXT                    -- JSON: timings, usage, errors
);
CREATE INDEX IF NOT EXISTS runs_user_day ON runs(user_id, created_at);
"""

DAY_S = 86400


def _day_start(now: float | None = None) -> float:
    now = now or time.time()
    return now - (now % DAY_S)  # UTC day


class AuditLog:
    def __init__(self, db: Database):
        self.db = db
        with db.tx() as conn:
            conn.executescript(SCHEMA)

    def record(self, run: dict) -> None:
        with self.db.tx() as conn:
            conn.execute(
                """INSERT INTO runs(request_id, created_at, user_id, role, question, depth, router, status, cache,
                                    routes, cost_usd, latency_ms, verification, detail)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run["request_id"],
                    run.get("created_at", time.time()),
                    run["user_id"],
                    run.get("role"),
                    run["question"],
                    run["depth"],
                    run.get("router"),
                    run["status"],
                    run.get("cache", "n/a"),
                    json.dumps(run.get("routes", [])),
                    run.get("cost_usd", 0.0),
                    run.get("latency_ms", 0.0),
                    json.dumps(run.get("verification")) if run.get("verification") else None,
                    json.dumps(run.get("detail", {})),
                ),
            )

    def spend_today(self, user_id: str) -> float:
        rows = self.db.query(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM runs WHERE user_id = ? AND created_at >= ?",
            (user_id, _day_start()),
        )
        return float(rows[0]["total"])

    def deep_runs_today(self, user_id: str) -> int:
        rows = self.db.query(
            """SELECT COUNT(*) AS n FROM runs WHERE user_id = ? AND created_at >= ? AND depth = 'deep'
               AND status IN ('answered', 'partial') AND cache != 'HIT'""",
            (user_id, _day_start()),
        )
        return int(rows[0]["n"])

    def recent(self, limit: int = 50, user_id: str | None = None) -> list[dict]:
        if user_id:
            rows = self.db.query("SELECT * FROM runs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit))
        else:
            rows = self.db.query("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        out = []
        for row in rows:
            item = dict(row)
            item["routes"] = json.loads(item["routes"])
            item["verification"] = json.loads(item["verification"]) if item["verification"] else None
            item["detail"] = json.loads(item["detail"]) if item["detail"] else {}
            out.append(item)
        return out

    def stats(self, user_id: str | None = None) -> dict:
        """Today's totals, for one user or (user_id=None) for everyone."""
        where, params = "created_at >= ?", [_day_start()]
        if user_id:
            where, params = where + " AND user_id = ?", [*params, user_id]
        rows = self.db.query(
            f"""SELECT COUNT(*) AS runs, COALESCE(SUM(cost_usd), 0) AS cost,
                       COALESCE(AVG(latency_ms), 0) AS avg_latency,
                       SUM(cache = 'HIT') AS hits, SUM(status = 'denied') AS denied
                FROM runs WHERE {where}""",
            params,
        )
        r = rows[0]
        return {
            "runs_today": r["runs"],
            "cost_today_usd": round(r["cost"], 4),
            "avg_latency_ms": round(r["avg_latency"], 1),
            "cache_hits_today": r["hits"] or 0,
            "denied_today": r["denied"] or 0,
        }
