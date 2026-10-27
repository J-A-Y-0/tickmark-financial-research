"""Run the router comparison and write a report.

    python -m evals.run                                # every configured router
    python -m evals.run --routers llm,jev,cascade      # a subset
    python -m evals.run --limit 5 --out /tmp/smoke     # quick smoke run

Identical conditions for every router: the sub-questions come from the case file (no
planner variance), the same sources, answer model and scoring rules are used, and the
answer cache is off. The report records the SHA-256 of the case file, so a reader can
check the cases were not edited after the results came in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from evals.scoring import aggregate, score_case
from tickmark.config import load_settings
from tickmark.db import Database
from tickmark.engine import AskRequest, Engine
from tickmark.models import Depth

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases" / "cases.jsonl"
RESULTS = ROOT / "results"


def code_version() -> str:
    """The commit the run used, marked dirty if the working tree had uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(routers: list[str], cases: list[dict], engine: Engine) -> tuple[list[dict], list[dict]]:
    rows, summaries = [], []
    for router in routers:
        scores = []
        for case in cases:
            request = AskRequest(
                case["user"],
                case["question"],
                Depth.QUICK,
                router=router,
                use_cache=False,
                plan=[s["text"] for s in case["sub_questions"]],
            )
            events = list(engine.ask(request))
            score = score_case(case, router, events)
            scores.append(score)
            mark = "ok " if not score.reasons else "FAIL"
            print(f"  [{router:8}] {mark} {case['id']} {'; '.join(score.reasons)[:110]}", file=sys.stderr)
        summaries.append(aggregate(router, scores))
        rows.extend(s.to_dict() for s in scores)
    return rows, summaries


def to_markdown(report: dict) -> str:
    lines = [
        f"# Router evaluation {report['run_id']}",
        "",
        f"{report['created_at']} · code `{report.get('code_version', 'unknown')}` · {report['cases']} cases (sha256 `{report['cases_sha256'][:12]}`) · "
        f"answer model `{report['answer_model']}` · planner fixed by case file · cache off · "
        f"web search: {report['web_search']}",
        "",
        "| Router | Route accuracy | Answer accuracy | Safety | Grounding | Route p50 | End-to-end p50 | Route $/case | Total $/case |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    pct = lambda v: "–" if v is None else f"{v:.0%}"  # noqa: E731
    for r in report["routers"]:
        lines.append(
            f"| {r['router']} | {pct(r['route_accuracy'])} | {pct(r['answer_accuracy'])} | {pct(r['safety_pass_rate'])} | "
            f"{pct(r['grounding_rate'])} | {r['route_ms_p50']:.0f} ms | {r['total_ms_p50']:.0f} ms | "
            f"${r['route_cost_per_case']:.5f} | ${r['total_cost_per_case']:.5f} |"
        )
    lines += ["", "## Failures", "", "| Router | Case | Category | Reasons |", "| --- | --- | --- | --- |"]
    for f in report["failures"]:
        lines.append(f"| {f['router']} | {f['case_id']} | {f['category']} | {'; '.join(f['reasons'])} |")
    if not report["failures"]:
        lines.append("| – | – | – | none |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routers", default="", help="comma-separated; default: every configured router")
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)

    settings = load_settings()
    with tempfile.TemporaryDirectory() as tmp:
        engine = Engine(settings, db=Database(Path(tmp) / "eval.sqlite3"))
        available = engine.available_routers()
        routers = [r.strip() for r in args.routers.split(",") if r.strip()] or available
        missing = [r for r in routers if r not in available]
        if missing:
            print(f"Not configured: {', '.join(missing)} (available: {', '.join(available)})", file=sys.stderr)
            return 2
        cases = load_cases(args.cases)[: args.limit or None]
        started = time.time()
        rows, summaries = run(routers, cases, engine)
        info = engine.describe()

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    report = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cases": len(cases),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "code_version": code_version(),
        "answer_model": info["answer_model"],
        "fast_model": info["fast_model"],
        "embeddings": info["embeddings"],
        "web_search": info["web_search"],
        "duration_s": round(time.time() - started, 1),
        "routers": summaries,
        "failures": [r for r in rows if r["reasons"]],
        "results": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{run_id}.json").write_text(json.dumps(report, indent=2))
    (args.out / f"{run_id}.md").write_text(to_markdown(report))
    (args.out / "latest.json").write_text(json.dumps(report, indent=2))
    print(to_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
