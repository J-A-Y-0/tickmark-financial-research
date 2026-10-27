"""Command line: ask a question, rebuild the index, or show the configuration.

python -m tickmark ask "What was Uber's revenue in 2021?" --role analyst
python -m tickmark ask "Should we stay Overweight on Uber?" --deep --router cascade
python -m tickmark index --force
python -m tickmark info
"""

from __future__ import annotations

import argparse
import json
import sys

from tickmark.config import load_settings
from tickmark.engine import AskRequest, Engine
from tickmark.models import Depth
from tickmark.retrieval.ingest import build_corpus_index

LABELS = {"filings": "SEC filings", "internal": "Internal research", "web": "Web"}


def _ask(engine: Engine, args) -> int:
    request = AskRequest(args.user, args.question, Depth.DEEP if args.deep else Depth.QUICK, args.router, not args.no_cache)
    status = "error"
    for event in engine.ask(request):
        kind = event["type"]
        if args.json:
            print(json.dumps(event))
            if kind == "done":
                status = event["status"]
            continue
        if kind == "plan":
            print(f"Plan ({event['planner']}):")
            for sub in event["sub_questions"]:
                print(f"  {sub['index']}. {sub['text']}")
        elif kind == "route":
            mark = "" if event["allowed"] else "  DENIED"
            print(f"  -> Q{event['sub_question']}: {LABELS[event['route']]} [{event['router']}] {event['reason']}{mark}")
        elif kind == "cache":
            print(f"Cache: {event['status']}")
        elif kind == "error":
            print(f"Error: {event['message']}", file=sys.stderr)
        elif kind == "sources" and event["sources"]:
            print()
        elif kind == "token":
            print(event["text"], end="", flush=True)
        elif kind == "verification":
            summary = event["summary"]
            print(f"\n\nVerified {summary['verified']}/{summary['sentences_checked']} sentences")
            for sentence in event["sentences"]:
                if sentence["verdict"] != "verified":
                    print(f"  ! {sentence['verdict']}: {sentence['text'][:100]}")
        elif kind == "done":
            status = event["status"]
            print(
                f"\nStatus {status} · cache {event['cache']} · ${event['cost_usd']:.5f} · {event['timings_ms'].get('total', 0):.0f} ms"
            )
    return 0 if status in {"answered", "partial", "denied"} else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tickmark", description="Permission-aware research engine")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--role", dest="user", default="analyst", help="access level: pm, analyst or consultant")
    ask.add_argument("--deep", action="store_true", help="deep research mode")
    ask.add_argument("--router", default=None, help="keyword, llm, jev or cascade")
    ask.add_argument("--no-cache", action="store_true")
    ask.add_argument("--json", action="store_true", help="print raw events as JSON lines")
    index = sub.add_parser("index", help="rebuild the corpus index")
    index.add_argument("--force", action="store_true")
    sub.add_parser("info", help="show the active configuration")
    args = parser.parse_args(argv)

    if args.command == "index":
        engine = Engine(load_settings(), build_index=False)
        rebuilt = build_corpus_index(engine.store, engine.settings.data_dir, force=args.force)
        print(("Rebuilt" if rebuilt else "Up to date") + f": {engine.store.chunk_count()} chunks")
        return 0
    engine = Engine.from_settings(load_settings())
    if args.command == "info":
        print(json.dumps(engine.describe(), indent=2))
        return 0
    return _ask(engine, args)


if __name__ == "__main__":
    sys.exit(main())
