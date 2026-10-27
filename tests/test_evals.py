"""The evaluation set and its scoring rules."""

import json

import pytest

from evals.run import load_cases, run, to_markdown
from evals.scoring import aggregate, figure_present, score_case
from tickmark.access import AccessPolicy
from tickmark.config import ROOT
from tickmark.models import Route

CASES = load_cases()


def test_case_file_is_well_formed():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    policy = AccessPolicy.load(ROOT / "data" / "access.json")
    for case in CASES:
        assert case["sub_questions"], case["id"]
        for sub in case["sub_questions"]:
            assert sub["routes"] and all(Route(r) for r in sub["routes"])
        assert case["expect_status"] in {"answered", "partial", "denied", "rejected"}
        if case["expect_status"] != "rejected":
            assert policy.user(case["user"]) is not None, case["id"]
        for figure in case.get("must_include_figures", []):
            figure_present(figure, figure)  # every expected figure parses


def test_case_file_covers_every_category():
    assert {c["category"] for c in CASES} == {"filings", "internal", "web", "multi_source", "ambiguous", "injection", "access"}
    assert any(len(c["sub_questions"]) > 1 and len({r for s in c["sub_questions"] for r in s["routes"]}) > 1 for c in CASES)


@pytest.mark.parametrize(
    "expected, answer, ok",
    [
        ("$17,455 million", "Revenue was $17.5 billion [1].", True),
        ("$17,455 million", "Revenue (in millions): 2021: 17,455.", True),
        ("$17,455 million", "Revenue was $11.1 billion.", False),
        ("828.3 million", "Lyft completed 828 million rides.", True),
        ("15%", "We model 15% growth.", True),
        ("15%", "We model 15 basis points.", False),
    ],
)
def test_figure_present(expected, answer, ok):
    assert figure_present(expected, answer) is ok


def _events(routes, answer, status="answered", sources=()):
    return [
        *(
            {"type": "route", "sub_question": i, "route": r, "latency_ms": 5.0, "usage": {"cost_usd": 0.0001}}
            for i, r in enumerate(routes, 1)
        ),
        {"type": "sources", "sources": [{"doc_id": d} for d in sources]},
        {"type": "token", "text": answer},
        {"type": "verification", "summary": {"grounding_rate": 1.0}},
        {"type": "done", "status": status, "cost_usd": 0.002, "timings_ms": {"total": 900.0}},
    ]


def test_scoring_dimensions():
    case = next(c for c in CASES if c["id"] == "M01")
    good = score_case(case, "llm", _events(["filings", "internal"], "Revenue was $17.5 billion [1]. Target $92 [2]."))
    assert good.route_hits == [True, True] and good.answer_correct and good.safe is None and not good.reasons
    bad = score_case(case, "llm", _events(["web", "internal"], "No figures."))
    assert bad.route_hits == [False, True] and bad.answer_correct is False and len(bad.reasons) == 2


def test_safety_scoring_catches_leaks_and_forbidden_sources():
    case = next(c for c in CASES if c["id"] == "X02")
    leak = score_case(case, "jev", _events(["internal"], "Atlas is a $40 million deal.", sources=["hpc-deal-atlas-2025-06"]))
    assert leak.safe is False and any("leaked" in r for r in leak.reasons)
    clean = score_case(case, "jev", _events(["internal"], "I could not find this."))
    assert clean.safe is True


def test_aggregate():
    case = next(c for c in CASES if c["id"] == "F01")
    scores = [score_case(case, "keyword", _events(["filings"], "Revenue was $17.5 billion [1]."))]
    summary = aggregate("keyword", scores)
    assert summary["route_accuracy"] == 1.0 and summary["answer_accuracy"] == 1.0
    assert summary["route_cost_per_case"] == pytest.approx(0.0001)


def test_harness_runs_end_to_end_offline(make_engine):
    subset = [c for c in CASES if c["id"] in {"F01", "M02", "X01", "X04"}]
    rows, summaries = run(["keyword"], subset, make_engine())
    assert len(rows) == 4 and summaries[0]["safety_pass_rate"] == 1.0
    report = {
        "run_id": "t",
        "created_at": "now",
        "cases": 4,
        "cases_sha256": "0" * 64,
        "answer_model": "x",
        "web_search": "none",
        "routers": summaries,
        "failures": [r for r in rows if r["reasons"]],
    }
    assert "| keyword |" in to_markdown(report)
    json.dumps(report)


def test_web_case_needs_a_web_source():
    case = next(c for c in CASES if c["id"] == "W03")
    internal_only = _events(["internal"], "The policy rate is 4%.", sources=["hpc-research-policy-2024"])
    for event in internal_only:
        if event["type"] == "sources":
            event["sources"] = [{"doc_id": "hpc-research-policy-2024", "kind": "internal"}]
    assert score_case(case, "keyword", internal_only).answer_correct is False
