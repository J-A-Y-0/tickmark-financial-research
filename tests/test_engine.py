"""The pipeline end to end, with scripted models."""

from tests.conftest import FakeChat, FakeWeb, events_of, plan_json
from tickmark.engine import AskRequest
from tickmark.models import Depth, Evidence, Route, RouteDecision, SubQuestion
from tickmark.synthesis import ExtractiveSynthesizer, Part, build_prompt, merge_sources


def run(engine, question, user="analyst", **kwargs):
    return list(engine.ask(AskRequest(user, question, **kwargs)))


def test_event_order(make_engine):
    events = run(make_engine(), "What was Uber's revenue in 2021, and what is our price target on Uber?")
    order = [e["type"] for e in events]
    first_token = order.index("token")
    assert order[0] == "start" and order[-1] == "done"
    assert order.index("plan") < order.index("route") < order.index("trace") < order.index("sources") < first_token
    assert order.index("verification") > max(i for i, t in enumerate(order) if t == "token")


def test_one_generation_call_for_a_multi_part_question(make_engine):
    chat = FakeChat(
        [
            plan_json("What was Uber's revenue in 2021?", "What is our price target on Uber?"),
            '{"route": "filings", "reason": "reported"}',
            '{"route": "internal", "reason": "house view"}',
        ],
        answer="Uber's revenue was $17.5 billion in 2021 [1]. Our price target is $92 [7].",
    )
    events = run(make_engine(chat=chat), "Uber revenue 2021 and our target?", router="llm")
    assert len(chat.stream_calls) == 1
    assert {r["route"] for r in events_of(events, "route")} == {"filings", "internal"}
    done = events_of(events, "done")[0]
    assert done["status"] == "answered"
    assert done["usage"]["calls"] == 4  # plan + 2 routes + 1 answer


def test_simple_question_uses_one_route(make_engine):
    chat = FakeChat(
        [plan_json("What was Lyft's revenue in 2024?"), '{"route": "filings", "reason": "x"}'],
        answer="Lyft revenue was $5.8 billion [1].",
    )
    events = run(make_engine(chat=chat), "What was Lyft's revenue in 2024?", router="llm")
    assert len(events_of(events, "route")) == 1 and len(chat.stream_calls) == 1


def test_sources_reach_the_prompt_and_numbers_are_verified(make_engine):
    chat = FakeChat(
        [plan_json("What was Lyft's net income in 2024?"), '{"route": "filings", "reason": "x"}'],
        answer="Lyft's 2024 net income was $22.8 million [3].",
    )
    events = run(make_engine(chat=chat), "Lyft net income 2024?", router="llm")
    system, prompt = chat.stream_calls[0]
    assert "Sources:" in prompt and "Lyft, Inc. Form 10-K" in prompt
    assert "untrusted data" in system
    summary = events_of(events, "verification")[0]["summary"]
    assert summary["sentences_checked"] == 1


def test_fabricated_citation_is_flagged_and_not_cached(make_engine):
    chat = FakeChat(
        [plan_json("Lyft revenue 2024?"), '{"route": "filings", "reason": "x"}'], answer="Revenue was $9 billion [42]."
    )
    engine = make_engine(chat=chat)
    events = run(engine, "Lyft revenue 2024?", router="llm")
    assert events_of(events, "verification")[0]["summary"]["fabricated_citations"] == 1
    assert engine.db.query("SELECT COUNT(*) AS n FROM answer_cache")[0]["n"] == 0


def test_generation_failure_reports_an_error_not_an_answer(make_engine):
    chat = FakeChat([plan_json("Lyft revenue 2024?"), '{"route": "filings", "reason": "x"}'], fail_stream=True)
    events = run(make_engine(chat=chat), "Lyft revenue 2024?", router="llm")
    assert events_of(events, "error")[0]["code"] == "generation_failed"
    assert events_of(events, "done")[0]["status"] == "error"


def test_planner_and_router_failures_degrade_gracefully(make_engine):
    events = run(make_engine(chat=FakeChat(fail=True, answer="unused")), "What is our price target on Uber?", router="llm")
    plan = events_of(events, "plan")[0]
    route = events_of(events, "route")[0]
    assert plan["fallback"] and route["fallback"] and route["route"] == "internal"


def test_web_failure_is_visible_and_the_answer_is_partial(make_engine):
    engine = make_engine(web=FakeWeb(ok=False))
    events = run(engine, "What is the latest news on DoorDash?")
    trace = events_of(events, "trace")[0]
    assert trace["ok"] is False and trace["error"] == "provider timeout"
    assert events_of(events, "done")[0]["status"] == "partial"


def test_web_results_are_cited(make_engine):
    web = FakeWeb(
        [
            Evidence(
                Route.WEB,
                "DoorDash news",
                "DoorDash announced a new grocery partnership this week.",
                url="https://example.com/a",
                locator="example.com",
            )
        ]
    )
    events = run(make_engine(web=web), "What is the latest news on DoorDash?")
    sources = events_of(events, "sources")[0]["sources"]
    assert sources[0]["kind"] == "web" and sources[0]["url"] == "https://example.com/a"
    assert "[1]" in "".join(e["text"] for e in events_of(events, "token"))


def test_deep_mode_cap_and_budget(make_engine):
    engine = make_engine(settings={"deep_daily_cap": 1})
    assert events_of(run(engine, "How did DoorDash grow in 2024?", depth=Depth.DEEP), "done")[0]["status"] == "answered"
    capped = run(engine, "How did Lyft grow in 2024?", depth=Depth.DEEP)
    assert events_of(capped, "error")[0]["code"] == "deep_cap"
    broke = make_engine(settings={"daily_budget_usd": 0.0})
    assert events_of(run(broke, "Uber revenue 2021?"), "error")[0]["code"] == "budget_exceeded"


def test_fixed_plan_is_used_verbatim(make_engine):
    events = run(make_engine(), "ignored", plan=["What was Uber's revenue in 2021?", "What is our price target on Uber?"])
    assert events_of(events, "plan")[0]["planner"] == "fixed"
    assert len(events_of(events, "route")) == 2


def test_input_validation(make_engine):
    engine = make_engine()
    assert events_of(run(engine, "   "), "error")[0]["code"] == "empty_question"
    assert events_of(run(engine, "x" * 1001), "error")[0]["code"] == "too_long"
    assert events_of(run(engine, "Uber?", router="jev"), "error")[0]["code"] == "router_unavailable"


def test_audit_log_records_routes_and_cost(make_engine):
    engine = make_engine()
    run(engine, "What was Lyft's net income in 2024 and what is our rating on Lyft?", user="consultant")
    row = engine.audit.recent()[0]
    assert row["user_id"] == "consultant" and row["status"] == "partial"
    assert [r["allowed"] for r in row["routes"]] == [True, False]


def test_estimate(make_engine):
    estimate = make_engine(chat=FakeChat()).estimate("analyst", Depth.DEEP)
    assert estimate["deep_runs_left_today"] == 5
    assert estimate["estimated_cost_usd"] is None  # fake model has no price


# ---- citation merging and prompt building ----------------------------------------------


def _part(i, route, denied=False):
    return Part(SubQuestion(i, f"Q{i}?"), RouteDecision(route, "r", "test"), denied=denied)


def test_merge_sources_dedupes_and_numbers_contiguously():
    a = Evidence(Route.FILINGS, "Uber 10-K", "text a", doc_id="u", locator="Item 7")
    b = Evidence(Route.FILINGS, "Uber 10-K", "text b", doc_id="u", locator="Item 1")
    w = Evidence(Route.WEB, "News", "text w", url="https://x")
    parts = [_part(1, Route.FILINGS), _part(2, Route.FILINGS), _part(3, Route.WEB)]
    sources = merge_sources(parts, {1: [a, b], 2: [b], 3: [w, w]})
    assert [s.number for s in sources] == [1, 2, 3]
    assert sources[1].sub_questions == [1, 2]
    assert [s.number for s in parts[1].sources] == [2]


def test_prompt_mentions_denied_parts_without_their_content():
    parts = [_part(1, Route.FILINGS), _part(2, Route.INTERNAL, denied=True)]
    sources = merge_sources(parts, {1: [Evidence(Route.FILINGS, "Lyft 10-K", "Net income 22.8", doc_id="l", locator="x")]})
    _, prompt = build_prompt("q", parts, sources, Depth.QUICK)
    assert "NOT RESEARCHED" in prompt and "[1] Lyft 10-K" in prompt


def test_extractive_writer_quotes_table_rows_with_their_columns():
    text = "2024 | 2023 | % Change\n(in millions, except percentages)\nNet income (loss) | 22.8 | (340.3) | 107%"
    parts = [_part(1, Route.FILINGS)]
    parts[0].sub = SubQuestion(1, "What was Lyft's net income in 2024?")
    sources = merge_sources(parts, {1: [Evidence(Route.FILINGS, "Lyft, Inc. Form 10-K", text, doc_id="l", locator="x")]})
    answer = "".join(ExtractiveSynthesizer().stream("q", parts, sources, Depth.QUICK))
    assert "2024: 22.8" in answer and answer.rstrip().endswith("[1]")


def test_time_limit_ends_with_an_honest_partial_answer(make_engine):
    events = run(make_engine(settings={"quick_timeout_s": 0.0}), "What was Lyft's net income in 2024?")
    done = events_of(events, "done")[0]
    assert done["status"] == "partial" and done["terminated"] == "timeout"
    assert "time limit" in "".join(e["text"] for e in events_of(events, "token"))


def test_percentage_tables_keep_their_meaning_when_quoted():
    from tickmark.synthesis import table_facts

    text = "The following table sets forth costs as a percentage of revenue:\n2024 | 2023\nNet income (loss) | 0.4% | (7.7)%"
    assert table_facts(text) == ["Net income (loss) (% of revenue): 2024: 0.4%; 2023: (7.7)%."]
