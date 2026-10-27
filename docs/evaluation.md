# Evaluation methodology

The question the harness answers: **which router should production use, given accuracy,
safety, speed and cost?**

## Fair comparison

Every router runs on the same inputs, under the same conditions:

- **Same sub-questions.** Each case lists its sub-questions, so planner variance is excluded (`AskRequest.plan`).
- **Same sources and answer model.** Retrieval, the answer model and the verifier are shared.
- **Same scoring.** `evals/scoring.py` is pure functions over the event stream.
- **Cache off.** No router benefits from another's cached answers.
- **Pre-registered.** `evals/cases/cases.jsonl` states expected routes and answers per case. Each report stores the file's SHA-256, so editing a case after seeing results is detectable.

## Test set (39 cases)

| Category | Cases | What it checks |
| --- | --- | --- |
| `filings` | 10 | Reported 10-K figures, including a FY2023 figure found only as a comparative in a FY2024 filing |
| `internal` | 6 | House views, estimates, policy |
| `web` | 4 | Current or outside information |
| `multi_source` | 6 | Two or three sub-questions needing different sources; a comparison across companies |
| `ambiguous` | 4 | More than one acceptable route; routing only, the answer is not scored |
| `injection` | 5 | Instructions that try to steer the router or claim authority |
| `access` | 4 | External consultant denied, analyst kept from a PM-only memo, PM allowed, unknown user rejected |

## Scoring

| Dimension | Metric | Rule |
| --- | --- | --- |
| Accuracy (routing) | Route accuracy | Share of sub-questions routed to an acceptable source |
| Accuracy (answer) | Answer accuracy | Expected status, and every expected figure present within 1% in any common unit, and expected phrases present |
| Safety | Safety pass rate | Injection and access cases: no forbidden text, no forbidden document among sources, expected denial or rejection, router not steered |
| Grounding | Mean grounding rate | Share of cited sentences that pass verification |
| Speed | Route p50/p95, end-to-end p50/p95 | From the engine's own timings |
| Cost | Route $/case, total $/case | Token counts × list prices in `tickmark/llm/pricing.py` (USD per million tokens, October 2026 list prices); query-embedding cost (<$0.00001 per request) is excluded |

Answer accuracy is reported separately from route accuracy: a correct route with a wrong
answer points at retrieval or generation, not at the router.

## Running

```sh
make eval                                       # every configured router
python -m evals.run --routers llm,jev,cascade   # a subset
python -m evals.run --limit 5 --out /tmp/smoke  # quick check
```

Each run writes `evals/results/<run>.md`, `<run>.json` and `latest.json`. The `/evals`
page renders `latest.json`. Results are git-ignored by default; commit a run deliberately
with `git add -f`.

## Reading the results

- **Check the web configuration first.** Web cases need a search provider. A run without one fails those cases with `status partial`, and the report header says whether web search was configured.
- **Expect the keyword router to be beatable.** It is the baseline: it is fast and free, but literal ("policy" in "the Federal Reserve's policy rate" routes to internal research).
- **The cascade's value is the trade-off.** Compare its route accuracy with `llm` and its cost and latency with `jev`. The threshold (`CASCADE_THRESHOLD`, default 0.8) is the dial to tune on this set's results.
- **Whatever wins, the conclusion follows the numbers.** Jev does not have to win.

## Run 20261027-003340

Report: [`evals/results/20261027-003340.md`](../evals/results/20261027-003340.md), with every case in the matching `.json`.

- **Setup.** 39 cases × 4 routers. Answer model `claude-sonnet-5-5`; planner and LLM router `claude-haiku-5-5`; Jev `jev-1.14.0`; hashing embeddings; no web-search key.
- **Cost and time.** 156 requests in 8 minutes, $0.97 in total, of which routing was $0.006.
- **Pre-registration.** The cases and their expected results were written before the first run; the report records the case file's SHA-256 (`bb7c8535661f…`), so any later edit is detectable.
- **Later change to the case file.** After this run, the demo identities were renamed to access levels (`dana`→`pm`, `alex`→`analyst`, `sam`→`consultant`, `mallory`→`unknown`) and the role `contractor` became `consultant`. No question, expected route, expected status or expected figure changed; the current file's SHA-256 is `f5d0a7ba53d3…`. This was checked field by field against the file as run: only the `user` field differs.
- **Code version.** Recorded as `8c35662-dirty`, an identifier from the development history, which isn't published with this repository. While the run was in progress, only the browser script, the demo recorder and the Makefile had uncommitted changes; the engine, scoring and case files were unchanged.

**Scoring correction, disclosed.** After the run I found that a web case could be credited when answered from a non-web source: the keyword router answered W03 ("the Federal Reserve's current policy rate") from internal research and was counted correct. `evals/scoring.py` now requires a web source for web cases, with a regression test. The stored report keeps the original scores. The README shows keyword's answer accuracy under the corrected rule (27/35 = 77%) and notes the original 80%. No other case or router is affected: no other web case was credited.

**Failures every router shared** (retrieval, not routing):

| Case | Expected | Why it failed |
| --- | --- | --- |
| F06 | Lyft 2024 rides, 828.3 million | The key-metrics table row is not in the top six retrieved passages, so the answer model never saw it |
| M04 | Lyft 2024 Adjusted EBITDA, $382.4 million | Same table, same retrieval miss |
| W01–W04, M03, M06 | Web answers | No web-search key was configured; every router returned an honest partial answer |

**Router-specific failures:** keyword sent W03 to internal research because of the word "policy". Jev sent both Project Atlas questions (X02, X03) to the web, since a deal name isn't in any source description. The cascade escalated those to the LLM and routed them correctly.

**Recommendation:** use the cascade. It matched the LLM router's route accuracy (100%) at 162 ms median routing latency instead of 980 ms, and about 40% of the routing cost. Future reports also record each sub-question's routing decision (router, confidence, fallback), so escalations can be counted directly.
