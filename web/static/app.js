// Tickmark client. Streams /api/ask (Server-Sent Events over a POST) into a thread of turns.
// Source text comes from the web and from uploads, so everything is escaped before formatting.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const LABEL = { filings: "SEC filings", internal: "Internal research", web: "Web" };
const VERDICT = {
  number_unverified: "Figure not found in source",
  weak_support: "Weakly supported",
  fabricated: "Cites a missing source",
  uncited: "No citation",
  contradicted: "Contradicted by source",
};
const ICON_OK = '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const ICON_WARN = '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.5l6 11H2z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M8 7v3M8 12v.2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
const seconds = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
const list = (items) => (items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`);

// ---- figures: the same matching rule as the server's verifier ------------------------------
const FIG = /(\$\s?)?\(?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\)?\s?(%|percent\b|trillion\b|billion\b|million\b|thousand\b)?/gi;
const SCALE = { thousand: 1e3, million: 1e6, billion: 1e9, trillion: 1e12 };
function figures(text) {
  const out = [];
  for (const m of text.matchAll(FIG)) {
    const unit = (m[4] || "").toLowerCase();
    const value = parseFloat(m[2].replace(/,/g, "") + (m[3] ? "." + m[3] : ""));
    const isYear = !m[1] && !unit && !m[3] && !m[2].includes(",") && value >= 1900 && value <= 2100;
    out.push({ index: m.index, length: m[0].length, value, decimals: (m[3] || "").length,
      scale: SCALE[unit] || 1, percent: unit === "%" || unit === "percent", dollar: !!m[1], isYear });
  }
  return out;
}
function sameFigure(claim, fig) {
  if (claim.percent !== fig.percent) return false;
  const target = claim.value * claim.scale;
  const tol = 0.5 * 10 ** -claim.decimals * claim.scale + 1e-9 * Math.max(1, target);
  const scales = fig.scale !== 1 || fig.percent ? [fig.scale] : [1, 1e3, 1e6, 1e9];
  return scales.some((s) => Math.abs(fig.value * s - target) <= tol);
}
const DATE = /\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}\b,?/gi;
const COMPOUND = /\b\d+(?:\.\d+)?-(?=[A-Za-z])/g; // "12-month", "3-year": descriptors, not figures
const claimFigures = (sentence) => figures(sentence.replace(/\[\d{1,3}\]/g, " ").replace(DATE, " ").replace(COMPOUND, " "))
  .filter((f) => !f.isYear && !(f.scale === 1 && !f.percent && !f.dollar && f.decimals === 0 && f.value <= 10));

// ---- markdown with sentence spans -----------------------------------------------------------
const SENT = /(?<=[.!?]["”’)]?)\s+(?=[A-Z0-9“"(\[*-])/;
function sentencesOf(line) {
  const out = [];
  for (const piece of line.split(SENT)) {
    const lead = piece.match(/^((?:\[\d{1,3}\]\s*)+)([\s\S]*)$/);
    if (lead && out.length) { out[out.length - 1] += " " + lead[1].trim(); if (lead[2].trim()) out.push(lead[2].trim()); }
    else if (piece.trim()) out.push(piece.trim());
  }
  return out;
}
const plain = (s) => s.replace(/\*\*/g, "").replace(/\s+/g, " ").trim();

function renderMarkdown(text, flags, known, verdicts = new Map()) {
  let html = "", list = false, para = [];
  const inline = (raw) => esc(raw)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|\W)_(.+?)_(?=\W|$)/g, "$1<em>$2</em>")
    .replace(/\[(\d{1,3})\]/g, (_, n) => `<a class="cite${known.size && !known.has(n) ? " bad" : ""}" href="#" data-n="${n}" aria-label="Open source ${n}">${n}</a>`);
  const spans = (line) => sentencesOf(line).map((s) => {
    const issue = flags.get(plain(s));
    return `<span class="s${issue ? " flag" : ""}" data-s="${esc(plain(s))}"${issue ? ` title="${esc(issue)}"` : ""}>${inline(s)}</span>`;
  }).join(" ");
  const tick = (line) => {
    const states = sentencesOf(line).map((s) => verdicts.get(plain(s))).filter(Boolean).map((v) => v.verdict);
    if (!states.length) return "";
    return states.every((v) => v === "verified") ? ' class="tick ok"' : ' class="tick warn"';
  };
  const flush = () => { if (para.length) { const line = para.join(" "); html += `<p${tick(line)}>${spans(line)}</p>`; para = []; } };
  const endList = () => { if (list) { html += "</ul>"; list = false; } };
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) { flush(); endList(); continue; }
    const h = line.match(/^#{1,4}\s+(.*)$/);
    const li = line.match(/^(?:[-*•]|\d+\.)\s+(.*)$/);
    const lead = line.match(/^\*\*([^*]+\?)\*\*$/);
    if (h) { flush(); endList(); html += `<h3>${inline(h[1])}</h3>`; }
    else if (lead) { flush(); endList(); html += `<span class="lead">${esc(lead[1])}</span>`; }
    else if (li) { flush(); if (!list) { html += "<ul>"; list = true; } html += `<li${tick(li[1])}>${spans(li[1])}</li>`; }
    else { endList(); para.push(line); }
  }
  flush(); endList();
  return html;
}

// ---- SSE over fetch -------------------------------------------------------------------------
async function* sse(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const data = frame.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
      if (data) yield JSON.parse(data);
    }
  }
}

// ---- sources grouped by document --------------------------------------------------------------
function shortTitle(s) {
  const filing = s.title.match(/^(\S+).*Form 10-K, fiscal year (\d{4})/);
  if (filing) return `${filing[1].replace(/,$/, "")} 10-K FY${filing[2]}`;
  if (s.kind === "web") return s.locator || s.title;
  return s.title;
}
function groupSources(sources) {
  const groups = new Map();
  for (const s of sources) {
    const key = s.kind === "web" ? s.url : s.doc_id;
    if (!groups.has(key)) groups.set(key, { kind: s.kind, title: shortTitle(s), numbers: [] });
    groups.get(key).numbers.push(s.n);
  }
  return [...groups.values()];
}

// ---- research page ----------------------------------------------------------------------------
function initResearch() {
  const form = $("#ask"), q = $("#q"), go = $("#go"), home = $("#home"), thread = $("#thread");
  const dock = $("#dock"), q2 = $("#q2"), drawer = $("#drawer");
  const turns = [];
  let running = false;

  const depth = () => $("input[name=depth]:checked").value;
  const autosize = () => { q.style.height = "auto"; q.style.height = `${Math.min(q.scrollHeight, 200)}px`; };
  q.addEventListener("input", autosize);
  q.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });

  async function refreshEstimate() {
    const hint = $("#estimate");
    if (depth() !== "deep") { hint.textContent = "Quick: one pass, splits only what you asked."; return; }
    hint.textContent = "Deep: plans 3–6 sub-questions.";
    try {
      const e = await (await fetch("/api/estimate?depth=deep")).json();
      const cost = e.estimated_cost_usd == null ? "" : ` About $${e.estimated_cost_usd.toFixed(3)} per question.`;
      hint.textContent = `Deep: plans 3–6 sub-questions.${cost} ${e.deep_runs_left_today} of ${e.deep_daily_cap} left today.`;
    } catch { /* the default hint stands */ }
  }
  $$("input[name=depth]").forEach((el) => el.addEventListener("change", refreshEstimate));

  $$(".suggestion").forEach((btn) => btn.addEventListener("click", () => {
    $(`#d-${btn.dataset.depth}`).checked = true;
    ask(btn.dataset.q);
  }));
  form.addEventListener("submit", (e) => { e.preventDefault(); if (q.value.trim()) ask(q.value.trim()); });
  dock.addEventListener("submit", (e) => { e.preventDefault(); if (q2.value.trim()) { const text = q2.value.trim(); q2.value = ""; ask(text); } });
  $("#new-q").addEventListener("click", () => {
    closeDrawer();
    turns.length = 0; thread.innerHTML = ""; thread.hidden = true; dock.hidden = true; home.hidden = false;
    history.pushState(null, "", "/");
    q.value = ""; autosize(); q.focus();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDrawer();
    if (e.key === "/" && !/input|textarea|select/i.test(document.activeElement.tagName)) { e.preventDefault(); (home.hidden ? q2 : q).focus(); }
  });

  // ---- one turn: a run sheet ----
  function stage(t, name, value, state) {
    const li = $(`[data-stage="${name}"]`, t.el);
    li.className = state;
    $(".stage-value", li).textContent = value;
  }

  function renderSteps(t) {
    const rows = [];
    if (t.plan.length) {
      rows.push(`<div class="step"><span class="step-name">Plan and route${t.planner ? ` · planner: ${esc(t.planner)}` : ""}</span>${t.plan.map((p) => {
        const r = p.route;
        const meta = !r ? "routing…" : r.allowed
          ? `${LABEL[r.route]} · ${esc(r.router)}${r.confidence != null ? ` ${Math.round(r.confidence * 100)}%` : ""}${r.fallback ? " · fallback" : ""} · ${seconds(r.latency_ms)}`
          : `${LABEL[r.route]} · denied for your role`;
        return `<div class="step-row ${r && !r.allowed ? "denied" : ""}"><span class="q">${p.index}. ${esc(p.text)}${r && r.reason ? `<small>${esc(r.reason)}</small>` : ""}</span><span class="meta">${meta}</span></div>`;
      }).join("")}</div>`);
    }
    if (t.searches.length) {
      rows.push(`<div class="step"><span class="step-name">Retrieve</span>${t.searches.map((x) =>
        `<div class="step-row ${x.ok ? "" : "bad"}"><span class="q">${x.sub_question}. ${LABEL[x.source]}${x.provider ? ` · ${esc(x.provider)}` : ""}</span><span class="meta">${x.ok ? `${x.results} passages · ${seconds(x.latency_ms)}${x.cached ? " · cached" : ""}` : esc(x.error)}</span></div>`).join("")}</div>`);
    }
    if (t.cache) rows.push(`<div class="step"><span class="step-name">Cache</span><div class="step-row"><span class="q">${esc(t.cache.status === "HIT" ? "Served from your role's cache partition" : t.cache.status === "MISS" ? "Checked your role's cache partition: no match" : `Skipped: ${t.cache.reason}`)}</span></div></div>`);
    if (t.done) {
      const d = t.done, tm = d.timings_ms || {};
      rows.push(`<div class="step"><span class="step-name">Timing and cost</span><div class="step-totals">${["plan", "route", "retrieve", "generate", "verify"].filter((k) => tm[k] != null).map((k) => `<span>${k} ${seconds(tm[k])}</span>`).join("")}<span>${d.usage.calls} model calls</span><span>${d.usage.input_tokens.toLocaleString()} tokens in · ${d.usage.output_tokens.toLocaleString()} out</span></div></div>`);
    }
    $(".steps", t.el).innerHTML = rows.join("");
  }

  function renderChips(t) {
    const block = $(".sources-block", t.el), box = $(".chips", t.el);
    block.hidden = !t.sources.length || !t.done;
    if (block.hidden) return;
    const cited = new Set([...t.answer.matchAll(/\[(\d{1,3})\]/g)].map((m) => Number(m[1])));
    const groups = groupSources(t.sources);
    const used = groups.filter((g) => g.numbers.some((n) => cited.has(n)));
    const rest = groups.filter((g) => !g.numbers.some((n) => cited.has(n)));
    const chip = (g) => {
      const nums = g.numbers.filter((n) => cited.has(n));
      const shown = nums.length ? nums : g.numbers;
      return `<button type="button" class="chip ${g.kind}" data-n="${shown[0]}" title="${esc(g.title)}"><span class="t">${esc(g.title)}</span><span class="n">${shown.slice(0, 4).join(" ")}${shown.length > 4 ? " …" : ""}</span></button>`;
    };
    box.innerHTML = used.map(chip).join("") +
      (rest.length ? `<button type="button" class="chip more" data-more>${t.showAll ? "Hide uncited" : `+${rest.length} retrieved, not cited`}</button>${t.showAll ? rest.map(chip).join("") : ""}` : "");
  }

  function renderAnswer(t) {
    $(".answer", t.el).innerHTML = renderMarkdown(t.answer, t.flags, new Set(t.sources.map((s) => String(s.n))), t.verdicts);
  }

  function renderLedger(t, v) {
    const rows = [];
    const sourceLabel = (n) => {
      const src = t.sources.find((s) => s.n === n);
      return src ? `<b>${esc(shortTitle(src))}</b><span>${esc(src.locator)}</span>` : `<b>Source ${n}</b><span>never retrieved</span>`;
    };
    const excerpt = (text) => { const p = plain(text).replace(/\s*\[\d{1,3}\]/g, ""); return p.length > 110 ? `${p.slice(0, 108)}…` : p; };
    for (const x of v.sentences) {
      const n = x.citations[0];
      if (x.figures && x.figures.length) {
        for (const f of x.figures) {
          const ok = f.source != null && x.verdict !== "contradicted";
          rows.push({ n: f.source ?? n, sentence: plain(x.text), ok, claim: `<b>${esc(f.claim)}</b><span>${esc(excerpt(x.text))}</span>`,
            source: f.source != null ? sourceLabel(f.source) : n ? sourceLabel(n) : "<span>No citation</span>",
            check: ok ? `Matches ${esc(f.found)}` : esc(VERDICT[x.verdict] || "Not found in source") });
        }
      } else {
        const ok = x.verdict === "verified";
        rows.push({ n, sentence: plain(x.text), ok, claim: `<span class="plain">${esc(excerpt(x.text))}</span>`,
          source: n ? sourceLabel(n) : "<span>No citation</span>", check: ok ? "Supported" : esc(VERDICT[x.verdict] || x.verdict) });
      }
    }
    const box = $(".ledger", t.el);
    box.hidden = !rows.length;
    const okCount = rows.filter((r) => r.ok).length;
    $(".ledger-count", t.el).textContent = `${okCount} of ${rows.length} ticked`;
    $(".ledger-rows", t.el).innerHTML = `<div class="ledger-row head" role="row"><span role="columnheader">Claim</span><span role="columnheader">Source</span><span role="columnheader">Check</span></div>` +
      rows.map((r) => `<button type="button" class="ledger-row ${r.ok ? "ok" : "warn"}" role="row" data-n="${r.n ?? ""}" data-s="${esc(r.sentence)}"><span class="c-claim" role="cell">${r.claim}</span><span class="c-src" role="cell">${r.source}</span><span class="c-check" role="cell">${r.ok ? ICON_OK : ICON_WARN}${r.check}</span></button>`).join("");
    return { ok: okCount, total: rows.length };
  }

  function applyVerification(t, v) {
    const flagged = v.sentences.filter((x) => x.verdict !== "verified");
    t.flags = new Map(flagged.map((x) => [plain(x.text), `${VERDICT[x.verdict] || x.verdict}: ${x.issues.join("; ")}`]));
    t.verdicts = new Map(v.sentences.map((x) => [plain(x.text), x]));
    const tally = renderLedger(t, v);
    if (!tally.total) stage(t, "verify", "No figures to check", "ok");
    else stage(t, "verify", `${tally.ok} of ${tally.total} ticked`, tally.ok === tally.total ? "ok" : "warn");
    renderAnswer(t);
  }

  function renderReceipt(t) {
    const d = t.done, box = $(".receipt", t.el);
    const model = thread.dataset.model, router = (t.plan.find((p) => p.route) || {}).route?.router || "";
    const cache = { HIT: "served from cache", MISS: "fresh run", BYPASS: "not cached" }[d.cache] || "";
    box.hidden = false;
    box.innerHTML = `<span>${esc(model)}</span><span>${esc(router)} router</span><span>${seconds((d.timings_ms || {}).total || 0)}</span><span>$${d.cost_usd.toFixed(4)}</span><span>${cache}</span><span>Logged to audit · ${esc(d.request_id)}</span>`;
  }

  function handle(t, e) {
    switch (e.type) {
      case "start":
        $(".run-meta", t.el).textContent = `${e.user.role_label} access · ${e.depth === "deep" ? "Deep" : "Quick"} run`;
        break;
      case "plan":
        t.plan = e.sub_questions.map((s) => ({ ...s, route: null }));
        t.planner = e.planner;
        stage(t, "plan", `${t.plan.length} ${t.plan.length === 1 ? "part" : "parts"}${e.fallback ? " · fallback" : ""}`, "ok");
        stage(t, "route", "Choosing sources…", "active");
        renderSteps(t);
        break;
      case "route": {
        const p = t.plan.find((x) => x.index === e.sub_question);
        if (p) p.route = e;
        if (t.plan.every((x) => x.route)) {
          const decided = t.plan.map((x) => x.route);
          const conf = decided.filter((r) => r.confidence != null).map((r) => r.confidence);
          const label = decided[0].router + (conf.length ? ` · ${Math.round(Math.min(...conf) * 100)}%` : "");
          stage(t, "route", label, decided.some((r) => r.fallback) ? "warn" : "ok");
          const allowed = decided.filter((r) => r.allowed).length;
          const blocked = decided.length - allowed;
          stage(t, "access", blocked ? `${blocked} restricted` : `${allowed} of ${decided.length} allowed`, blocked ? "locked" : "ok");
          if (allowed) stage(t, "retrieve", "Searching…", "active");
        }
        renderSteps(t);
        break;
      }
      case "cache":
        t.cache = e;
        if (e.status === "HIT") stage(t, "retrieve", "From cache", "ok");
        renderSteps(t);
        break;
      case "trace": t.searches.push(e); renderSteps(t); break;
      case "sources": {
        t.sources = e.sources;
        const failed = t.searches.some((x) => !x.ok);
        if (!(t.cache && t.cache.status === "HIT")) stage(t, "retrieve", `${e.sources.length} passages`, failed ? "warn" : e.sources.length ? "ok" : "warn");
        stage(t, "verify", "Writing…", "active");
        break;
      }
      case "token":
        t.answer += e.text;
        $(".answer", t.el).classList.add("streaming");
        renderAnswer(t);
        break;
      case "verification":
        $(".answer", t.el).classList.remove("streaming");
        applyVerification(t, e);
        break;
      case "error":
        $(".answer", t.el).insertAdjacentHTML("afterend", `<p class="callout error">${esc(e.message)}</p>`);
        $$(".stages li.active, .stages li:not([class])", t.el).forEach((li) => { li.className = "fail"; $(".stage-value", li).textContent = "Stopped"; });
        break;
      case "done":
        t.done = e;
        $(".answer", t.el).classList.remove("streaming");
        if (e.status === "denied") ["retrieve", "verify"].forEach((n) => stage(t, n, "Not run", "skipped"));
        $$(".stages li.active", t.el).forEach((li) => { li.className = "ok"; });
        renderChips(t); renderAnswer(t); renderSteps(t); renderReceipt(t);
        break;
    }
  }

  async function ask(question) {
    if (running) return;
    running = true;
    const el = $("#turn-tpl").content.firstElementChild.cloneNode(true);
    const t = { el, question, answer: "", sources: [], plan: [], searches: [], flags: new Map(), verdicts: new Map(), done: null, cache: null, showAll: false };
    turns.push(t);
    $(".turn-q", el).textContent = question;
    stage(t, "plan", "Planning…", "active");
    $(".steps-toggle", el).addEventListener("click", (ev) => {
      const steps = $(".steps", el), open = steps.hidden;
      steps.hidden = !open; ev.target.setAttribute("aria-expanded", String(open)); ev.target.textContent = open ? "Hide run details" : "Show run details";
    });
    $(".chips", el).addEventListener("click", (ev) => {
      const chip = ev.target.closest(".chip");
      if (!chip) return;
      if (chip.hasAttribute("data-more")) { t.showAll = !t.showAll; renderChips(t); return; }
      openDrawer(t, Number(chip.dataset.n), "");
    });
    $(".ledger-rows", el).addEventListener("click", (ev) => {
      const row = ev.target.closest(".ledger-row[data-n]");
      if (row && row.dataset.n) openDrawer(t, Number(row.dataset.n), row.dataset.s);
    });
    $(".answer", el).addEventListener("click", (ev) => {
      const c = ev.target.closest(".cite");
      if (!c) return;
      ev.preventDefault();
      const s = c.closest(".s");
      openDrawer(t, Number(c.dataset.n), s ? s.dataset.s : "", c);
    });

    home.hidden = true; thread.hidden = false; dock.hidden = false;
    thread.appendChild(el);
    el.scrollIntoView({ behavior: turns.length > 1 ? "smooth" : "auto", block: "start" });
    const url = new URL(location.href);
    url.searchParams.set("q", question);
    if (depth() === "deep") url.searchParams.set("depth", "deep"); else url.searchParams.delete("depth");
    history.replaceState(null, "", url);
    go.disabled = true;
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, depth: depth(), router: $("#router").value }),
      });
      if (!response.ok) throw new Error(`the server answered HTTP ${response.status}`);
      for await (const e of sse(response)) handle(t, e);
      if (!t.done) throw new Error("the stream ended early");
    } catch (err) {
      handle(t, { type: "error", message: `The request didn't finish: ${err.message}. Check that the server is running, then ask again.` });
    } finally {
      running = false; go.disabled = false;
      $(".answer", el).classList.remove("streaming");
      q2.focus({ preventScroll: true });
    }
  }

  // ---- source drawer ----
  function passageHtml(source, sentence) {
    const passage = source.passage || source.snippet || "";
    const claims = claimFigures(sentence || "");
    const marks = figures(passage).filter((f) => claims.some((c) => sameFigure(c, f)));
    let html = "", last = 0;
    for (const f of marks) { html += esc(passage.slice(last, f.index)) + `<mark>${esc(passage.slice(f.index, f.index + f.length))}</mark>`; last = f.index + f.length; }
    html += esc(passage.slice(last));
    if (!marks.length && sentence) {
      const words = new Set(plain(sentence).toLowerCase().match(/[a-z]{4,}/g) || []);
      const parts = passage.split(/(?<=[.!?])\s+/);
      let best = -1, score = 0;
      parts.forEach((p, i) => { const s = (p.toLowerCase().match(/[a-z]{4,}/g) || []).filter((w) => words.has(w)).length; if (s > score) { score = s; best = i; } });
      html = parts.map((p, i) => (i === best && score >= 2 ? `<mark>${esc(p)}</mark>` : esc(p))).join(" ");
    }
    if (passage.includes(" | ")) html = html.split("\n").map((line) => `<span class="row">${line}</span>`).join("");
    return { html, matched: marks.length };
  }
  function openDrawer(t, n, sentence, cite) {
    const source = t.sources.find((s) => s.n === n);
    $$(".cite.on, .chip.on, .ledger-row.on").forEach((c) => c.classList.remove("on"));
    if (cite) cite.classList.add("on");
    if (!source) {
      $("#drawer-title").textContent = `Source ${n} was never retrieved`;
      $("#drawer-loc").textContent = "The answer cites a number that isn't in its source list.";
      $("#drawer-body").innerHTML = "";
      $("#drawer-foot").innerHTML = '<span class="bad">Treat the cited claim as unverified.</span>';
    } else {
      const { html, matched } = passageHtml(source, sentence);
      const verdict = sentence ? t.verdicts.get(sentence) : null;
      $("#drawer-title").textContent = source.title;
      $("#drawer-loc").textContent = `[${source.n}] ${source.locator}`;
      $("#drawer-body").innerHTML = html;
      const state = verdict
        ? verdict.verdict === "verified" ? `<span class="ok">${claimFigures(sentence).length ? "Figure found in this passage" : "Claim supported by this passage"}</span>` : `<span class="bad">${esc(VERDICT[verdict.verdict] || verdict.verdict)}</span>`
        : `<span class="muted">${matched ? "Matching figures highlighted" : "The passage the answer read"}</span>`;
      $("#drawer-foot").innerHTML = `${state}<a href="${esc(source.url)}" ${source.kind === "web" ? 'target="_blank" rel="noopener noreferrer"' : ""}>${source.kind === "web" ? "Open page" : "Open full document"} →</a>`;
      const firstMark = $("#drawer-body mark");
      requestAnimationFrame(() => firstMark && firstMark.scrollIntoView({ block: "center" }));
    }
    drawer.hidden = false;
    document.body.classList.add("drawer-open");
  }
  function closeDrawer() {
    drawer.hidden = true;
    document.body.classList.remove("drawer-open");
    $$(".cite.on").forEach((c) => c.classList.remove("on"));
  }
  $("#drawer-close").addEventListener("click", closeDrawer);

  // ---- deep links: /?q=...&depth=deep asks on load ----
  const params = new URLSearchParams(location.search);
  if (params.get("depth") === "deep") $("#d-deep").checked = true;
  refreshEstimate();
  if (params.get("q")) ask(params.get("q"));
}

// ---- library page ---------------------------------------------------------------------------
function initLibrary() {
  $$(".filters button").forEach((btn) => btn.addEventListener("click", () => {
    $$(".filters button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    const kind = btn.dataset.filter;
    $$("[data-kind]").forEach((row) => { row.hidden = kind !== "all" && row.dataset.kind !== kind; });
  }));

  const form = $("#upload");
  if (!form) return;
  const input = $("#file"), status = $("#upload-status"), chosen = $("#chosen");
  $("#choose").addEventListener("click", () => input.click());
  ["dragenter", "dragover"].forEach((t) => form.addEventListener(t, (e) => { e.preventDefault(); form.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => form.addEventListener(t, (e) => { e.preventDefault(); form.classList.remove("over"); }));
  form.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; input.dispatchEvent(new Event("change")); } });
  input.addEventListener("change", () => { chosen.textContent = input.files[0] ? input.files[0].name : "Drop a PDF, Markdown or text file here"; });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!input.files.length) { status.textContent = "Choose a file first."; return; }
    status.textContent = "Uploading…";
    const response = await fetch("/api/documents", { method: "POST", body: new FormData(form) });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) { status.textContent = body.detail || `Upload failed (HTTP ${response.status}).`; return; }
    status.textContent = "Added. Indexing…";
    for (let i = 0; i < 120; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      const doc = (await (await fetch("/api/documents")).json()).find((d) => d.doc_id === body.doc_id);
      if (!doc) continue;
      status.textContent = doc.status === "indexed" ? "Indexed and searchable." : doc.status === "failed" ? `Couldn't index: ${doc.error}` : `${doc.status[0].toUpperCase()}${doc.status.slice(1)}… ${doc.progress}%`;
      if (doc.status === "indexed" || doc.status === "failed") { setTimeout(() => location.reload(), 900); return; }
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;
  if (page === "index") initResearch();
  if (page === "library") initLibrary();
  document.addEventListener("click", (e) => { $$("details.who[open], details.settings[open]").forEach((d) => { if (!d.contains(e.target)) d.open = false; }); });
});
