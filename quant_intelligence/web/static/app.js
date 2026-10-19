"use strict";
const $ = (s) => document.querySelector(s);
const view = $("#view");
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const chip = (text, tone = "") => `<span class="chip ${tone}">${esc(text)}</span>`;
const num = (x, d = 2) => (x == null ? "n/a" : Number(x).toFixed(d));
const pct = (x, d = 1, signed = true) => (x == null ? "n/a" : `${signed && x > 0 ? "+" : ""}${Number(x).toFixed(d)}%`);
const sgn = (x) => (x == null ? "" : x > 0 ? "pos" : x < 0 ? "neg" : "");
const title = (s) => String(s ?? "").replace(/_/g, " ").replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase());
const day = (iso) => (iso ? new Date(iso).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }) : "n/a");

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}
const errorBox = (e) => `<div class="errbox"><b>Could not load data.</b><br>${esc(e.message || e)}</div>`;
const skeleton = (msg) => `<div class="stack"><div class="skel"></div><div class="skel"></div><div class="note">${esc(msg)}</div></div>`;

/* ---------- shared pieces ---------- */
const STATUS_TONE = { SELECTED: "good", "TIE-BREAK": "warn", ELIGIBLE: "info", REJECTED: "" };
function expCell(v) {
  if (v == null) return `<span class="mono mute">n/a</span>`;
  const f = Math.max(-1, Math.min(v, 1));
  const left = f >= 0 ? 50 : 50 + f * 50;
  return `<div class="exp"><div class="tr"><div class="mid"></div><div class="f" style="left:${left}%;width:${Math.abs(f) * 50}%;background:${v > 0 ? "var(--good)" : "var(--bad)"}"></div></div><span class="val ${sgn(v)}">${v > 0 ? "+" : ""}${v.toFixed(2)}R</span></div>`;
}
function rankingTable(rows, { why = false, click = false, selected = null } = {}) {
  const body = rows.map((r) => `<tr class="${r.Status === "SELECTED" || r.Status === "TIE-BREAK" ? "sel" : ""} ${click ? "click" : ""}" data-s="${esc(r.Strategy)}">
    <td><b>${esc(r.Strategy)}</b></td><td>${chip(r.Status, STATUS_TONE[r.Status])}</td>
    <td>${expCell(r["Cond. expectancy (R, net of costs)"])}</td>
    <td class="mono">${esc(r["Win rate (historical)"] ?? "n/a")}</td><td class="mono">${esc(r["Analogue sample"])}</td>
    <td>${esc(title(r.Confidence))}</td><td class="mono">${r["Max drawdown (R)"] == null ? "n/a" : r["Max drawdown (R)"].toFixed(1) + "R"}</td>
    ${why ? `<td class="why">${esc(r.Why)}</td>` : ""}</tr>`).join("");
  const head = ["Strategy", "Status", "Net expectancy", "Win rate", "Sample", "Confidence", "Max DD"].concat(why ? ["Why"] : []);
  return `<div class="tw"><table class="t"><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>
  <div class="note">Historical, cost-adjusted conditional statistics from similar past market states. Not a prediction. Small samples: read the confidence column.</div>`;
}
const chartLayout = (h, yfmt) => ({ height: h, margin: { l: 56, r: 10, t: 6, b: 28 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#6b7280", size: 11 }, hovermode: "x unified", showlegend: true, legend: { orientation: "h", y: 1.12 },
  xaxis: { showgrid: false, linecolor: "#e5e7eb" }, yaxis: { gridcolor: "#eef0f3", zeroline: false, tickformat: yfmt } });
const plot = (id, data, layout) => Plotly.newPlot(id, data, layout, { displayModeBar: false, responsive: true });

/* ---------- nav chips ---------- */
let lastOverview = null;
function navChips(o) {
  const chips = [];
  if (o) {
    chips.push(chip("Data " + o.data_quality, o.data_quality === "OK" ? "good" : "bad"));
    if (o.provenance) chips.push(chip(o.provenance.mode, o.provenance.mode === "LIVE" ? "good" : "info"));
    chips.push(chip(o.mode.label, "info"));
  }
  $("#navchips").innerHTML = chips.join("");
}

/* ---------- dashboard ---------- */
async function dashboard() {
  view.innerHTML = skeleton("Running the research pipeline on current market data. This can take up to 20 seconds the first time.");
  let o;
  try { o = await api("/api/overview"); } catch (e) { view.innerHTML = errorBox(e); return; }
  lastOverview = o; navChips(o);
  const age = o.age_minutes;
  const ageText = age == null ? "age n/a" : age > 120 ? `${(age / 60).toFixed(1)} h old` : `${Math.max(age, 0).toFixed(0)} min old`;
  const fresh = age != null && age <= o.max_order_age_minutes;
  const d = o.decision;
  const cls = d.no_trade ? "" : d.tie_break ? "tie" : "go";
  const probs = Object.entries(o.regime.probabilities || {}).sort((a, b) => b[1] - a[1]).slice(0, 6);
  view.innerHTML = `
  <div class="hero"><div><h1>Quant Strategy Intelligence</h1>
    <div class="sub"><b>${esc(o.instrument)}</b> &middot; last bar ${esc(new Date(o.last_bar).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" }))} (${esc(ageText)}) &middot; research-first terminal for Indian index derivatives</div></div>
  </div>
  <div class="grid4">
    <div class="card kpi"><div class="l">Market regime</div><div class="v">${esc(title(o.regime.label))}</div><div class="s">top score share ${num(o.regime.confidence * 100, 0)}% (heuristic, not a probability)</div></div>
    <div class="card kpi"><div class="l">Data quality</div><div class="v ${o.data_quality === "OK" ? "pos" : "neg"}">${esc(o.data_quality)}</div><div class="s">${o.data_quality_issues.length} issue(s)</div></div>
    <div class="card kpi"><div class="l">Data age</div><div class="v ${fresh ? "pos" : "neg"}">${age == null ? "n/a" : age > 120 ? (age / 60).toFixed(1) + " h" : Math.max(age, 0).toFixed(0) + " min"}</div><div class="s">orders need data under ${o.max_order_age_minutes} min old</div></div>
    <div class="card kpi"><div class="l">Eligible now</div><div class="v">${o.counts.eligible} / ${o.counts.evaluated}</div><div class="s">strategies clearing edge and confidence bars</div></div>
  </div>
  ${o.provenance && o.provenance.mode !== "LIVE" ? `<div class="note" style="margin:-6px 0 12px"><b>${esc(o.provenance.mode)}</b>: ${esc(o.provenance.mode_meaning)} Source: ${esc(o.provenance.source)}.</div>` : ""}
  ${o.data_quality_issues.map((i) => `<div class="warnbox">${esc(i)}</div>`).join("")}
  <div class="card banner ${cls}"><div class="k">Decision${d.tie_break ? " - tie-break" : ""}</div>
    <div class="t">${d.no_trade ? "NO TRADE" : esc(d.strategy)}</div><div class="r">${esc(d.reason)}</div></div>
  <div class="note" style="margin:-6px 0 16px">A decision is not an order: it still has to pass the independent risk engine and a real setup must trigger.</div>
  <div class="cols"><div><h2>Strategy ranking</h2>${rankingTable(o.strategies, { why: false, click: true })}</div>
    <div class="card"><h2>Regime scores</h2>${probs.map(([n, p], i) => `<div class="prob ${i === 0 ? "top" : ""}"><span>${esc(title(n))}</span><div class="bar"><div style="width:${(p * 100).toFixed(1)}%"></div></div><span class="n">${(p * 100).toFixed(0)}%</span></div>`).join("")}
    <div class="note">Normalised rule-based scores, not calibrated probabilities. Regimes overlap, and the regime is not used by the ranking.</div></div></div>`;
  view.querySelectorAll("tr.click").forEach((tr) => tr.addEventListener("click", () => { sessionStorage.setItem("strategy", tr.dataset.s); location.hash = "#/strategies"; }));
}

/* ---------- strategies ---------- */
async function strategies() {
  view.innerHTML = skeleton("Loading strategies...");
  let o;
  try { o = lastOverview || (lastOverview = await api("/api/overview")); } catch (e) { view.innerHTML = errorBox(e); return; }
  navChips(o);
  let sel = sessionStorage.getItem("strategy") || o.strategies[0].Strategy;
  let q = "";
  view.innerHTML = `<div class="hero"><div><h1>Strategies</h1><div class="sub">Why each of the ${o.strategies.length} strategies is selected, eligible or rejected right now.</div></div></div>
  <div class="wb"><div><input type="search" id="q" placeholder="Search strategies..."><div class="list" id="slist" style="margin-top:10px"></div></div><div id="sdetail"></div></div>`;
  const paintList = () => {
    $("#slist").innerHTML = o.strategies.filter((r) => r.Strategy.toLowerCase().includes(q)).map((r) => `<div class="run ${r.Strategy === sel ? "sel" : ""}" data-s="${esc(r.Strategy)}">
      <div class="top"><span>${esc(r.Strategy)}</span>${chip(r.Status, STATUS_TONE[r.Status])}</div>
      <div class="meta"><span>E: ${r["Cond. expectancy (R, net of costs)"] == null ? "n/a" : r["Cond. expectancy (R, net of costs)"].toFixed(2) + "R"}</span><span>n=${esc(r["Analogue sample"])}</span></div></div>`).join("");
    $("#slist").querySelectorAll(".run").forEach((el) => el.addEventListener("click", () => { sel = el.dataset.s; sessionStorage.setItem("strategy", sel); paintList(); paintDetail(); }));
  };
  const kv = (obj) => Object.entries(obj || {}).filter(([, v]) => v == null || typeof v !== "object").map(([k, v]) => `<span>${esc(title(k))}</span><span>${v == null ? "n/a" : typeof v === "number" ? Number(v.toFixed(4)) : esc(v)}</span>`).join("");
  const paintDetail = async () => {
    const box = $("#sdetail");
    box.innerHTML = skeleton("Loading...");
    try {
      const s = await api(`/api/strategies/${encodeURIComponent(sel)}`);
      const r = s.ranking || {};
      box.innerHTML = `<div class="stack"><div class="card"><div class="head2"><div><h2 style="font-size:1.5rem">${esc(s.name)}</h2><div class="mute">${esc(r.Why || "")}</div></div>${chip(r.Status || "", STATUS_TONE[r.Status])}</div>
        <div class="kpis6"><div><div class="l">Net expectancy</div><div class="v ${sgn(r["Cond. expectancy (R, net of costs)"])}">${r["Cond. expectancy (R, net of costs)"] == null ? "n/a" : num(r["Cond. expectancy (R, net of costs)"], 3) + "R"}</div></div>
        <div><div class="l">Std. error</div><div class="v">${r["Std. error (R)"] == null ? "n/a" : num(r["Std. error (R)"], 3) + "R"}</div></div>
        <div><div class="l">Win rate</div><div class="v">${esc(r["Win rate (historical)"] ?? "n/a")}</div></div>
        <div><div class="l">Analogue sample</div><div class="v">${esc(r["Analogue sample"] ?? "n/a")}</div></div>
        <div><div class="l">Distinct days</div><div class="v">${esc(r["Distinct days"] ?? "n/a")}</div></div>
        <div><div class="l">Max drawdown</div><div class="v">${r["Max drawdown (R)"] == null ? "n/a" : num(r["Max drawdown (R)"], 1) + "R"}</div></div></div>
        <div class="note">${esc(s.note)}</div></div>
        <div class="cols" style="grid-template-columns:1fr 1fr"><div class="card"><h2>Conditional (similar past states, net of costs)</h2><div class="kv">${kv(s.conditional_metrics) || "<span>No data</span><span></span>"}</div></div>
        <div class="card"><h2>Global backtest (unconditional)</h2><div class="kv">${kv(s.global_metrics) || "<span>No data</span><span></span>"}</div></div></div>
        <div class="card"><h2>Score components</h2><div class="kv">${kv(s.score_components) || "<span>Not scored (insufficient history)</span><span></span>"}</div></div></div>`;
    } catch (e) { box.innerHTML = errorBox(e); }
  };
  $("#q").addEventListener("input", (e) => { q = e.target.value.toLowerCase(); paintList(); });
  paintList(); paintDetail();
}

/* ---------- backtests (workbench) ---------- */
async function backtests() {
  view.innerHTML = skeleton("Loading backtest runs...");
  let data;
  try { data = await api("/api/backtests"); } catch (e) { view.innerHTML = errorBox(e); return; }
  let runs = data.runs, sel = runs[0]?.run_id, q = "", f = "All";
  view.innerHTML = `<div class="wb"><div><h2 style="font-size:1.3rem;margin-bottom:2px">Backtest Workbench</h2><div class="mute" id="cnt"></div>
    <input type="search" id="q" placeholder="Search runs, strategy, instrument..." style="margin-top:10px">
    <div class="seg" id="seg">${["All", "Win", "Loss"].map((x) => `<button data-f="${x}" class="${x === "All" ? "on" : ""}">${x}</button>`).join("")}</div>
    <div class="list" id="rlist"></div></div><div id="rdetail"></div></div>`;
  const shown = () => runs.filter((r) => (`${r.strategy} ${r.instrument} ${r.split} ${r.run_id}`).toLowerCase().includes(q) &&
    (f === "All" || (f === "Win" ? (r.return_pct || 0) > 0 : (r.return_pct || 0) < 0)));
  const paintList = () => {
    const s = shown();
    $("#cnt").textContent = `${runs.length} runs total`;
    $("#rlist").innerHTML = s.length ? s.map((r) => `<div class="run ${r.run_id === sel ? "sel" : ""}" data-id="${esc(r.run_id)}">
      <div class="top"><span>${esc(r.strategy)}</span><span class="mono ${sgn(r.return_pct)}">${pct(r.return_pct)}</span></div>
      <div class="mute" style="font-size:.78rem">${esc(r.instrument)} &middot; ${esc(title(r.split))}</div>
      <div class="meta"><span>DD: ${pct(r.max_dd_pct)}</span><span>WR: ${r.win_rate == null ? "n/a" : (r.win_rate * 100).toFixed(0) + "%"}</span><span>${r.n_trades} trades</span></div>
      <div class="mute" style="font-size:.74rem;margin-top:5px">${day(r.start)} &rarr; ${day(r.end)}</div></div>`).join("") : `<div class="note">No runs match.</div>`;
    $("#rlist").querySelectorAll(".run").forEach((el) => el.addEventListener("click", () => { sel = el.dataset.id; paintList(); paintDetail(); }));
  };
  const paintDetail = async () => {
    const box = $("#rdetail");
    if (!sel) { box.innerHTML = `<div class="card">No backtest runs stored yet.</div>`; return; }
    box.innerHTML = skeleton("Loading run...");
    try {
      const d = await api(`/api/backtests/${encodeURIComponent(sel)}`);
      box.innerHTML = `<div class="stack"><div class="card"><div class="head2"><div><h2 style="font-size:1.5rem;margin:0">${esc(d.strategy)} ${chip("completed")}</h2>
        <div class="mute">${esc(d.instrument)} &middot; ${esc(title(d.split))} &middot; ${day(d.start)} &rarr; ${day(d.end)}</div></div><div class="big ${sgn(d.return_pct)}">${pct(d.return_pct)}</div></div>
        <div class="kpis6"><div><div class="l">Max DD</div><div class="v neg">${pct(d.max_dd_pct)}</div></div>
        <div><div class="l">Sharpe (daily)</div><div class="v ${sgn(d.sharpe_daily)}">${d.sharpe_daily == null ? "n/a" : num(d.sharpe_daily)}</div></div>
        <div><div class="l">Win rate</div><div class="v">${d.win_rate == null ? "n/a" : (d.win_rate * 100).toFixed(1) + "%"}</div></div>
        <div><div class="l">Trades</div><div class="v">${d.n_trades}</div></div>
        <div><div class="l">Initial</div><div class="v">Rs ${Math.round(d.initial).toLocaleString("en-IN")}</div></div>
        <div><div class="l">Final</div><div class="v ${sgn((d.final ?? d.initial) - d.initial)}">${d.final == null ? "n/a" : "Rs " + Math.round(d.final).toLocaleString("en-IN")}</div></div></div>
        <div class="note">${d.notes.map(esc).join(" ")}</div></div>
        <div class="card"><h2>Equity curve</h2><div class="mute" style="font-size:.82rem">Account value after each closed trade${d.benchmark ? " vs buy-and-hold" : " (no cached prices cover this run, so no benchmark is shown)"}.</div><div id="c1" class="chart"></div></div>
        <div class="card"><h2>Drawdown</h2><div id="c2" class="chart"></div></div>
        <div class="card"><h2>Daily returns</h2><div id="c3" class="chart"></div></div></div>`;
      const traces = [{ x: d.equity.x, y: d.equity.y, name: "Strategy", mode: "lines", line: { color: "#111827", width: 2 }, fill: "tozeroy", fillcolor: "rgba(17,24,39,.05)" }];
      if (d.benchmark) traces.push({ x: d.benchmark.x, y: d.benchmark.y, name: "Benchmark (buy & hold)", mode: "lines", line: { color: "#9ca3af", width: 1.5, dash: "dash" } });
      const ys = traces.flatMap((t) => t.y), lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.15 || d.initial * 0.01;
      const l1 = chartLayout(300, ",.0f"); l1.yaxis.range = [lo - pad, hi + pad];
      plot("c1", traces, l1);
      plot("c2", [{ x: d.drawdown.x, y: d.drawdown.y, name: "Drawdown", mode: "lines", line: { color: "#dc2626", width: 1.5 }, fill: "tozeroy", fillcolor: "rgba(220,38,38,.12)" }], Object.assign(chartLayout(220, ".1f"), { showlegend: false }));
      plot("c3", [{ x: d.daily.x, y: d.daily.y, type: "bar", name: "Daily return", marker: { color: d.daily.y.map((v) => (v >= 0 ? "#059669" : "#dc2626")) } }], Object.assign(chartLayout(220, ".1f"), { showlegend: false }));
    } catch (e) { box.innerHTML = errorBox(e); }
  };
  $("#q").addEventListener("input", (e) => { q = e.target.value.toLowerCase(); paintList(); });
  $("#seg").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { f = b.dataset.f; $("#seg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); paintList(); }));
  paintList(); paintDetail();
}

/* ---------- risk ---------- */
async function risk() {
  view.innerHTML = skeleton("Loading risk engine...");
  let r;
  try { r = await api("/api/risk"); } catch (e) { view.innerHTML = errorBox(e); return; }
  const ev = r.recent_events.map((e) => `<tr><td class="mono">${esc(new Date(e.time).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" }))}</td>
    <td>${chip(e.type, e.type === "VETO" ? "bad" : "good")}</td><td>${esc(e.reason)}</td><td class="mono mute">${esc(e.decision_id)}</td></tr>`).join("");
  view.innerHTML = `<div class="hero"><div><h1>Independent Risk Engine</h1><div class="sub">${esc(r.principle)}</div></div><div>${chip(r.mode.label, "info")}</div></div>
  <h2>Mandatory checks</h2><div class="tw"><table class="t"><thead><tr><th>Check</th><th>Rule</th><th></th></tr></thead><tbody>
  ${r.checks.map((c) => `<tr><td><b>${esc(c.name)}</b></td><td>${esc(c.rule)}</td><td>${chip("MANDATORY", "info")}</td></tr>`).join("")}</tbody></table></div>
  <h2 style="margin-top:22px">Recent verdicts</h2>
  ${ev ? `<div class="tw"><table class="t"><thead><tr><th>Time</th><th>Verdict</th><th>Reason</th><th>Decision id</th></tr></thead><tbody>${ev}</tbody></table></div>` : `<div class="card mute">No risk verdicts recorded yet.</div>`}
  <div class="note">This website is read-only: it has no way to place, change or cancel an order.</div>`;
}

/* ---------- method ---------- */
function method() {
  view.innerHTML = `<div class="hero"><div><h1>Method and honesty notes</h1><div class="sub">What this system does, and what it does not claim.</div></div></div>
  <div class="grid4" style="grid-template-columns:repeat(auto-fit,minmax(320px,1fr))">
  <div class="card"><h2>What the ranking means</h2><p>Each strategy is backtested, then scored on how it behaved in the most similar past market states (nearest-neighbour analogues), net of modelled costs. The score uses a shrunk, lower-bound estimate of the edge and is penalised for drawdown and thin samples.</p></div>
  <div class="card"><h2>What it does not claim</h2><p>It does not predict price. The walk-forward evaluation has not shown the selector beating a random pick, and a tie-break selection carries no evidence of an edge. Treat every number as historical.</p></div>
  <div class="card"><h2>NO TRADE is an answer</h2><p>When data is stale or degraded, nothing clears the edge and confidence bars, or the top strategies are statistically tied, the system says NO TRADE and shows why.</p></div>
  <div class="card"><h2>Independent risk engine</h2><p>Risk checks run separately from the ranker and can veto any trade. A failed mandatory check always means NO TRADE.</p></div>
  <div class="card"><h2>Known limits</h2><p>Option premiums in backtests are modelled, not real option history. Option-chain strategies (PCR, OI, IV, unusual volume) have no historical setups and cannot be ranked. Regime probabilities are uncalibrated.</p></div>
  <div class="card"><h2>Safety</h2><p>Paper trading only. This website is read-only and has no order endpoints.</p></div></div>`;
}

/* ---------- router ---------- */
const routes = { dashboard, strategies, backtests, risk, method };
function route() {
  const name = (location.hash.replace("#/", "") || "dashboard").split("?")[0];
  const fn = routes[name] || dashboard;
  document.querySelectorAll("#links a").forEach((a) => a.classList.toggle("on", a.dataset.r === (routes[name] ? name : "dashboard")));
  window.scrollTo(0, 0);
  fn();
}
window.addEventListener("hashchange", route);
route();
