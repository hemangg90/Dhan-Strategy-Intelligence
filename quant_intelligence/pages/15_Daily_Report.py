"""Daily Report - how the system worked each day and which strategies are earning.

Built automatically after each market close (NSE ~15:40, MCX just after midnight) from the stored trades, the
scan decision log, risk events and system events, and kept permanently when DATABASE_URL points at a durable
database. Pick any past day; download Excel / Markdown / JSON. Strategy verdicts are worded only as strongly as the
sample size allows: with under 30 closed trades no strategy is called good or bad.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from quant_intelligence.database.db import is_durable
from quant_intelligence.reports import eod, export
from quant_intelligence.reports.performance import MIN_TRADES
from quant_intelligence.ui import format as F
from quant_intelligence.ui.charts import equity_curve_chart, pnl_bar_chart, r_ci_chart
from quant_intelligence.ui.components import empty_state, kpi_row, page_header
from quant_intelligence.ui.state import init_session_state
from quant_intelligence.ui.theme import apply_theme
from quant_intelligence.utils.timeutil import now_ist

apply_theme()
init_session_state()
page_header(
    "Daily Report",
    "How the system worked each day and which strategies are earning - generated after every market close, kept day by day.",
)

if not is_durable():
    st.warning(
        "Reports and trade history are stored in a local SQLite file, which Streamlit Cloud erases on every reboot or sleep. "
        "Set DATABASE_URL (a free Neon/Supabase Postgres) in the app's Secrets to keep every day - see the README.",
        icon="⚠️",
    )

today = now_ist().date()
c1, c2, c3 = st.columns([1.2, 1.4, 1.6])
scope = c1.segmented_control("Market", list(eod.SCOPES), default="ALL", key="rep_scope") or "ALL"
stored = eod.list_reports(scope)
dates = sorted({r["date"] for r in stored} | {today}, reverse=True)
chosen = c2.selectbox("Report date", dates, format_func=lambda d: f"{d:%a %d %b %Y}" + (" (today)" if d == today else ""), key="rep_date")
c3.write("")
if c3.button("Generate / refresh this report now", key="rep_gen"):
    with st.spinner("Building the report..."):
        eod.generate_and_save(chosen, scope)
    st.rerun()

record = eod.load_report(chosen, scope)
_older_version = record is not None and not eod.is_complete(record["content_json"])
if _older_version:
    # A report stored by an older version lacks fields this page needs: rebuild it from the trade records instead of crashing.
    st.warning("The stored report for this day was saved by an older version and is incomplete, so it is rebuilt below from "
               "the trade records. Press 'Generate / refresh this report now' to replace the stored copy.")
    record = None
preview = record is None
if preview:
    with st.spinner("Building a preview..."):
        built = eod.build_report(chosen, scope)
    rep, markdown = built["content_json"], built["content_markdown"]
    st.info("Preview - this day's report has not been stored yet. It is created automatically after the close, or press the button above.")
else:
    rep, markdown = record["content_json"], record["content_markdown"]
    st.caption(f"Stored report, generated {record['generated_at']:%d %b %Y %H:%M} UTC.")

s = rep["summary"]
t, cum = s["today"], s["cumulative"]
prev = s.get("previous")
kpi_row([
    {"label": "Trades closed", "value": str(t["trades"]), "delta": f"{t['wins']}W / {t['losses']}L" if t["trades"] else "none"},
    {"label": "Net P&L", "value": F.inr(t["net"], signed=True), "tone": F.tone(t["net"]),
     "help": "After brokerage, statutory charges and GST",
     "delta": (f"prev {F.inr(prev['net'], signed=True)}" if prev else None)},
    {"label": "Win rate", "value": f"{t['win_rate']:.0f}%" if t["win_rate"] is not None else "–",
     "help": (f"95% interval {t['win_ci_low']:.0f}-{t['win_ci_high']:.0f}%" if t["trades"] else None)},
    {"label": "Profit factor", "value": ("∞" if t["profit_factor"] == float("inf") else f"{t['profit_factor']:.2f}") if t["profit_factor"] is not None else "–"},
    {"label": "Cumulative net", "value": F.inr(cum["net"], signed=True), "tone": F.tone(cum["net"]), "delta": f"{cum['trades']} trades"},
    {"label": "Open positions", "value": str(s["open_positions"])},
])

d1, d2, d3, d4, _ = st.columns([1, 1, 1, 1.4, 2])
d1.download_button("Excel", export.to_excel(rep), file_name=f"daily_report_{rep['date']}_{scope}.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
d2.download_button("Markdown", markdown.encode("utf-8"), file_name=f"daily_report_{rep['date']}_{scope}.md", mime="text/markdown", key="dl_md")
d3.download_button("JSON", export.to_json(rep), file_name=f"daily_report_{rep['date']}_{scope}.json", mime="application/json", key="dl_json")
d4.download_button("Printable (HTML)", export.markdown_to_html(markdown, f"Daily report {rep['date']} {scope}").encode("utf-8"),
                   file_name=f"daily_report_{rep['date']}_{scope}.html", mime="text/html", key="dl_html",
                   help="Open in a browser and print to PDF.")
st.caption("Trade results in this report are " + (rep.get("labels") or {}).get("trades", "SIMULATED (paper trading)")
           + "; expected R is " + (rep.get("labels") or {}).get("expected_r", "HISTORICAL") + ".")

tab_sum, tab_str, tab_sig, tab_rej, tab_brk, tab_exp, tab_health, tab_days = st.tabs(
    ["How things went", "Strategies", "Signals & risk", "Not traded & data", "Breakdowns", "Expected vs realised", "System health", "All days"]
)

with tab_sum:
    for line in rep["narrative"]:
        st.markdown(f"- {line}")
    with st.expander("How to read this report"):
        for c in rep["caveats"]:
            st.markdown(f"- {c}")
    with st.expander("Methodology"):
        for m in rep.get("methodology") or ["This stored report predates the methodology section; press 'Generate / refresh' to add it."]:
            st.markdown(f"- {m}")


def _strategy_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _table(container, rows: list[dict]) -> None:
    """A compact table, or the word 'none' (a plain statement: Streamlit would print the value of a conditional expression)."""
    if rows:
        container.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        container.caption("none")


with tab_str:
    st.markdown(f"**{rep['leader']['text']}**")
    view = st.radio("Period", ["All time", "Last 30 days", "Today"], horizontal=True, key="rep_period", label_visibility="collapsed")
    key = {"All time": "all_time", "Last 30 days": next((k for k in rep["strategies"] if k.startswith("last_")), "all_time"), "Today": "today"}[view]
    df = _strategy_frame(rep["strategies"][key])
    if df.empty:
        empty_state("No closed trades in this period", "Strategies appear here once their trades close.")
    else:
        st.plotly_chart(r_ci_chart(df), width="stretch", config={"displayModeBar": False})
        st.caption(f"Hollow marker = fewer than {MIN_TRADES} trades (too few to conclude). A whisker that crosses zero means "
                   "the result cannot be told apart from no edge.")
        df["win_range"] = [f"{lo:.0f}-{hi:.0f}%" if lo == lo and hi == hi else "" for lo, hi in zip(df["win_ci_low"], df["win_ci_high"])]
        st.dataframe(
            df[["strategy", "trades", "verdict_text", "net_pnl", "expectancy", "avg_r", "t_stat", "win_rate", "win_range",
                "profit_factor", "payoff", "max_drawdown"]],
            width="stretch", hide_index=True,
            column_config={
                "strategy": "Strategy", "trades": "Trades", "verdict_text": st.column_config.TextColumn("What the evidence supports", width="large"),
                "net_pnl": st.column_config.NumberColumn("Net P&L (₹)", format="%.0f"),
                "expectancy": st.column_config.NumberColumn("Per trade (₹)", format="%.0f"),
                "avg_r": st.column_config.NumberColumn("Mean R", format="%+.2f"),
                "t_stat": st.column_config.NumberColumn("t-stat", format="%.2f"),
                "win_rate": st.column_config.NumberColumn("Win rate", format="%.0f%%"),
                "win_range": "Win 95% range",
                "profit_factor": st.column_config.NumberColumn("Profit factor", format="%.2f"),
                "payoff": st.column_config.NumberColumn("Payoff", format="%.2f"),
                "max_drawdown": st.column_config.NumberColumn("Max drawdown (₹)", format="%.0f"),
            },
        )

with tab_sig:
    f = rep["funnel"]
    kpi_row([
        {"label": "Bars evaluated", "value": str(f["bars_evaluated"])}, {"label": "Setups triggered", "value": str(f["signals"])},
        {"label": "Filled", "value": str(f["filled"])}, {"label": "Stopped by risk engine", "value": str(f["vetoed"])},
        {"label": "Rejected by broker", "value": str(f["rejected"])}, {"label": "Tie-break decisions", "value": str(f["tie_break"])},
    ])
    if not f["bars_evaluated"]:
        empty_state("No scan decisions recorded for this day", "The auto-traders were not running, or the market was closed.")
    a, b = st.columns(2)
    with a:
        st.markdown("**Why the system stood aside**")
        _table(st, f["no_trade_reasons"])
        st.markdown("**Risk-engine vetoes (reasons)**")
        _table(st, rep["risk"]["vetoes"])
    with b:
        st.markdown("**Signals by strategy**")
        _table(st, f["by_strategy"])
        st.markdown("**Rejected orders (reasons)**")
        _table(st, rep["orders"]["reject_reasons"])

with tab_rej:
    st.markdown("**Setups that did not become trades**")
    rej = rep.get("rejected_setups")
    if rej is None:
        st.caption("This stored report predates this section; press 'Generate / refresh this report now'.")
    elif not rej:
        st.caption("none")
    else:
        st.dataframe(pd.DataFrame(rej), width="stretch", hide_index=True)
    st.markdown("**Data quality findings while the data was loaded**")
    dq = rep.get("data_quality")
    if dq is None:
        st.caption("This stored report predates this section.")
    else:
        st.caption(f"{dq['loads']} data loads recorded, {dq['not_ok']} not OK. A bar that fails the gate produces NO TRADE.")
        _table(st, dq["items"])

with tab_brk:
    dims = {"Instrument": "instrument", "Market": "market", "Calls vs puts": "option_type", "Exit reason": "exit_reason",
            "Entry hour": "hour", "Weekday": "weekday", "TIE-BREAK vs normal": "tag_label"}
    pick = st.segmented_control("Split by", list(dims), default="Instrument", key="rep_dim") or "Instrument"
    bdf = pd.DataFrame(rep["breakdowns"][dims[pick]])
    if bdf.empty:
        empty_state("No closed trades yet")
    else:
        bdf[dims[pick]] = bdf[dims[pick]].astype(str)
        st.plotly_chart(pnl_bar_chart(bdf, dims[pick]), width="stretch", config={"displayModeBar": False})
        st.dataframe(bdf, width="stretch", hide_index=True, column_config={
            "trades": "Trades", "win_rate": st.column_config.NumberColumn("Win rate", format="%.0f%%"),
            "net_pnl": st.column_config.NumberColumn("Net P&L (₹)", format="%.0f"),
            "expectancy": st.column_config.NumberColumn("Per trade (₹)", format="%.0f"),
            "avg_r": st.column_config.NumberColumn("Mean R", format="%+.2f")})

with tab_exp:
    edf = pd.DataFrame(rep["expected_vs_realised"])
    st.caption("The ranker picks a strategy because it expects a positive R. This compares that expectation at entry with what the "
               "trades actually delivered. A large negative gap means the backtest is overstating the strategy.")
    if edf.empty:
        empty_state("No trades with a recorded expectation yet", "New trades store the ranker's expected R at entry; older ones do not.")
    else:
        st.dataframe(edf, width="stretch", hide_index=True, column_config={
            "strategy": "Strategy", "trades": "Trades", "expected_r": st.column_config.NumberColumn("Expected R", format="%+.2f"),
            "realised_r": st.column_config.NumberColumn("Realised R", format="%+.2f"),
            "gap": st.column_config.NumberColumn("Gap", format="%+.2f"), "gap_se": st.column_config.NumberColumn("Gap std err", format="%.2f"),
            "gap_t": st.column_config.NumberColumn("t-stat", format="%.2f")})

with tab_health:
    h = rep["health"]
    kpi_row([
        {"label": "Dhan token", "value": h["token"]["state"].upper(),
         "delta": (f"expires {h['token']['expires_at'][:16].replace('T', ' ')}" if h["token"]["expires_at"] else None)},
        {"label": "Log errors", "value": str(h["errors"])}, {"label": "Log warnings", "value": str(h["warnings"])},
        {"label": "Orders", "value": str(rep["orders"]["total"]), "delta": f"{rep['orders']['rejected']} rejected"},
        {"label": "Unresolved positions", "value": str(h["unresolved"]["stale"] + h["unresolved"]["open"]),
         "help": "Left open when the app stopped: no exit, no result"},
    ])
    a, b = st.columns(2)
    a.markdown("**Warnings/errors by component**")
    _table(a, h["by_component"])
    b.markdown("**Most frequent messages**")
    _table(b, h["top_messages"])

with tab_days:
    days = pd.DataFrame(stored)
    if days.empty:
        empty_state("No stored reports yet", "One is stored after each market close.")
    else:
        curve = days.sort_values("date").assign(time=lambda d: d["date"], trade=lambda d: d["date"].astype(str), pnl=lambda d: d["net_pnl"],
                                                equity=lambda d: d["net_pnl"].cumsum())
        st.markdown("**Cumulative net P&L by day**")
        st.plotly_chart(equity_curve_chart(curve[["time", "trade", "pnl", "equity"]], by_time=True), width="stretch", config={"displayModeBar": False})
        st.dataframe(days[["date", "trades", "net_pnl", "win_rate", "cumulative"]], width="stretch", hide_index=True, column_config={
            "date": "Date", "trades": "Trades", "net_pnl": st.column_config.NumberColumn("Net P&L (₹)", format="%.0f"),
            "win_rate": st.column_config.NumberColumn("Win rate", format="%.0f%%"),
            "cumulative": st.column_config.NumberColumn("Cumulative (₹, as of that day)", format="%.0f")})
