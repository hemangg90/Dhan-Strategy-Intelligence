"""Streamlit panel: scan a watchlist and paper-trade it, all as options.

Used for the NSE index+stock watchlist (`ENGINE`) and for MCX commodities (`COMMODITY_RUNNER`).
Auto trading runs in a background thread owned by the runner (execution/engine.py), so it keeps running
when you switch pages or close the tab, until you press Stop. NSE and MCX are separate paper accounts, each
with its own capital, risk limits, drawdown and kill switch.

Layout: control bar -> live account/market tiles -> market watch (price, change, day range, sparkline,
scan result per instrument) -> tabs for positions, scan results, activity feed and today's performance.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.config.watchlist import WATCHLIST_COMMODITIES, WATCHLIST_STOCKS
from quant_intelligence.data.data_keeper import DATA_KEEPER
from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import Order, Position, RiskEvent
from quant_intelligence.execution.capital import capital_summary
from quant_intelligence.execution.engine import ENGINE, ScanRunner
from quant_intelligence.ui import format as F
from quant_intelligence.ui.charts import equity_curve_chart, pnl_bar_chart
from quant_intelligence.ui.components import chip, empty_state, kpi_row
from quant_intelligence.ui.market_data import load_snapshot
from quant_intelligence.ui.open_positions import live_open, render_open_positions
from quant_intelligence.ui.position_views import (
    activity_frame, equity_curve, orders_frame, pnl_by, positions_frame, trade_stats,
)
from quant_intelligence.ui.tables import trades_table
from quant_intelligence.utils.market_calendar import session_status
from quant_intelligence.utils.timeutil import now_ist

_CFG = {"displayModeBar": False}
_STATUS_LABEL = {
    "NO_TRADE": "– No trade", "WAITING_FOR_SETUP": "◔ Waiting for setup", "SETUP_TRIGGERED": "▲ Setup triggered",
    "NO_SELECTION": "– Nothing selected", "CHAIN_ERROR": "✖ Chain error", "DATA_ERROR": "✖ Data error", "MONITOR_ERROR": "✖ Monitor error",
}


def _universe(commodities: bool) -> tuple[list[str], list[str], list[dict]]:
    """(index symbols, stock/commodity symbols, one metadata dict per instrument for the market watch)."""
    if commodities:
        symbols = [r["symbol"] for r in WATCHLIST_COMMODITIES]
        meta = [{"symbol": r["symbol"], "name": r["name"], "group": r["sector"], "lot": r["lot_size"]} for r in WATCHLIST_COMMODITIES]
        return [], symbols, meta
    indices = list(SETTINGS.option_underlyings)
    meta = [{"symbol": s, "name": s, "group": "Index", "lot": None} for s in indices]
    meta += [{"symbol": r["symbol"], "name": r["name"], "group": r["sector"], "lot": None} for r in WATCHLIST_STOCKS]
    return indices, [r["symbol"] for r in WATCHLIST_STOCKS], meta


# ---------------------------------------------------------------------------------------------- live header
@st.fragment(run_every="5s")
def _live_header(runner: ScanRunner, market: str, symbols: list[str]) -> None:
    """Runner status line and the account / market tiles. Display only: leaving the page never affects the worker."""
    status = runner.status()
    feed = DATA_KEEPER.status()
    stale_here = [s for s in feed["not_ok"] if s in symbols]
    if feed["paused"]:
        st.warning(f"Candles are not being refreshed: {feed['error']}")
        stale_here = []  # the banner above already says why; do not repeat it per instrument
    if stale_here:
        issues = [i for sym in stale_here for i in DATA_KEEPER.instruments[sym]["issues"]]
        reason = next((i for i in issues if "refresh failed" in i), issues[0] if issues else None)
        # Say it plainly instead of leaving a silent old answer: what is stale, why, and that it is being retried.
        st.info(
            f"Candles for {len(stale_here)} of {len(symbols)} {market} instruments are not current "
            f"({', '.join(stale_here[:4])}{'...' if len(stale_here) > 4 else ''}). The data keeper retries every "
            f"{DATA_KEEPER.interval}s" + (f" - {reason}" if reason else ".")
        )
    s = session_status(now_ist(), runner.profile)
    eod = max(0.0, s["change_in"].total_seconds() / 60.0 - SETTINGS.eod_square_off_minutes) if s["open"] else None
    chips = [chip("RUNNING" if status["running"] else "STOPPED", "good" if status["running"] else "muted")]
    if status["running"]:
        chips.append(chip(f"{status['cycles']} cycles · every {SETTINGS.auto_trade_refresh_seconds}s", "info"))
    if status["last_cycle_at"]:
        chips.append(chip(f"last scan {status['last_cycle_at']:%H:%M:%S} IST", "muted", icon=""))
    if eod is not None and runner.market_hours_only and SETTINGS.eod_square_off:
        chips.append(chip(f"square-off in {F.duration(dt.timedelta(minutes=eod))}", "warning"))
    st.markdown(" ".join(chips), unsafe_allow_html=True)
    st.caption(f"Status: {status['last_status']}")
    if status["last_error"]:
        st.warning(f"Last error: {status['last_error']}")

    cap = capital_summary(runner.owner.broker)
    open_df, _ = live_open()
    mine = open_df[open_df["market"] == market] if len(open_df) else open_df
    unreal = float(mine["unrealised_net"].sum(skipna=True)) if len(mine) else 0.0
    today = _today_closed(market)
    realised = float(today["net_pnl"].sum()) if len(today) else 0.0
    required_now = sum(r.get("capital_required") or 0 for r in runner.last_rows if r.get("status") == "SETUP_TRIGGERED")
    day = realised + unreal
    kpi_row([
        {"label": f"{market} day P&L (net)", "value": F.inr(day, signed=True), "tone": F.tone(day),
         "delta": f"{F.inr(realised, signed=True)} realised",
         "help": f"Closed {market} trades today after charges + open {market} positions marked to live prices"},
        {"label": f"Open {market} positions", "value": str(len(mine)),
         "delta": f"{F.inr(float(mine['invested'].sum()) if len(mine) else 0)} invested"},
        {"label": f"{market} capital used", "value": F.inr(cap["capital_used"]),
         "delta": f"{cap['capital_used'] / cap['account_value'] * 100:.1f}% of account" if cap["account_value"] else None},
        {"label": "Available", "value": F.inr(cap["available"]), "help": f"Available in the {market} paper account (NSE and MCX are separate accounts)"},
        {"label": "Needed by signals", "value": F.inr(required_now),
         "help": "Premium x quantity for setups that triggered in the last scan"},
    ])
    try:
        from quant_intelligence.execution.auto_trader import risk_budget
        from quant_intelligence.risk.risk_engine import limits_for

        budget, mult, why = risk_budget(runner.owner.account_state())
        st.caption(
            f"Loss limit: a stopped {market} trade risks at most {F.inr(budget)} ({limits_for(market).max_risk_per_trade_pct:g}% of equity, "
            f"whole lots; the {market} limit, set on its Risk Control page" + (f"; cut to {mult:.0%} now - {why}" if mult < 1 else "")
            + "). An instrument whose single lot would lose more than that at its stop is skipped (see 'NO SIZE' in the scan detail)."
        )
    except Exception:
        pass
    if cap["sell_premium_value"]:
        st.caption(f"Open SELL positions carry {F.inr(cap['sell_premium_value'])} of premium value; exchange margin is not modelled.")


def _today_closed(market: str) -> pd.DataFrame:
    start = dt.datetime.combine(now_ist().date(), dt.time.min)
    with get_session() as session:
        rows = session.query(Position).filter(Position.status == "CLOSED", Position.closed_at >= start).all()
    df = positions_frame(rows, now=now_ist())
    return df[df["market"] == market] if len(df) else df


# ---------------------------------------------------------------------------------------------- market watch
@st.fragment(run_every="10s")
def _market_watch(runner: ScanRunner, meta: list[dict], timeframe: str) -> None:
    results = {r.get("symbol"): r for r in runner.last_rows}
    picked = runner.mover_selection if runner.mover_config is not None else None
    rows = []
    for m in meta:
        snap = load_snapshot(m["symbol"], timeframe)
        res = results.get(m["symbol"], {})
        span = (snap["high"] - snap["low"]) if snap else 0
        rows.append(
            {
                "Heatmap": ({"LONG": "▲ calls only", "SHORT": "▼ puts only"}.get(picked.allowed_direction(m["symbol"]), "–")
                            if picked is not None else None),
                "Symbol": m["symbol"],
                "Name": m["name"],
                "Group": m["group"],
                "Last": snap["last"] if snap else None,
                "Chg %": snap["change_pct"] if snap else None,
                "Day range": ((snap["last"] - snap["low"]) / span * 100.0) if snap and span else None,
                "Trend": snap["spark"] if snap else None,
                "Lot": m["lot"],
                "Scan": _STATUS_LABEL.get(res.get("status"), res.get("status") or "–"),
                "Strategy": res.get("strategy") if res.get("strategy") not in (None, "-") else "",
                "Needs": res.get("capital_required"),
            }
        )
    df = pd.DataFrame(rows)
    df["Needs"] = pd.to_numeric(df["Needs"], errors="coerce")
    if df["Heatmap"].isna().all():
        df = df.drop(columns=["Heatmap"])
    if df["Lot"].isna().all():
        df = df.drop(columns=["Lot"])
    if df["Needs"].isna().all():
        df = df.drop(columns=["Needs"])
    cfg = {
        "Last": st.column_config.NumberColumn("Last", format="%.2f", help="Last closed bar from the candle cache"),
        "Chg %": st.column_config.NumberColumn("Chg %", format="%+.2f%%"),
        "Day range": st.column_config.ProgressColumn("Day range", min_value=0, max_value=100, format="%.0f%%",
                                                      help="Where the last price sits between today's low (0%) and high (100%)"),
        "Trend": st.column_config.LineChartColumn("Trend", help="Last ~4 hours of closes"),
        "Needs": st.column_config.NumberColumn("Needs ₹", format="%.0f", help="Capital the triggered setup needs"),
    }
    cfg = {k: v for k, v in cfg.items() if k in df.columns}
    st.dataframe(df, column_config=cfg, hide_index=True, width="stretch", height=min(38 * (len(df) + 1) + 3, 520))
    st.caption("Prices are the last closed bar from the candle cache (no extra Dhan requests). The Scan column is the "
               "last auto-trader scan of each instrument.")


# ---------------------------------------------------------------------------------------------- tabs
@st.fragment(run_every="5s")
def _scan_results(runner: ScanRunner) -> None:
    rows = runner.last_rows
    if not rows:
        empty_state("No scan has run yet", "Press 'Run one cycle now' or Start auto trading.")
        return
    df = pd.DataFrame(rows)
    counts = df["status"].fillna("-").value_counts()
    st.markdown(
        " ".join([
            chip(f"{int(counts.get('SETUP_TRIGGERED', 0))} setups", "good" if counts.get("SETUP_TRIGGERED", 0) else "muted"),
            chip(f"{int(counts.get('WAITING_FOR_SETUP', 0))} waiting", "warning" if counts.get("WAITING_FOR_SETUP", 0) else "muted"),
            chip(f"{int(counts.get('NO_TRADE', 0))} no trade", "muted"),
            chip(f"{int(sum(counts.get(k, 0) for k in ('DATA_ERROR', 'CHAIN_ERROR', 'MONITOR_ERROR')))} errors",
                 "critical" if any(counts.get(k, 0) for k in ("DATA_ERROR", "CHAIN_ERROR", "MONITOR_ERROR")) else "muted"),
        ]),
        unsafe_allow_html=True,
    )
    show = pd.DataFrame(
        {
            "Symbol": df["symbol"], "Type": df["type"],
            "Result": df["status"].map(lambda s: _STATUS_LABEL.get(s, s)),
            "Strategy": df["strategy"].replace("-", ""),
            "Needs ₹": df.get("capital_required"), "Used ₹": df.get("capital_used"), "Detail": df["detail"],
        }
    )
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"Needs ₹": st.column_config.NumberColumn(format="%.0f"),
                                "Used ₹": st.column_config.NumberColumn(format="%.0f")})


def _activity(symbols: set[str]) -> None:
    with get_session() as session:
        orders = orders_frame(session.query(Order).order_by(Order.id.desc()).limit(300).all())
        risk = session.query(RiskEvent).order_by(RiskEvent.id.desc()).limit(100).all()
        closed = positions_frame(
            session.query(Position).filter(Position.status == "CLOSED").order_by(Position.id.desc()).limit(200).all(),
            now=now_ist(),
        )
    feed = activity_frame(orders, risk, closed, limit=40, instruments=symbols)
    if feed.empty:
        empty_state("No activity yet", "Orders, risk-engine vetoes and exits appear here as they happen.")
        return
    icon = {"good": "✓", "warning": "▲", "critical": "✖", "neutral": "–"}
    show = pd.DataFrame({
        "Time": pd.to_datetime(feed["time"]).dt.strftime("%d %b %H:%M:%S"),
        "": feed["tone"].map(icon),
        "Type": feed["kind"],
        "Event": feed["text"],
    })
    st.dataframe(show, hide_index=True, width="stretch", height=min(38 * (len(show) + 1) + 3, 560))
    st.caption("Risk-engine vetoes are account-wide (they carry no instrument).")


def _performance(market: str) -> None:
    today = _today_closed(market)
    if today.empty:
        empty_state(f"No closed {market} trades today", "Performance appears after the first position closes.")
        return
    s = trade_stats(today)
    kpi_row([
        {"label": "Trades", "value": str(s["trades"]), "delta": f"{s['wins']}W / {s['losses']}L"},
        {"label": "Net P&L", "value": F.inr(s["net"], signed=True), "tone": F.tone(s["net"]),
         "delta": f"gross {F.inr(s['gross'], signed=True)} · charges {F.inr(s['charges'])}"},
        {"label": "Win rate", "value": f"{s['win_rate']:.0f}%"},
        {"label": "Expectancy / trade", "value": F.inr(s["expectancy"], signed=True), "tone": F.tone(s["expectancy"])},
    ])
    a, b = st.columns([3, 2])
    with a:
        st.markdown("**Cumulative net P&L today**")
        st.plotly_chart(equity_curve_chart(equity_curve(today), height=260), width="stretch", config=_CFG)
    with b:
        st.markdown("**Net P&L by instrument**")
        st.plotly_chart(pnl_bar_chart(pnl_by(today, "instrument"), "instrument"), width="stretch", config=_CFG)
    st.dataframe(trades_table(today.sort_values("closed_at", ascending=False)), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------------------------- main entry
def render_multi_instrument_panel(runner: ScanRunner = ENGINE, commodities: bool = False) -> None:
    """`commodities=True` scans the MCX watchlist with `runner` (normally COMMODITY_RUNNER);
    otherwise the NSE indices + stock watchlist."""
    key = f"{runner.name}_"
    market = "MCX" if commodities else "NSE"
    index_symbols, stock_symbols, meta = _universe(commodities)
    timeframe = st.session_state.get("timeframe", SETTINGS.default_timeframe)
    lookback_days = st.session_state.get("lookback_days", 60)
    status = runner.status()

    # ---- controls -------------------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            f"**{'Commodities' if commodities else 'Indices & stocks'} auto-trader** "
            f"· {len(meta)} instruments · every trade is an OPTION · session {runner.profile.label}"
        )
        b1, b2, b3, b4 = st.columns([1, 1, 1, 2], vertical_alignment="center")
        if not status["running"]:
            if b1.button("Start auto trading", type="primary", key=key + "start", width="stretch"):
                runner.start(index_symbols, stock_symbols, timeframe, lookback_days,
                             market_hours_only=st.session_state.get(key + "market_only", runner.market_hours_only))
                st.rerun()
        elif b1.button("Stop auto trading", type="primary", key=key + "stop", width="stretch"):
            runner.stop()
            st.rerun()
        if b2.button("Run one cycle now", disabled=status["running"], key=key + "once", width="stretch"):
            with st.spinner("Scanning all instruments..."):
                runner.run_cycle(index_symbols, stock_symbols, timeframe, lookback_days)
        if b3.button("Square off", key=key + "squareoff", width="stretch",
                     help=f"Close every open {market} position now at its latest price"):
            with st.spinner("Closing positions..."):
                closed = runner.square_off_now()
            st.toast(f"Closed {len(closed)} position(s)" if closed else "Nothing closed (no open positions, or no live price)")
        b4.checkbox(
            f"Only trade in session (Mon-Fri {runner.profile.label}); square off {SETTINGS.eod_square_off_minutes} min before close",
            value=runner.market_hours_only, disabled=status["running"], key=key + "market_only",
        )
        st.caption(
            "Runs on the server in the background - it keeps going when you change pages or close the tab, until you "
            "press Stop (or the server restarts / the cloud app sleeps). Each cycle runs the strategy pipeline per "
            "instrument and fetches an option chain only when a setup triggers."
        )

    cfg = runner.mover_config
    if cfg is not None:
        st.info(f"Heatmap selection is ON: only the top {cfg.n} gainers and losers per group are scanned (up-movers take calls, "
                f"down-movers puts; min move {cfg.min_abs_pct:g}%). Change it on the NSE Heatmap page.")
    _live_header(runner, market, [m["symbol"] for m in meta])

    st.markdown("#### Market watch")
    _market_watch(runner, meta, timeframe)

    tab_pos, tab_scan, tab_act, tab_perf = st.tabs(["Positions", "Scan results", "Activity", "Today's performance"])
    with tab_pos:
        render_open_positions(market=market, key=f"{key}pos", toolbar=True)
    with tab_scan:
        _scan_results(runner)
    with tab_act:
        _activity({m["symbol"] for m in meta})
    with tab_perf:
        _performance(market)
