"""Reusable trading-UI pieces: chips, status bar, KPI tiles, P&L text, empty states, position cards."""
from __future__ import annotations

import datetime as dt
import html
import math

import pandas as pd
import streamlit as st

from quant_intelligence.brokers.dhan_rate_limit import LIMITER
from quant_intelligence.config.credentials import token_status
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data.data_keeper import DATA_KEEPER
from quant_intelligence.execution.engine import COMMODITY_RUNNER, ENGINE
from quant_intelligence.ui import format as F
from quant_intelligence.utils.market_calendar import session_status
from quant_intelligence.utils.market_profile import MCX, NSE
from quant_intelligence.utils.timeutil import now_ist

_ICON = {"good": "●", "warning": "▲", "serious": "▲", "critical": "✖", "info": "●",
         "neutral": "○", "muted": "○"}


# ---------------------------------------------------------------------------------------------- atoms
def chip(text: str, tone: str = "muted", icon: str | None = None) -> str:
    """A small pill. The icon makes the state readable without colour."""
    glyph = _ICON.get(tone, "") if icon is None else icon
    prefix = f"{glyph} " if glyph else ""
    return f'<span class="qi-chip {tone}">{prefix}{html.escape(str(text))}</span>'


def pnl_html(value: float | None, pct: float | None = None) -> str:
    """P&L as colour + triangle + sign (+ optional % return) - the sign and triangle carry meaning without colour."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '<span class="qi-pnl neutral">–</span>'
    sign = "-" if value < 0 else ("+" if value > 0 else "")
    pct_html = f"<small>{F.pct(pct)}</small>" if pct is not None else ""
    return f'<span class="qi-pnl {F.tone(value)}">{F.arrow(value)} {sign}{F.inr(abs(value))}{pct_html}</span>'


def empty_state(title: str, hint: str = "") -> None:
    st.markdown(f'<div class="qi-empty"><b>{html.escape(title)}</b>{html.escape(hint)}</div>', unsafe_allow_html=True)


def kpi_row(items: list[dict], border: bool = True) -> None:
    """A row of bordered stat tiles. item: {label, value, delta?, tone?, help?, spark?}.
    `tone` ('good'/'critical'/'neutral') picks the delta colour; a sparkline is optional."""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        delta_color = {"good": "normal", "critical": "inverse", "neutral": "off"}.get(it.get("tone", "neutral"), "off")
        with col:
            kwargs = dict(label=it["label"], value=it["value"], delta=it.get("delta"), delta_color=delta_color,
                          help=it.get("help"), border=border,
                          delta_arrow="off")  # subtext is a note, not a direction; colour alone reflects the tone
            if it.get("spark") and len(it["spark"]) > 1:
                kwargs.update(chart_data=list(it["spark"]), chart_type="area")
            st.metric(**kwargs)


# ---------------------------------------------------------------------------------------------- status bar
def _session_chip(label: str, profile) -> str:
    s = session_status(now_ist(), profile)
    return chip(f"{label} {'open' if s['open'] else 'closed'} · {s['label']}", "good" if s["open"] else "muted")


def dhan_chip() -> str:
    snap = LIMITER.snapshot(token=SETTINGS.dhan_access_token)
    if not (SETTINGS.dhan_client_id and SETTINGS.dhan_access_token):
        return chip("Dhan: not configured", "muted")
    ts = token_status(SETTINGS.dhan_access_token)
    if ts["state"] == "expired":
        return chip(f"Dhan: token expired {ts['expires_at']:%d %b %H:%M} - update it", "critical")
    if ts["state"] == "expiring":
        return chip(f"Dhan: token expires in {F.duration(ts['delta'])}", "warning")
    if snap["auth_block_remaining"] > 0:
        return chip("Dhan: token rejected - update it", "critical")
    if snap["cooldown_remaining"] > 0:
        return chip(f"Dhan: rate-limited {snap['cooldown_remaining']:.0f}s", "warning")
    age = snap["last_ok_age_sec"]
    if age is None:
        return chip("Dhan: ready", "info")
    return chip(f"Dhan: connected · {F.duration(dt.timedelta(seconds=age))} ago", "good")


def data_chip() -> str:
    """State of the background data keeper: are the watchlist candles current?"""
    if not SETTINGS.data_keeper_enabled:
        return chip("Data feed: refresh on demand", "muted")
    ds = DATA_KEEPER.status()
    if not ds["running"] or ds["rounds"] == 0:
        return chip("Data feed: starting", "info")
    if ds["paused"]:
        return chip("Data feed: paused - token expired", "warning")
    if ds["not_ok"]:
        return chip(f"Data feed: {len(ds['not_ok'])} of {ds['tracked']} stale", "warning")
    newest = ds["newest_bar"]
    return chip(f"Data feed: current{f' \u00b7 newest bar {newest:%d %b %H:%M}' if newest is not None else ''}", "good")


def _runner_chip(label: str, runner) -> str:
    s = runner.status()
    if s["running"]:
        return chip(f"{label} auto: running · {s['cycles']} cycles", "good")
    return chip(f"{label} auto: stopped", "muted")


def status_bar_html() -> str:
    mode = "LIVE" if SETTINGS.live_mode_fully_authorized else "PAPER"
    parts = [
        f'<span class="qi-clock">{now_ist():%H:%M:%S} IST</span>',
        _session_chip("NSE", NSE),
        _session_chip("MCX", MCX),
        '<span class="qi-sep"></span>',
        dhan_chip(),
        data_chip(),
        '<span class="qi-sep"></span>',
        _runner_chip("NSE", ENGINE),
        _runner_chip("MCX", COMMODITY_RUNNER),
        '<span class="qi-sep"></span>',
        chip(f"{mode} mode", "critical" if mode == "LIVE" else "info"),
    ]
    if ENGINE.kill_switch:
        parts.append(chip("KILL SWITCH ON", "critical"))
    return f'<div class="qi-statusbar">{"".join(parts)}</div>'


@st.fragment(run_every="10s")
def status_bar() -> None:
    """Market sessions, Dhan connection, auto-traders and mode. Re-renders itself every 10 s."""
    st.markdown(status_bar_html(), unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    status_bar()


# ---------------------------------------------------------------------------------------------- positions
def range_bar_html(stop: float, entry: float, target: float, ltp: float | None) -> str:
    """Stop-loss ... target track with the entry tick, a fill from entry to the live price (green when in
    profit, red when not) and a marker at the live price. Works for BUY (stop<target) and SELL (stop>target)."""
    span = target - stop
    if not span:
        return ""

    def at(x: float) -> float:
        return max(0.0, min(100.0, (x - stop) / span * 100.0))

    entry_pos = at(entry)
    parts = [f'<div class="qi-range"><div class="track"></div><div class="tick" style="left:{entry_pos:.1f}%"></div>']
    if ltp is not None:
        ltp_pos = at(ltp)
        profit = (ltp - entry) * (1 if target > stop else -1) >= 0
        color = "var(--qi-good)" if profit else "var(--qi-critical)"
        left, width = sorted([entry_pos, ltp_pos])[0], abs(ltp_pos - entry_pos)
        parts.append(f'<div class="fill" style="left:{left:.1f}%;width:{width:.1f}%;background:{color}"></div>')
        parts.append(f'<div class="ltp" style="left:{ltp_pos:.1f}%;background:{color}"></div>')
    parts.append("</div>")
    parts.append(
        f'<div class="qi-range-labels"><span>✖ Stop {F.num(stop, 2)}</span>'
        f"<span>Entry {F.num(entry, 2)}</span><span>Target {F.num(target, 2)} ◎</span></div>"
    )
    return "".join(parts)


def _held(row) -> str:
    return F.duration(row["held"]) if row.get("held") is not None and not pd.isna(row.get("held")) else "–"


def position_card(row: pd.Series, key: str, on_exit=None) -> None:
    """One open position as a trading-app card: contract, side, live P&L, quantities, charges, stop/target bar."""
    ltp = None if pd.isna(row.get("ltp", float("nan"))) else float(row["ltp"])
    pnl = None if pd.isna(row.get("unrealised", float("nan"))) else float(row["unrealised"])
    net = None if pd.isna(row.get("unrealised_net", float("nan"))) else float(row["unrealised_net"])
    pct = None if pd.isna(row.get("unrealised_pct", float("nan"))) else float(row["unrealised_pct"])
    side_tone = "info" if row["side"] == "BUY" else "serious"
    head = (
        f'<div class="qi-pos-head"><span class="qi-sym">{html.escape(str(row["contract"]))}</span>'
        f'{chip(row["side"], side_tone, icon="")}{chip(row["market"], "muted", icon="")}'
        f'{chip(row["strategy"] or "-", "muted", icon="")}'
        f'{chip(row["tag"], "warning", icon="") if row.get("tag") else ""}'
        f'<span class="qi-pos-pnl">{pnl_html(pnl, pct)}</span></div>'
    )
    grid = (
        '<div class="qi-grid">'
        f'<div><span>Qty</span><b>{int(row["qty"])}</b></div>'
        f'<div><span>Avg price</span><b>{F.num(row["entry"], 2)}</b></div>'
        f'<div><span>LTP</span><b>{F.num(ltp, 2)}</b></div>'
        f'<div><span>Invested</span><b>{F.inr(row["invested"])}</b></div>'
        f'<div><span>Est. charges</span><b>{F.inr(row.get("unrealised_charges"), 0)}</b></div>'
        f'<div><span>Net after charges</span><b>{F.inr(net, 0, signed=True)}</b></div>'
        f'<div><span>Held</span><b>{_held(row)}</b></div>'
        "</div>"
    )
    bar = range_bar_html(float(row["stop"]), float(row["entry"]), float(row["target"]), ltp)
    with st.container(border=True):
        if on_exit is not None:
            body, side = st.columns([6, 1], vertical_alignment="center")
            body.markdown(f'<div class="qi-pos">{head}{grid}{bar}</div>', unsafe_allow_html=True)
            if side.button("Exit", key=f"exit_{key}", help="Close this paper position at the live price"):
                on_exit(row["position_id"])
        else:
            st.markdown(f'<div class="qi-pos">{head}{grid}{bar}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------- quotes
def day_range_html(low: float, high: float, last: float) -> str:
    span = high - low
    pos = 50.0 if not span else max(0.0, min(100.0, (last - low) / span * 100.0))
    return (f'<div class="qi-dayrange"><div class="track"></div><div class="dot" style="left:{pos:.1f}%"></div></div>'
            f'<div class="qi-dayrange-labels"><span>Low {F.num(low, 2)}</span><span>High {F.num(high, 2)}</span></div>')


def quote_header(symbol: str, snap: dict | None, tags: list[str] | None = None) -> None:
    """Instrument title + last price + day change + OHLC strip + day-range marker (from the cached candles)."""
    tag_html = "".join(chip(t, "muted", icon="") for t in (tags or []))
    if not snap:
        st.markdown(f'<div class="qi-quote"><span class="qi-sym">{html.escape(symbol)}</span> {tag_html}'
                    '<div class="qi-quote-sub">No cached candles yet - run the pipeline once to load them.</div></div>',
                    unsafe_allow_html=True)
        return
    chg, pct = snap["change"], snap["change_pct"]
    sign = "+" if chg > 0 else ("-" if chg < 0 else "")
    st.markdown(
        f'<div class="qi-quote"><div><span class="qi-sym">{html.escape(symbol)}</span> {tag_html}</div>'
        f'<div class="qi-quote-main"><span class="big">{F.num(snap["last"], 2)}</span>'
        f'<span class="qi-pnl {F.tone(chg)}">{F.arrow(chg)} {sign}{F.num(abs(chg), 2)} ({F.pct(pct)})</span></div>'
        f'<div class="qi-quote-sub">O {F.num(snap["open"], 2)} · H {F.num(snap["high"], 2)} · '
        f'L {F.num(snap["low"], 2)} · prev close {F.num(snap["prev_close"], 2)} · '
        f'last closed bar {snap["as_of"]:%d %b %H:%M}</div>'
        f'{day_range_html(snap["low"], snap["high"], snap["last"])}</div>',
        unsafe_allow_html=True,
    )
