"""Where a result's data came from and how current it is, as a label a user can read at a glance.

LIVE        the exchange session is open and the latest bar closed within about one bar interval
DELAYED     the session is open but the latest bar is older than that (feed lag, failed refresh, rate limit)
HISTORICAL  the session is closed: the data is the last completed session, not the live market
SIMULATED   the data (or the outcome) was produced by a simulation, not observed (paper fills, synthetic candles)

This is a statement about the data's currency, not about its quality; quality (OK / DEGRADED / FAIL) is the separate gate.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from quant_intelligence.utils.market_calendar import session_status
from quant_intelligence.utils.market_profile import NSE, MarketProfile

LIVE, DELAYED, HISTORICAL, SIMULATED = "LIVE", "DELAYED", "HISTORICAL", "SIMULATED"
MODES = (LIVE, DELAYED, HISTORICAL, SIMULATED)
LIVE_TOLERANCE_BARS = 1.5  # a closed bar may be up to this many intervals behind the clock and still count as live
SIMULATED_SOURCES = {"synthetic", "simulated", "paper"}

MODE_MEANING = {
    LIVE: "Session open and the latest bar is current.",
    DELAYED: "Session open but the latest bar is older than expected.",
    HISTORICAL: "Session closed: this is the last completed session, not the live market.",
    SIMULATED: "Produced by a simulation (paper trading or synthetic candles), not observed.",
}


def classify_mode(
    last_bar: dt.datetime | pd.Timestamp | None,
    now: dt.datetime,
    profile: MarketProfile = NSE,
    tf_minutes: int = 5,
    source: str | None = None,
) -> str:
    """The currency label for data whose latest bar STARTED at `last_bar` (naive IST), judged at `now` (naive IST)."""
    if source and source.lower() in SIMULATED_SOURCES:
        return SIMULATED
    if last_bar is None or pd.isna(last_bar):
        return HISTORICAL
    now = now.replace(tzinfo=None)
    if not session_status(now, profile)["open"]:
        return HISTORICAL
    bar_end = pd.Timestamp(last_bar).to_pydatetime().replace(tzinfo=None) + dt.timedelta(minutes=tf_minutes)
    lag = (now - bar_end).total_seconds() / 60.0
    return LIVE if lag <= tf_minutes * LIVE_TOLERANCE_BARS else DELAYED


def build_provenance(metadata: dict, now: dt.datetime, profile: MarketProfile = NSE, tf_minutes: int = 5) -> dict:
    """One dict for the UI/website: source, mode, last bar, age and quality, from a DataManager metadata dict."""
    last_bar = metadata.get("end_ts")
    report = metadata.get("quality_report") or {}
    source = str(metadata.get("source") or "unknown")
    mode = classify_mode(last_bar, now, profile, tf_minutes, source)
    age = None
    if last_bar is not None and not pd.isna(last_bar):
        age = (now.replace(tzinfo=None) - pd.Timestamp(last_bar).to_pydatetime().replace(tzinfo=None)).total_seconds() / 60.0
    return {
        "source": source,
        "mode": mode,
        "mode_meaning": MODE_MEANING[mode],
        "last_bar": last_bar,
        "age_minutes": age,
        "quality_status": metadata.get("quality_status"),
        "quality_category": report.get("category"),
        "rows": metadata.get("n_rows"),
        "timeframe": metadata.get("timeframe"),
        "refresh_failed": any("refresh failed" in str(i) for i in report.get("issues", [])),
    }


def strip_text(p: dict) -> str:
    """One line for a header: 'HISTORICAL · cache · last bar 16 Oct 15:25 (2d 18h old) · quality OK'."""
    if not p:
        return "data provenance unavailable"
    age = p.get("age_minutes")
    if age is None:
        age_txt = "age unknown"
    elif age < 90:
        age_txt = f"{age:.0f} min old"
    elif age < 2880:
        age_txt = f"{age / 60:.1f} h old"
    else:
        age_txt = f"{age / 1440:.1f} days old"
    last = p.get("last_bar")
    last_txt = pd.Timestamp(last).strftime("%d %b %H:%M") if last is not None and not pd.isna(last) else "n/a"
    return f"{p['mode']} · source {p['source']} · last bar {last_txt} ({age_txt}) · quality {p.get('quality_status') or 'n/a'}"
