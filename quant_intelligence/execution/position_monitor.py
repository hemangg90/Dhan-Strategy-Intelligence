"""
Position Monitor & Re-evaluation.

After a position is opened, this module continuously checks: has stop/target
been hit (paper broker fills), has the market regime changed enough that the
selected strategy's conditional edge should be flagged, and does risk state
require intervention. It does NOT automatically switch strategies mid-trade -
it only flags reassessment events for the user/UI, and closes positions when
their own stop/target is actually hit (that is basic trade management, not
strategy switching).
"""
from __future__ import annotations

import datetime as dt
from quant_intelligence.utils.timeutil import now_ist

import pandas as pd

from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.options.option_selector import option_segment_for
from quant_intelligence.regimes.regime_engine import classify_regime
from quant_intelligence.utils.logging_utils import log_event


def _fetch_quotes(api_client, positions: list[dict], max_age: float = 1.0, stale_ok_for: float = 0.0) -> dict:
    """Live quotes for option positions keyed by security_id (as str). One LTP call covering
    every exchange segment involved, so NSE and MCX positions are both priced. Goes through the
    shared quote cache, so tabs and background runners asking within `max_age` seconds share a single
    Dhan request (Dhan allows only 1 quote request per second for the whole account)."""
    by_segment: dict[str, list[int]] = {}
    for p in positions:
        if not p.get("security_id"):
            continue
        segment = option_segment_for(p.get("underlying") or p.get("instrument"))
        by_segment.setdefault(segment, []).append(int(p["security_id"]))
    if not by_segment:
        return {}
    from quant_intelligence.brokers.dhan_cache import QUOTE_CACHE

    data = QUOTE_CACHE.get_quotes(
        api_client, {seg: sorted(set(ids)) for seg, ids in by_segment.items()},
        max_age=max_age, stale_ok_for=stale_ok_for,
    )
    quotes: dict = {}
    for seg_quotes in data.values():
        quotes.update({str(k): v for k, v in (seg_quotes or {}).items()})
    return quotes


def monitor_positions(broker: PaperBroker, latest_bar: pd.Series) -> list[dict]:
    """Check all open non-option paper positions against the latest bar's high/low for
    stop/target hits. Option positions are denominated in premium, not the underlying's
    price, and are monitored separately by `monitor_option_positions`."""
    events = []
    for pos in broker.get_open_positions():
        if pos.get("option_type"):
            continue
        direction = pos["direction"]
        stop = pos["stop_price"]
        target = pos["target_price"]
        pos["last_price"] = float(latest_bar["close"])

        hit_stop = (latest_bar["low"] <= stop) if direction == "LONG" else (latest_bar["high"] >= stop)
        hit_target = (latest_bar["high"] >= target) if direction == "LONG" else (latest_bar["low"] <= target)

        if hit_stop:
            # A bar that opened beyond the stop gapped through it: the exit fills at the worse open, not at the stop.
            bar_open = float(latest_bar["open"]) if "open" in latest_bar else stop
            stop_fill = min(stop, bar_open) if direction == "LONG" else max(stop, bar_open)
            closed = broker.close_position(pos["position_id"], stop_fill, "STOP")
            events.append({"type": "STOP_HIT", "position": closed})
            log_event("position_monitor", f"Stop hit for {pos['position_id']}", level="INFO")
        elif hit_target:
            closed = broker.close_position(pos["position_id"], target, "TARGET")
            events.append({"type": "TARGET_HIT", "position": closed})
            log_event("position_monitor", f"Target hit for {pos['position_id']}", level="INFO")

    return events


def monitor_option_positions(broker: PaperBroker, api_client) -> list[dict]:
    """Check all open option positions against LIVE premium (via the Dhan LTP feed),
    since an option's stop/target is set in premium terms, not underlying terms (see
    quant_intelligence/options/premium_model.py). `api_client` is a DhanApiClient."""
    events = []
    option_positions = [p for p in broker.get_open_positions() if p.get("option_type")]
    if not option_positions:
        return events

    priced = [p for p in option_positions if p.get("security_id")]  # a list: two positions on one contract are both checked
    if not priced:
        return events

    quotes = _fetch_quotes(api_client, option_positions, max_age=1.0)  # exits: fresh, never stale

    for pos in priced:
        quote = quotes.get(str(pos["security_id"]))
        if not quote:
            continue
        premium = quote.get("last_price")
        if premium is None:
            continue
        pos["last_price"] = float(premium)  # marks the position to market for the risk engine's equity

        transaction = pos.get("transaction") or ("BUY" if pos["direction"] == "LONG" else "SELL")
        stop, target = pos["stop_price"], pos["target_price"]

        if transaction == "BUY":
            hit_stop = premium <= stop
            hit_target = premium >= target
        else:
            hit_stop = premium >= stop
            hit_target = premium <= target

        if hit_stop:
            # The premium is only sampled periodically, so it may already be past the stop: exit at the worse price.
            stop_fill = min(stop, float(premium)) if transaction == "BUY" else max(stop, float(premium))
            closed = broker.close_position(pos["position_id"], stop_fill, "STOP")
            events.append({"type": "STOP_HIT", "position": closed})
            log_event("position_monitor", f"Option premium stop hit for {pos['position_id']}", level="INFO")
        elif hit_target:
            closed = broker.close_position(pos["position_id"], target, "TARGET")
            events.append({"type": "TARGET_HIT", "position": closed})
            log_event("position_monitor", f"Option premium target hit for {pos['position_id']}", level="INFO")

    return events


def fetch_option_ltp_map(broker: PaperBroker, api_client) -> dict:
    """Live LTP for every open option position, keyed by security_id. Returns {} if
    there are no open option positions or the API client isn't configured."""
    option_positions = [p for p in broker.get_open_positions() if p.get("option_type")]
    security_ids = {p["security_id"] for p in option_positions if p.get("security_id")}
    if not security_ids:
        return {}

    # Display only: the cache TTL applies, and a rate-limited fetch falls back to recent prices.
    quotes = _fetch_quotes(api_client, option_positions, max_age=SETTINGS.dhan.quote_cache_ttl_sec, stale_ok_for=30.0)
    ltp_map = {}
    for sid in security_ids:
        quote = quotes.get(str(sid))
        if quote and quote.get("last_price") is not None:
            ltp_map[sid] = quote["last_price"]
    for p in option_positions:
        if p.get("security_id") in ltp_map:
            p["last_price"] = float(ltp_map[p["security_id"]])
    return ltp_map


def positions_with_live_ltp(broker: PaperBroker, ltp_map: dict) -> list[dict]:
    """Open positions annotated with `current_ltp` and `unrealized_pnl` for display.
    Non-option (underlying) positions and positions with no LTP quote yet pass through
    with those fields set to None."""
    rows = []
    for pos in broker.get_open_positions():
        row = dict(pos)
        ltp = ltp_map.get(pos.get("security_id")) if pos.get("option_type") else None
        row["current_ltp"] = ltp
        if ltp is not None:
            signed_field = pos.get("transaction") or pos["direction"]
            direction_sign = 1 if signed_field in ("LONG", "BUY") else -1
            row["unrealized_pnl"] = direction_sign * (ltp - pos["entry_price"]) * pos["quantity"]
        else:
            row["unrealized_pnl"] = None
        rows.append(row)
    return rows


def reassess_regime_validity(position: dict, current_features: dict, entry_regime_label: str) -> dict:
    """Flag (but do not act on) a regime change since entry - informational for the UI/trader."""
    current_label, probs = classify_regime(current_features)
    changed = current_label != entry_regime_label
    result = {
        "position_id": position.get("position_id"),
        "entry_regime": entry_regime_label,
        "current_regime": current_label,
        "regime_changed": changed,
        "current_regime_confidence": max(probs.values()) if probs else None,
        "checked_at": now_ist(),
    }
    if changed:
        log_event(
            "position_monitor",
            f"Regime changed since entry for position {position.get('position_id')}: "
            f"{entry_regime_label} -> {current_label}",
            level="WARNING",
        )
    return result
