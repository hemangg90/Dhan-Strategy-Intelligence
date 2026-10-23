"""
Liquidity zones and liquidity events (sweeps, accepted breakouts, failed breakouts).

IMPORTANT: OHLCV data cannot show where stop orders actually rest. A "liquidity zone" here is a STRUCTURAL HYPOTHESIS:
a price where many traders would plausibly have placed stops or entries (just beyond an obvious swing, an equal high or
low, the prior session's or week's extreme, or the opening range). An event at such a zone is a pattern in price, not
evidence of who was triggered. Nothing in this module produces an order; events are context for a strategy's own setup.

Zones (`build_zones`)
  SWING_HIGH / SWING_LOW   confirmed swing points (known only after their confirmation bar)
  EQUAL_HIGHS / EQUAL_LOWS two or more confirmed swings within `equal_atr` x ATR of each other (level = the extreme)
  PDH / PDL                prior session high / low (known from the session's first bar)
  WEEK_HIGH / WEEK_LOW     extreme of up to the 5 sessions before today (known from today's first bar)
  OR_HIGH / OR_LOW         opening range of today (known once `opening_range_minutes` have passed)
  Side BUY_SIDE = liquidity above a high; SELL_SIDE = below a low.

Events (`detect_events`) - a breach starts an episode; the episode is judged over a window of `confirm_bars` + 1 bars:
  breach            the bar's wick goes beyond the level by more than `breach_atr` x ATR
  BREAKOUT_ACCEPTED `accept_bars` consecutive closes beyond the level
  FAILED_BREAKOUT   at least one close beyond the level, then a close back at or inside it within the window
  SWEEP_CONFIRMED   wick-only breach (no close beyond), then a close at least `reclaim_atr` x ATR back inside the level
  SWEEP_UNCONFIRMED wick-only breach and the window ended without that reclaim
  UNRESOLVED        the window has not ended and none of the above has happened yet (or it ended with closes beyond
                    the level but fewer than `accept_bars`)
Each event has `known_index/known_at` (the bar that settled it), expires `event_valid_bars` later, and is INVALIDATED if
a later bar closes beyond the episode's extreme. A swept level is consumed: it produces one episode.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_intelligence.features.market_structure import StructureConfig, StructureReport, _atr, analyze_structure

BUY_SIDE, SELL_SIDE = "BUY_SIDE", "SELL_SIDE"
SWEEP_CONFIRMED, SWEEP_UNCONFIRMED = "SWEEP_CONFIRMED", "SWEEP_UNCONFIRMED"
BREAKOUT_ACCEPTED, FAILED_BREAKOUT, UNRESOLVED = "BREAKOUT_ACCEPTED", "FAILED_BREAKOUT", "UNRESOLVED"
ACTIVE, EXPIRED, INVALIDATED = "ACTIVE", "EXPIRED", "INVALIDATED"

IMPLICATION = {
    (SWEEP_CONFIRMED, BUY_SIDE): "highs taken then rejected: bearish context, not a signal",
    (SWEEP_CONFIRMED, SELL_SIDE): "lows taken then rejected: bullish context, not a signal",
    (SWEEP_UNCONFIRMED, BUY_SIDE): "wick above the level without rejection: no read",
    (SWEEP_UNCONFIRMED, SELL_SIDE): "wick below the level without rejection: no read",
    (BREAKOUT_ACCEPTED, BUY_SIDE): "price accepted above the level: bullish continuation context",
    (BREAKOUT_ACCEPTED, SELL_SIDE): "price accepted below the level: bearish continuation context",
    (FAILED_BREAKOUT, BUY_SIDE): "close above the level did not hold: bearish context",
    (FAILED_BREAKOUT, SELL_SIDE): "close below the level did not hold: bullish context",
}


@dataclass(frozen=True)
class LiquidityConfig:
    equal_atr: float = 0.15
    breach_atr: float = 0.05
    reclaim_atr: float = 0.10
    confirm_bars: int = 3
    accept_bars: int = 2
    event_valid_bars: int = 12
    lookback_bars: int = 400
    opening_range_minutes: int = 30
    week_sessions: int = 5


@dataclass(frozen=True)
class LiquidityZone:
    source: str
    side: str
    level: float
    touches: int
    known_index: int  # first bar at which the zone exists
    known_at: pd.Timestamp


@dataclass(frozen=True)
class LiquidityEvent:
    kind: str
    side: str
    level: float
    source: str
    breach_index: int
    known_index: int
    known_at: pd.Timestamp
    extreme: float  # highest high (BUY_SIDE) or lowest low (SELL_SIDE) reached in the episode
    status: str  # ACTIVE / EXPIRED / INVALIDATED at the decision bar
    expires_index: int
    note: str


@dataclass
class LiquidityReport:
    as_of_index: int
    zones: list[LiquidityZone]
    events: list[LiquidityEvent]
    standing: list[LiquidityZone]  # zones nothing has breached yet

    def active_events(self) -> list[LiquidityEvent]:
        return [e for e in self.events if e.status == ACTIVE]


def build_zones(df: pd.DataFrame, structure: StructureReport, atr_value: float, cfg: LiquidityConfig) -> list[LiquidityZone]:
    """Zones that exist at `structure.as_of_index` (df must end at that bar)."""
    zones: list[LiquidityZone] = []
    ts = df["timestamp"].tolist()
    last = len(df) - 1
    start = max(0, last - cfg.lookback_bars)
    tol = cfg.equal_atr * atr_value

    for kind, side, eq_name, single_name in (("HIGH", BUY_SIDE, "EQUAL_HIGHS", "SWING_HIGH"), ("LOW", SELL_SIDE, "EQUAL_LOWS", "SWING_LOW")):
        swings = [s for s in structure.swings if s.kind == kind and s.index >= start]
        swings.sort(key=lambda s: s.price)
        groups: list[list] = []
        for s in swings:
            if groups and s.price - groups[-1][-1].price <= tol:
                groups[-1].append(s)
            else:
                groups.append([s])
        for g in groups:
            known = max(g, key=lambda s: s.confirmed_index)
            if len(g) >= 2:
                level = max(s.price for s in g) if kind == "HIGH" else min(s.price for s in g)
                zones.append(LiquidityZone(eq_name, side, level, len(g), known.confirmed_index, known.known_at))
            else:
                zones.append(LiquidityZone(single_name, side, g[0].price, 1, g[0].confirmed_index, g[0].known_at))

    dates = df["timestamp"].dt.date
    order = list(dict.fromkeys(dates.tolist()))
    today = order[-1]
    today_rows = np.flatnonzero((dates == today).to_numpy())
    today_first = int(today_rows[0])
    prior = order[:-1]
    if prior:
        prev = df[dates == prior[-1]]
        zones.append(LiquidityZone("PDH", BUY_SIDE, float(prev["high"].max()), 1, today_first, ts[today_first]))
        zones.append(LiquidityZone("PDL", SELL_SIDE, float(prev["low"].min()), 1, today_first, ts[today_first]))
        if len(prior) >= 2:
            span = df[dates.isin(prior[-cfg.week_sessions:])]
            zones.append(LiquidityZone("WEEK_HIGH", BUY_SIDE, float(span["high"].max()), 1, today_first, ts[today_first]))
            zones.append(LiquidityZone("WEEK_LOW", SELL_SIDE, float(span["low"].min()), 1, today_first, ts[today_first]))
    open_ts = ts[today_first]
    or_idx = [i for i in today_rows if (ts[i] - open_ts) < pd.Timedelta(minutes=cfg.opening_range_minutes)]
    if or_idx and (ts[last] - open_ts) >= pd.Timedelta(minutes=cfg.opening_range_minutes):  # only once the range is complete
        window = df.iloc[or_idx]
        known_i = max(or_idx) + 1
        if known_i <= last:
            zones.append(LiquidityZone("OR_HIGH", BUY_SIDE, float(window["high"].max()), 1, known_i, ts[known_i]))
            zones.append(LiquidityZone("OR_LOW", SELL_SIDE, float(window["low"].min()), 1, known_i, ts[known_i]))
    return zones


def _episode(df_arrays, zone: LiquidityZone, atr: np.ndarray, cfg: LiquidityConfig, t: int) -> LiquidityEvent | None:
    h, l, c, ts = df_arrays
    buy = zone.side == BUY_SIDE
    level = zone.level
    sign = 1.0 if buy else -1.0
    # first bar after the zone exists whose wick goes beyond the level by more than the breach tolerance
    j0 = None
    for j in range(zone.known_index, t + 1):
        a = atr[j] if np.isfinite(atr[j]) else 0.0
        beyond = (h[j] - level) if buy else (level - l[j])
        if beyond > cfg.breach_atr * a and beyond > 0:
            j0 = j
            break
    if j0 is None:
        return None
    end = min(j0 + cfg.confirm_bars, t)
    window_over = (j0 + cfg.confirm_bars) <= t
    extreme = max(h[j0: end + 1]) if buy else min(l[j0: end + 1])
    consecutive = 0
    seen_close_beyond = False
    kind = None
    known_index = None
    for j in range(j0, end + 1):
        a = atr[j] if np.isfinite(atr[j]) else 0.0
        close_beyond = (c[j] > level) if buy else (c[j] < level)
        if close_beyond:
            seen_close_beyond = True
            consecutive += 1
            if consecutive >= cfg.accept_bars:
                kind, known_index = BREAKOUT_ACCEPTED, j
                break
        else:
            consecutive = 0
            if seen_close_beyond:
                kind, known_index = FAILED_BREAKOUT, j
                break
            back = (level - c[j]) if buy else (c[j] - level)
            if back >= cfg.reclaim_atr * a:
                kind, known_index = SWEEP_CONFIRMED, j
                break
    if kind is None:
        if not window_over:
            kind, known_index = UNRESOLVED, t
        elif seen_close_beyond:
            kind, known_index = UNRESOLVED, end  # closes beyond the level but not enough to call it accepted
        else:
            kind, known_index = SWEEP_UNCONFIRMED, end
    if known_index is not None and kind in (BREAKOUT_ACCEPTED, FAILED_BREAKOUT, SWEEP_CONFIRMED):
        extreme = max(h[j0: known_index + 1]) if buy else min(l[j0: known_index + 1])
    expires = known_index + cfg.event_valid_bars
    status = ACTIVE
    for j in range(known_index + 1, t + 1):
        if (buy and c[j] > extreme) or ((not buy) and c[j] < extreme):
            status = INVALIDATED
            break
    if status == ACTIVE and t > expires:
        status = EXPIRED
    note = IMPLICATION.get((kind, zone.side), "price is interacting with the level; undecided")
    return LiquidityEvent(kind, zone.side, level, zone.source, j0, known_index, ts[known_index], float(extreme), status, expires, note)


def detect_events(df: pd.DataFrame, zones: list[LiquidityZone], atr: np.ndarray, cfg: LiquidityConfig, t: int) -> list[LiquidityEvent]:
    arrays = (df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float), df["timestamp"].tolist())
    events = []
    for z in zones:
        ev = _episode(arrays, z, atr, cfg, t)
        if ev is not None:
            events.append(ev)
    events.sort(key=lambda e: e.known_index)
    return events


def analyze_liquidity(df: pd.DataFrame, structure_cfg: StructureConfig | None = None, cfg: LiquidityConfig | None = None,
                      as_of_index: int | None = None) -> LiquidityReport:
    """Zones and events as known at `as_of_index` (default the last bar). Only bars up to it are read."""
    cfg = cfg or LiquidityConfig()
    structure_cfg = structure_cfg or StructureConfig()
    t = len(df) - 1 if as_of_index is None else int(as_of_index)
    data = df.iloc[: t + 1].reset_index(drop=True)
    structure = analyze_structure(data, structure_cfg)
    atr = _atr(data, structure_cfg.atr_period)
    atr_now = float(atr[-1]) if np.isfinite(atr[-1]) else 0.0
    if atr_now <= 0:
        return LiquidityReport(t, [], [], [])
    zones = build_zones(data, structure, atr_now, cfg)
    events = detect_events(data, zones, atr, cfg, len(data) - 1)
    breached = {(e.source, e.level) for e in events}
    standing = [z for z in zones if (z.source, z.level) not in breached]
    return LiquidityReport(t, zones, events, standing)
