"""
Support / resistance and liquidity levels, and the check a trade must pass against them.

`find_levels` builds the levels from price history (and, when given, the option chain's OI peaks):
  PDH / PDL      prior session high / low                                    weight 2
  WEEK_HIGH/LOW  high / low of the prior sessions (up to 5)                  weight 2.5
  DAY_HIGH/LOW   today's range so far                                        weight 1.5
  SWING_HIGH/LOW fractal turning points (3 bars each side); weight 1 + 0.5 per extra touch
  EQ_HIGHS/LOWS  liquidity pools: two or more swing points (or PDH/day highs) within a small tolerance of each
                 other - the resting stops sit just beyond them                weight 2 + 0.5 per extra touch
  PIVOT_*        floor pivots from the prior session (P, R1, S1, R2, S2)     weight 1
  OI_CALL_PEAK / OI_PUT_PEAK  the strike with the most call / put open interest (context only)   weight 1
  CONSOLIDATION_HIGH/LOW  the edges of the last 20 bars when they fit inside 2.5 ATR (a range)         weight 1.5
Levels within 0.1 ATR of each other merge into one (kinds joined, weights added, capped at 4).

Each level then carries its own record (all from bars up to the decision bar):
  reactions        independent excursions to the level that were followed by a reversal of at least 0.75 ATR; the consecutive
                   bars of one excursion count once, so lingering at a level is not repeated evidence
  bars_since_reaction  recency of the latest one
  behavior         the latest excursion's outcome: REJECTED (price turned back), BROKE (closed through by 0.75 ATR or more),
                   OPEN (not yet resolved) or NONE (never tested)
  htf              the level comes from a higher timeframe (prior day, prior week, floor pivots)
  confluence       other levels within 0.25 ATR of it (independent sources agreeing)
These describe a level; they do not change its weight or whether it counts as significant.

`assess_trade` is the check. Only "significant" levels (weight >= 1.5) count.
  * Room: the nearest significant level against the trade must be at least `min_room_mult` x the stop distance
    away, else the trade is vetoed ("no room"). A target that would run past that level is pulled back to just
    short of it.
  * Stop: when a significant level sits between the stop and the entry (and at least `MIN_STOP_ATR` x ATR from
    entry), the stop moves to just beyond it - a structure stop that is never wider than the original.
  * Notes: entry at a supporting level, and a recent sweep of a level followed by a reclaim.
Pure logic on a DataFrame (timestamp, open, high, low, close), no network calls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

SIGNIFICANT_WEIGHT = 1.5
MIN_STOP_ATR = 0.5  # a structure stop closer than this to the entry is noise, so it is not used
MERGE_ATR = 0.1
SWING_BARS = 3
LOOKBACK_SESSIONS = 5
MAX_WEIGHT = 4.0


@dataclass(frozen=True)
class Level:
    price: float
    kind: str
    weight: float
    touches: int = 1
    reactions: int = 0
    bars_since_reaction: int | None = None
    behavior: str = "NONE"
    htf: bool = False
    confluence: int = 0

    @property
    def significant(self) -> bool:
        return self.weight >= SIGNIFICANT_WEIGHT

    def label(self) -> str:
        return f"{self.kind.replace('_', ' ').title()} {self.price:g}"


def _valid(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def _swings(high: pd.Series, low: pd.Series, n: int) -> tuple[list[float], list[float]]:
    """Fractal swing highs/lows: a bar whose high (low) is the strict extreme of the n bars on each side. The last n
    bars cannot be confirmed yet, so they are never swing points."""
    highs, lows = [], []
    h, l = high.tolist(), low.tolist()
    for i in range(n, len(h) - n):
        window_h, window_l = h[i - n: i + n + 1], l[i - n: i + n + 1]
        if h[i] == max(window_h) and window_h.count(h[i]) == 1:
            highs.append(h[i])
        if l[i] == min(window_l) and window_l.count(l[i]) == 1:
            lows.append(l[i])
    return highs, lows


def _cluster(prices: list[float], tol: float) -> list[list[float]]:
    groups: list[list[float]] = []
    for p in sorted(prices):
        if groups and p - groups[-1][-1] <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return groups


def find_levels(ohlcv: pd.DataFrame, atr: float, chain_context: dict | None = None) -> list[Level]:
    """All levels for the latest bar of `ohlcv` (see the module docstring). Empty when the data is unusable."""
    if not _valid(atr) or ohlcv is None or not hasattr(ohlcv, "columns"):
        return []
    needed = {"timestamp", "high", "low", "close"}
    if not needed <= set(ohlcv.columns) or len(ohlcv) < 2 * SWING_BARS + 2:
        return []
    df = ohlcv.sort_values("timestamp").reset_index(drop=True)
    dates = df["timestamp"].dt.date
    order = list(dict.fromkeys(dates.tolist()))
    today = order[-1]
    prior = order[:-1][-LOOKBACK_SESSIONS:]
    tol = max(0.15 * atr, 1e-9)
    raw: list[Level] = []

    today_df = df[dates == today]
    if len(today_df) >= 2:
        raw.append(Level(float(today_df["high"].max()), "DAY_HIGH", 1.5))
        raw.append(Level(float(today_df["low"].min()), "DAY_LOW", 1.5))

    if prior:
        last = df[dates == prior[-1]]
        ph, pl, pc = float(last["high"].max()), float(last["low"].min()), float(last["close"].iloc[-1])
        raw.append(Level(ph, "PDH", 2.0))
        raw.append(Level(pl, "PDL", 2.0))
        p = (ph + pl + pc) / 3.0
        rng = ph - pl
        for kind, price in (("PIVOT_P", p), ("PIVOT_R1", 2 * p - pl), ("PIVOT_S1", 2 * p - ph),
                            ("PIVOT_R2", p + rng), ("PIVOT_S2", p - rng)):
            raw.append(Level(price, kind, 1.0))
        if len(prior) >= 3:
            span = df[dates.isin(prior)]
            raw.append(Level(float(span["high"].max()), "WEEK_HIGH", 2.5))
            raw.append(Level(float(span["low"].min()), "WEEK_LOW", 2.5))

    recent = df[dates.isin(prior + [today])]
    swing_highs, swing_lows = _swings(recent["high"], recent["low"], SWING_BARS)
    # Liquidity pools: swing points (plus the session extremes already found) that line up within the tolerance.
    for side, points, single_kind, pool_kind in (("H", swing_highs, "SWING_HIGH", "EQ_HIGHS"), ("L", swing_lows, "SWING_LOW", "EQ_LOWS")):
        extremes = [lv.price for lv in raw if lv.kind in (("PDH", "DAY_HIGH") if side == "H" else ("PDL", "DAY_LOW"))]
        for group in _cluster(points + extremes, tol):
            touches = len(group)
            members = [q for q in group if q in points]
            if not members:
                continue  # only session extremes in this cluster: they are already levels
            price = max(group) if side == "H" else min(group)
            if touches >= 2:
                raw.append(Level(price, pool_kind, min(2.0 + 0.5 * (touches - 2), MAX_WEIGHT), touches))
            else:
                raw.append(Level(price, single_kind, 1.0, 1))

    recent_bars = df.tail(20)
    if len(recent_bars) == 20 and float(recent_bars["high"].max() - recent_bars["low"].min()) <= 2.5 * atr:
        raw.append(Level(float(recent_bars["high"].max()), "CONSOLIDATION_HIGH", 1.5))
        raw.append(Level(float(recent_bars["low"].min()), "CONSOLIDATION_LOW", 1.5))

    if chain_context:
        # Open interest is positioning context, not proof of support or resistance: these levels are never "significant", so
        # they cannot veto a trade or move a stop. They are listed so a reader can see where the positions are.
        if _valid(chain_context.get("call_oi_peak")):
            raw.append(Level(float(chain_context["call_oi_peak"]), "OI_CALL_PEAK", 1.0))
        if _valid(chain_context.get("put_oi_peak")):
            raw.append(Level(float(chain_context["put_oi_peak"]), "OI_PUT_PEAK", 1.0))

    return _enrich(_merge(raw, MERGE_ATR * atr), df, atr)


HTF_KINDS = ("PDH", "PDL", "WEEK_", "PIVOT_")
TOUCH_ATR, REVERSAL_ATR, REACTION_LOOK_BARS, REACTION_LOOKBACK = 0.15, 0.75, 10, 400


def _reaction_record(level_price: float, high, low, close, atr: float) -> tuple[int, int | None, str]:
    """(reactions, bars since the latest, latest behaviour) for one price over the given bars. See the module docstring."""
    n = len(close)
    tol, rev = TOUCH_ATR * atr, REVERSAL_ATR * atr
    contact = [(low[i] - tol <= level_price <= high[i] + tol) for i in range(n)]
    reactions, last_reaction_end, behavior = 0, None, "NONE"
    i = 1
    while i < n:
        if not contact[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and contact[i + 1]:
            i += 1
        end = i
        i += 1
        if start == 0:
            continue
        from_below = close[start - 1] < level_price
        window_end = min(end + REACTION_LOOK_BARS, n - 1)
        if end == n - 1 or window_end == end:
            behavior = "OPEN"
            continue
        after_lo, after_hi = min(low[end + 1: window_end + 1]), max(high[end + 1: window_end + 1])
        after_close = close[end + 1: window_end + 1]
        if from_below:
            broke = max(after_close) >= level_price + rev
            turned = after_lo <= level_price - rev
            came_back = after_close[-1] < level_price
        else:
            broke = min(after_close) <= level_price - rev
            turned = after_hi >= level_price + rev
            came_back = after_close[-1] > level_price
        if broke and not (turned and came_back):
            behavior = "BROKE"
        elif turned:
            reactions += 1
            last_reaction_end = end
            behavior = "REJECTED"
        elif window_end - end < REACTION_LOOK_BARS:
            behavior = "OPEN"
    since = None if last_reaction_end is None else (n - 1 - last_reaction_end)
    return reactions, since, behavior


def _enrich(levels: list[Level], df: pd.DataFrame, atr: float) -> list[Level]:
    import dataclasses

    tail = df.tail(REACTION_LOOKBACK)
    high, low, close = tail["high"].to_numpy(float), tail["low"].to_numpy(float), tail["close"].to_numpy(float)
    out = []
    for lv in levels:
        reactions, since, behavior = _reaction_record(lv.price, high, low, close, atr)
        confluence = sum(1 for o in levels if o is not lv and abs(o.price - lv.price) <= 0.25 * atr)
        out.append(dataclasses.replace(lv, reactions=reactions, bars_since_reaction=since, behavior=behavior,
                                       htf=any(k in lv.kind for k in HTF_KINDS), confluence=confluence))
    return out


def _merge(levels: list[Level], tol: float) -> list[Level]:
    merged: list[Level] = []
    for lv in sorted((x for x in levels if _valid(x.price)), key=lambda x: x.price):
        if merged and lv.price - merged[-1].price <= tol:
            prev = merged[-1]
            kinds = prev.kind if lv.kind in prev.kind.split("+") else f"{prev.kind}+{lv.kind}"
            top = lv if lv.weight > prev.weight else prev
            merged[-1] = Level(top.price, kinds, min(prev.weight + lv.weight, MAX_WEIGHT), prev.touches + lv.touches)
        else:
            merged.append(lv)
    return merged


@dataclass
class LevelAssessment:
    ok: bool
    reason: str  # why the trade is vetoed (empty when ok)
    stop: float
    target: float
    stop_changed: bool = False
    target_capped: bool = False
    opposing: Level | None = None
    supporting: Level | None = None
    room_r: float | None = None  # distance to the opposing level in stop distances
    notes: list[str] = field(default_factory=list)
    levels_found: int = 0

    @property
    def summary(self) -> str:
        bits = []
        if self.opposing is not None:
            bits.append(f"{self.opposing.label()} ahead ({self.room_r:.1f}R)")
        if self.stop_changed and self.supporting is not None:
            bits.append(f"stop behind {self.supporting.label()}")
        if self.target_capped:
            bits.append("target held short of the level")
        bits.extend(self.notes)
        return "Levels: " + ("; ".join(bits) if bits else f"{self.levels_found} levels, nothing in the way")


def _recent_sweep(ohlcv: pd.DataFrame, levels: list[Level], sign: int, atr: float) -> str | None:
    """A level on the trade's own side that one of the last 3 bars pierced and the latest close is back beyond: the
    resting stops were run and price came back (a liquidity sweep and reclaim)."""
    if ohlcv is None or len(ohlcv) < 3:
        return None
    tail = ohlcv.tail(3)
    close = float(tail["close"].iloc[-1])
    for lv in levels:
        if not lv.significant:
            continue
        if sign > 0 and lv.price < close and float(tail["low"].min()) < lv.price - 0.02 * atr:
            return f"swept {lv.label()} and reclaimed it"
        if sign < 0 and lv.price > close and float(tail["high"].max()) > lv.price + 0.02 * atr:
            return f"swept {lv.label()} and reclaimed it"
    return None


def assess_trade(direction: str, entry: float, stop: float, target: float, levels: list[Level], atr: float,
                 min_room_mult: float = 1.5, buffer_atr: float = 0.15, ohlcv: pd.DataFrame | None = None) -> LevelAssessment:
    """Check a setup (prices on the underlying) against `levels`; see the module docstring.

    Works in a space where the trade is long (prices multiplied by `sign`), so SHORT is the mirror image."""
    base = LevelAssessment(True, "", stop, target, levels_found=len(levels))
    if direction not in ("LONG", "SHORT") or not all(_valid(x) for x in (entry, stop, target, atr)):
        return base
    sign = 1 if direction == "LONG" else -1
    e, s, t = sign * entry, sign * stop, sign * target
    stop_dist = e - s
    if stop_dist <= 0 or t <= e:
        return base
    buf = buffer_atr * atr
    eps = 0.05 * atr
    sig = [(sign * lv.price, lv) for lv in levels if lv.significant]

    # Room to the nearest significant level against the trade.
    ahead = sorted((p, lv) for p, lv in sig if p > e + eps)
    if ahead:
        p, lv = ahead[0]
        base.opposing, base.room_r = lv, (p - e) / stop_dist
        if p - e < min_room_mult * stop_dist:
            base.ok = False
            base.reason = (f"no room: {lv.label()} is only {p - e:.2f} away ({base.room_r:.1f}x the {stop_dist:.2f} stop distance, "
                           f"needs {min_room_mult:g}x)")
            return base
        if t > p - buf:
            capped = p - buf
            if capped - e < 1.2 * stop_dist:
                base.ok = False
                base.reason = f"no room: {lv.label()} leaves less than 1.2x the stop distance once the buffer is kept"
                return base
            base.target, base.target_capped = sign * capped, True

    # Structure stop: the nearest significant level between the stop and the entry, at least MIN_STOP_ATR from entry.
    between = sorted(((p, lv) for p, lv in sig if s < p - buf and (e - (p - buf)) >= MIN_STOP_ATR * atr and p < e), reverse=True)
    if between:
        p, lv = between[0]
        base.stop, base.stop_changed, base.supporting = sign * (p - buf), True, lv

    below = [(p, lv) for p, lv in sig if p <= e and e - p <= 0.5 * atr]
    if below:
        base.notes.append(f"entry at {max(below, key=lambda x: x[0])[1].label()}")
    sweep = _recent_sweep(ohlcv, levels, sign, atr) if ohlcv is not None else None
    if sweep:
        base.notes.append(sweep)
    return base
