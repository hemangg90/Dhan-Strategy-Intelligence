"""
Market structure: swing points, Break of Structure (BOS), Market Structure Shift (MSS), displacement and Fair Value Gaps.

Every definition is objective and configurable (`StructureConfig`). Every event carries two moments:
  * the bar it happened on, and
  * `known_at`: the earliest bar at which it could legitimately be known.
A swing high at bar i needs `swing_bars` bars after it to be recognised, so its `known_at` is bar i + swing_bars; a
future bar can never confirm a historical pivot early. `analyze_structure(df, as_of_index=t)` uses only bars <= t, and
only swings whose confirmation bar is <= t, so the answer at t cannot change when later bars arrive.

Definitions
-----------
Swing high (low): a bar whose high (low) is strictly above (below) the previous `swing_bars` highs (lows) and at least
    the next `swing_bars`. Confirmed `swing_bars` bars later.
BOS:  a bar CLOSES beyond (or, with break_on="wick", trades beyond) the most recent confirmed swing in the direction of
    the prevailing trend (or with no trend yet, the first break). Each swing can be broken once.
MSS:  the same break, but AGAINST the prevailing trend: a close above the last confirmed swing high while the trend is
    down (or below the last swing low while it is up). The prevailing trend then flips. This is the common simplified
    definition (it does not require a prior displacement leg, which is reported separately).
Displacement: a bar whose body is at least `displacement_atr` x ATR and which closes in the outer
    `displacement_close_frac` of its range in its direction.
Fair Value Gap: three bars where bar i-2 and bar i do not overlap: bullish if low[i] > high[i-2] (zone high[i-2]..low[i]),
    bearish if high[i] < low[i-2] (zone high[i]..low[i-2]); known at bar i; zone at least `fvg_min_atr` x ATR.
    Status at the decision bar: UNMITIGATED (never traded into), PARTIAL (traded part-way in), MITIGATED (the whole
    zone traded through by a wick), INVALIDATED (a bar CLOSED through the far side) or EXPIRED (older than
    `fvg_max_age_bars` without being mitigated).

Limits: these are price-pattern definitions on OHLCV. They do not show who is positioned, only what price did.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

UP, DOWN, UNDEFINED = "UP", "DOWN", "UNDEFINED"
BOS, MSS = "BOS", "MSS"
BULL, BEAR = "BULL", "BEAR"
UNMITIGATED, PARTIAL, MITIGATED, INVALIDATED, EXPIRED = "UNMITIGATED", "PARTIAL", "MITIGATED", "INVALIDATED", "EXPIRED"


@dataclass(frozen=True)
class StructureConfig:
    swing_bars: int = 3
    break_on: str = "close"  # "close" or "wick"
    atr_period: int = 14
    displacement_atr: float = 1.5
    displacement_close_frac: float = 0.25
    fvg_min_atr: float = 0.1
    fvg_max_age_bars: int = 60

    def __post_init__(self):
        if self.swing_bars < 1:
            raise ValueError("swing_bars must be at least 1")
        if self.break_on not in ("close", "wick"):
            raise ValueError("break_on must be 'close' or 'wick'")
        if not 0 < self.displacement_close_frac <= 0.5:
            raise ValueError("displacement_close_frac must be in (0, 0.5]")


@dataclass(frozen=True)
class SwingPoint:
    kind: str  # HIGH / LOW
    price: float
    index: int  # bar of the extreme
    timestamp: pd.Timestamp
    confirmed_index: int  # first bar at which the swing is known
    known_at: pd.Timestamp


@dataclass(frozen=True)
class StructureEvent:
    kind: str  # BOS / MSS
    direction: str  # BULL / BEAR
    level: float  # price of the swing that was broken
    swing_index: int
    break_index: int
    detected_at: pd.Timestamp
    known_at: pd.Timestamp
    initial: bool = False  # the first break with no prevailing trend


@dataclass(frozen=True)
class Displacement:
    direction: str
    index: int
    timestamp: pd.Timestamp
    body_atr: float


@dataclass(frozen=True)
class FairValueGap:
    direction: str  # BULL / BEAR
    low: float
    high: float
    index: int  # the third bar, when the gap is known
    known_at: pd.Timestamp
    status: str = UNMITIGATED
    fill_pct: float = 0.0
    status_index: int | None = None

    @property
    def open(self) -> bool:
        """Still a live zone: not mitigated, invalidated or expired."""
        return self.status in (UNMITIGATED, PARTIAL)


@dataclass
class StructureReport:
    as_of_index: int
    as_of: pd.Timestamp
    trend: str
    swings: list[SwingPoint] = field(default_factory=list)  # confirmed by as_of
    events: list[StructureEvent] = field(default_factory=list)
    displacements: list[Displacement] = field(default_factory=list)
    fvgs: list[FairValueGap] = field(default_factory=list)
    last_swing_high: SwingPoint | None = None
    last_swing_low: SwingPoint | None = None

    @property
    def last_event(self) -> StructureEvent | None:
        return self.events[-1] if self.events else None

    def open_fvgs(self) -> list[FairValueGap]:
        return [g for g in self.fvgs if g.open]


def _atr(df: pd.DataFrame, period: int) -> np.ndarray:
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean().to_numpy()


def detect_swings(df: pd.DataFrame, swing_bars: int = 3) -> list[SwingPoint]:
    """All swing points in `df`, each with the bar at which it becomes known. Pass a frame that already ends at the
    decision bar (or use `analyze_structure(..., as_of_index=)`); swings whose confirmation bar lies beyond it are not
    returned because their `swing_bars` following bars do not exist."""
    n = swing_bars
    high, low = df["high"].to_numpy(float), df["low"].to_numpy(float)
    ts = df["timestamp"].tolist()
    out: list[SwingPoint] = []
    for i in range(n, len(df) - n):
        before_h, after_h = high[i - n: i], high[i + 1: i + n + 1]
        before_l, after_l = low[i - n: i], low[i + 1: i + n + 1]
        c = i + n
        if high[i] > before_h.max() and high[i] >= after_h.max():
            out.append(SwingPoint("HIGH", float(high[i]), i, ts[i], c, ts[c]))
        if low[i] < before_l.min() and low[i] <= after_l.min():
            out.append(SwingPoint("LOW", float(low[i]), i, ts[i], c, ts[c]))
    out.sort(key=lambda s: (s.confirmed_index, s.index))
    return out


def _scan(data: pd.DataFrame, cfg: StructureConfig, swings: list[SwingPoint]):
    """One causal pass over the bars: swings become known at their confirmation bar, a close through the latest known swing
    is a BOS or MSS. Returns (events, final trend, final last high, final last low, per-bar state) where per-bar state is a
    list of (trend, last_high, last_low, index of the latest event or -1) AFTER each bar."""
    ts = data["timestamp"].tolist()
    h, c, l = data["high"].to_numpy(float), data["close"].to_numpy(float), data["low"].to_numpy(float)
    events: list[StructureEvent] = []
    trend = UNDEFINED
    last_high: SwingPoint | None = None
    last_low: SwingPoint | None = None
    broken: set[tuple[str, int]] = set()
    pending = list(swings)
    p = 0
    last_event = -1
    state = []
    for i in range(len(data)):
        while p < len(pending) and pending[p].confirmed_index <= i:
            sw = pending[p]
            if sw.kind == "HIGH":
                last_high = sw
            else:
                last_low = sw
            p += 1
        up_price = h[i] if cfg.break_on == "wick" else c[i]
        dn_price = l[i] if cfg.break_on == "wick" else c[i]
        if last_high is not None and ("HIGH", last_high.index) not in broken and up_price > last_high.price:
            kind = MSS if trend == DOWN else BOS
            events.append(StructureEvent(kind, BULL, last_high.price, last_high.index, i, ts[i], ts[i], initial=(trend == UNDEFINED)))
            broken.add(("HIGH", last_high.index))
            trend = UP
            last_event = i
        elif last_low is not None and ("LOW", last_low.index) not in broken and dn_price < last_low.price:
            kind = MSS if trend == UP else BOS
            events.append(StructureEvent(kind, BEAR, last_low.price, last_low.index, i, ts[i], ts[i], initial=(trend == UNDEFINED)))
            broken.add(("LOW", last_low.index))
            trend = DOWN
            last_event = i
        state.append((trend, last_high, last_low, last_event))
    return events, trend, last_high, last_low, state


def structure_feature_frame(df: pd.DataFrame, cfg: StructureConfig | None = None) -> pd.DataFrame:
    """Causal per-bar structure features for every bar of `df`, in ONE pass (so it is cheap enough to attach to every
    historical observation). Row i uses only information known at bar i - the same values `analyze_structure(df, as_of_index=i)`
    would give. Columns:
      structure_trend_code      +1 trend up, -1 down, 0 undefined
      dist_swing_high_atr       (last confirmed swing high - close) / ATR   (NaN until one exists)
      dist_swing_low_atr        (close - last confirmed swing low) / ATR
      bars_since_structure_event  bars since the latest BOS/MSS, capped at 50 (50 when none)
      last_event_is_mss         1 if the latest event was an MSS, else 0
    """
    cfg = cfg or StructureConfig()
    missing = {"timestamp", "open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    data = df.reset_index(drop=True)
    swings = detect_swings(data, cfg.swing_bars)
    events, _, _, _, state = _scan(data, cfg, swings)
    atr = _atr(data, cfg.atr_period)
    close = data["close"].to_numpy(float)
    by_break = {e.break_index: e for e in events}
    n = len(data)
    out = {k: np.full(n, np.nan) for k in ("structure_trend_code", "dist_swing_high_atr", "dist_swing_low_atr")}
    since = np.full(n, 50.0)
    mss = np.zeros(n)
    for i, (trend, hi, lo, last_event) in enumerate(state):
        out["structure_trend_code"][i] = {UP: 1.0, DOWN: -1.0}.get(trend, 0.0)
        if np.isfinite(atr[i]) and atr[i] > 0:
            if hi is not None:
                out["dist_swing_high_atr"][i] = (hi.price - close[i]) / atr[i]
            if lo is not None:
                out["dist_swing_low_atr"][i] = (close[i] - lo.price) / atr[i]
        if last_event >= 0:
            since[i] = min(i - last_event, 50)
            mss[i] = 1.0 if by_break[last_event].kind == MSS else 0.0
    frame = pd.DataFrame({"timestamp": data["timestamp"], **out, "bars_since_structure_event": since, "last_event_is_mss": mss})
    return frame


def analyze_structure(df: pd.DataFrame, cfg: StructureConfig | None = None, as_of_index: int | None = None) -> StructureReport:
    """Structure as known at bar `as_of_index` (default the last bar). Needs columns timestamp, open, high, low, close."""
    cfg = cfg or StructureConfig()
    missing = {"timestamp", "open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if len(df) == 0:
        raise ValueError("no bars")
    t = len(df) - 1 if as_of_index is None else int(as_of_index)
    if not 0 <= t < len(df):
        raise ValueError("as_of_index outside the data")
    data = df.iloc[: t + 1].reset_index(drop=True)  # nothing after the decision bar is ever looked at
    ts = data["timestamp"].tolist()
    o, h, l, c = (data[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = _atr(data, cfg.atr_period)

    swings = detect_swings(data, cfg.swing_bars)
    events, trend, last_high, last_low, _ = _scan(data, cfg, swings)

    displacements: list[Displacement] = []
    for i in range(len(data)):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0 or body < cfg.displacement_atr * a:
            continue
        if c[i] > o[i] and (h[i] - c[i]) <= cfg.displacement_close_frac * rng:
            displacements.append(Displacement(BULL, i, ts[i], float(body / a)))
        elif c[i] < o[i] and (c[i] - l[i]) <= cfg.displacement_close_frac * rng:
            displacements.append(Displacement(BEAR, i, ts[i], float(body / a)))

    fvgs: list[FairValueGap] = []
    for i in range(2, len(data)):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= cfg.fvg_min_atr * a:
            fvgs.append(_evaluate_fvg(BULL, float(h[i - 2]), float(l[i]), i, ts[i], data, t, cfg))
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) >= cfg.fvg_min_atr * a:
            fvgs.append(_evaluate_fvg(BEAR, float(h[i]), float(l[i - 2]), i, ts[i], data, t, cfg))

    return StructureReport(
        as_of_index=t, as_of=ts[t], trend=trend, swings=swings, events=events, displacements=displacements, fvgs=fvgs,
        last_swing_high=last_high, last_swing_low=last_low,
    )


def _evaluate_fvg(direction: str, lo: float, hi: float, i: int, known_at, data: pd.DataFrame, t: int, cfg: StructureConfig) -> FairValueGap:
    """Walk the bars after the gap up to the decision bar and settle its status."""
    height = hi - lo
    h, l, c = data["high"].to_numpy(float), data["low"].to_numpy(float), data["close"].to_numpy(float)
    fill = 0.0
    status, status_index = UNMITIGATED, None
    for j in range(i + 1, t + 1):
        if direction == BULL:
            if c[j] < lo:
                status, status_index = INVALIDATED, j
                break
            fill = max(fill, min((hi - l[j]) / height, 1.0)) if l[j] < hi else fill
            if l[j] <= lo:
                status, status_index = MITIGATED, j
                fill = 1.0
                break
        else:
            if c[j] > hi:
                status, status_index = INVALIDATED, j
                break
            fill = max(fill, min((h[j] - lo) / height, 1.0)) if h[j] > lo else fill
            if h[j] >= hi:
                status, status_index = MITIGATED, j
                fill = 1.0
                break
        if fill > 0 and status == UNMITIGATED:
            status, status_index = PARTIAL, j
    if status in (UNMITIGATED, PARTIAL) and (t - i) > cfg.fvg_max_age_bars:
        status, status_index = EXPIRED, i + cfg.fvg_max_age_bars + 1
    return FairValueGap(direction, lo, hi, i, known_at, status, float(fill), status_index)
