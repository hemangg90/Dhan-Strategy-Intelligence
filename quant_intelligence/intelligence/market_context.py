"""
MarketContext: the underlying's state at one decision bar, as one typed, reproducible object.

Built from price history only. `info_cutoff` is the last bar used; nothing after it is read, and the structure and
liquidity detectors only act on bars up to it (see `features/market_structure.py`). `parameters` records every setting
that shaped the result, so the same inputs and parameters rebuild the same context.

The regime scores are the regime engine's HEURISTIC scores (they sum to 1 but are not calibrated probabilities);
they are stored under `regime["scores_heuristic"]` so nobody mistakes them for probabilities.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_intelligence.features.levels import Level, find_levels
from quant_intelligence.features.liquidity import LiquidityConfig, LiquidityReport, analyze_liquidity
from quant_intelligence.features.market_structure import StructureConfig, StructureReport, analyze_structure
from quant_intelligence.features.vwap_volume import volume_profile, vwap_features
from quant_intelligence.intelligence.status import DataStatus, critical, good, warning, worst

MIN_BARS = 60


@dataclass(frozen=True)
class MarketContext:
    timestamp: pd.Timestamp
    underlying: str
    timeframe: str
    info_cutoff: pd.Timestamp
    first_bar: pd.Timestamp
    bars_used: int
    last_close: float | None
    regime: dict
    trend: dict
    volatility: dict
    volume: dict
    vwap: dict
    volume_profile: dict
    levels: tuple[Level, ...]
    nearest: dict  # nearest significant support / resistance with distance in ATR
    structure: StructureReport
    liquidity: LiquidityReport
    status: DataStatus
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.underlying:
            raise ValueError("a MarketContext needs an underlying")
        if self.info_cutoff > self.timestamp:
            raise ValueError("information cutoff is after the decision time")

    @property
    def atr(self) -> float | None:
        return self.volatility.get("atr_14")

    def as_record(self) -> dict:
        """A JSON-safe snapshot for storage and research reproducibility."""
        s = self.structure
        return _json_safe({
            "timestamp": self.timestamp, "underlying": self.underlying, "timeframe": self.timeframe,
            "info_cutoff": self.info_cutoff, "first_bar": self.first_bar, "bars_used": self.bars_used, "last_close": self.last_close,
            "regime": self.regime, "trend": self.trend, "volatility": self.volatility, "volume": self.volume,
            "vwap": self.vwap, "volume_profile": self.volume_profile,
            "levels": [dataclasses.asdict(lv) for lv in self.levels], "nearest": self.nearest,
            "structure": {
                "trend": s.trend, "last_swing_high": s.last_swing_high.price if s.last_swing_high else None,
                "last_swing_low": s.last_swing_low.price if s.last_swing_low else None,
                "events": [dataclasses.asdict(e) for e in s.events[-5:]],
                "open_fvgs": [dataclasses.asdict(g) for g in s.open_fvgs()],
                "displacements": [dataclasses.asdict(d) for d in s.displacements[-5:]],
            },
            "liquidity": {
                "standing": [dataclasses.asdict(z) for z in self.liquidity.standing],
                "active_events": [dataclasses.asdict(e) for e in self.liquidity.active_events()],
            },
            "status": self.status.to_dict(), "parameters": self.parameters,
        })


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def validate_ohlcv_for_context(ohlcv: pd.DataFrame, max_age_minutes: float | None = None,
                               now: pd.Timestamp | None = None) -> DataStatus:
    """The checks a context needs before anything is computed from the bars."""
    comp = "price_history"
    if ohlcv is None or not hasattr(ohlcv, "columns"):
        return critical(comp, "no price history")
    missing = {"timestamp", "open", "high", "low", "close"} - set(ohlcv.columns)
    if missing:
        return critical(comp, f"missing columns {sorted(missing)}")
    if len(ohlcv) < MIN_BARS:
        return critical(comp, f"only {len(ohlcv)} bars; at least {MIN_BARS} are needed")
    ts = ohlcv["timestamp"]
    if not ts.is_monotonic_increasing:
        return critical(comp, "timestamps are not in order")
    if ts.duplicated().any():
        return critical(comp, "duplicate timestamps")
    prices = ohlcv[["open", "high", "low", "close"]].tail(MIN_BARS).to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        return critical(comp, "invalid prices (missing, zero or negative) in the recent bars")
    if (ohlcv["high"].tail(MIN_BARS) < ohlcv["low"].tail(MIN_BARS)).any():
        return critical(comp, "a bar has its high below its low")
    if max_age_minutes is not None and now is not None:
        age = (now - ts.iloc[-1]).total_seconds() / 60.0
        if age > max_age_minutes:
            return warning(comp, f"the last bar is {age:.0f} minutes old (limit {max_age_minutes:g})")
    return good(comp)


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _nearest(levels: list[Level], price: float, atr: float | None) -> dict:
    out = {"support": None, "resistance": None}
    sig = [lv for lv in levels if lv.significant]
    below = [lv for lv in sig if lv.price <= price]
    above = [lv for lv in sig if lv.price > price]
    for key, pool, pick in (("support", below, max), ("resistance", above, min)):
        if pool:
            lv = pick(pool, key=lambda x: x.price)
            dist = abs(lv.price - price)
            out[key] = {"price": lv.price, "kind": lv.kind, "weight": lv.weight, "distance": dist,
                        "distance_atr": (dist / atr) if atr else None}
    return out


def build_market_context(
    ohlcv: pd.DataFrame,
    *,
    underlying: str,
    timeframe: str = "5min",
    features: pd.DataFrame | None = None,
    regime: dict | None = None,
    chain_context: dict | None = None,
    data_status: DataStatus | None = None,
    structure_cfg: StructureConfig | None = None,
    liquidity_cfg: LiquidityConfig | None = None,
    max_bars: int = 1500,
    max_age_minutes: float | None = None,
    now: pd.Timestamp | None = None,
) -> MarketContext:
    """Assemble the context at the last bar of `ohlcv`. A CRITICAL status (bad or too little data) returns a context
    whose `status` says so and whose analytics are empty - callers must check `status.ok` before using it."""
    structure_cfg = structure_cfg or StructureConfig()
    liquidity_cfg = liquidity_cfg or LiquidityConfig()
    params = {"structure": dataclasses.asdict(structure_cfg), "liquidity": dataclasses.asdict(liquidity_cfg),
              "max_bars": max_bars, "max_age_minutes": max_age_minutes}
    check = validate_ohlcv_for_context(ohlcv, max_age_minutes, now)
    statuses = [check] + ([data_status] if data_status is not None else [])
    if not check.ok:
        ts = ohlcv["timestamp"].iloc[-1] if ohlcv is not None and len(ohlcv) and "timestamp" in ohlcv else pd.Timestamp(0)
        empty_struct = StructureReport(0, ts, "UNDEFINED")
        return MarketContext(ts, underlying, timeframe, ts, ts, 0, None, {}, {}, {}, {}, {}, {}, (), {}, empty_struct,
                             LiquidityReport(0, [], [], []), worst("market_context", statuses), params)

    df = ohlcv.tail(max_bars).reset_index(drop=True)
    structure = analyze_structure(df, structure_cfg)
    liquidity = analyze_liquidity(df, structure_cfg, liquidity_cfg)

    row = {}
    if features is not None and len(features):
        row = features.iloc[-1].to_dict()
    atr = _num(row.get("atr_14"))
    if atr is None:  # features not supplied: derive the ATR from the bars themselves
        prev = df["close"].shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
        atr = _num(tr.rolling(14, min_periods=14).mean().iloc[-1])
    price = float(df["close"].iloc[-1])
    levels = find_levels(df, atr, chain_context) if atr else []

    regime = regime or {}
    regime_block = {"label": regime.get("label"), "axes": regime.get("axes", {}), "clarity": _num(regime.get("clarity")),
                    "age_bars": regime.get("age_bars"), "scores_heuristic": regime.get("scores", {}),
                    "note": "scores are the regime engine's heuristic scores, not calibrated probabilities"}
    trend_block = {"structure_trend": structure.trend, "trend_z": _num(row.get("trend_z")), "momentum_z": _num(row.get("momentum_z")),
                   "trend_slope": _num(row.get("trend_slope")), "market_structure_feature": _num(row.get("market_structure"))}
    vol_block = {"atr_14": atr, "atr_pct_of_price": (atr / price) if atr else None,
                 "atr_percentile_100": _num(row.get("atr_percentile_100")), "vol_expansion": _num(row.get("vol_expansion")),
                 "realized_vol_20": _num(row.get("realized_vol_20"))}
    rel = _num(row.get("relative_volume"))
    volume_block = {"relative_volume": rel, "expanding": (rel > 1.5) if rel is not None else None}
    vwap = vwap_features(df, atr)
    profile = volume_profile(df)

    unavailable = tuple(k for k, v in (("vwap", vwap), ("volume_profile", profile)) if not v["available"])
    if unavailable:
        statuses.append(warning("volume", "no usable volume: " + vwap["reason"], unavailable))
    if atr is None:
        statuses.append(warning("volatility", "ATR is not available yet", ("atr_14",)))
    ts = df["timestamp"].iloc[-1]
    return MarketContext(
        timestamp=ts, underlying=underlying, timeframe=timeframe, info_cutoff=ts, first_bar=df["timestamp"].iloc[0],
        bars_used=len(df), last_close=price, regime=regime_block, trend=trend_block, volatility=vol_block, volume=volume_block, vwap=vwap,
        volume_profile=profile, levels=tuple(levels), nearest=_nearest(levels, price, atr), structure=structure,
        liquidity=liquidity, status=worst("market_context", statuses) if len(statuses) > 1 else statuses[0], parameters=params,
    )


def context_from_pipeline(output, chain=None, **kw) -> MarketContext:
    """Build the context for a `PipelineOutput` (the application's own pipeline result)."""
    chain_ctx = None
    if chain is not None and hasattr(chain, "strikes"):
        from quant_intelligence.options.contract_scoring import chain_context

        chain_ctx = chain_context(chain)
    regime = {"label": getattr(output, "regime_label", None), "axes": getattr(output, "regime_axes", {}),
              "clarity": getattr(output, "regime_clarity", None), "age_bars": getattr(output, "regime_age_bars", None),
              "scores": getattr(output, "regime_probabilities", {})}
    quality = getattr(output, "data_quality_status", "OK")
    status = None if quality == "OK" else (critical if quality == "FAIL" else warning)(
        "data_quality", f"data quality status is {quality}")
    kw.setdefault("timeframe", (getattr(output, "provenance", None) or {}).get("timeframe") or "5min")
    return build_market_context(output.ohlcv, underlying=output.instrument, features=getattr(output, "features", None),
                                regime=regime, chain_context=chain_ctx, data_status=status, **kw)
