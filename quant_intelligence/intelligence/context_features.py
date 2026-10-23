"""
Turn a MarketContext / OptionsContext into the flat feature dict the analogue engine compares.

The names match `analogues.analogue_engine.STRUCTURE_FEATURES` and `OPTIONS_FEATURES`. A value that is not available is
simply absent (None): the analogue engine then leaves that feature out of the comparison and says so, instead of
comparing against an invented number.

The structure features here equal the last row of `features.market_structure.structure_feature_frame`, so the live decision
and the historical observations are described in the same terms.
"""
from __future__ import annotations

import math

from quant_intelligence.features.market_structure import MSS
from quant_intelligence.intelligence.market_context import MarketContext
from quant_intelligence.intelligence.options_context import OptionsContext


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def market_context_features(ctx: MarketContext) -> dict:
    if not ctx.status.ok or ctx.bars_used == 0:
        return {}
    s = ctx.structure
    atr = ctx.atr
    price = ctx.last_close
    out: dict = {"structure_trend_code": {"UP": 1.0, "DOWN": -1.0}.get(s.trend, 0.0)}
    if atr and price is not None:
        if s.last_swing_high is not None:
            out["dist_swing_high_atr"] = (s.last_swing_high.price - price) / atr
        if s.last_swing_low is not None:
            out["dist_swing_low_atr"] = (price - s.last_swing_low.price) / atr
    ev = s.last_event
    out["bars_since_structure_event"] = float(min(s.as_of_index - ev.break_index, 50)) if ev is not None else 50.0
    out["last_event_is_mss"] = 1.0 if ev is not None and ev.kind == MSS else 0.0
    return out


def options_context_features(ctx: OptionsContext) -> dict:
    if not ctx.status.ok or not ctx.quotes:
        return {}
    out: dict = {}
    if ctx.iv.available and ctx.iv.iv_rank is not None:
        out["iv_rank"] = ctx.iv.iv_rank
    if ctx.atm_iv is not None:
        out["atm_iv"] = ctx.atm_iv
    if ctx.iv_vs_realized.get("available"):
        out["iv_minus_rv"] = ctx.iv_vs_realized["iv_minus_rv"]
    if ctx.oi.pcr_oi is not None:
        out["pcr_oi"] = ctx.oi.pcr_oi
    if ctx.oi.pcr_oi_window is not None:
        out["pcr_oi_window"] = ctx.oi.pcr_oi_window
    spread = ctx.liquidity.get("median_spread_pct")
    if spread is not None:
        out["atm_spread_pct"] = spread
    if ctx.expiry_info.available and ctx.expiry_info.trading_days is not None:
        out["dte_trading_days"] = ctx.expiry_info.trading_days
    return {k: v for k, v in ((k, _num(v)) for k, v in out.items()) if v is not None}


def current_feature_vector(base: dict, market: MarketContext | None = None, options: OptionsContext | None = None) -> dict:
    """The analogue query: the pipeline's own market-state features plus whatever the two contexts add."""
    out = dict(base)
    if market is not None:
        out.update(market_context_features(market))
    if options is not None:
        out.update(options_context_features(options))
    return out
