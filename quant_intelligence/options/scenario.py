"""
Scenario analysis for one option contract: what would one lot make or lose if the underlying, its volatility and the clock
move, after realistic fills and costs?

The contract is re-priced with the application's own model (`options/black_scholes.py`; BSM for NSE/BSE, Black-76 for MCX;
European exercise, flat volatility) at the implied volatility solved from its own price (`options/greeks.py`). Exit fills
are conservative: the re-priced value less half the quoted spread (as a share of price) and the slippage ticks; entry is
`ExecutionQuality.entry_fill`. Charges come from the contract-note model.

Two scenario sets, both labelled as assumptions, not forecasts:
  neutral distribution   spot moves ~ N(0, sigma_move) over the horizon, discretised at -2, -1, -0.5, 0, +0.5, +1, +2 sigma
                         with normal weights; implied volatility shifts by -15%, 0, +15% (relative) with weights 25/50/25.
                         sigma_move = the stored volatility-model forecast for the horizon when a fresh one exists, else
                         ATR x sqrt(bars) (random-walk scaling). The mean of this set is what the contract earns with NO
                         view: for a bought option it is normally negative (decay and costs), and that is the honest base.
  strategy path          the trade as the strategy states it: the underlying reaches its target (probability p) or its
                         stop (1 - p) after the horizon. p is the comparable observations' share of positive outcomes;
                         its lower bound (Wilson, at the observations' effective sample size) is used for ranking, so a
                         thin sample cannot make a contract look good. Not available when p is not given.
Downside = the 10% tail (CVaR10) of the neutral set, and the worst scenario. For a BOUGHT option the loss cannot exceed the
premium paid plus costs; for a WRITTEN option the worst scenario is only the worst one tried, not a bound.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np

from quant_intelligence.config.expiry_rules import time_to_expiry_years
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.charges import option_trade_charges
from quant_intelligence.options import black_scholes as bs
from quant_intelligence.options.contracts import OptionQuote
from quant_intelligence.options.execution_quality import ExecutionQuality
from quant_intelligence.options.greeks import GreekSet, rates_for
from quant_intelligence.utils.market_profile import profile_for

SPOT_Z = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
IV_SHIFTS = ((-0.15, 0.25), (0.0, 0.50), (0.15, 0.25))


@dataclass(frozen=True)
class ScenarioInputs:
    spot: float
    atr: float  # underlying ATR per bar
    now: dt.datetime
    horizon_bars: int = 12
    bar_minutes: float = 5.0
    sigma_move: float | None = None  # underlying move over the horizon (price units); None = ATR x sqrt(bars)
    direction: str = "LONG"
    target_price: float | None = None  # underlying prices of the strategy path
    stop_price: float | None = None
    win_prob: float | None = None
    effective_n: float | None = None


@dataclass(frozen=True)
class ScenarioResult:
    available: bool
    reason: str = ""
    entry: float | None = None
    quantity: int | None = None
    move_basis: str = ""
    sigma_move: float | None = None
    expected_neutral: float | None = None  # per unit, net
    expected_neutral_return: float | None = None  # as a share of the entry premium
    cvar10: float | None = None  # per unit, net, neutral set
    worst: float | None = None
    best: float | None = None
    pnl_target: float | None = None
    pnl_stop: float | None = None
    win_prob: float | None = None
    win_prob_lower: float | None = None
    expected_strategy: float | None = None  # at win_prob
    expected_strategy_lower: float | None = None  # at the lower bound: the figure used for ranking
    expected_strategy_return: float | None = None
    objective: float | None = None
    rows: tuple = ()
    assumptions: tuple[str, ...] = field(default_factory=tuple)


def wilson_lower(p: float, n: float, z: float = 1.64) -> float:
    """Lower end of a Wilson interval for a proportion p seen over n effective observations (one-sided about 95%)."""
    if n <= 0:
        return 0.0
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def _weights(zs) -> np.ndarray:
    w = np.exp(-0.5 * np.asarray(zs, float) ** 2)
    return w / w.sum()


def _exit_fill(value: float, rel_spread_pct: float | None, transaction: str) -> float:
    """Conservative fill when closing at re-priced `value`: give up half the spread and the slippage ticks."""
    half = (rel_spread_pct or 0.0) / 200.0
    tick = SETTINGS.costs.slippage_ticks * SETTINGS.costs.tick_size
    if transaction == "BUY":
        return max(value * (1 - half) - tick, 0.0)
    return value * (1 + half) + tick


def evaluate_scenarios(quote: OptionQuote, greeks: GreekSet, inputs: ScenarioInputs, eq: ExecutionQuality,
                       transaction: str = "BUY", quantity: int | None = None) -> ScenarioResult:
    if not greeks.available or greeks.iv is None or greeks.t_years is None:
        return ScenarioResult(False, "scenario analysis needs the contract's Greeks: " + (greeks.reason or "unavailable"))
    if eq.entry_fill is None or eq.entry_fill <= 0:
        return ScenarioResult(False, "no conservative entry fill (the quote is unusable)")
    if not (inputs.spot > 0 and inputs.atr and inputs.atr > 0):
        return ScenarioResult(False, "the underlying price or ATR is not available")
    quantity = quantity or quote.lot_size or 1
    kind = bs.CALL if quote.option_type == "CE" else bs.PUT
    r, q, _ = rates_for(quote.underlying)
    profile = profile_for(quote.underlying)
    horizon_years = (inputs.horizon_bars * inputs.bar_minutes / profile.session_minutes) / SETTINGS.trading_days_per_year
    t_after = max(greeks.t_years - horizon_years, 0.0)
    sigma_move = inputs.sigma_move
    basis = "volatility-model forecast"
    if sigma_move is None or not sigma_move > 0:
        sigma_move = inputs.atr * math.sqrt(max(inputs.horizon_bars, 1))
        basis = "ATR x sqrt(bars)"
    entry = eq.entry_fill
    sign = 1.0 if transaction == "BUY" else -1.0

    def pnl_at(spot_after: float, iv_mult: float) -> float:
        if spot_after <= 0:
            spot_after = 1e-9
        value = bs.price(spot_after, quote.strike, t_after, r, q, max(greeks.iv * iv_mult, 1e-4), kind)
        exit_ = _exit_fill(value, eq.rel_spread_pct, transaction)
        charges = option_trade_charges(entry, exit_, quantity, transaction, profile.name).total / quantity
        return sign * (exit_ - entry) - charges

    rows, pnls, wts = [], [], []
    spot_w = _weights(SPOT_Z)
    for zi, z in enumerate(SPOT_Z):
        for shift, iv_w in IV_SHIFTS:
            s_after = inputs.spot + z * sigma_move
            pnl = pnl_at(s_after, 1.0 + shift)
            w = float(spot_w[zi] * iv_w)
            rows.append({"scenario": f"spot {z:+.1f}σ, IV {shift:+.0%}", "spot": s_after, "iv_shift": shift, "pnl_per_unit": pnl, "weight": w})
            pnls.append(pnl)
            wts.append(w)
    pnls_a, wts_a = np.asarray(pnls), np.asarray(wts)
    mean = float(np.average(pnls_a, weights=wts_a))
    order = np.argsort(pnls_a)
    cum = np.cumsum(wts_a[order])
    tail = order[cum <= 0.10 + 1e-12]
    if len(tail) == 0:
        tail = order[:1]
    cvar10 = float(np.average(pnls_a[tail], weights=wts_a[tail]))

    pnl_target = pnl_stop = p = p_lb = e_strat = e_lb = None
    if inputs.target_price and inputs.stop_price:
        pnl_target = pnl_at(inputs.target_price, 1.0)
        pnl_stop = pnl_at(inputs.stop_price, 1.0)
        rows.append({"scenario": "strategy target", "spot": inputs.target_price, "iv_shift": 0.0, "pnl_per_unit": pnl_target, "weight": None})
        rows.append({"scenario": "strategy stop", "spot": inputs.stop_price, "iv_shift": 0.0, "pnl_per_unit": pnl_stop, "weight": None})
        if inputs.win_prob is not None and 0 <= inputs.win_prob <= 1:
            p = inputs.win_prob
            p_lb = wilson_lower(p, inputs.effective_n or 0.0)
            e_strat = p * pnl_target + (1 - p) * pnl_stop
            e_lb = p_lb * pnl_target + (1 - p_lb) * pnl_stop
    tail_ret = max(0.0, -cvar10) / entry
    objective = None
    if e_lb is not None:
        objective = e_lb / entry - 0.25 * tail_ret
    assumptions = (
        f"spot moves ~ N(0, {sigma_move:.2f}) over {inputs.horizon_bars} bars (basis: {basis}); heuristic, not a forecast",
        "IV shifts of -15% / 0 / +15% with weights 25/50/25: a heuristic",
        f"re-priced with {greeks.model} at the implied volatility solved from the contract's price ({greeks.iv:.1%}); flat vol, European",
        "exit fills give up half the quoted spread and the slippage ticks; charges from the contract-note model",
        "strategy path: target with probability p, stop with 1-p; p's lower bound is a Wilson bound at the effective sample size",
        "ranking objective = lower-bound expected net return on premium - 0.25 x tail loss (CVaR10) as a share of premium: a judgement, not fitted",
    )
    return ScenarioResult(
        True, "", entry, quantity, basis, sigma_move, mean, mean / entry, cvar10, float(pnls_a.min()), float(pnls_a.max()), pnl_target,
        pnl_stop, p, p_lb, e_strat, e_lb, (e_strat / entry) if e_strat is not None else None, objective, tuple(rows), assumptions,
    )
