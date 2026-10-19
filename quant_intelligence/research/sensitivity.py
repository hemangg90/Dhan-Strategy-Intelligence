"""Parameter sensitivity: does a strategy's result survive reasonable changes to its settings?

Each numeric parameter is moved to 0.8x and 1.2x of its value (others held), the strategy is back-tested again on the same
data, and the net R per trade is compared. A result that flips sign, or disappears, when a threshold moves 20% was probably
fitted to the data rather than found in it. Descriptive: it re-uses the same sample, so it cannot prove an edge is real; it
can only show fragility. Nothing is persisted and nothing feeds a decision.
"""
from __future__ import annotations

import pandas as pd

from quant_intelligence.backtesting.engine import market_for, net_r_multiple, run_backtest
from quant_intelligence.strategies.registry import get_strategy

FACTORS = (0.8, 1.0, 1.2)
MIN_TRADES_FOR_SIGN = 5


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scaled(value, factor):
    scaled = value * factor
    if isinstance(value, int):
        scaled = max(1, int(round(scaled)))
    return scaled


def _run(strategy_name: str, params: dict, ohlcv, features, instrument: str, quantity: int) -> dict:
    try:
        strategy = get_strategy(strategy_name, params)
    except ValueError as exc:  # an invalid combination (e.g. thresholds crossing) is a result, not a crash
        return {"n_trades": 0, "mean_net_r": None, "note": f"invalid setting: {exc}"}
    result = run_backtest(strategy, ohlcv, features, instrument, split="SENSITIVITY", quantity=quantity, persist=False)
    market = market_for(instrument)
    net = [net_r_multiple(t, quantity, market) for t in result.trades if t.r_multiple is not None]
    return {"n_trades": len(net), "mean_net_r": float(sum(net) / len(net)) if net else None, "note": None}


def parameter_sensitivity(strategy_name: str, ohlcv: pd.DataFrame, features: pd.DataFrame, instrument: str,
                          quantity: int, base_params: dict | None = None, factors=FACTORS) -> dict:
    """Returns {"rows": [...], "base": {...}, "sign_survives": bool | None}. `rows` has one entry per parameter x factor."""
    base_strategy = get_strategy(strategy_name, base_params)
    base_parameters = dict(base_strategy.parameters)
    base = _run(strategy_name, base_parameters, ohlcv, features, instrument, quantity)
    rows = []
    for name, value in base_parameters.items():
        if not _numeric(value):
            continue
        for factor in factors:
            if factor == 1.0:
                continue
            new_value = _scaled(value, factor)
            if new_value == value:
                continue
            outcome = _run(strategy_name, {**base_parameters, name: new_value}, ohlcv, features, instrument, quantity)
            rows.append({"parameter": name, "factor": factor, "value": new_value, **outcome})
    comparable = [r for r in rows if r["mean_net_r"] is not None and r["n_trades"] >= MIN_TRADES_FOR_SIGN]
    sign_survives = None
    if base["mean_net_r"] is not None and base["n_trades"] >= MIN_TRADES_FOR_SIGN and comparable:
        base_positive = base["mean_net_r"] > 0
        sign_survives = all((r["mean_net_r"] > 0) == base_positive for r in comparable)
    return {"strategy": strategy_name, "base": {**base, "parameters": base_parameters}, "rows": rows,
            "sign_survives": sign_survives, "variants_compared": len(comparable)}
