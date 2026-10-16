"""Shared performance-metric formulas, defined once.

Several modules used to carry their own copies of these (and they disagreed on conventions). Each function states its
definition and what it returns for degenerate input; undefined values are None, never silently 0 or infinity.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _clean(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError("metrics inputs must be finite numbers")
    return arr


def max_drawdown(cumulative: Sequence[float], start: float = 0.0) -> float:
    """Worst peak-to-trough fall (a number <= 0) of a cumulative P&L / equity path that BEGINS at `start`.

    The starting value counts as a peak: a strategy whose first trades lose has already drawn down from its starting
    equity. (Measuring from the first trade's value, as earlier code did, under-reports exactly those drawdowns.)
    An empty path has no drawdown: 0.0."""
    path = _clean(cumulative)
    if path.size == 0:
        return 0.0
    full = np.concatenate([[float(start)], path])
    return float((full - np.maximum.accumulate(full)).min())


def profit_factor(pnl: Sequence[float]) -> float | None:
    """Gross profit / gross loss over per-trade P&L. None when undefined: no trades, or no losing trade
    (an infinite ratio is not a number to report)."""
    arr = _clean(pnl)
    gross_loss = float(-arr[arr < 0].sum())
    return float(arr[arr > 0].sum() / gross_loss) if gross_loss > 0 else None


def payoff_ratio(pnl: Sequence[float]) -> float | None:
    """Average winning trade / |average losing trade|. Break-even trades (exactly 0) are neither. None if there is no
    winner or no loser."""
    arr = _clean(pnl)
    wins, losses = arr[arr > 0], arr[arr < 0]
    if wins.size == 0 or losses.size == 0:
        return None
    return float(wins.mean() / abs(losses.mean()))


def expectancy(pnl: Sequence[float]) -> float | None:
    """Mean P&L per trade. None for no trades."""
    arr = _clean(pnl)
    return float(arr.mean()) if arr.size else None


def annualised_sharpe(period_returns: Sequence[float], periods_per_year: int = 252, min_periods: int = 10) -> float | None:
    """mean / sample standard deviation (ddof = 1) x sqrt(periods_per_year), risk-free rate 0. None with fewer than
    `min_periods` observations or zero variance."""
    arr = _clean(period_returns)
    if arr.size < max(min_periods, 2):
        return None
    sd = float(arr.std(ddof=1))
    return float(arr.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else None
