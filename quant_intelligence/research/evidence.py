"""Extra evidence shown next to a strategy's ranking. Descriptive statistics about PAST trades; none of it feeds selection.

* holdout        in-sample vs the chronologically last part of the strategy's own trades
* regime         net R grouped by the (causal, smoothed) regime label on the entry bar
* frequency      trades per trading day
* cost drag      gross minus net average R
* multiple tests how large an edge must look to stand out after choosing the best of N strategies (Sidak)
* vs winner      how far a runner-up's edge sits below the winner's, in combined standard errors
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from quant_intelligence.analytics.metrics import profit_factor

HOLDOUT_FRACTION = 0.30
MIN_HOLDOUT_TRADES = 5


def holdout_split(net_r: np.ndarray, holdout_fraction: float = HOLDOUT_FRACTION) -> dict:
    """`net_r` in chronological order. The last `holdout_fraction` of trades is the holdout; None when either side is too small."""
    n = len(net_r)
    cut = n - int(round(n * holdout_fraction))
    in_sample, holdout = net_r[:cut], net_r[cut:]
    if len(in_sample) < MIN_HOLDOUT_TRADES or len(holdout) < MIN_HOLDOUT_TRADES:
        return {"in_sample_n": len(in_sample), "holdout_n": len(holdout), "in_sample_mean_r": None, "holdout_mean_r": None}
    return {"in_sample_n": len(in_sample), "holdout_n": len(holdout),
            "in_sample_mean_r": float(in_sample.mean()), "holdout_mean_r": float(holdout.mean())}


def regime_breakdown(net_r: np.ndarray, labels: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for label in dict.fromkeys(labels):
        sel = np.array([lab == label for lab in labels])
        r = net_r[sel]
        out[label] = {"n": int(sel.sum()), "mean_net_r": float(r.mean()), "win_rate": float((r > 0).mean())}
    return out


def sidak_required_z(n_tests: int, alpha: float = 0.05) -> float | None:
    """One-sided z a single estimate needs so that, across `n_tests` independent comparisons, the chance that ANY of them
    reaches it by luck stays at `alpha`. 1.645 for one test; larger as the number of strategies grows."""
    if n_tests < 1:
        return None
    per_test = 1.0 - (1.0 - alpha) ** (1.0 / n_tests)
    return float(NormalDist().inv_cdf(1.0 - per_test))


def versus_winner(winner, other) -> dict | None:
    """Edge gap in combined standard errors between two StrategyScore objects, or None when either lacks a standard error."""
    if None in (winner.edge_mean, winner.edge_stderr, other.edge_mean, other.edge_stderr):
        return None
    se = math.hypot(winner.edge_stderr, other.edge_stderr)
    gap = winner.edge_mean - other.edge_mean
    return {"edge_gap_r": float(gap), "combined_se": float(se), "z": float(gap / se) if se > 0 else None}


def build_strategy_evidence(trades, net_r: np.ndarray, gross_r: np.ndarray, entry_ts, regime_labels: list[str],
                            n_trading_days: int, net_pnls=None) -> dict:
    """`trades`/`net_r`/`gross_r`/`entry_ts`/`regime_labels` are aligned. Sorted chronologically here."""
    n = len(net_r)
    if n == 0:
        return {"n_trades": 0}
    order = np.argsort(pd.to_datetime(list(entry_ts)).to_numpy(), kind="stable")
    net, gross = np.asarray(net_r, dtype=float)[order], np.asarray(gross_r, dtype=float)[order]
    labels = [regime_labels[i] for i in order]
    out = {
        "n_trades": n,
        "trades_per_day": float(n / n_trading_days) if n_trading_days else None,
        "mean_net_r": float(net.mean()),
        "mean_gross_r": float(gross.mean()),
        "cost_drag_r": float(gross.mean() - net.mean()),
        "profit_factor_r": profit_factor(net),
        "holdout": holdout_split(net),
        "by_regime": regime_breakdown(net, labels),
    }
    return out
