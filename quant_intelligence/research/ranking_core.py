"""Shared ranking core.

One place that turns a strategy's closed trades + the current market state into a
StrategyScore. Used by research/pipeline.py (live decisions) and by
ranking/evaluate_ranker.py (walk-forward evaluation), so what is evaluated is exactly what
runs live - a change here is measured by the evaluator automatically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_intelligence.analogues.analogue_engine import (
    ALL_FEATURES,
    INFORMATIVE_P_MAX,
    MIN_DAYS_FOR_HIGH_CONFIDENCE,
    MIN_DAYS_FOR_MEDIUM_CONFIDENCE,
    _cluster_robust_stderr,
    analogue_search,
    conditional_metrics_from_analogues,
    conditioning_test,
    default_config,
)
from quant_intelligence.backtesting.engine import _compute_run_metrics, net_r_multiple
from quant_intelligence.ranking.ranking_engine import StrategyScore, score_strategy


@dataclass
class Assessment:
    analogues: pd.DataFrame
    conditional: dict
    metrics: dict
    score: StrategyScore


def observations_from_trades(
    trades, feat_by_ts: pd.DataFrame, strategy_name: str, quantity: int, market: str = "NSE"
) -> pd.DataFrame:
    """Flatten each trade's entry-time features + outcome into one row per historical
    observation (the analogue engine's input). `feat_by_ts` is features.set_index("timestamp").
    `r_multiple` is NET of costs - what the strategy actually earns - with the gross value kept
    in `gross_r_multiple`."""
    rows = []
    for t in trades:
        if t.setup.timestamp not in feat_by_ts.index:
            continue
        feat_row = feat_by_ts.loc[t.setup.timestamp]
        row = {c: feat_row.get(c) for c in ALL_FEATURES}
        row["r_multiple"] = net_r_multiple(t, quantity, market)
        row["gross_r_multiple"] = t.r_multiple
        row["outcome"] = t.outcome
        row["entry_timestamp"] = t.setup.timestamp
        row["exit_timestamp"] = getattr(t, "exit_timestamp", None)
        row["strategy_name"] = strategy_name
        rows.append(row)
    return pd.DataFrame(rows)


def _stability(observations: pd.DataFrame, n_folds: int = 4, min_trades: int = 3) -> dict:
    """Is the edge positive across time? Split the strategy's closed trades into `n_folds`
    chronological folds and count the share with a positive mean net R (folds with too few trades
    are ignored). A strategy that made all its money in one stretch is not a stable edge."""
    if len(observations) < n_folds * min_trades or "entry_timestamp" not in observations.columns:
        return {"stability_folds": 0, "stability_positive_frac": None}
    ordered = observations.sort_values("entry_timestamp")["r_multiple"].to_numpy()
    means = [c.mean() for c in np.array_split(ordered, n_folds) if len(c) >= min_trades]
    if not means:
        return {"stability_folds": 0, "stability_positive_frac": None}
    return {"stability_folds": len(means), "stability_positive_frac": float(np.mean([m > 0 for m in means]))}


def assess_observations(
    strategy_name: str,
    observations: pd.DataFrame,
    current_features: dict,
    metrics: dict,
    top_k: int = 30,
    config=None,
) -> Assessment:
    cfg = config or default_config()
    analogues, index = analogue_search(current_features, observations, top_k=top_k, config=cfg)
    conditional = conditional_metrics_from_analogues(analogues)
    conditional["strategy_name"] = strategy_name
    conditional["analogue_method"] = cfg.name
    conditional.update(_stability(observations))
    if len(observations):
        conditional["global_expected_r"] = float(observations["r_multiple"].mean())
    if cfg.test_information and index is not None and conditional.get("expected_r") is not None:
        _apply_information_test(conditional, index, observations, top_k)
    score = score_strategy(conditional, metrics.get("max_drawdown_r"))
    return Assessment(analogues=analogues, conditional=conditional, metrics=metrics, score=score)


def _apply_information_test(conditional: dict, index, observations: pd.DataFrame, top_k: int) -> None:
    """Add the conditioning test to `conditional`; if the analogues add no detectable information, replace the
    conditional estimate by the strategy's overall (day-clustered) estimate, which is then the better-supported one."""
    r_all = pd.to_numeric(observations["r_multiple"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(r_all)
    test = conditioning_test(index, r_all[ok] if ok.all() else np.nan_to_num(r_all), conditional["expected_r"], top_k=top_k)
    conditional["conditioning_p_value"] = test["p_value"]
    conditional["analogues_informative"] = None if test["p_value"] is None else bool(test["p_value"] <= INFORMATIVE_P_MAX)
    conditional["estimate_basis"] = "analogues"
    if test["p_value"] is None or test["p_value"] <= INFORMATIVE_P_MAX:
        return
    r = pd.Series(r_all[ok])
    days = pd.to_datetime(observations["entry_timestamp"]).dt.date.to_numpy()[ok]
    n_days = int(pd.Series(days).nunique())
    conditional["analogue_expected_r"] = conditional["expected_r"]
    conditional.update({
        "estimate_basis": "overall",
        "expected_r": float(r.mean()),
        "prob_positive_return": float((r > 0).mean()),
        "win_rate": float((r > 0).mean()),
        "expected_r_stderr": (lambda se: None if np.isnan(se) else float(se))(_cluster_robust_stderr(r.to_numpy(), days)),
        "n_days": n_days,
        "confidence_label": "HIGH" if n_days >= MIN_DAYS_FOR_HIGH_CONFIDENCE else "MEDIUM" if n_days >= MIN_DAYS_FOR_MEDIUM_CONFIDENCE else "LOW",
    })


def assess_strategy(
    strategy_name: str,
    trades,
    feat_by_ts: pd.DataFrame,
    current_features: dict,
    quantity: int,
    metrics: dict | None = None,
    top_k: int = 30,
    market: str = "NSE",
    config=None,
) -> Assessment:
    observations = observations_from_trades(trades, feat_by_ts, strategy_name, quantity, market)
    metrics = metrics if metrics is not None else _compute_run_metrics(list(trades), quantity, market)
    return assess_observations(strategy_name, observations, current_features, metrics, top_k=top_k, config=config)
