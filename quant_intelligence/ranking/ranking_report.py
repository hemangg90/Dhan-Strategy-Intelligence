"""
The strategy ranking, laid out so the evidence behind each rank can be read.

Per strategy: how many comparable observations back the figures, win rate, mean and median net R, expectancy with a 95%
interval, profit factor, tail loss, drawdown, the cost drag, the confidence label, stability across time, and the
strategy's record in the current regime. Strategies are NOT ranked by win rate: the rank is the existing documented score
(`ranking/ranking_engine.py`: a lower bound of the shrunk conditional edge, plus confidence and probability terms, less a
drawdown penalty). The report shows the lower bound and each component so the effect of a weight can be read off.

Definitions
  expectancy         mean net R of the comparable (analogue) observations; net = after the application's cost model
  95% interval       mean +/- 1.96 x stderr, where the stderr clusters by trading day (observations on one day are one piece
                     of evidence); None when the engine could not estimate it
  profit factor      sum of winning net R / |sum of losing net R| over the same observations
  tail loss (CVaR5)  mean of the worst 5% of net R outcomes (at least the worst one)
  options-specific   not available: the application has no historical option prices, so strategy results are underlying
                     setups with a modelled premium, never real option fills. The report says so on every row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OPTIONS_NOTE = "unavailable - no historical option prices; premium P&L is modelled, not observed"


def cvar(r, alpha: float = 0.05) -> float | None:
    values = np.sort(np.asarray([x for x in r if x is not None and np.isfinite(x)], float))
    if len(values) == 0:
        return None
    k = max(1, int(np.ceil(alpha * len(values))))
    return float(values[:k].mean())


def profit_factor_of(r) -> float | None:
    values = np.asarray([x for x in r if x is not None and np.isfinite(x)], float)
    losses = -values[values < 0].sum()
    wins = values[values > 0].sum()
    if len(values) == 0:
        return None
    if losses == 0:
        return None if wins == 0 else float("inf")
    return float(wins / losses)


def _round(x, n=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def strategy_report_rows(strategy_intel, ranking, regime_label: str | None = None) -> list[dict]:
    """One row per strategy, best-ranked first. `strategy_intel` is `PipelineOutput.strategy_intel`; `ranking` its RankingDecision."""
    selected = None if getattr(ranking, "is_no_trade", False) else getattr(ranking, "selected_strategy", None)
    rows = []
    for si in strategy_intel:
        cond, glob, score = si.conditional_metrics, si.global_metrics, si.score
        analogues = si.analogues if isinstance(si.analogues, pd.DataFrame) else pd.DataFrame()
        r = analogues["r_multiple"].dropna().to_numpy(float) if "r_multiple" in analogues.columns else np.array([])
        se = cond.get("expected_r_stderr")
        mean = cond.get("expected_r")
        ev = getattr(si, "evidence", None) or {}
        regime_row = (ev.get("by_regime") or {}).get(regime_label) if regime_label else None
        comps = dict(score.components or {})
        rows.append({
            "strategy": si.strategy_name,
            "selected": si.strategy_name == selected,
            "eligible": bool(score.eligible),
            "why_not_eligible": score.ineligibility_reason,
            "rank_score": _round(score.score),
            "edge_lower_bound": _round(comps.get("expected_r_component") / 0.5) if comps.get("expected_r_component") is not None else None,
            "n_comparable": int(cond.get("sample_size") or 0),
            "n_independent_days": cond.get("n_days"),
            "effective_n": _round(cond.get("n_effective"), 1),
            "confidence": cond.get("confidence_label"),
            "win_rate": _round(cond.get("win_rate")),
            "mean_net_r": _round(mean),
            "median_net_r": _round(cond.get("median_r")),
            "ci95_low": _round(mean - 1.96 * se) if mean is not None and se is not None else None,
            "ci95_high": _round(mean + 1.96 * se) if mean is not None and se is not None else None,
            "profit_factor": _round(profit_factor_of(r), 2),
            "tail_loss_cvar5_r": _round(cvar(r)),
            "max_drawdown_r": _round(score.max_drawdown),
            "cost_drag_r": _round((ev or {}).get("cost_drag_r")),
            "unconditional_mean_net_r": _round(cond.get("global_expected_r")),
            "estimate_basis": cond.get("estimate_basis", "analogues"),
            "analogues_informative": cond.get("analogues_informative"),
            "stability_positive_folds": _round(cond.get("stability_positive_frac"), 2),
            "global_trades": glob.get("n_trades"),
            "regime": regime_label,
            "regime_record": regime_row,
            "options_specific": OPTIONS_NOTE,
            "score_components": comps,
        })
    rows.sort(key=lambda x: (not x["eligible"], -(x["rank_score"] if x["rank_score"] is not None else -1e9)))
    return rows


def no_trade_explanation(ranking, rows: list[dict]) -> list[str]:
    """Plain-language reasons the system stood aside; empty when a strategy was selected."""
    if not getattr(ranking, "is_no_trade", False):
        return []
    lines = [str(getattr(ranking, "reason", "") or "No strategy cleared the bar.")]
    for row in rows:
        if not row["eligible"] and row["why_not_eligible"]:
            lines.append(f"{row['strategy']}: {row['why_not_eligible']}")
    if rows and not any(r["eligible"] for r in rows):
        lines.append("No strategy is eligible, so standing aside is the output - not an error.")
    return lines
