"""Pure helpers that turn a PipelineOutput into the explainable tables on the Decision Center.

Only values that exist are shown; anything missing is None (rendered "n/a"), never a placeholder number.
Everything here is a HISTORICAL statistic about past trades, not a prediction.
"""
from __future__ import annotations

from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES

NO_HISTORY_REASON = (
    "Needs a live option chain (PCR / OI / IV); it has no historical setups, so it cannot be ranked or backtested."
)


def strategy_status(score, selected: str | None, tie_break: bool = False) -> tuple[str, str]:
    """(status, why) for one ranked strategy. Status is SELECTED, TIE-BREAK, ELIGIBLE or REJECTED."""
    if score.strategy_name == selected:
        if tie_break:
            return "TIE-BREAK", "Selected only because the top strategies were statistically tied (no proven edge over the runner-up)."
        return "SELECTED", "Highest historical conditional score among eligible strategies."
    if score.eligible:
        if selected is None:
            return "ELIGIBLE", "Clears the edge and confidence thresholds, but no strategy was selected (see the decision reason)."
        return "ELIGIBLE", "Clears the edge and confidence thresholds but ranks below the selected strategy."
    if score.strategy_name in CHAIN_AWARE_STRATEGY_NAMES:
        return "REJECTED", NO_HISTORY_REASON
    return "REJECTED", score.ineligibility_reason or "Does not clear the minimum edge / confidence thresholds."


def explanation_rows(output) -> list[dict]:
    """One row per strategy, ranked order, with the statistics behind the ranking."""
    by_name = {si.strategy_name: si for si in output.strategy_intel}
    rows = []
    for score in output.ranking.ranked:
        si = by_name.get(score.strategy_name)
        cond = si.conditional_metrics if si else {}
        status, why = strategy_status(score, output.ranking.selected_strategy, output.ranking.tie_break)
        rows.append(
            {
                "Strategy": score.strategy_name,
                "Status": status,
                "Why": why,
                "Score": round(score.score, 3),
                "Cond. expectancy (R, net of costs)": _r(score.expected_r),
                "Std. error (R)": _r(score.edge_stderr),
                "Win rate (historical)": _pct(score.prob_positive_return),
                "Analogue sample": score.sample_size,
                "Distinct days": cond.get("n_days"),
                "Confidence": score.confidence_label,
                "Max drawdown (R)": _r(score.max_drawdown, 1),
            }
        )
    return rows


def global_rows(output) -> list[dict]:
    """Unconditional backtest statistics; flagged gross because they are not cost-adjusted."""
    rows = []
    for si in output.strategy_intel:
        g = si.global_metrics
        n = g.get("n_trades", 0)
        cost_adj = g.get("cost_adjusted_expectancy")
        rows.append(
            {
                "Strategy": si.strategy_name,
                "Backtest trades": n,
                "Expected R (gross)": _r(g.get("expected_r")) if n else None,
                "Win rate (gross)": _pct(g.get("win_rate")) if n else None,
                "Cost-adj. expectancy (Rs/trade)": round(cost_adj, 1) if n and cost_adj is not None else None,
            }
        )
    return rows


def _r(x, nd: int = 3):
    return None if x is None or x != x else round(float(x), nd)


def _pct(x):
    return None if x is None or x != x else f"{float(x) * 100:.1f}%"
