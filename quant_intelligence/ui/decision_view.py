"""Pure helpers that turn a PipelineOutput into the explainable tables on the Decision Center.

Only values that exist are shown; anything missing is None (rendered "n/a"), never a placeholder number.
Everything here is a HISTORICAL statistic about past trades, not a prediction.
"""
from __future__ import annotations

from quant_intelligence.research.evidence import sidak_required_z, versus_winner
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
        gap = _gap_to_winner(output, score)
        if gap is not None and status in ("ELIGIBLE", "REJECTED") and score.eligible:
            why += f" Its edge is {gap['edge_gap_r']:.2f} R ({gap['z']:.1f} standard errors) below the winner's." if gap["z"] is not None else ""
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


def _gap_to_winner(output, score):
    selected = output.ranking.selected_strategy
    if selected is None or score.strategy_name == selected:
        return None
    winner = next((s for s in output.ranking.ranked if s.strategy_name == selected), None)
    return versus_winner(winner, score) if winner is not None else None


def evidence_rows(output) -> list[dict]:
    """Profit factor, trade frequency, cost drag and in-sample vs holdout for each strategy (descriptive, past trades)."""
    rows = []
    for si in output.strategy_intel:
        ev = getattr(si, "evidence", None) or {}
        hold = ev.get("holdout") or {}
        n = ev.get("n_trades", 0)
        rows.append({
            "Strategy": si.strategy_name,
            "Trades": n,
            "Trades / day": _r(ev.get("trades_per_day"), 2) if n else None,
            "Profit factor (net R)": _r(ev.get("profit_factor_r"), 2) if n else None,
            "Net R / trade": _r(ev.get("mean_net_r")) if n else None,
            "Cost drag (R / trade)": _r(ev.get("cost_drag_r")) if n else None,
            "In-sample net R": _r(hold.get("in_sample_mean_r")),
            "Holdout net R (last 30%)": _r(hold.get("holdout_mean_r")),
            "Holdout trades": hold.get("holdout_n"),
        })
    return rows


def regime_breakdown_rows(output, strategy: str) -> list[dict]:
    si = next((s for s in output.strategy_intel if s.strategy_name == strategy), None)
    by_regime = ((getattr(si, "evidence", None) or {}).get("by_regime") or {}) if si else {}
    return [{"Regime at entry": k.replace("_", " ").title(), "Trades": v["n"], "Net R / trade": _r(v["mean_net_r"]),
             "Win rate": _pct(v["win_rate"])} for k, v in sorted(by_regime.items(), key=lambda kv: -kv[1]["n"])]


def multiple_testing_note(output) -> str | None:
    """How many strategies were compared, and the z a single winner needs after that many tries. Informational only."""
    n = sum(1 for s in output.strategy_intel if (s.conditional_metrics or {}).get("expected_r") is not None)
    z = sidak_required_z(n)
    if n < 2 or z is None:
        return None
    selected = output.ranking.selected_strategy
    top = next((s for s in output.ranking.ranked if s.strategy_name == selected), None)
    seen = ""
    if top is not None and top.edge_mean is not None and top.edge_stderr:
        seen = f" The selected strategy's edge is {top.edge_mean / top.edge_stderr:.1f} standard errors above zero."
    return (f"{n} strategies were compared. When the best of {n} is picked, an edge needs to be about {z:.1f} standard errors "
            f"above zero (not 1.6) to be unlikely to be luck.{seen} Shown for context; it does not change the selection.")


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
