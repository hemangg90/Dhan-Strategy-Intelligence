"""The ranking report: sample size, uncertainty, tail loss, profit factor, and the NO TRADE explanation."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.ranking.ranking_report import OPTIONS_NOTE, cvar, no_trade_explanation, profit_factor_of, strategy_report_rows


def _intel(name, r, eligible=True, score=0.3, stderr=0.1, reason=None, **cond):
    analogues = pd.DataFrame({"r_multiple": r})
    conditional = {"sample_size": len(r), "n_days": 12, "expected_r": float(np.mean(r)), "expected_r_stderr": stderr, "median_r": float(np.median(r)),
                   "win_rate": float(np.mean(np.asarray(r) > 0)), "confidence_label": "MEDIUM", "global_expected_r": 0.05,
                   "n_effective": 18.5, "estimate_basis": "analogues", "analogues_informative": True, "stability_positive_frac": 0.75}
    conditional.update(cond)
    score_obj = SimpleNamespace(score=score, eligible=eligible, ineligibility_reason=reason, max_drawdown=-3.2,
                                components={"expected_r_component": 0.1})
    return SimpleNamespace(strategy_name=name, conditional_metrics=conditional, global_metrics={"n_trades": 90}, analogues=analogues,
                           score=score_obj, evidence={"cost_drag_r": 0.04, "by_regime": {"TRENDING": {"n": 20, "mean_net_r": 0.2}}})


R = [1.0, -1.0, 2.0, -0.5, -2.0, 0.5, 1.5, -1.0, 0.3, 0.2]


def test_cvar_is_the_mean_of_the_worst_tail_and_never_empty():
    assert cvar([-3.0, -1.0, 1.0, 2.0], 0.5) == pytest.approx(-2.0)
    assert cvar([5.0]) == 5.0 and cvar([]) is None and cvar([float("nan")]) is None
    assert cvar(list(range(-100, 0)), 0.05) == pytest.approx(np.mean(range(-100, -95)))


def test_profit_factor_handles_no_losses_and_no_wins():
    assert profit_factor_of([2.0, -1.0, -1.0]) == pytest.approx(1.0)
    assert profit_factor_of([1.0, 2.0]) == float("inf") and profit_factor_of([-1.0]) == 0.0 and profit_factor_of([]) is None


def test_each_row_reports_sample_size_uncertainty_tail_and_costs():
    rows = strategy_report_rows([_intel("Momentum", R)], SimpleNamespace(is_no_trade=False, selected_strategy="Momentum"), "TRENDING")
    row = rows[0]
    assert row["n_comparable"] == 10 and row["n_independent_days"] == 12 and row["effective_n"] == 18.5 and row["confidence"] == "MEDIUM"
    mean = float(np.mean(R))
    assert row["mean_net_r"] == pytest.approx(mean, abs=1e-3) and row["ci95_low"] == pytest.approx(mean - 0.196, abs=1e-3)
    assert row["ci95_high"] == pytest.approx(mean + 0.196, abs=1e-3)
    assert row["tail_loss_cvar5_r"] == -2.0 and row["profit_factor"] == pytest.approx(5.5 / 4.5, abs=0.01)
    assert row["max_drawdown_r"] == -3.2 and row["cost_drag_r"] == 0.04 and row["unconditional_mean_net_r"] == 0.05
    assert row["selected"] is True and row["regime_record"] == {"n": 20, "mean_net_r": 0.2}
    assert row["options_specific"] == OPTIONS_NOTE and row["edge_lower_bound"] == pytest.approx(0.2)


def test_a_missing_standard_error_gives_no_interval_not_a_made_up_one():
    row = strategy_report_rows([_intel("A", R, stderr=None)], SimpleNamespace(is_no_trade=True, reason="x"), None)[0]
    assert row["ci95_low"] is None and row["ci95_high"] is None and row["selected"] is False and row["regime_record"] is None


def test_eligible_strategies_come_first_and_are_ordered_by_rank_score_not_win_rate():
    high_win = _intel("HighWinRate", [0.1] * 9 + [-3.0], score=0.1)  # 90% winners, negative expectancy shape
    solid = _intel("Solid", R, score=0.4)
    ineligible = _intel("Ineligible", R, eligible=False, score=0.9, reason="Insufficient historical analogues (4)")
    rows = strategy_report_rows([high_win, ineligible, solid], SimpleNamespace(is_no_trade=False, selected_strategy="Solid"), None)
    assert [r["strategy"] for r in rows] == ["Solid", "HighWinRate", "Ineligible"]
    assert rows[1]["win_rate"] == 0.9 and rows[0]["win_rate"] < rows[1]["win_rate"]


def test_no_trade_explains_itself_and_lists_each_ineligible_strategy():
    ranking = SimpleNamespace(is_no_trade=True, selected_strategy=None, reason="Strategies are statistically tied")
    rows = strategy_report_rows([_intel("A", R, eligible=False, reason="Confidence 'LOW' below minimum 'MEDIUM'"),
                                 _intel("B", R, eligible=False, reason="Edge not stable over time")], ranking, None)
    lines = no_trade_explanation(ranking, rows)
    assert lines[0] == "Strategies are statistically tied" and "A: Confidence 'LOW' below minimum 'MEDIUM'" in lines
    assert "B: Edge not stable over time" in lines and any("standing aside is the output" in line for line in lines)
    assert no_trade_explanation(SimpleNamespace(is_no_trade=False), rows) == []


def test_a_strategy_with_too_few_observations_still_gets_a_row_with_the_reason():
    thin = SimpleNamespace(strategy_name="Thin", conditional_metrics={"sample_size": 3, "confidence_label": "INSUFFICIENT_DATA", "expected_r": None},
                           global_metrics={}, analogues=pd.DataFrame(), score=SimpleNamespace(score=0.0, eligible=False, ineligibility_reason="Insufficient", max_drawdown=None, components={}),
                           evidence=None)
    row = strategy_report_rows([thin], SimpleNamespace(is_no_trade=True, reason="r"), None)[0]
    assert row["n_comparable"] == 3 and row["mean_net_r"] is None and row["profit_factor"] is None and row["tail_loss_cvar5_r"] is None
