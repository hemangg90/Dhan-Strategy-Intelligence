from types import SimpleNamespace

from quant_intelligence.ranking.ranking_engine import StrategyScore
from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES
from quant_intelligence.ui.decision_view import explanation_rows, global_rows, strategy_status


def _score(name, eligible=True, reason=None, expected_r=0.2, se=0.1):
    return StrategyScore(name, 0.3 if eligible else 0.0, expected_r, 0.55, "MEDIUM", 40, -3.0, eligible, reason,
                         edge_mean=expected_r, edge_stderr=se)


def _output(scores, selected=None, tie=False):
    intel = [SimpleNamespace(strategy_name=s.strategy_name, conditional_metrics={"n_days": 9},
                             global_metrics={"n_trades": 12, "expected_r": 0.1, "win_rate": 0.5,
                                             "cost_adjusted_expectancy": 120.0}) for s in scores]
    ranking = SimpleNamespace(ranked=scores, selected_strategy=selected, tie_break=tie)
    return SimpleNamespace(strategy_intel=intel, ranking=ranking)


def test_status_selected_eligible_rejected_and_tie_break():
    a, b = _score("A"), _score("B")
    c = _score("C", eligible=False, reason="Insufficient historical analogues (3) to estimate conditional edge")
    assert strategy_status(a, "A")[0] == "SELECTED"
    assert strategy_status(a, "A", tie_break=True)[0] == "TIE-BREAK"
    assert strategy_status(b, "A")[0] == "ELIGIBLE"
    status, why = strategy_status(c, "A")
    assert status == "REJECTED" and "Insufficient historical analogues" in why


def test_chain_aware_strategies_are_explained_not_silently_unranked():
    name = next(iter(CHAIN_AWARE_STRATEGY_NAMES))
    status, why = strategy_status(_score(name, eligible=False, reason="x"), None)
    assert status == "REJECTED" and "option chain" in why


def test_rows_show_real_values_and_none_for_missing_never_placeholders():
    scores = [_score("A"), _score("B", eligible=False, reason="r", expected_r=None, se=None)]
    rows = explanation_rows(_output(scores, selected="A"))
    assert rows[0]["Cond. expectancy (R, net of costs)"] == 0.2 and rows[0]["Distinct days"] == 9
    assert rows[1]["Cond. expectancy (R, net of costs)"] is None and rows[1]["Std. error (R)"] is None


def test_global_rows_are_labelled_gross_and_blank_without_trades():
    out = _output([_score("A")])
    out.strategy_intel[0].global_metrics = {"n_trades": 0}
    row = global_rows(out)[0]
    assert row["Expected R (gross)"] is None and row["Win rate (gross)"] is None
