"""Selector comparison: paired chronological differences with day-clustered intervals, and honest 'unavailable' for options."""
import numpy as np
import pandas as pd
import pytest

from quant_intelligence.ranking.evaluate_ranker import Decision
from quant_intelligence.ranking.selector_comparison import UNAVAILABLE_OPTIONS, compare, paired_outcomes


def _decisions(values, start="2030-01-02 09:15", per_day=10):
    """values: net R per bar (None = stood aside)."""
    out = []
    for i, v in enumerate(values):
        ts = pd.Timestamp(start) + pd.Timedelta(days=i // per_day) + pd.Timedelta(minutes=5 * (i % per_day))
        out.append(Decision(instrument="NIFTY", timestamp=ts, day=str(ts.date()), selected=None if v is None else "Momentum",
                            n_signals=1, selected_net_r=v, pick_net_r=0.0))
    return out


def test_outcomes_are_paired_by_bar_and_a_stand_aside_counts_as_zero():
    a = _decisions([1.0, None, -1.0, 0.5])
    b = _decisions([0.5, 0.2, -1.0, None])
    p = paired_outcomes(a, b)
    assert list(p["a"]) == [1.0, 0.0, -1.0, 0.5] and list(p["b"]) == [0.5, 0.2, -1.0, 0.0]
    assert list(p["a_traded"]) == [True, False, True, True] and list(p["b_traded"]) == [True, True, True, False]


def test_only_bars_both_selectors_saw_are_compared():
    a, b = _decisions([1.0, 1.0, 1.0, 1.0]), _decisions([1.0, 1.0])
    assert len(paired_outcomes(a, b)) == 2
    assert compare([], b)["status"] == "NO_COMMON_BARS"


def test_identical_selectors_are_not_distinguishable_with_zero_difference():
    rng = np.random.default_rng(1)
    vals = list(rng.normal(0, 1, 120))
    res = compare(_decisions(vals), _decisions(vals))
    assert res["status"] == "OK" and res["paired_difference_mean_net_r_per_bar"] == 0.0 and res["verdict"] == "not distinguishable"
    assert res["bars_where_they_differ"] == 0 and "in-sample" in res["caveat"]


def test_a_noisy_small_difference_is_not_called_better():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 100)
    other = base + rng.normal(0.02, 1.0, 100)  # a tiny mean shift buried in noise
    res = compare(_decisions(list(base)), _decisions(list(other)))
    assert res["verdict"] == "not distinguishable"
    assert res["paired_difference_ci95"][0] < 0 < res["paired_difference_ci95"][1]


def test_a_consistent_improvement_across_many_days_is_called_better_and_the_reverse_worse():
    rng = np.random.default_rng(3)
    base = list(rng.normal(0, 0.5, 400))
    better = [v + 0.5 + rng.normal(0, 0.05) for v in base]
    res = compare(_decisions(base), _decisions(better))
    assert res["verdict"] == "B better" and res["paired_difference_ci95"][0] > 0 and res["trading_days"] == 40
    assert compare(_decisions(better), _decisions(base))["verdict"] == "B worse"


def test_the_statistics_describe_each_selector_and_count_only_real_trades():
    a = _decisions([1.0, -2.0, None, 1.0])
    b = _decisions([None, None, None, None])
    res = compare(a, b)
    assert res["A"]["trades"] == 3 and res["B"]["trades"] == 0 and res["A"]["win_rate"] == pytest.approx(2 / 3)
    assert res["A"]["profit_factor"] == pytest.approx(1.0) and res["A"]["tail_loss_cvar5_r"] == -2.0
    assert res["A"]["max_drawdown_r"] == pytest.approx(2.0) and res["B"]["mean_net_r_per_trade"] is None


def test_the_options_selectors_are_reported_unavailable_with_the_reason():
    assert "historical option-chain features do not exist" in UNAVAILABLE_OPTIONS


def test_compare_selectors_runs_a_and_b_and_reports_c_and_d_unavailable(monkeypatch):
    from quant_intelligence.ranking import evaluate_ranker, selector_comparison

    calls = []

    def fake(ohlcv, instrument, analogue_config=None, extra_frame=None, **kw):
        calls.append((analogue_config.name, extra_frame is not None))
        return _decisions([0.5, -0.5, 1.0, None] * 5)

    monkeypatch.setattr(evaluate_ranker, "evaluate_instrument", fake)
    rng = np.random.default_rng(0)
    n = 80
    closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({"timestamp": pd.date_range("2030-01-02 09:15", periods=n, freq="5min"), "open": closes, "high": closes + 0.3,
                       "low": closes - 0.3, "close": closes, "volume": 1000})
    res = selector_comparison.compare_selectors(df, "NIFTY")
    assert [c[1] for c in calls] == [False, True] and calls[1][0] == "context-structure"  # only B gets the structure features
    assert res["A_vs_B"]["status"] == "OK" and res["A_vs_B"]["verdict"] == "not distinguishable"
    assert res["C"]["status"] == "UNAVAILABLE" == res["D"]["status"] and "historical option-chain" in res["C"]["reason"]


def test_evaluate_instrument_refuses_an_extra_frame_that_is_not_one_row_per_bar():
    from quant_intelligence.ranking.evaluate_ranker import evaluate_instrument

    rng = np.random.default_rng(1)
    n = 700
    closes = 100 + np.cumsum(rng.normal(0, 0.3, n))
    df = pd.DataFrame({"timestamp": pd.date_range("2030-01-02 09:15", periods=n, freq="5min"), "open": closes, "high": closes + 0.2,
                       "low": closes - 0.2, "close": closes, "volume": 1000})
    with pytest.raises(ValueError, match="one row per bar"):
        evaluate_instrument(df, "NIFTY", extra_frame=pd.DataFrame({"timestamp": df["timestamp"].head(10), "x": 1.0}))
