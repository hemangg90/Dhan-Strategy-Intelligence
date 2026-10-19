"""Provenance labels and the extra ranking evidence (display-only statistics about past trades)."""
import datetime as dt
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.data import provenance as prov
from quant_intelligence.ranking.ranking_engine import StrategyScore
from quant_intelligence.research import evidence as ev
from quant_intelligence.ui import decision_view as dv
from quant_intelligence.utils.market_profile import MCX, NSE

MON = dt.datetime(2026, 10, 19)  # a Monday; Saturday 17 Oct and Sunday 18 Oct are closed


# ------------------------------------------------------------------------------------ provenance
@pytest.mark.parametrize("now, last_bar, expected", [
    (MON.replace(hour=10, minute=3), MON.replace(hour=9, minute=55), prov.LIVE),        # 9:55 bar closed 10:00, 3 min ago
    (MON.replace(hour=10, minute=3), MON.replace(hour=10, minute=0), prov.LIVE),        # bar still forming: lag negative
    (MON.replace(hour=10, minute=30), MON.replace(hour=10, minute=0), prov.DELAYED),    # open, last bar 25 min stale
    (MON.replace(hour=10, minute=30), MON.replace(hour=9, minute=20), prov.DELAYED),
    (MON.replace(hour=16, minute=0), MON.replace(hour=15, minute=25), prov.HISTORICAL),  # closed for the day
    (dt.datetime(2026, 10, 17, 11, 0), dt.datetime(2026, 10, 16, 15, 25), prov.HISTORICAL),  # Saturday
    (MON.replace(hour=8, minute=0), dt.datetime(2026, 10, 16, 15, 25), prov.HISTORICAL),  # before the open
])
def test_mode_known_answers(now, last_bar, expected):
    assert prov.classify_mode(last_bar, now, NSE, 5) == expected


def test_simulated_source_wins_and_missing_bar_is_historical():
    assert prov.classify_mode(MON.replace(hour=10, minute=3), MON.replace(hour=9, minute=55), NSE, 5, source="synthetic") == prov.SIMULATED
    assert prov.classify_mode(None, MON.replace(hour=10), NSE) == prov.HISTORICAL


def test_mcx_is_live_in_the_evening_when_nse_is_closed():
    now = MON.replace(hour=19, minute=2)
    assert prov.classify_mode(MON.replace(hour=18, minute=55), now, MCX, 5) == prov.LIVE
    assert prov.classify_mode(MON.replace(hour=18, minute=55), now, NSE, 5) == prov.HISTORICAL


def test_build_provenance_carries_source_age_and_quality():
    meta = {"source": "cache", "end_ts": pd.Timestamp("2026-10-16 15:25"), "quality_status": "DEGRADED", "n_rows": 3000,
            "timeframe": "5min", "quality_report": {"category": "STALE", "issues": ["latest refresh failed: timeout"]}}
    p = prov.build_provenance(meta, dt.datetime(2026, 10, 17, 11, 0), NSE, 5)
    assert p["mode"] == prov.HISTORICAL and p["source"] == "cache" and p["quality_category"] == "STALE"
    assert p["refresh_failed"] is True and p["age_minutes"] == pytest.approx(19 * 60 + 35)
    assert "HISTORICAL" in prov.strip_text(p) and "cache" in prov.strip_text(p)
    assert prov.strip_text({}) == "data provenance unavailable"


# -------------------------------------------------------------------------------------- evidence
def test_holdout_split_known_answer_and_small_samples():
    assert ev.holdout_split(np.array([1.0] * 7 + [-1.0] * 3))["holdout_mean_r"] is None  # 3 holdout trades: too few
    h = ev.holdout_split(np.array([1.0] * 14 + [-1.0] * 6))  # 20 trades: 14 in-sample, 6 holdout
    assert (h["in_sample_n"], h["holdout_n"]) == (14, 6)
    assert h["in_sample_mean_r"] == pytest.approx(1.0) and h["holdout_mean_r"] == pytest.approx(-1.0)


def test_regime_breakdown_counts_sum_to_trades():
    net = np.array([1.0, -1.0, 2.0, 0.5, -0.5])
    out = ev.regime_breakdown(net, ["RANGE", "RANGE", "TREND_UP", "TREND_UP", "TREND_UP"])
    assert sum(v["n"] for v in out.values()) == 5
    assert out["RANGE"]["mean_net_r"] == pytest.approx(0.0) and out["TREND_UP"]["win_rate"] == pytest.approx(2 / 3)


def test_sidak_known_answers():
    assert ev.sidak_required_z(1) == pytest.approx(1.6449, abs=1e-3)
    n = 18
    assert ev.sidak_required_z(n) == pytest.approx(NormalDist().inv_cdf(0.95 ** (1 / n)))
    assert ev.sidak_required_z(n) > ev.sidak_required_z(5) > ev.sidak_required_z(1)
    assert ev.sidak_required_z(0) is None


def _score(name, mean, se, eligible=True):
    return StrategyScore(name, 0.3, mean, 0.55, "MEDIUM", 40, -3.0, eligible, None, edge_mean=mean, edge_stderr=se)


def test_versus_winner_gap_in_standard_errors():
    g = ev.versus_winner(_score("W", 0.40, 0.10), _score("X", 0.10, 0.10))
    assert g["edge_gap_r"] == pytest.approx(0.30) and g["z"] == pytest.approx(0.30 / np.hypot(0.1, 0.1))
    assert ev.versus_winner(_score("W", 0.4, 0.1), _score("X", None, None)) is None


def test_build_evidence_is_chronological_and_reports_frequency_and_cost_drag():
    ts = pd.date_range("2026-08-03 10:00", periods=20, freq="1D")
    net = np.array([1.0] * 14 + [-1.0] * 6)
    order = np.random.default_rng(0).permutation(20)  # shuffled input must not matter
    out = ev.build_strategy_evidence([None] * 20, net[order], net[order] + 0.1, ts[order], ["RANGE"] * 20, n_trading_days=40)
    assert out["trades_per_day"] == pytest.approx(0.5) and out["cost_drag_r"] == pytest.approx(0.1)
    assert out["holdout"]["holdout_mean_r"] == pytest.approx(-1.0)  # the LAST six by time, not by input position
    assert ev.build_strategy_evidence([], np.array([]), np.array([]), [], [], 10) == {"n_trades": 0}


# ----------------------------------------------------------------------------- decision view
def _output(selected="A"):
    scores = [_score("A", 0.40, 0.10), _score("B", 0.10, 0.10), _score("C", None, None, eligible=False)]
    evid = {"n_trades": 20, "trades_per_day": 0.5, "profit_factor_r": 1.4, "mean_net_r": 0.05, "cost_drag_r": 0.08,
            "holdout": {"in_sample_mean_r": 0.1, "holdout_mean_r": -0.2, "holdout_n": 6},
            "by_regime": {"RANGE": {"n": 12, "mean_net_r": 0.1, "win_rate": 0.5}, "TREND_UP": {"n": 8, "mean_net_r": -0.1, "win_rate": 0.4}}}
    intel = [SimpleNamespace(strategy_name=s.strategy_name, conditional_metrics={"n_days": 9, "expected_r": s.expected_r},
                             global_metrics={"n_trades": 20}, evidence=evid) for s in scores]
    return SimpleNamespace(strategy_intel=intel, ranking=SimpleNamespace(ranked=scores, selected_strategy=selected, tie_break=False))


def test_runner_up_explanation_names_the_gap_to_the_winner():
    rows = {r["Strategy"]: r for r in dv.explanation_rows(_output())}
    assert "standard errors) below the winner" in rows["B"]["Why"] and "below the winner" not in rows["A"]["Why"]
    assert "below the winner" not in dv.explanation_rows(_output(selected=None))[1]["Why"]


def test_evidence_rows_and_regime_rows():
    row = dv.evidence_rows(_output())[0]
    assert row["Profit factor (net R)"] == 1.4 and row["Holdout net R (last 30%)"] == -0.2 and row["Trades / day"] == 0.5
    reg = dv.regime_breakdown_rows(_output(), "A")
    assert [r["Regime at entry"] for r in reg] == ["Range", "Trend Up"] and reg[0]["Trades"] == 12


def test_multiple_testing_note_is_informational_and_absent_for_one_strategy():
    note = dv.multiple_testing_note(_output())
    assert note and "does not change the selection" in note
    out = _output()
    out.strategy_intel = out.strategy_intel[:1]
    assert dv.multiple_testing_note(out) is None


def test_rows_survive_outputs_without_evidence():
    out = _output()
    for s in out.strategy_intel:
        del s.evidence
    assert dv.evidence_rows(out)[0]["Trades"] == 0 and dv.regime_breakdown_rows(out, "A") == []
