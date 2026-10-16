"""Regression tests for the trade simulator and run metrics (hand-built OHLC, no market data)."""
import math

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.backtesting.engine import daily_sharpe_sortino, run_backtest
from quant_intelligence.strategies.base_strategy import BaseStrategy, Setup, TradeResult
from quant_intelligence.strategies.registry import get_strategy


def _bars(rows):
    """rows: (open, high, low, close). Five-minute bars from 09:15 on one day."""
    ts = pd.date_range("2026-01-05 09:15", periods=len(rows), freq="5min")
    return pd.DataFrame({"timestamp": ts, "open": [r[0] for r in rows], "high": [r[1] for r in rows],
                         "low": [r[2] for r in rows], "close": [r[3] for r in rows], "volume": 1000.0})


def _long(entry=100.0, stop=99.0, target=102.0):
    return Setup(pd.Timestamp("2026-01-05 09:15"), "LONG", entry, stop, target)


def _short(entry=100.0, stop=101.0, target=98.0):
    return Setup(pd.Timestamp("2026-01-05 09:15"), "SHORT", entry, stop, target)


SIM = get_strategy("Momentum")


def test_a_stop_that_is_touched_fills_at_the_stop():
    bars = _bars([(100, 100, 100, 100), (99.6, 100.0, 98.9, 99.2)])
    t = SIM.simulate_trade(_long(), bars, 0)
    assert (t.exit_price, t.exit_reason, round(t.r_multiple, 2)) == (99.0, "STOP", -1.0)


def test_a_long_stop_gapped_through_fills_at_the_worse_open():
    bars = _bars([(100, 100, 100, 100), (97.5, 98.0, 97.0, 97.8)])  # opens 1.5 below the stop
    t = SIM.simulate_trade(_long(), bars, 0)
    assert t.exit_price == 97.5 and t.exit_reason == "STOP_GAP"
    assert round(t.r_multiple, 2) == -2.5  # worse than the planned -1R, as it would be in the market


def test_a_short_stop_gapped_through_fills_at_the_worse_open():
    bars = _bars([(100, 100, 100, 100), (102.4, 103.0, 102.0, 102.5)])
    t = SIM.simulate_trade(_short(), bars, 0)
    assert t.exit_price == 102.4 and t.exit_reason == "STOP_GAP" and round(t.r_multiple, 1) == -2.4


def test_stop_and_target_in_the_same_bar_resolves_to_the_stop():
    bars = _bars([(100, 100, 100, 100), (100.2, 102.5, 98.5, 100.0)])
    t = SIM.simulate_trade(_long(), bars, 0)
    assert t.exit_reason == "STOP" and t.exit_price == 99.0 and t.outcome == "LOSS"


def test_a_target_fills_at_the_target_price():
    bars = _bars([(100, 100, 100, 100), (100.5, 102.3, 100.4, 102.0)])
    t = SIM.simulate_trade(_long(), bars, 0)
    assert (t.exit_price, t.exit_reason, t.outcome) == (102.0, "TARGET", "WIN")


def test_the_entry_bar_itself_is_never_checked_against_the_stop():
    bars = _bars([(100, 101, 90, 100), (100, 100.5, 99.5, 100)])  # the entry bar has a huge range
    t = SIM.simulate_trade(_long(), bars, 0, max_holding_bars=1)
    assert t.exit_reason == "TIME" and t.exit_price == 100


def test_a_trade_the_data_cuts_off_is_flagged_data_end_not_closed():
    bars = _bars([(100, 100, 100, 100), (100, 100.5, 99.6, 100.2), (100.2, 100.6, 99.8, 100.4)])
    t = SIM.simulate_trade(_long(), bars, 0, max_holding_bars=75)
    assert t.exit_reason == "DATA_END"
    assert SIM.simulate_trade(_long(), bars, 2).exit_reason == "DATA_END"  # a setup on the very last bar


def test_a_trade_that_reaches_max_holding_is_a_genuine_time_exit():
    bars = _bars([(100, 100, 100, 100)] + [(100.2, 100.6, 99.8, 100.4)] * 6)
    assert SIM.simulate_trade(_long(), bars, 0, max_holding_bars=3).exit_reason == "TIME"


class _OneSetupAtEnd(BaseStrategy):
    name = "TestOnly"

    def generate_historical_setups(self, ohlcv, features):
        ts = ohlcv["timestamp"]
        return [Setup(ts.iloc[1], "LONG", 100.0, 99.0, 102.0), Setup(ts.iloc[-3], "LONG", 100.0, 99.0, 102.0)]

    def check_setup(self, market_state, recent_ohlcv):
        return None


def test_run_backtest_excludes_trades_still_open_when_the_data_ends():
    rows = [(100, 100, 100, 100)] + [(100.3, 100.6, 99.7, 100.2)] * 4 + [(100.2, 102.5, 100.0, 102.2)] + [(100.3, 100.6, 99.7, 100.2)] * 4
    bars = _bars(rows)
    feats = pd.DataFrame({"timestamp": bars["timestamp"]})
    res = run_backtest(_OneSetupAtEnd(), bars, feats, "NIFTY", split="TEST", max_holding_bars=75, persist=False)
    assert res.metrics["n_unresolved_excluded"] == 1
    assert res.metrics["n_trades"] == 1 and all(t.exit_reason != "DATA_END" for t in res.trades)


def _trade(day, hour=10):
    ts = pd.Timestamp(f"2026-01-{day:02d} {hour}:00")
    return TradeResult(Setup(ts, "LONG", 100, 99, 102), ts, 100, 0.0, 1, 0, 0, 0, "SCRATCH", "TIME")


def test_sharpe_needs_enough_days_and_uses_daily_net_r():
    few = [_trade(d) for d in range(1, 6)]
    s, so = daily_sharpe_sortino(few, np.array([1.0, -0.5, 0.5, 0.2, -0.1]))
    assert math.isnan(s) and math.isnan(so)

    trades = [_trade(d) for d in range(1, 13)]
    r = np.array([0.5, -0.2, 0.3, -0.1, 0.4, 0.2, -0.3, 0.6, 0.1, -0.4, 0.2, 0.3])
    sharpe, sortino = daily_sharpe_sortino(trades, r)
    assert sharpe == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
    dd = math.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert sortino == pytest.approx(r.mean() / dd * math.sqrt(252))


def test_trades_closed_on_one_day_are_one_daily_observation():
    """Per-trade annualisation would treat these 24 trades as 24 days; the daily series has only 12 points."""
    trades = [_trade(d, h) for d in range(1, 13) for h in (10, 14)]
    r = np.tile([0.4, -0.1], 12)
    daily = pd.Series(r).groupby(pd.to_datetime([t.exit_timestamp for t in trades]).normalize()).sum()
    assert len(daily) == 12
    sharpe, _ = daily_sharpe_sortino(trades, r)
    assert sharpe == pytest.approx(daily.mean() / daily.std(ddof=1) * math.sqrt(252)) if daily.std(ddof=1) > 0 else math.isnan(sharpe)


class _BadGeometry(BaseStrategy):
    name = "TestBadGeometry"

    def generate_historical_setups(self, ohlcv, features):
        ts = ohlcv["timestamp"]
        return [
            Setup(ts.iloc[1], "LONG", 100.0, 99.0, 98.0),   # target BELOW a long entry
            Setup(ts.iloc[2], "SHORT", 100.0, 101.0, 102.0),  # target ABOVE a short entry
            Setup(ts.iloc[3], "LONG", 100.0, 100.0, 102.0),  # zero stop distance
            Setup(ts.iloc[4], "LONG", 100.0, 99.0, 101.5),   # valid
        ]

    def check_setup(self, market_state, recent_ohlcv):
        return None


def test_setups_with_a_target_or_stop_on_the_wrong_side_are_not_simulated_as_trades():
    rows = [(100, 100, 100, 100)] * 4 + [(100.0, 100.4, 99.8, 100.2), (100.2, 101.8, 100.1, 101.6)] + [(101.6, 101.7, 101.5, 101.6)] * 80
    bars = _bars(rows)
    feats = pd.DataFrame({"timestamp": bars["timestamp"]})
    res = run_backtest(_BadGeometry(), bars, feats, "NIFTY", split="TEST", persist=False)
    assert res.metrics["n_invalid_setups_skipped"] == 3
    assert res.metrics["n_trades"] == 1 and res.trades[0].exit_reason == "TARGET" and res.trades[0].r_multiple > 0
