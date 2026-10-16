"""Known-answer and property tests for the financial calculations.

Every expected value comes from an independent implementation written in this file (plain loops / hand arithmetic),
never from the production function under test.
"""
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from quant_intelligence.analytics.metrics import (
    annualised_sharpe, expectancy, max_drawdown, payoff_ratio, profit_factor,
)
from quant_intelligence.features import feature_engine as fe
from quant_intelligence.features.feature_engine import compute_features, infer_bar_minutes


def _bars(closes, start="2026-03-02 09:15", freq="5min", spread=0.5):
    ts = pd.date_range(start, periods=len(closes), freq=freq)
    c = pd.Series(closes, dtype=float)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + spread, "low": c - spread, "close": c, "volume": 1000.0})


# ======================================================================= shared metrics
def test_max_drawdown_known_answers():
    assert max_drawdown([]) == 0.0
    assert max_drawdown([1, 2, 3]) == 0.0  # only gains
    assert max_drawdown([-1, -2]) == -2.0  # the start (0) is a peak: the fall is 2, not 1
    assert max_drawdown([1, -2, 0]) == -3.0  # path 0,1,-2,0: peak 1 -> trough -2
    assert max_drawdown([5, 3, 4, 1]) == -4.0  # peak 5 -> trough 1
    assert max_drawdown([110, 120], start=100) == 0.0  # a rising equity path
    assert max_drawdown([90, 80], start=100) == -20.0  # equity path, not P&L path


def _brute_force_drawdown(values, start):
    path = [start] + list(values)
    worst = 0.0
    for i in range(len(path)):
        for j in range(i, len(path)):
            worst = min(worst, path[j] - path[i])
    return worst


@settings(max_examples=150, deadline=None)
@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), max_size=40), st.floats(min_value=-1e3, max_value=1e3))
def test_max_drawdown_matches_an_exhaustive_search_and_is_never_positive(values, start):
    got = max_drawdown(values, start)
    assert got <= 0.0
    assert got == pytest.approx(_brute_force_drawdown(values, start), abs=1e-6)


def test_max_drawdown_rejects_non_finite_input():
    with pytest.raises(ValueError):
        max_drawdown([1.0, float("nan")])


def test_profit_factor_payoff_expectancy_and_their_undefined_cases():
    pnl = [10.0, -5.0, 20.0, -5.0, 0.0]
    assert profit_factor(pnl) == pytest.approx(30.0 / 10.0)
    assert payoff_ratio(pnl) == pytest.approx(15.0 / 5.0)  # the 0 trade is neither a win nor a loss
    assert expectancy(pnl) == pytest.approx(4.0)
    assert profit_factor([5.0, 6.0]) is None  # no losing trade: not infinity
    assert payoff_ratio([-1.0, -2.0]) is None and payoff_ratio([1.0, 2.0]) is None
    assert profit_factor([]) is None and expectancy([]) is None


def test_sharpe_uses_the_sample_standard_deviation_and_needs_enough_observations():
    r = np.array([0.5, -0.2, 0.3, -0.1, 0.4, 0.2, -0.3, 0.6, 0.1, -0.4, 0.2, 0.3])
    expected = r.mean() / math.sqrt(((r - r.mean()) ** 2).sum() / (len(r) - 1)) * math.sqrt(252)
    assert annualised_sharpe(r) == pytest.approx(expected)
    assert annualised_sharpe(r[:5]) is None  # fewer than 10 observations
    assert annualised_sharpe([1.0] * 12) is None  # zero variance


# ======================================================================= indicators vs independent references
def _reference_rsi(close, period):
    """Wilder smoothing as a plain recursion, seeded with the first average gain / loss (pandas ewm adjust=False)."""
    gains = [None] + [max(close[i] - close[i - 1], 0.0) for i in range(1, len(close))]
    losses = [None] + [max(close[i - 1] - close[i], 0.0) for i in range(1, len(close))]
    ag = al = None
    out = [float("nan")] * len(close)
    for i in range(1, len(close)):
        ag = gains[i] if ag is None else ag * (1 - 1 / period) + gains[i] / period
        al = losses[i] if al is None else al * (1 - 1 / period) + losses[i] / period
        if i >= period:  # min_periods observations seen
            if ag == 0 and al == 0:
                out[i] = float("nan")
            elif al == 0:
                out[i] = 100.0
            else:
                out[i] = 100 - 100 / (1 + ag / al)
    return out


@pytest.mark.parametrize("period", [2, 14])
def test_rsi_matches_an_independent_wilder_recursion(period):
    rng = np.random.default_rng(3)
    close = (100 + np.cumsum(rng.normal(0, 1, 120))).tolist()
    got = fe._rsi(pd.Series(close), period).tolist()
    ref = _reference_rsi(close, period)
    for g, r in zip(got, ref):
        assert (math.isnan(g) and math.isnan(r)) or g == pytest.approx(r, abs=1e-9)


def test_rsi_edge_cases_all_gains_all_losses_and_no_movement():
    assert fe._rsi(pd.Series(np.arange(1.0, 40.0)), 14).iloc[-1] == 100.0
    assert fe._rsi(pd.Series(np.arange(40.0, 1.0, -1.0)), 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)
    flat = fe._rsi(pd.Series([100.0] * 30), 2)
    assert flat.isna().all(), "no movement is undefined, not overbought"


@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(min_value=1.0, max_value=1e5, allow_nan=False), min_size=20, max_size=80))
def test_rsi_always_lies_between_0_and_100(prices):
    rsi = fe._rsi(pd.Series(prices), 14).dropna()
    assert ((rsi >= -1e-9) & (rsi <= 100 + 1e-9)).all()


def test_atr_is_the_simple_mean_of_true_range_over_14_bars():
    rng = np.random.default_rng(5)
    close = 100 + np.cumsum(rng.normal(0, 1, 60))
    df = _bars(close.tolist(), spread=0.7)
    feats = compute_features(df)
    high, low, cl = df["high"].tolist(), df["low"].tolist(), df["close"].tolist()
    tr = [high[0] - low[0]] + [max(high[i] - low[i], abs(high[i] - cl[i - 1]), abs(low[i] - cl[i - 1])) for i in range(1, len(cl))]
    for i in range(13, len(cl)):
        assert feats["atr_14"].iloc[i] == pytest.approx(sum(tr[i - 13:i + 1]) / 14)
    assert feats["atr_14"].iloc[:13].isna().all()  # warm-up: no value before 14 true ranges exist


def test_ema_and_macd_match_the_recursive_definition():
    rng = np.random.default_rng(8)
    close = (100 + np.cumsum(rng.normal(0, 1, 260))).tolist()
    feats = compute_features(_bars(close))

    def ema(values, span):
        alpha, out, prev = 2 / (span + 1), [], None
        for v in values:
            prev = v if prev is None else alpha * v + (1 - alpha) * prev
            out.append(prev)
        return out

    e20, e50, e200 = ema(close, 20), ema(close, 50), ema(close, 200)
    for i in (25, 120, 259):
        assert feats["ema_20"].iloc[i] == pytest.approx(e20[i])
    assert feats["ema_50"].iloc[120] == pytest.approx(e50[120]) and feats["ema_200"].iloc[259] == pytest.approx(e200[259])
    macd = [a - b for a, b in zip(ema(close, 12), ema(close, 26))]
    signal = ema(macd, 9)
    assert feats["macd_line"].iloc[259] == pytest.approx(macd[259])
    assert feats["macd_signal"].iloc[259] == pytest.approx(signal[259])
    assert feats["macd_hist"].iloc[259] == pytest.approx(macd[259] - signal[259])


def test_vwap_resets_each_session_and_is_volume_weighted():
    day1 = _bars([100, 102, 101], start="2026-03-02 09:15")
    day2 = _bars([200, 198], start="2026-03-03 09:15")
    df = pd.concat([day1, day2], ignore_index=True)
    df["volume"] = [10.0, 30.0, 60.0, 5.0, 15.0]
    feats = compute_features(df)
    typical = ((df["high"] + df["low"] + df["close"]) / 3).tolist()
    v = df["volume"].tolist()
    assert feats["vwap"].iloc[2] == pytest.approx(sum(t * w for t, w in zip(typical[:3], v[:3])) / sum(v[:3]))
    assert feats["vwap"].iloc[3] == pytest.approx(typical[3])  # a new session starts from scratch
    assert feats["vwap"].iloc[4] == pytest.approx((typical[3] * 5 + typical[4] * 15) / 20)


def test_trend_slope_of_a_straight_line_is_its_slope_over_price():
    close = [100 + 0.5 * i for i in range(40)]
    feats = compute_features(_bars(close))
    assert feats["trend_slope"].iloc[30] == pytest.approx(0.5 / close[30])  # OLS slope per bar, as a fraction of price


def test_bollinger_bands_use_the_SAMPLE_standard_deviation_convention():
    """Pins the current convention on purpose. Classic Bollinger Bands use the POPULATION std (ddof=0); this code uses
    pandas' default sample std (ddof=1), so bands are sqrt(20/19) = 2.6 % wider. Changing it alters strategy signals,
    so it is a deliberate decision: see docs/CALCULATION_INVENTORY.md."""
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.normal(0, 1, 60))
    feats = compute_features(_bars(close.tolist()))
    window = close[40:60]
    assert feats["bb_mid_20"].iloc[59] == pytest.approx(window.mean())
    assert feats["bb_upper_20"].iloc[59] == pytest.approx(window.mean() + 2 * window.std(ddof=1))
    assert feats["bb_lower_20"].iloc[59] == pytest.approx(window.mean() - 2 * window.std(ddof=1))


def test_cpr_and_camarilla_use_the_previous_sessions_high_low_close():
    day1 = _bars([100, 104, 98, 101], start="2026-03-02 09:15", spread=1.0)  # H=105, L=97, C=101
    day2 = _bars([102, 103], start="2026-03-03 09:15", spread=1.0)
    feats = compute_features(pd.concat([day1, day2], ignore_index=True))
    h, l, c = 105.0, 97.0, 101.0
    pivot, bc = (h + l + c) / 3, (h + l) / 2
    row = feats.iloc[4]
    assert row["cpr_pivot"] == pytest.approx(pivot) and row["cpr_bc"] == pytest.approx(bc)
    assert row["cpr_tc"] == pytest.approx(2 * pivot - bc)
    assert row["camarilla_r3"] == pytest.approx(c + (h - l) * 1.1 / 4) and row["camarilla_s3"] == pytest.approx(c - (h - l) * 1.1 / 4)
    assert row["camarilla_r4"] == pytest.approx(c + (h - l) * 1.1 / 2) and row["camarilla_s4"] == pytest.approx(c - (h - l) * 1.1 / 2)
    assert feats["cpr_pivot"].iloc[:4].isna().all()  # the first session has no previous session


def test_donchian_channel_excludes_the_current_bar():
    rng = np.random.default_rng(4)
    close = (100 + np.cumsum(rng.normal(0, 1, 50))).tolist()
    df = _bars(close)
    feats = compute_features(df)
    i = 40
    assert feats["donchian_high_20"].iloc[i] == pytest.approx(df["high"].iloc[i - 20:i].max())
    assert feats["donchian_low_20"].iloc[i] == pytest.approx(df["low"].iloc[i - 20:i].min())


def test_returns_use_log_differences_and_momentum_is_a_simple_return():
    close = [100.0, 101.0, 103.0, 102.0, 104.0] + [105.0] * 30
    feats = compute_features(_bars(close))
    assert feats["return_1"].iloc[2] == pytest.approx(math.log(103 / 101))
    assert feats["return_5"].iloc[5] == pytest.approx(math.log(105 / 100))
    assert feats["momentum_20"].iloc[20] == pytest.approx(close[20] / close[0] - 1)


def test_realized_volatility_is_annualised_by_the_bars_actually_in_a_year():
    rng = np.random.default_rng(6)
    closes = (100 + np.cumsum(rng.normal(0, 0.5, 80))).tolist()
    five = compute_features(_bars(closes, freq="5min"))
    fifteen = compute_features(_bars(closes, freq="15min"))
    log_ret = np.diff(np.log(closes))
    window = log_ret[-20:]
    bars_per_session_5, bars_per_session_15 = 375 / 5, 375 / 15  # NSE session = 375 minutes
    assert five["realized_vol_20"].iloc[-1] == pytest.approx(window.std(ddof=1) * math.sqrt(252 * bars_per_session_5))
    # the same price path sampled every 15 minutes has 3x fewer bars a year: its annualisation factor is sqrt(3) smaller
    assert fifteen["realized_vol_20"].iloc[-1] == pytest.approx(window.std(ddof=1) * math.sqrt(252 * bars_per_session_15))


def test_bar_interval_is_inferred_ignoring_overnight_gaps():
    two_days = pd.concat([_bars([1, 2, 3], start="2026-03-02 09:15", freq="15min"), _bars([1, 2], start="2026-03-03 09:15", freq="15min")])
    assert infer_bar_minutes(two_days["timestamp"].reset_index(drop=True)) == 15.0
    assert infer_bar_minutes(pd.Series([pd.Timestamp("2026-03-02 09:15")])) == 5.0  # nothing to infer from: documented default


# ======================================================================= costs and R
def test_option_costs_match_an_independent_calculation():
    from quant_intelligence.backtesting.engine import _apply_costs
    from quant_intelligence.config.settings import SETTINGS
    from quant_intelligence.strategies.base_strategy import Setup, TradeResult

    c = SETTINGS.costs
    entry_underlying, gross, qty = 22000.0, 40.0, 75  # an underlying move of +40 points
    trade = TradeResult(Setup(pd.Timestamp("2026-03-02 10:00"), "LONG", entry_underlying, 21960.0, 22080.0),
                        pd.Timestamp("2026-03-02 10:30"), 22040.0, 1.0, 6, 0, 0, gross, "WIN", "TARGET")
    entry_prem = max(c.assumed_option_premium_pct / 100 * entry_underlying, 0.05)
    exit_prem = max(entry_prem + c.assumed_option_delta * gross, 0.05)
    buy, sell = entry_prem * qty, exit_prem * qty
    brokerage = 2 * c.brokerage_per_order_inr
    txn = (buy + sell) * c.opt_txn_rate_nse
    sebi = (buy + sell) * c.sebi_rate
    fees = sell * c.opt_stt_sell_rate_nse + txn + sebi + buy * c.stamp_buy_rate + c.gst_rate * (brokerage + txn + sebi)
    slippage = c.slippage_ticks * c.tick_size * qty * 2
    expected_net = (exit_prem - entry_prem) * qty - brokerage - fees - slippage
    commissions, got_fees, got_slip, net = _apply_costs(trade, entry_underlying, qty)
    assert (commissions, got_fees, got_slip) == pytest.approx((brokerage, fees, slippage))
    assert net == pytest.approx(expected_net)


def test_a_bought_option_cannot_lose_more_than_its_premium():
    from quant_intelligence.backtesting.engine import _apply_costs
    from quant_intelligence.config.settings import SETTINGS
    from quant_intelligence.strategies.base_strategy import Setup, TradeResult

    c, qty = SETTINGS.costs, 50
    entry_prem = c.assumed_option_premium_pct / 100 * 22000.0
    crash = -10 * entry_prem / c.assumed_option_delta  # an underlying move far larger than the whole premium
    trade = TradeResult(Setup(pd.Timestamp("2026-03-02 10:00"), "LONG", 22000.0, 21000.0, 23000.0),
                        pd.Timestamp("2026-03-02 11:00"), 22000.0 + crash, -5.0, 10, 0, 0, crash, "LOSS", "STOP")
    option_pnl_before_costs = _apply_costs(trade, 22000.0, qty)[3]
    assert option_pnl_before_costs > -(entry_prem * qty) - 1e4  # bounded by the premium paid plus costs, not by delta x move


def test_net_r_is_net_pnl_over_amount_risked():
    from quant_intelligence.backtesting.engine import _apply_costs, net_r_multiple
    from quant_intelligence.config.settings import SETTINGS
    from quant_intelligence.strategies.base_strategy import Setup, TradeResult

    qty = 75
    trade = TradeResult(Setup(pd.Timestamp("2026-03-02 10:00"), "LONG", 22000.0, 21980.0, 22040.0),
                        pd.Timestamp("2026-03-02 10:20"), 22040.0, 2.0, 4, 0, 0, 40.0, "WIN", "TARGET")
    risk = SETTINGS.costs.assumed_option_delta * 20.0 * qty
    assert net_r_multiple(trade, qty) == pytest.approx(_apply_costs(trade, 22000.0, qty)[3] / risk)


# ======================================================================= statistics used by the ranker
def test_cluster_robust_standard_error_known_answer():
    from quant_intelligence.analogues.analogue_engine import _cluster_robust_stderr

    values = np.array([1.0, 3.0, 2.0, 6.0])
    days = np.array(["a", "a", "b", "b"])
    resid = values - values.mean()  # mean 3: residuals -2, 0, -1, 3 -> day sums -2 and 2
    expected = math.sqrt(2 / 1 * (4 + 4)) / 4
    assert _cluster_robust_stderr(values, days) == pytest.approx(expected)
    assert math.isnan(_cluster_robust_stderr(values, np.array(["a"] * 4)))  # one cluster tells us nothing
    assert _cluster_robust_stderr(np.array([1.0, 3.0]), None) == pytest.approx(math.sqrt(2) / math.sqrt(2) )  # sd/sqrt(n)


def test_shrinkage_is_the_normal_normal_posterior():
    from quant_intelligence.ranking.ranking_engine import SHRINKAGE_TAU, _shrunk_edge

    mean, se, prior = 0.4, 0.2, 0.1
    tau2, se2 = SHRINKAGE_TAU**2, se**2
    post_var = 1 / (1 / tau2 + 1 / se2)
    post_mean = post_var * (prior / tau2 + mean / se2)
    got_mean, got_se = _shrunk_edge(mean, se, prior)
    assert got_mean == pytest.approx(post_mean) and got_se == pytest.approx(math.sqrt(post_var))
    assert _shrunk_edge(0.4, None, 0.1) == (0.4, None)  # no standard error: nothing to weigh the prior against


# ======================================================================= position sizing
def _account(equity):
    from quant_intelligence.risk.risk_engine import AccountState

    return AccountState(equity=equity, peak_equity=equity, daily_pnl=0.0, open_positions_count=0, trades_today=0)


def test_position_size_known_answer():
    from quant_intelligence.config.settings import SETTINGS
    from quant_intelligence.execution.auto_trader import size_position

    equity, entry, stop, lot = 1_000_000.0, 100.0, 90.0, 75
    budget = equity * SETTINGS.risk.max_risk_per_trade_pct * 0.9 / 100
    lots = int((budget / 10.0) // lot)
    cap = int((equity * SETTINGS.risk.max_capital_per_trade_pct / 100) // (entry * lot))
    assert size_position(_account(equity), entry, stop, lot) == min(lots, cap) * lot


@settings(max_examples=80, deadline=None)
@given(st.floats(min_value=1e5, max_value=1e8), st.floats(min_value=1.0, max_value=5e3), st.floats(min_value=0.5, max_value=0.99),
       st.integers(min_value=1, max_value=2000))
def test_size_is_whole_lots_within_the_risk_budget_and_grows_with_equity(equity, entry, stop_frac, lot):
    from quant_intelligence.config.settings import SETTINGS
    from quant_intelligence.execution.auto_trader import size_position

    stop = entry * stop_frac
    qty = size_position(_account(equity), entry, stop, lot)
    assert qty % lot == 0
    assert qty * abs(entry - stop) <= equity * SETTINGS.risk.max_risk_per_trade_pct / 100 + 1e-6  # never above the risk budget
    assert size_position(_account(equity * 2), entry, stop, lot) >= qty  # more equity never shrinks the size


# ======================================================================= risk engine: invalid input never approves
_BAD = st.one_of(st.just(float("nan")), st.just(float("inf")), st.just(0.0), st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False))


@settings(max_examples=120, deadline=None)
@given(_BAD, st.sampled_from(["entry", "stop", "target"]))
def test_a_non_finite_or_non_positive_price_is_never_approved(bad, field):
    from quant_intelligence.risk.risk_engine import ProposedTrade, evaluate_trade

    prices = {"entry": 100.0, "stop": 99.0, "target": 103.0}
    prices[field] = bad
    trade = ProposedTrade("Momentum", "LONG", prices["entry"], prices["stop"], prices["target"], 50, relative_volume=1.0)
    assert not evaluate_trade(_account(1_000_000), trade).approved


@settings(max_examples=80, deadline=None)
@given(st.integers(max_value=0, min_value=-1000))
def test_a_non_positive_quantity_is_never_approved(qty):
    from quant_intelligence.risk.risk_engine import ProposedTrade, evaluate_trade

    assert not evaluate_trade(_account(1_000_000), ProposedTrade("Momentum", "LONG", 100.0, 99.0, 103.0, qty)).approved


def test_gap_percent_is_open_over_previous_close_and_held_for_the_session():
    day1 = _bars([100, 101], start="2026-03-02 09:15")  # previous close 101
    day2 = _bars([103, 104, 105], start="2026-03-03 09:15")  # opens at 103
    feats = compute_features(pd.concat([day1, day2], ignore_index=True))
    expected = (103 - 101) / 101 * 100
    assert feats["gap_pct"].iloc[2:].tolist() == pytest.approx([expected] * 3)
    assert math.isnan(feats["gap_pct"].iloc[0])  # no previous session


def test_supertrend_follows_price_in_clean_trends():
    up = compute_features(_bars([100 + 0.4 * i for i in range(120)]))
    assert (up["supertrend_direction"].dropna().iloc[-30:] == 1).all()
    assert (up["supertrend"].iloc[-30:] < _bars([100 + 0.4 * i for i in range(120)])["close"].iloc[-30:]).all()  # support below price
    down = compute_features(_bars([200 - 0.4 * i for i in range(120)]))
    assert (down["supertrend_direction"].dropna().iloc[-30:] == -1).all()
    assert (down["supertrend"].iloc[-30:] > _bars([200 - 0.4 * i for i in range(120)])["close"].iloc[-30:]).all()  # resistance above


@pytest.mark.parametrize("equity,entry,stop,lot", [
    (float("nan"), 100.0, 90.0, 1), (float("inf"), 100.0, 90.0, 1), (0.0, 100.0, 90.0, 1), (-5.0, 100.0, 90.0, 1),
    (1e6, float("nan"), 90.0, 1), (1e6, 100.0, float("inf"), 1), (1e6, 100.0, 100.0, 1), (1e6, 100.0, 90.0, 0),
])
def test_sizing_refuses_invalid_inputs_instead_of_crashing_or_guessing(equity, entry, stop, lot):
    from quant_intelligence.execution.auto_trader import size_position

    assert size_position(_account(equity), entry, stop, lot) == 0
