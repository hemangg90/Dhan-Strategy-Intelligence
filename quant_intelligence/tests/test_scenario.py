"""Scenario P&L of an option contract: repricing, fills, costs, tails and the strategy-path objective."""
import datetime as dt

import numpy as np
import pytest

from quant_intelligence.config.expiry_rules import time_to_expiry_years
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.options import black_scholes as bs
from quant_intelligence.options.contracts import OptionQuote
from quant_intelligence.options.execution_quality import ExecutionLimits, assess_execution_quality
from quant_intelligence.options.greeks import GreekSet, compute_greeks, rates_for
from quant_intelligence.options.scenario import ScenarioInputs, evaluate_scenarios, wilson_lower
from quant_intelligence.utils.market_profile import NSE

NOW = dt.datetime(2030, 1, 7, 10, 0)
EXPIRY = "2030-01-31"
LIMITS = ExecutionLimits(max_spread_pct=8.0, max_quote_age_sec=120.0, min_premium=1.0, min_top_qty_lots=1.0)
SPOT = 24000.0


def _price(strike, kind, sigma=0.18, expiry=EXPIRY):
    r, q, _ = rates_for("NIFTY")
    t = time_to_expiry_years(NOW, dt.date.fromisoformat(expiry), NSE, SETTINGS.trading_days_per_year)
    return bs.price(SPOT, strike, t, r, q, sigma, "CALL" if kind == "CE" else "PUT")


def _quote(strike=24000.0, kind="CE", spread_pct=1.0, expiry=EXPIRY, **kw):
    mid = _price(strike, kind, expiry=expiry)
    half = mid * spread_pct / 200
    base = dict(underlying="NIFTY", exchange_segment="NSE_FNO", security_id="1", option_type=kind, strike=strike, expiry=expiry,
                lot_size=75, bid=mid - half, ask=mid + half, bid_qty=750.0, ask_qty=750.0, ltp=mid, volume=1000.0, oi=9000.0,
                oi_change=0.0, iv=18.0, delta=None, gamma=None, theta=None, vega=None, prev_close=None, quote_time=NOW, source="t",
                unavailable=())
    base.update(kw)
    return OptionQuote(**base)


def _run(quote, win_prob=0.5, n_eff=100.0, transaction="BUY", target=24150.0, stop=23920.0, **inputs):
    greeks = compute_greeks(quote, SPOT, NOW)
    eq = assess_execution_quality(quote, NOW, transaction, quote.lot_size, LIMITS)
    base = dict(spot=SPOT, atr=40.0, now=NOW, horizon_bars=12, direction="LONG", target_price=target, stop_price=stop,
                win_prob=win_prob, effective_n=n_eff)
    base.update(inputs)
    return evaluate_scenarios(quote, greeks, ScenarioInputs(**base), eq, transaction, quote.lot_size)


def test_wilson_lower_bound_is_below_the_estimate_and_tightens_with_data():
    assert wilson_lower(0.6, 10) < wilson_lower(0.6, 100) < wilson_lower(0.6, 10_000) < 0.6
    assert wilson_lower(0.6, 10_000) == pytest.approx(0.6, abs=0.01)
    assert wilson_lower(0.5, 0) == 0.0


def test_a_scenario_run_has_a_grid_a_strategy_path_and_stated_assumptions():
    res = _run(_quote())
    assert res.available and res.quantity == 75 and res.move_basis == "ATR x sqrt(bars)" and res.sigma_move == pytest.approx(40 * np.sqrt(12))
    grid = [r for r in res.rows if r["weight"] is not None]
    assert len(grid) == 21 and sum(r["weight"] for r in grid) == pytest.approx(1.0)
    assert {r["scenario"] for r in res.rows} >= {"strategy target", "strategy stop"}
    assert len(res.assumptions) >= 5 and any("heuristic" in a for a in res.assumptions)


def test_a_bought_option_has_no_edge_without_a_view_and_costs_make_it_worse():
    res = _run(_quote())
    assert res.expected_neutral < 0  # decay and costs: a long option needs a move
    assert res.cvar10 <= res.expected_neutral <= res.best and res.worst <= res.cvar10
    wide = _run(_quote(spread_pct=6.0))
    assert wide.expected_neutral < res.expected_neutral and wide.entry > res.entry


def test_the_loss_on_a_bought_option_cannot_exceed_the_premium_plus_costs():
    res = _run(_quote(strike=24400.0))
    assert res.worst >= -(res.entry + 20)  # charges are a few rupees per unit at most here
    assert res.worst < 0


def test_the_target_pays_and_the_stop_loses_and_a_higher_delta_strike_gains_more_per_move():
    atm, otm = _run(_quote(24000.0)), _run(_quote(24300.0))
    assert atm.pnl_target > 0 > atm.pnl_stop
    assert atm.pnl_target > otm.pnl_target  # in rupees per unit, a nearer-the-money option moves more for the same underlying move
    assert otm.entry < atm.entry  # but the far strike is cheaper


def test_the_ranking_objective_uses_the_lower_bound_of_the_win_probability():
    thin, thick = _run(_quote(), win_prob=0.6, n_eff=8), _run(_quote(), win_prob=0.6, n_eff=2000)
    assert thin.win_prob_lower < thick.win_prob_lower < 0.6
    assert thin.expected_strategy_lower < thick.expected_strategy_lower and thin.objective < thick.objective
    assert thin.expected_strategy == pytest.approx(thick.expected_strategy)  # the point estimate is the same; the lower bound differs


def test_without_a_win_probability_there_is_no_strategy_expectation_or_objective():
    res = _run(_quote(), win_prob=None)
    assert res.available and res.objective is None and res.expected_strategy is None and res.pnl_target is not None
    none = _run(_quote(), target=None, stop=None)
    assert none.pnl_target is None and none.objective is None


def test_a_forecast_sigma_replaces_the_atr_scaling():
    res = _run(_quote(), sigma_move=120.0)
    assert res.sigma_move == 120.0 and res.move_basis == "volatility-model forecast"


def test_a_horizon_past_expiry_reprices_to_intrinsic_value():
    near = _quote(24000.0, expiry="2030-01-08")
    res = _run(near, horizon_bars=200, target=24300.0, stop=23800.0)
    assert res.available and res.pnl_target > 0  # deep in the money at expiry is worth its intrinsic value


def test_a_written_option_profits_from_decay_and_loses_when_the_market_runs():
    quote = _quote(24300.0)
    res = _run(quote, transaction="SELL", win_prob=0.5, target=23900.0, stop=24400.0)
    assert res.available and res.entry < quote.bid + 1  # sold at or below the bid
    assert res.pnl_stop < res.pnl_target  # a rally through the strike costs the writer
    assert res.expected_neutral > _run(quote).expected_neutral  # the writer collects the decay the buyer pays


def test_unusable_inputs_are_reported_not_defaulted():
    q = _quote()
    eq = assess_execution_quality(q, NOW, "BUY", 75, LIMITS)
    inputs = ScenarioInputs(SPOT, 40.0, NOW)
    assert "Greeks" in evaluate_scenarios(q, GreekSet(False, "no price"), inputs, eq).reason
    no_quote = assess_execution_quality(_quote(bid=None, ask=None), NOW, "BUY", 75, LIMITS)
    assert "entry fill" in evaluate_scenarios(q, compute_greeks(q, SPOT, NOW), inputs, no_quote).reason
    assert "ATR" in evaluate_scenarios(q, compute_greeks(q, SPOT, NOW), ScenarioInputs(SPOT, 0.0, NOW), eq).reason
