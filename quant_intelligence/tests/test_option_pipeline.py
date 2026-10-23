"""The option pipeline end to end: validated setup -> staged contract selection -> independent risk engine -> paper order."""
import dataclasses
import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.config.expiry_rules import time_to_expiry_years
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.database.db import get_session, init_db
from quant_intelligence.database.models import Position, ScanDecision
from quant_intelligence.execution import auto_trader
from quant_intelligence.execution.setup_detector import SETUP_TRIGGERED
from quant_intelligence.intelligence.market_context import build_market_context
from quant_intelligence.options import black_scholes as bs
from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow
from quant_intelligence.options.contract_scoring import select_best_contract
from quant_intelligence.options.option_selector import OptionSelectionError
from quant_intelligence.options.scenario import ScenarioInputs
from quant_intelligence.risk.options_risk import exposure_from_positions
from quant_intelligence.risk.risk_engine import AccountState, ProposedTrade, evaluate_trade, limits_for
from quant_intelligence.strategies.base_strategy import Setup
from quant_intelligence.utils.market_profile import NSE

EXPIRY = "2030-02-28"


# ------------------------------------------------------------------ fixtures and builders
@pytest.fixture
def flags():
    """Switch the production defaults back on for one test and restore the test-suite values afterwards."""
    names = ["setup_validation", "option_selection", "option_require_quote", "reconcile_before_order", "levels_enabled"]
    saved = {n: getattr(SETTINGS, n) for n in names}
    r = limits_for("NSE")
    risk_names = ["max_vega_pct", "max_theta_pct", "max_net_delta_pct", "max_premium_at_risk_pct", "max_underlying_premium_pct",
                  "min_trading_days_to_expiry", "max_order_lots"]
    saved_risk = {n: getattr(r, n) for n in risk_names}
    for n, v in (("setup_validation", True), ("option_selection", "SCORED"), ("option_require_quote", True),
                 ("reconcile_before_order", False), ("levels_enabled", False)):
        object.__setattr__(SETTINGS, n, v)
    init_db()
    from quant_intelligence.reports import decision_log

    decision_log._seen.clear()  # the decision log memoises (instrument, bar) pairs across tests
    with get_session() as s:
        s.query(Position).delete()
        s.query(ScanDecision).delete()
    yield r
    for n, v in saved.items():
        object.__setattr__(SETTINGS, n, v)
    for n, v in saved_risk.items():
        object.__setattr__(r, n, v)
    with get_session() as s:
        s.query(Position).delete()


def _frame(n_last=40, seed=5):
    rng = np.random.default_rng(seed)
    stamps = []
    for i, d in enumerate(["2030-01-02", "2030-01-03", "2030-01-04", "2030-01-07"]):
        stamps += list(pd.date_range(f"{d} 09:15", periods=75 if i < 3 else n_last, freq="5min"))
    closes = 24000 + np.cumsum(rng.normal(0, 25, len(stamps)))  # a realistic 5-minute range (ATR ~ 35 points)
    rows, prev = [], closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + abs(rng.normal(0, 3)), min(prev, c) - abs(rng.normal(0, 3)), c))
        prev = c
    return pd.DataFrame({"timestamp": stamps, "open": [r[0] for r in rows], "high": [r[1] for r in rows], "low": [r[2] for r in rows],
                         "close": [r[3] for r in rows], "volume": rng.integers(500, 1500, len(stamps))})


def _chain(spot, now, spread_pct=0.3, fetched_at=None, with_quotes=True):
    t = time_to_expiry_years(now, dt.date.fromisoformat(EXPIRY), NSE, SETTINGS.trading_days_per_year)
    atm = round(spot / 50) * 50
    rows = []
    for k in range(-4, 5):
        strike = atm + 50 * k
        c = bs.price(spot, strike, t, SETTINGS.risk_free_rate, SETTINGS.dividend_yield, 0.18, "CALL")
        p = bs.price(spot, strike, t, SETTINGS.risk_free_rate, SETTINGS.dividend_yield, 0.18, "PUT")
        h = spread_pct / 200
        cd = min(max(0.5 - (strike - spot) / 800.0, 0.05), 0.95)
        rows.append(StrikeRow(
            strike=float(strike), ce_ltp=c, ce_oi=9000 + 100 * k, ce_oi_change=50, ce_volume=900, ce_iv=18.0, ce_delta=cd,
            ce_security_id=f"{strike}C", pe_ltp=p, pe_oi=9500, pe_oi_change=-20, pe_volume=800, pe_iv=18.0, pe_delta=cd - 1.0,
            pe_security_id=f"{strike}P",
            ce_bid=c * (1 - h) if with_quotes else None, ce_ask=c * (1 + h) if with_quotes else None,
            pe_bid=p * (1 - h) if with_quotes else None, pe_ask=p * (1 + h) if with_quotes else None,
            ce_bid_qty=900, ce_ask_qty=900, pe_bid_qty=900, pe_ask_qty=900, ce_prev_close=c * 0.97, pe_prev_close=p * 1.01))
    return ChainSnapshot("NIFTY", EXPIRY, spot, rows, fetched_at=fetched_at if fetched_at is not None else now, lot_size=75,
                         exchange_segment="NSE_FNO")


class Market:
    """dict-like market state with `.get(key, default)`."""

    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        v = self.values.get(key, default)
        return default if v is None else v


class Env:
    def __init__(self, monkeypatch, direction="LONG", spread_pct=0.3, equity=5_000_000.0):  # one NIFTY lot is ~Rs 50k of premium
        self.df = _frame()
        self.now = self.df["timestamp"].iloc[-1].to_pydatetime() + dt.timedelta(minutes=1)
        self.spot = float(self.df["close"].iloc[-1])
        self.chain = _chain(self.spot, self.now, spread_pct)
        ctx = build_market_context(self.df, underlying="NIFTY")
        self.atr = ctx.atr
        sign = 1 if direction == "LONG" else -1
        self.setup = Setup(self.df["timestamp"].iloc[-1], direction, self.spot, self.spot - sign * self.atr,
                           self.spot + sign * 2 * self.atr, {"transaction": "BUY"})
        ranked = [SimpleNamespace(strategy_name="Momentum", expected_r=0.2, confidence_label="MEDIUM")]
        ranking = SimpleNamespace(is_no_trade=False, tie_break=False, selected_strategy="Momentum", reason="", ranked=ranked)
        intel = [SimpleNamespace(strategy_name="Momentum", conditional_metrics={"prob_positive_return": 0.58, "n_effective": 40.0, "sample_size": 40})]
        self.output = SimpleNamespace(
            instrument="NIFTY", ohlcv=self.df, features=None, market_state=Market(atr_14=self.atr, relative_volume=1.0, realized_vol_20=0.16),
            ranking=ranking, data_quality_status="OK", timestamp=self.df["timestamp"].iloc[-1], strategy_intel=intel,
            regime_label="TRENDING", regime_axes={}, regime_clarity=0.5, regime_age_bars=3, regime_probabilities={"TRENDING": 1.0})
        self.broker = PaperBroker(starting_capital=equity)
        self.equity = equity
        monkeypatch.setattr(auto_trader, "get_strategy", lambda n: object())
        monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, **k: SimpleNamespace(status=SETUP_TRIGGERED, setup=self.setup))
        monkeypatch.setattr(auto_trader, "get_underlying_info", lambda u: {"lot_size": 75})
        monkeypatch.setattr(auto_trader, "now_ist", lambda: self.now)
        monkeypatch.setattr("quant_intelligence.risk.risk_engine.now_ist", lambda: self.now)  # one clock for the whole decision

    def account(self):
        return AccountState(equity=self.equity, peak_equity=self.equity, daily_pnl=0.0, open_positions_count=len(self.broker.get_open_positions()),
                            trades_today=0, exposure_by_strategy={}, total_exposure=0.0, broker_connected=True, kill_switch_engaged=False,
                            option_exposure=exposure_from_positions(self.broker.get_open_positions()))

    def run(self, chain=None):
        return auto_trader.run_auto_option_cycle(self.output, "NIFTY", self.broker, client=None, chain=chain or self.chain, account=self.account())


# ------------------------------------------------------------------ contract selection: strict quotes and scenario ranking
def test_a_contract_without_a_two_sided_quote_is_not_eligible_when_quotes_are_required(flags):
    now = dt.datetime(2030, 1, 7, 10, 0)
    chain = _chain(24000.0, now, with_quotes=False)
    with pytest.raises(OptionSelectionError) as err:
        select_best_contract(chain, "LONG", "BUY", 75, now=now, require_quote=True)
    assert "NO SUITABLE CONTRACT" in str(err.value) and "no two-sided quote" in str(err.value)
    assert select_best_contract(chain, "LONG", "BUY", 75, now=now, require_quote=False).contract.security_id  # the old tolerant mode


def test_a_stale_snapshot_leaves_no_suitable_contract(flags):
    now = dt.datetime(2030, 1, 7, 10, 0)
    chain = _chain(24000.0, now, fetched_at=now - dt.timedelta(seconds=600))
    with pytest.raises(OptionSelectionError) as err:
        select_best_contract(chain, "LONG", "BUY", 75, now=now, require_quote=True)
    assert "stale" in str(err.value)


def test_the_contract_is_priced_at_the_executable_side_and_keeps_the_last_price(flags):
    now = dt.datetime(2030, 1, 7, 10, 0)
    chain = _chain(24000.0, now)
    buy = select_best_contract(chain, "LONG", "BUY", 75, now=now).contract
    row = next(r for r in chain.strikes if r.strike == buy.strike)
    assert buy.premium == pytest.approx(row.ce_ask) and buy.last_price == pytest.approx(row.ce_ltp) and buy.premium > buy.last_price
    sell = select_best_contract(chain, "LONG", "SELL", 75, now=now).contract
    assert sell.premium < sell.last_price  # a writer sells at the bid


def _scenario(chain, now, atr=40.0, **kw):
    base = dict(spot=chain.spot_price, atr=atr, now=now, horizon_bars=12, direction="LONG", target_price=chain.spot_price + 2 * atr,
                stop_price=chain.spot_price - atr, win_prob=0.58, effective_n=40.0)
    base.update(kw)
    return ScenarioInputs(**base)


def test_scenarios_rank_the_candidates_by_the_lower_bound_objective(flags):
    now = dt.datetime(2030, 1, 7, 10, 0)
    chain = _chain(24000.0, now)
    choice = select_best_contract(chain, "LONG", "BUY", 75, now=now, scenario=_scenario(chain, now))
    assert choice.basis == "scenario objective" and choice.scored.scenario.available
    objectives = [s.objective for s in choice.ranking]
    assert all(o is not None for o in objectives) and objectives == sorted(objectives, reverse=True)
    assert "expected net" in choice.summary and "ranked by scenario objective" in choice.summary
    assert choice.scored.scenario.win_prob_lower < 0.58  # the thin-sample haircut


def test_without_scenario_inputs_the_quality_score_ranks(flags):
    now = dt.datetime(2030, 1, 7, 10, 0)
    choice = select_best_contract(_chain(24000.0, now), "LONG", "BUY", 75, now=now)
    assert choice.basis == "quality score" and choice.scored.scenario is None


def test_a_strike_over_the_capital_cap_is_rejected_with_the_reason(flags):
    now = dt.datetime(2030, 1, 7, 10, 0)
    chain = _chain(24000.0, now)
    with pytest.raises(OptionSelectionError) as err:
        select_best_contract(chain, "LONG", "BUY", 75, 1_000.0, now=now)
    assert "over the per-trade cap" in str(err.value)


# ------------------------------------------------------------------ the full validated cycle
def test_a_valid_setup_flows_through_selection_risk_and_a_paper_fill(flags, monkeypatch):
    env = Env(monkeypatch)
    seen = {}
    real = auto_trader.evaluate_trade
    monkeypatch.setattr(auto_trader, "evaluate_trade", lambda acct, trade: seen.setdefault("trade", trade) and real(acct, trade))
    res = env.run()
    assert res.order is not None and res.order.status == "FILLED", res.reason
    trade = seen["trade"]
    assert trade.option["greeks"]["spot"] == env.spot and trade.option["trading_days_to_expiry"] > 30
    assert trade.bid and trade.ask and trade.ask > trade.bid and trade.quote_time == env.now and trade.lot_size == 75
    pos = env.broker.get_open_positions()[0]
    assert set(pos["entry_greeks"]) == {"delta", "gamma", "vega", "theta", "spot"} and pos["entry_greeks"]["delta"] > 0
    assert pos["entry_price"] >= trade.ask  # filled at the ask or worse, never at the last price
    assert "ranked by scenario objective" in res.reason or res.contract_review.endswith("scenario objective)")
    row = exposure_from_positions(env.broker.get_open_positions())
    assert row["premium_at_risk"] > 0 and row["unknown_positions"] == 0 and row["vega"] > 0 and row["theta"] < 0


def test_the_greeks_are_stored_with_the_position_in_the_database(flags, monkeypatch):
    env = Env(monkeypatch)
    assert env.run().order.status == "FILLED"
    with get_session() as s:
        row = s.query(Position).first()
        assert row.entry_greeks and row.entry_greeks["spot"] == env.spot


def test_a_setup_that_is_not_active_never_reaches_selection_or_risk(flags, monkeypatch):
    env = Env(monkeypatch)
    monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, **k: SimpleNamespace(status="WAITING_FOR_SETUP", setup=None))
    res = env.run()
    assert res.order is None and res.risk_decision_id is None


def test_stale_option_data_stops_the_trade_at_validation(flags, monkeypatch):
    env = Env(monkeypatch)
    stale = _chain(env.spot, env.now, fetched_at=env.now - dt.timedelta(minutes=10))
    res = env.run(stale)
    assert res.order is None and res.reason.startswith("SETUP VALIDATION [OPTIONS_DATA]") and "old" in res.reason
    with get_session() as s:
        row = s.query(ScanDecision).first()
        assert row.reason_class == "OPTIONS_DATA" and row.status == "SKIPPED"


def test_a_session_that_is_closed_stops_the_trade(flags, monkeypatch):
    env = Env(monkeypatch)
    monkeypatch.setattr(auto_trader, "now_ist", lambda: dt.datetime(2030, 1, 7, 16, 30))
    res = env.run()
    assert res.order is None and "SETUP VALIDATION [SESSION]" in res.reason


def test_the_risk_engine_vetoes_on_portfolio_vega_and_nothing_reaches_the_broker(flags, monkeypatch):
    env = Env(monkeypatch)
    object.__setattr__(flags, "max_vega_pct", 0.001)
    res = env.run()
    assert res.order is None and res.reason.startswith("RISK ENGINE VETO") and "Vega" in res.reason
    assert env.broker.get_open_positions() == []


def test_the_risk_engine_vetoes_an_option_whose_greeks_are_unavailable(flags, monkeypatch):
    env = Env(monkeypatch)
    object.__setattr__(flags, "max_theta_pct", 5.0)
    monkeypatch.setattr(auto_trader, "compute_greeks", lambda q, s, n: SimpleNamespace(available=False, reason="x", trading_days=30.0))
    res = env.run()
    assert res.order is None and "Greeks of the proposed option are unavailable" in res.reason


def test_a_second_position_is_checked_against_the_open_book(flags, monkeypatch):
    env = Env(monkeypatch)
    assert env.run().order.status == "FILLED"
    one = exposure_from_positions(env.broker.get_open_positions())
    # the first position's vega is the whole book; a limit just above it forbids adding another similar position
    object.__setattr__(flags, "max_vega_pct", one["vega"] / env.equity * 100 * 1.3)
    other = Env(monkeypatch)
    other.broker = env.broker
    other.output.instrument = "BANKNIFTY"
    chain2 = dataclasses.replace(env.chain, underlying="BANKNIFTY")
    monkeypatch.setattr(auto_trader, "get_underlying_info", lambda u: {"lot_size": 75})
    res = auto_trader.run_auto_option_cycle(other.output, "BANKNIFTY", env.broker, None, chain2, other.account())
    assert res.order is None and ("Vega" in res.reason or "already" in res.reason or "VALIDATION" in res.reason)


def test_a_book_that_disagrees_with_the_database_blocks_new_orders(flags, monkeypatch):
    env = Env(monkeypatch)
    object.__setattr__(SETTINGS, "reconcile_before_order", True)
    first = env.run()
    assert first.order.status == "FILLED"
    with get_session() as s:
        s.query(Position).delete()  # the saved copy is lost
    env.broker.positions.clear()
    env.broker.positions["ghost"] = {"position_id": "ghost", "status": "OPEN", "security_id": "9", "transaction": "BUY", "quantity": 75,
                                     "instrument": "BANKNIFTY", "entry_price": 1.0}
    again = Env(monkeypatch)
    again.broker = env.broker
    res = again.run()
    assert res.order is None and res.reason.startswith("RECONCILIATION FAILED")


# ------------------------------------------------------------------ the risk engine on its own
def _account(**kw):
    base = dict(equity=1_000_000.0, peak_equity=1_000_000.0, daily_pnl=0.0, open_positions_count=0, trades_today=0, exposure_by_strategy={},
                total_exposure=0.0, broker_connected=True, kill_switch_engaged=False)
    base.update(kw)
    return AccountState(**base)


def _option_trade(**kw):
    now = dt.datetime(2030, 1, 7, 10, 0)
    base = dict(strategy_name="Momentum", direction="LONG", entry_price=100.0, stop_price=90.0, target_price=130.0, quantity=75,
                relative_volume=1.0, quote_time=now, long_premium=True, instrument="NIFTY", bid=99.0, ask=101.0, lot_size=75,
                option={"underlying": "NIFTY", "transaction": "BUY", "greeks": {"delta": 0.5, "gamma": 0.001, "vega": 10.0, "theta": -8.0,
                                                                              "spot": 24000.0}, "trading_days_to_expiry": 5.0})
    base.update(kw)
    return ProposedTrade(**base)


NOW = dt.datetime(2030, 1, 7, 10, 0, 30)


def test_an_option_trade_needs_a_quote_time_and_a_fresh_one(flags):
    assert evaluate_trade(_account(), _option_trade(quote_time=None), NOW).reason.startswith("Option quote time unknown")
    stale = evaluate_trade(_account(), _option_trade(quote_time=NOW - dt.timedelta(seconds=300)), NOW)
    assert not stale.approved and "stale" in stale.reason and "max 120s" in stale.reason
    assert evaluate_trade(_account(), _option_trade(), NOW).approved


def test_an_option_trade_needs_a_quote_and_a_spread_within_the_option_limit(flags):
    no_quote = evaluate_trade(_account(), _option_trade(bid=None, ask=None), NOW)
    assert not no_quote.approved and "Spread unknown" in no_quote.reason
    wide = evaluate_trade(_account(), _option_trade(bid=80.0, ask=120.0, entry_price=100.0), NOW)
    assert not wide.approved and "exceeds max 8%" in wide.reason
    ok = evaluate_trade(_account(), _option_trade(bid=98.0, ask=102.0, stop_price=80.0), NOW)  # 4% spread: fine for an option (was 2% for all)
    assert ok.approved


def test_the_option_checks_are_skipped_for_a_trade_that_is_not_an_option(flags):
    plain = ProposedTrade("Momentum", "LONG", 100.0, 99.0, 103.0, 50, 1.0, instrument="TCS")
    assert evaluate_trade(_account(), plain, NOW).approved


def test_the_expiry_and_order_size_limits(flags):
    object.__setattr__(flags, "min_trading_days_to_expiry", 0.5)
    close = _option_trade()
    close.option["trading_days_to_expiry"] = 0.2
    assert "trading days to expiry" in evaluate_trade(_account(), close, NOW).reason
    object.__setattr__(flags, "min_trading_days_to_expiry", 0.0)
    object.__setattr__(flags, "max_order_lots", 1)
    assert "exceeds the maximum order size" in evaluate_trade(_account(), _option_trade(quantity=150), NOW).reason
