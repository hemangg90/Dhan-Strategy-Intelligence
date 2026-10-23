"""Setup validation: a ranking or an analogue is not an entry; only an active, fresh, tradeable setup with room proceeds."""
import dataclasses
import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.setup_detector import SETUP_TRIGGERED, WAITING_FOR_SETUP
from quant_intelligence.execution.setup_validation import NO_TRADE, VALID, WAIT, validate_setup
from quant_intelligence.features.levels import Level
from quant_intelligence.features.liquidity import ACTIVE, BREAKOUT_ACCEPTED, SELL_SIDE, LiquidityEvent
from quant_intelligence.intelligence.market_context import build_market_context
from quant_intelligence.intelligence.options_context import build_options_context
from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow
from quant_intelligence.options.execution_quality import ExecutionLimits
from quant_intelligence.strategies.base_strategy import Setup

@pytest.fixture(autouse=True)
def levels_on():
    """The suite switches the level check off for older tests; this module is about it."""
    original = SETTINGS.levels_enabled
    object.__setattr__(SETTINGS, "levels_enabled", True)
    yield
    object.__setattr__(SETTINGS, "levels_enabled", original)


LIMITS = ExecutionLimits(max_spread_pct=8.0, max_quote_age_sec=120.0, min_premium=1.0, min_top_qty_lots=1.0)
EXPIRY = "2030-02-28"


def _frame(n_last=40, seed=5):
    rng = np.random.default_rng(seed)
    days = ["2030-01-02", "2030-01-03", "2030-01-04", "2030-01-07"]
    stamps = []
    for i, d in enumerate(days):
        count = 75 if i < 3 else n_last
        stamps += list(pd.date_range(f"{d} 09:15", periods=count, freq="5min"))
    n = len(stamps)
    closes = 24000 + np.cumsum(rng.normal(0, 8, n))
    rows, prev = [], closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + abs(rng.normal(0, 3)), min(prev, c) - abs(rng.normal(0, 3)), c))
        prev = c
    return pd.DataFrame({"timestamp": stamps, "open": [r[0] for r in rows], "high": [r[1] for r in rows], "low": [r[2] for r in rows],
                         "close": [r[3] for r in rows], "volume": rng.integers(500, 1500, n)})


def _row(strike, spot):
    mid_c, mid_p = max(spot - strike, 0) + 90.0, max(strike - spot, 0) + 90.0
    return StrikeRow(
        strike=float(strike), ce_ltp=mid_c, ce_oi=9000, ce_oi_change=50, ce_volume=900, ce_iv=18.0, ce_delta=0.5, ce_security_id=f"{strike}C",
        pe_ltp=mid_p, pe_oi=9500, pe_oi_change=-20, pe_volume=800, pe_iv=19.0, pe_delta=-0.5, pe_security_id=f"{strike}P",
        ce_bid=mid_c * 0.99, ce_ask=mid_c * 1.01, pe_bid=mid_p * 0.99, pe_ask=mid_p * 1.01, ce_bid_qty=900, ce_ask_qty=900,
        pe_bid_qty=900, pe_ask_qty=900)


def _chain(spot, now, wide=False):
    atm = round(spot / 50) * 50
    rows = [_row(atm + 50 * k, spot) for k in range(-4, 5)]
    if wide:
        rows = [dataclasses.replace(r, ce_bid=r.ce_ltp * 0.6, ce_ask=r.ce_ltp * 1.4, pe_bid=r.pe_ltp * 0.6, pe_ask=r.pe_ltp * 1.4) for r in rows]
    return ChainSnapshot("NIFTY", EXPIRY, spot, rows, fetched_at=now, lot_size=75, exchange_segment="NSE_FNO")


class Env:
    """A consistent decision moment: market context, options context, a setup and a ranking."""

    def __init__(self, direction="LONG", n_last=40, wide=False):
        self.df = _frame(n_last)
        self.now = self.df["timestamp"].iloc[-1].to_pydatetime() + dt.timedelta(minutes=1)
        self.market = build_market_context(self.df, underlying="NIFTY")
        spot = float(self.df["close"].iloc[-1])
        self.options = build_options_context(_chain(spot, self.now, wide), now=self.now, limits=LIMITS)
        atr = self.market.atr
        sign = 1 if direction == "LONG" else -1
        self.setup = Setup(self.df["timestamp"].iloc[-1], direction, spot, spot - sign * atr, spot + sign * 2 * atr, {"transaction": "BUY"})
        self.status = SimpleNamespace(status=SETUP_TRIGGERED, setup=self.setup)
        self.ranking = SimpleNamespace(is_no_trade=False, tie_break=False, selected_strategy="Momentum", reason="")

    def validate(self, **kw):
        args = dict(ranking=self.ranking, setup_status=self.status, market=self.market, options=self.options, now=self.now,
                    strategy_name="Momentum", transaction="BUY")
        args.update(kw)
        return validate_setup(**args)

    def with_levels(self, *levels):
        self.market = dataclasses.replace(self.market, levels=tuple(levels))
        return self


def test_a_live_setup_on_fresh_data_with_room_and_contracts_is_valid():
    env = Env()
    res = env.validate()
    assert res.valid and res.status == VALID and res.reasons == [] and res.message == ""
    assert all(res.checks[k] for k in ("ranking_tradable", "market_data_ok", "session_open", "bar_fresh", "setup_active",
                                       "setup_not_expired", "options_data_ok", "eligible_contracts"))


def test_a_no_trade_ranking_blocks_everything_else():
    env = Env()
    env.ranking = SimpleNamespace(is_no_trade=True, tie_break=False, selected_strategy=None, reason="strategies are tied")
    res = env.validate()
    assert res.status == NO_TRADE and res.code == "RANKING" and "tied" in res.reasons[0].message


def test_bad_market_data_blocks_with_the_reason():
    env = Env()
    env.market = build_market_context(env.df.head(10), underlying="NIFTY")
    res = env.validate()
    assert res.status == NO_TRADE and res.code == "DATA_QUALITY" and "bars" in res.message


def test_a_closed_session_blocks():
    env = Env()
    res = env.validate(now=dt.datetime(2030, 1, 7, 16, 5))
    assert res.status == NO_TRADE and res.code == "SESSION"
    weekend = env.validate(now=dt.datetime(2030, 1, 5, 11, 0))
    assert weekend.code == "SESSION"


def test_a_stale_last_bar_blocks():
    env = Env()
    res = env.validate(now=env.now + dt.timedelta(minutes=60))
    assert res.status == NO_TRADE and res.code == "STALE_DATA" and res.bar_age_minutes > SETTINGS.setup_max_bar_age_min


def test_a_setup_that_is_not_active_waits_and_no_setup_is_no_trade():
    env = Env()
    env.status = SimpleNamespace(status=WAITING_FOR_SETUP, setup=None)
    res = env.validate()
    assert res.status == WAIT and res.code == "SETUP_NOT_ACTIVE"
    env.status = SimpleNamespace(status="NO_SETUP", setup=None)
    assert env.validate().status == NO_TRADE


def test_a_setup_that_fired_too_long_ago_has_expired():
    env = Env()
    old = dataclasses.replace(env.setup, timestamp=env.df["timestamp"].iloc[-1 - (SETTINGS.setup_valid_bars + 3)])
    env.status = SimpleNamespace(status=SETUP_TRIGGERED, setup=old)
    res = env.validate()
    assert res.status == WAIT and res.code == "SETUP_EXPIRED"


def test_missing_or_unusable_option_data_blocks():
    env = Env()
    assert env.validate(options=None).code == "OPTIONS_DATA"
    stale_chain = _chain(float(env.df["close"].iloc[-1]), env.now - dt.timedelta(minutes=30))
    env.options = build_options_context(stale_chain, now=env.now, limits=LIMITS)
    res = env.validate()
    assert res.status == NO_TRADE and res.code == "OPTIONS_DATA" and "old" in res.message
    assert Env().validate(options=None, require_options=False).valid


def test_no_eligible_contract_blocks():
    env = Env(wide=True)
    res = env.validate()
    assert res.status == NO_TRADE and res.code == "NO_ELIGIBLE_CONTRACT"


def test_a_resistance_right_above_the_entry_leaves_no_room():
    env = Env()
    env.with_levels(Level(env.setup.entry_price + 0.3 * env.market.atr, "PDH", 2.0))
    res = env.validate()
    assert res.status == NO_TRADE and res.code == "NO_ROOM" and res.message.startswith("LEVEL CHECK: no room")


def test_a_level_beyond_the_target_moves_the_target_back_and_keeps_the_original_setup_intact():
    env = Env()
    entry, atr = env.setup.entry_price, env.market.atr
    env.with_levels(Level(entry + 1.6 * atr, "PDH", 2.0))
    res = env.validate()
    assert res.valid and res.setup is not env.setup and res.setup.target_price < env.setup.target_price
    assert env.setup.target_price == pytest.approx(entry + 2 * atr) and "levels" in res.setup.meta and res.level_review.startswith("Levels:")


def test_the_level_check_is_skipped_for_written_options_and_level_native_strategies():
    env = Env()
    env.with_levels(Level(env.setup.entry_price + 0.3 * env.market.atr, "PDH", 2.0))
    assert env.validate(transaction="SELL").valid
    assert env.validate(strategy_name="Camarilla Pivot Reversal").valid


def test_advisories_are_warnings_that_do_not_block():
    env = Env()
    env.market.structure.trend = "DOWN"  # structure points down against a long trend-following setup
    event = LiquidityEvent(BREAKOUT_ACCEPTED, SELL_SIDE, 23950.0, "PDL", 10, 12, env.market.timestamp, 23940.0, ACTIVE, 30, "x")
    env.market.liquidity.events.append(event)
    env.market = dataclasses.replace(env.market, volatility={**env.market.volatility, "atr_percentile_100": 0.99})
    res = env.validate()
    codes = {w.code for w in res.warnings}
    assert res.valid and {"STRUCTURE_OPPOSES", "LIQUIDITY_OPPOSES", "EXTREME_VOLATILITY"} <= codes


def test_a_short_setup_is_checked_in_its_own_direction():
    env = Env(direction="SHORT")
    env.with_levels(Level(env.setup.entry_price - 0.3 * env.market.atr, "PDL", 2.0))
    assert env.validate().code == "NO_ROOM"
    env.with_levels(Level(env.setup.entry_price + 0.3 * env.market.atr, "PDH", 2.0))  # behind the trade: irrelevant
    assert env.validate().valid


def test_levels_disabled_turns_the_room_check_off():
    env = Env()
    env.with_levels(Level(env.setup.entry_price + 0.3 * env.market.atr, "PDH", 2.0))
    original = SETTINGS.levels_enabled
    object.__setattr__(SETTINGS, "levels_enabled", False)
    try:
        assert env.validate().valid
    finally:
        object.__setattr__(SETTINGS, "levels_enabled", original)
