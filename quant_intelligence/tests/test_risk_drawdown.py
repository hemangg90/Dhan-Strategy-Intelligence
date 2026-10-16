"""Drawdown controls: marked-to-market equity, multi-day peak, size throttle and the new vetoes."""
import datetime as dt

import pytest

from quant_intelligence.execution import engine as engine_module
from quant_intelligence.execution.auto_trader import size_position
from quant_intelligence.risk.risk_engine import (
    AccountState,
    ProposedTrade,
    evaluate_trade,
    loss_streak,
    risk_multiplier,
)

NOW = dt.datetime(2026, 10, 5, 12, 0)


def _account(**overrides):
    base = dict(equity=1_000_000, peak_equity=1_000_000, daily_pnl=0.0, open_positions_count=0, trades_today=0)
    base.update(overrides)
    return AccountState(**base)


def _trade(**overrides):
    base = dict(strategy_name="Momentum", direction="LONG", entry_price=100.0, stop_price=99.0,
                target_price=103.0, quantity=50, relative_volume=1.0, instrument="NIFTY")
    base.update(overrides)
    return ProposedTrade(**base)


def _closed(*pnls, strategy="Momentum", last_at=NOW - dt.timedelta(minutes=5)):
    n = len(pnls)
    return [{"strategy": strategy, "net_pnl": p, "closed_at": last_at - dt.timedelta(minutes=10 * (n - 1 - i))}
            for i, p in enumerate(pnls)]


# ---------------------------------------------------------------- size throttle
@pytest.mark.parametrize("equity,expected", [(980_000, 1.0), (945_000, 0.625), (920_100, 0.25)])
def test_size_shrinks_linearly_with_drawdown(equity, expected):
    m, _ = risk_multiplier(_account(equity=equity))
    assert m == pytest.approx(expected, abs=0.01)


def test_half_size_after_half_the_daily_loss_or_a_two_trade_losing_streak():
    assert risk_multiplier(_account(daily_pnl=-16_000))[0] == 0.5  # 1.6% of a 3% limit
    assert risk_multiplier(_account(closed_today=_closed(-100, -200)))[0] == 0.5
    assert risk_multiplier(_account(closed_today=_closed(-100, 50)))[0] == 1.0  # a winner ends the streak


def test_size_position_follows_the_throttle_and_still_passes_the_risk_check():
    # Wide stop, so the risk budget (not the 5% capital cap) decides the size.
    full = size_position(_account(), 100.0, 50.0)
    throttled_account = _account(equity=945_000)
    throttled = size_position(throttled_account, 100.0, 50.0)
    assert full == 180 and 0 < throttled < full * 0.7
    trade = _trade(entry_price=100.0, stop_price=50.0, quantity=throttled)
    assert evaluate_trade(throttled_account, trade, now=NOW).approved
    # the old full size is now too large
    assert not evaluate_trade(throttled_account, _trade(entry_price=100.0, stop_price=50.0, quantity=full), now=NOW).approved


# ---------------------------------------------------------------- vetoes
def test_max_open_positions():
    d = evaluate_trade(_account(open_positions_count=4), _trade(), now=NOW)
    assert not d.approved and "4 positions already open" in d.reason


def test_third_same_direction_index_position_is_vetoed_but_other_bets_are_allowed():
    open_longs = [{"instrument": "BANKNIFTY", "direction": "LONG", "risk": 1000.0},
                  {"instrument": "SENSEX", "direction": "LONG", "risk": 1000.0}]
    acct = _account(open_positions_count=2, open_positions=open_longs)
    d = evaluate_trade(acct, _trade(instrument="NIFTY", direction="LONG"), now=NOW)
    assert not d.approved and "group INDICES" in d.reason
    short = _trade(instrument="NIFTY", direction="SHORT", stop_price=101.0, target_price=97.0)
    assert evaluate_trade(acct, short, now=NOW).approved
    assert evaluate_trade(acct, _trade(instrument="RELIANCE", direction="LONG"), now=NOW).approved


def test_bank_stocks_form_one_group_and_group_risk_is_capped():
    acct = _account(open_positions_count=1, open_positions=[{"instrument": "HDFCBANK", "direction": "SHORT", "risk": 15_000.0}])
    d = evaluate_trade(acct, _trade(instrument="ICICIBANK", direction="LONG", entry_price=100, stop_price=90, quantity=600), now=NOW)
    assert not d.approved and "Open risk in group" in d.reason  # 15k + 6k = 2.1% > 2%


def test_losing_streak_pauses_entries_for_an_hour():
    acct = _account(closed_today=_closed(-1, -1, -1, strategy="Other"))
    d = evaluate_trade(acct, _trade(), now=NOW)
    assert not d.approved and "paused until" in d.reason
    later = NOW + dt.timedelta(minutes=56)  # 61 min after the third loss
    assert evaluate_trade(acct, _trade(quantity=10), now=later).approved


def test_strategy_stops_for_the_day_after_two_losses():
    acct = _account(closed_today=_closed(-1, 5, -1, 5))  # streak broken, but Momentum lost twice
    d = evaluate_trade(acct, _trade(), now=NOW)
    assert not d.approved and "done for the day" in d.reason
    assert evaluate_trade(acct, _trade(strategy_name="Breakout"), now=NOW).approved


def test_profit_lock_after_giving_back_half_of_a_good_day():
    locked = _account(day_peak_pnl=20_000, daily_pnl=9_000)  # peaked at 2%, gave back 55%
    d = evaluate_trade(locked, _trade(), now=NOW)
    assert not d.approved and "Profit lock" in d.reason
    assert evaluate_trade(_account(day_peak_pnl=20_000, daily_pnl=12_000), _trade(), now=NOW).approved
    assert evaluate_trade(_account(day_peak_pnl=10_000, daily_pnl=0.0), _trade(), now=NOW).approved  # never hit 1.5%


def test_loss_streak_counts_only_the_trailing_losers():
    assert loss_streak(_account(closed_today=_closed(-1, 2, -1, -1))) == 2


def test_live_positions_map_to_the_underlying_view():
    from quant_intelligence.ui.state import live_position_view

    def view(qty, opt):
        return live_position_view({"tradingSymbol": "NIFTY-Oct2026-25000-CE", "netQty": qty, "drvOptionType": opt})

    assert view(75, "CALL") == {"instrument": "NIFTY", "direction": "LONG", "risk": 0.0}
    assert view(-75, "CALL")["direction"] == "SHORT"
    assert view(75, "PUT")["direction"] == "SHORT"
    assert view(-75, "PUT")["direction"] == "LONG"


# ---------------------------------------------------------------- engine bookkeeping
@pytest.fixture
def fresh_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_module, "_PEAK_RESET_FILE", tmp_path / "peak.json")
    eng = engine_module.TradingEngine()
    eng._restored = True  # don't touch the test database
    return eng


def _open_position(eng, pid, entry, qty, last=None, transaction="BUY"):
    eng.broker.positions[pid] = {
        "position_id": pid, "instrument": "NIFTY", "underlying": "NIFTY", "strategy_name": "Momentum",
        "direction": "LONG", "quantity": qty, "entry_price": entry, "stop_price": entry * 0.5,
        "target_price": entry * 2, "status": "OPEN", "transaction": transaction, "option_type": "CE",
        "last_price": last,
    }


def test_open_losses_count_toward_equity_and_the_daily_loss_limit(fresh_engine):
    _open_position(fresh_engine, "P1", entry=200.0, qty=1000, last=160.0)  # -40k open
    acct = fresh_engine.account_state()
    assert acct.unrealised_pnl == -40_000
    assert acct.equity == fresh_engine.broker.cash - 40_000
    d = evaluate_trade(acct, _trade(), now=NOW)
    assert not d.approved and "Daily loss" in d.reason  # 4% open loss > 3%, nothing has closed yet


def test_drawdown_peak_survives_a_day_roll(fresh_engine, monkeypatch):
    fresh_engine.peak_equity = 1_050_000
    fresh_engine.broker.cash = 990_000
    monkeypatch.setattr(engine_module, "today_ist", lambda: fresh_engine._day + dt.timedelta(days=1))
    acct = fresh_engine.account_state()
    assert acct.peak_equity == 1_050_000  # yesterday's high-water mark still counts
    assert acct.daily_pnl == 0.0 and acct.trades_today == 0


def test_restart_rebuilds_equity_and_peak_from_all_closed_trades():
    history = [(NOW - dt.timedelta(days=3), 30_000), (NOW - dt.timedelta(days=2), -50_000), (NOW, -10_000)]
    equity, peak = engine_module._equity_and_peak(1_000_000, history, None)
    assert (equity, peak) == (970_000, 1_030_000)


def test_manual_peak_reset_restarts_the_high_water_mark(fresh_engine):
    fresh_engine.peak_equity = 1_100_000
    new_peak = fresh_engine.reset_drawdown_peak()
    assert new_peak == fresh_engine.broker.cash and fresh_engine.account_state().peak_equity == new_peak
    saved = engine_module._load_peak_reset()
    history = [(saved["at"] - dt.timedelta(days=1), 100_000), (saved["at"] + dt.timedelta(hours=1), 5_000)]
    equity, peak = engine_module._equity_and_peak(1_000_000, history, saved)
    assert peak == max(saved["equity"], 1_105_000)  # the pre-reset high is ignored; later closes count
