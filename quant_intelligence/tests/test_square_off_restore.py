"""End-of-session square-off and restart recovery of the paper book."""
import datetime as dt
import time

from quant_intelligence.brokers.base_broker import OrderRequest
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.database.db import get_session, init_db
from quant_intelligence.database.models import Position
from quant_intelligence.execution import engine as engine_module
from quant_intelligence.execution.square_off import EOD_REASON, in_close_window, square_off_positions
from quant_intelligence.utils.market_profile import MCX, NSE
from quant_intelligence.utils.timeutil import today_ist


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _order(instrument, security_id, transaction="BUY"):
    return OrderRequest(
        strategy_name="t", instrument=instrument, direction="LONG", quantity=100, order_type="MARKET",
        price=100.0, stop_price=50.0, target_price=200.0, security_id=security_id, exchange_segment="X",
        product_type="INTRADAY", transaction_type=transaction, option_type="CE", strike=1.0, expiry="2099-01-01",
        lot_size=1,
    )


class _Client:
    def __init__(self, quotes, configured=True):
        self.quotes, self.configured, self.asked = quotes, configured, None

    def is_configured(self):
        return self.configured

    def get_ltp(self, payload):
        self.asked = payload
        return {"data": {"NSE_FNO": {k: {"last_price": v} for k, v in self.quotes.items() if k in ("11", "12")},
                         "MCX_COMM": {k: {"last_price": v} for k, v in self.quotes.items() if k in ("21",)}}}


def test_close_window_covers_only_the_last_minutes_of_a_trading_day():
    tue = dt.date(2026, 9, 29)
    at = lambda h, m, d=tue: dt.datetime.combine(d, dt.time(h, m))
    assert not in_close_window(at(15, 24), NSE, 5)
    assert in_close_window(at(15, 25), NSE, 5) and in_close_window(at(15, 29), NSE, 5)
    assert not in_close_window(at(15, 30), NSE, 5)  # market closed: is_market_open covers this case
    assert in_close_window(at(23, 50), MCX, 5) and not in_close_window(at(23, 49), MCX, 5)
    assert not in_close_window(at(15, 27, dt.date(2026, 9, 26)), NSE, 5)  # Saturday


def test_square_off_closes_only_that_markets_positions_at_the_live_price():
    broker = PaperBroker(1_000_000)
    broker.place_order(_order("NIFTY", "11"), market_price=100.0)                      # NSE, winner
    broker.place_order(_order("TCS", "12", "SELL"), market_price=100.0)                # NSE, short, no quote later
    broker.place_order(_order("CRUDEOIL", "21"), market_price=100.0)                   # MCX
    pnl = []

    client = _Client({"11": 110.0, "21": 90.0})  # no quote for security 12
    closed = square_off_positions(broker, client, NSE, _NoLock(), pnl.append)

    assert [c["instrument"] for c in closed] == ["NIFTY"]
    assert closed[0]["exit_reason"] == EOD_REASON and closed[0]["exit_price"] == 110.0
    assert pnl == [closed[0]["net_pnl"]] and pnl[0] > 0
    open_now = {p["instrument"] for p in broker.get_open_positions()}
    assert open_now == {"TCS", "CRUDEOIL"}  # unpriced one is left open, never closed at a made-up price
    assert "MCX_COMM" not in (client.asked or {})  # the MCX position was not even priced by the NSE runner

    closed = square_off_positions(broker, _Client({"21": 90.0}), MCX, _NoLock(), pnl.append)
    assert [c["instrument"] for c in closed] == ["CRUDEOIL"] and closed[0]["net_pnl"] < 0


def test_square_off_does_nothing_without_credentials():
    broker = PaperBroker(1_000_000)
    broker.place_order(_order("NIFTY", "11"), market_price=100.0)
    assert square_off_positions(broker, _Client({}, configured=False), NSE, _NoLock(), lambda x: None) == []
    assert len(broker.get_open_positions()) == 1


def test_runner_squares_off_when_the_market_is_closed_but_only_in_market_hours_mode(monkeypatch):
    monkeypatch.setattr(engine_module, "is_market_open", lambda **kw: False)
    for market_hours_only, expected in ((True, True), (False, False)):
        eng = engine_module.TradingEngine()
        calls = []
        monkeypatch.setattr(eng, "square_off_now", lambda calls=calls: calls.append(1) or [])
        monkeypatch.setattr(
            engine_module, "run_multi_instrument_cycle", lambda *a, **k: ([{"status": "NO_TRADE"}], 0.0)
        )
        monkeypatch.setattr(engine_module, "DhanApiClient", lambda: object())
        eng.start([], ["TCS"], "5min", 30, interval_seconds=0.05, market_hours_only=market_hours_only)
        deadline = time.time() + 3
        while not calls and expected and time.time() < deadline:
            time.sleep(0.02)
        time.sleep(0.15)
        status_while_running = eng.last_status
        eng.stop()
        assert bool(calls) is expected
        if expected:
            assert "Market closed" in status_while_running


def _row(position_id, opened, status, **kw):
    return Position(
        position_id=position_id, instrument=kw.pop("instrument", "CRUDEOIL"), strategy_name="t", direction="LONG",
        quantity=100, entry_price=100.0, stop_price=90.0, target_price=120.0, status=status, opened_at=opened,
        mode="PAPER", underlying=kw.pop("instrument", "CRUDEOIL"), security_id="21", option_type="CE", strike=1.0,
        expiry="2099-01-01", transaction="BUY", **kw,
    )


def test_restore_rebuilds_todays_book_and_marks_older_open_rows_stale():
    init_db()
    today = dt.datetime.combine(today_ist(), dt.time(9, 30))
    yesterday = today - dt.timedelta(days=1)
    with get_session() as s:
        s.query(Position).delete()
        s.add_all([
            _row("OLD-OPEN", yesterday, "OPEN"),
            _row("TODAY-OPEN", today, "OPEN"),
            _row("TODAY-WIN", today, "CLOSED", net_pnl=500.0, closed_at=today, exit_price=105.0),
            _row("TODAY-LOSS", today, "CLOSED", net_pnl=-120.0, closed_at=today, exit_price=98.8),
            _row("OLD-CLOSED", yesterday, "CLOSED", net_pnl=9999.0, closed_at=yesterday),
        ])

    # CRUDEOIL is an MCX instrument: the NSE account must not pick up its book (the accounts are separate).
    nse = engine_module.TradingEngine()
    nse.restore_from_db()
    assert not nse.broker.get_open_positions() and nse.trades_today == 0 and nse.broker.cash == nse.broker.capital
    with get_session() as s:
        assert s.query(Position).filter_by(position_id="OLD-OPEN").one().status == "OPEN"  # not NSE's to mark stale

    eng = engine_module.TradingEngine("commodities", engine_module.MCX)
    eng.restore_from_db()

    open_ids = {p["position_id"] for p in eng.broker.get_open_positions()}
    assert open_ids == {"TODAY-OPEN"}  # reloaded, so stops/targets and square-off see it again
    assert eng.trades_today == 3 and eng.daily_pnl == 380.0  # daily limits survive a restart
    # Cash carries ALL realised P&L (yesterday's too), so drawdown is measured across days after a restart.
    assert eng.broker.cash == eng.broker.capital + 9999.0 + 380.0
    assert eng.peak_equity >= eng.broker.cash
    with get_session() as s:
        old = s.query(Position).filter_by(position_id="OLD-OPEN").one()
        assert old.status == "STALE" and old.exit_reason == "STALE_ON_RESTART"  # kept, never deleted
        assert s.query(Position).filter_by(position_id="OLD-CLOSED").one().status == "CLOSED"

    eng.restore_from_db()  # idempotent: a second call (every page render) changes nothing
    assert len(eng.broker.get_open_positions()) == 1 and eng.trades_today == 3
