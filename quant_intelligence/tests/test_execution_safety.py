"""Order-level safety: risk approval is required, spent once, and unknown outcomes block further orders."""
import dataclasses
from unittest.mock import MagicMock

import pytest
import requests

from quant_intelligence.brokers import dhan_broker
from quant_intelligence.brokers.base_broker import OrderRequest
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.risk import risk_engine
from quant_intelligence.risk.risk_engine import AccountState, ProposedTrade, consume_approval, evaluate_trade

CONFIRM = "YES_I_UNDERSTAND_THE_RISK"


def _live_broker(monkeypatch):
    live = dataclasses.replace(SETTINGS, trading_mode="LIVE", trading_live_confirm=CONFIRM, demo_lockdown=False,
                               dhan_client_id="1000000001", dhan_access_token="tok")
    monkeypatch.setattr(dhan_broker, "SETTINGS", live)
    broker = dhan_broker.DhanBroker()
    broker.client = MagicMock()
    broker.client.place_order.return_value = {"orderId": "9", "orderStatus": "PENDING"}
    return broker


def _order(decision_id):
    return OrderRequest(strategy_name="Momentum", instrument="NIFTY", direction="LONG", quantity=65, price=50.0,
                        stop_price=45.0, target_price=60.0,
                        decision_id=decision_id, security_id="777", exchange_segment="NSE_FNO", transaction_type="BUY")


def _approved_decision(**overrides):
    """Approve exactly the trade that `_order` submits (premium 50, stop 45, target 60, 65 units of NIFTY)."""
    acct = AccountState(equity=1_000_000, peak_equity=1_000_000, daily_pnl=0.0, open_positions_count=0, trades_today=0)
    fields = dict(strategy_name="Momentum", direction="LONG", entry_price=50.0, stop_price=45.0, target_price=60.0, quantity=65,
                  relative_volume=1.0, instrument="NIFTY", long_premium=True)
    fields.update(overrides)
    d = evaluate_trade(acct, ProposedTrade(**fields))
    assert d.approved, d.reason
    return d.decision_id


def test_a_live_order_without_a_risk_approval_is_rejected_before_any_network_call(monkeypatch):
    broker = _live_broker(monkeypatch)
    for decision_id in (None, "", "RISK-never-issued"):
        ack = broker.place_order(_order(decision_id))
        assert ack.status == "REJECTED" and "risk-engine approval" in ack.reject_reason
    broker.client.place_order.assert_not_called()


def test_a_vetoed_trade_gets_no_approval(monkeypatch):
    broker = _live_broker(monkeypatch)
    acct = AccountState(equity=1_000_000, peak_equity=1_000_000, daily_pnl=0.0, open_positions_count=0, trades_today=0,
                        kill_switch_engaged=True)
    d = evaluate_trade(acct, ProposedTrade("Momentum", "LONG", 100.0, 99.0, 103.0, 50, relative_volume=1.0))
    assert not d.approved
    assert broker.place_order(_order(d.decision_id)).status == "REJECTED"
    broker.client.place_order.assert_not_called()


def test_an_approval_is_spent_once_so_it_cannot_be_replayed(monkeypatch):
    broker = _live_broker(monkeypatch)
    decision_id = _approved_decision()
    assert broker.place_order(_order(decision_id)).status == "PENDING"
    assert broker.place_order(_order(decision_id)).status == "REJECTED"  # duplicate submission
    assert broker.client.place_order.call_count == 1


def test_an_expired_approval_is_rejected(monkeypatch):
    decision_id = _approved_decision()
    assert consume_approval(decision_id, max_age_sec=-1.0) is False  # older than allowed
    assert consume_approval(decision_id) is False  # and it was spent by the attempt above


def test_a_timeout_leaves_the_outcome_unknown_and_blocks_every_later_order(monkeypatch):
    broker = _live_broker(monkeypatch)
    broker.client.place_order.side_effect = requests.Timeout("read timed out")
    first = broker.place_order(_order(_approved_decision()))
    assert first.status == "UNKNOWN" and "order book" in first.reject_reason
    broker.client.place_order.side_effect = None
    second = broker.place_order(_order(_approved_decision()))
    assert second.status == "REJECTED" and "UNKNOWN outcome" in second.reject_reason
    assert broker.client.place_order.call_count == 1  # never retried, never duplicated


def test_paper_pnl_is_net_of_costs_and_records_the_gross():
    broker = PaperBroker(starting_capital=1_000_000)
    order = OrderRequest(strategy_name="X", instrument="NIFTY", direction="LONG", quantity=65, price=100.0,
                         option_type="CE", transaction_type="BUY", security_id="1", stop_price=90.0, target_price=120.0)
    broker.place_order(order, market_price=100.0)
    pid = broker.get_open_positions()[0]["position_id"]
    entry = broker.positions[pid]["entry_price"]
    closed = broker.close_position(pid, exit_price=120.0, reason="TARGET")
    assert closed["gross_pnl"] == pytest.approx((120.0 - entry) * 65)
    assert closed["costs"] > 2 * SETTINGS.costs.brokerage_per_order_inr  # brokerage plus taxes and levies
    assert closed["net_pnl"] == pytest.approx(closed["gross_pnl"] - closed["costs"])
    assert closed["costs_model"] == "OPTION_PREMIUM"
    assert broker.cash == pytest.approx(1_000_000 + closed["net_pnl"])


def test_a_losing_paper_trade_loses_more_than_its_gross_loss():
    broker = PaperBroker(starting_capital=1_000_000)
    order = OrderRequest(strategy_name="X", instrument="NIFTY", direction="LONG", quantity=65, price=100.0,
                         option_type="CE", transaction_type="BUY", security_id="1")
    broker.place_order(order, market_price=100.0)
    pid = broker.get_open_positions()[0]["position_id"]
    closed = broker.close_position(pid, exit_price=90.0, reason="STOP")
    assert closed["net_pnl"] < closed["gross_pnl"] < 0


# ------------------------------------------------------------- stop fills in the position monitors
def _open_equity_long(broker, stop=99.0):
    order = OrderRequest(strategy_name="X", instrument="TCS", direction="LONG", quantity=10, price=100.0,
                         stop_price=stop, target_price=105.0)
    broker.place_order(order, market_price=100.0)
    return broker.get_open_positions()[0]["position_id"]


def test_the_equity_monitor_fills_a_gapped_stop_at_the_worse_open():
    import pandas as pd

    from quant_intelligence.execution.position_monitor import monitor_positions

    broker = PaperBroker(starting_capital=1_000_000)
    pid = _open_equity_long(broker)
    gapped = pd.Series({"open": 97.0, "high": 98.0, "low": 96.5, "close": 97.5})
    monitor_positions(broker, gapped)
    assert broker.positions[pid]["status"] == "CLOSED" and broker.positions[pid]["exit_price"] == 97.0


def test_the_equity_monitor_fills_an_ordinary_stop_at_the_stop():
    import pandas as pd

    from quant_intelligence.execution.position_monitor import monitor_positions

    broker = PaperBroker(starting_capital=1_000_000)
    pid = _open_equity_long(broker)
    monitor_positions(broker, pd.Series({"open": 99.6, "high": 100.0, "low": 98.5, "close": 99.0}))
    assert broker.positions[pid]["exit_price"] == 99.0


def test_the_option_monitor_exits_at_the_worse_of_stop_and_sampled_premium():
    from quant_intelligence.execution import position_monitor

    broker = PaperBroker(starting_capital=1_000_000)
    broker.place_order(OrderRequest(strategy_name="X", instrument="NIFTY", direction="LONG", quantity=65, price=100.0,
                                    stop_price=80.0, target_price=130.0, option_type="CE", transaction_type="BUY",
                                    security_id="555"), market_price=100.0)
    pid = broker.get_open_positions()[0]["position_id"]
    original = position_monitor._fetch_quotes
    position_monitor._fetch_quotes = lambda *a, **k: {"555": {"last_price": 70.0}}  # premium fell well past the stop
    try:
        position_monitor.monitor_option_positions(broker, api_client=object())
    finally:
        position_monitor._fetch_quotes = original
    assert broker.positions[pid]["exit_price"] == 70.0 and broker.positions[pid]["exit_reason"] == "STOP"


# ------------------------------------------------------------- live account state
class _FakeLiveBroker:
    def __init__(self, positions, balance=1_000_000.0):
        self._positions, self._balance = positions, balance

    def get_fund_limit(self):
        return {"availabelBalance": self._balance}

    def get_positions(self):
        return self._positions

    def is_connected(self):
        return True


def _trade_for_risk():
    return ProposedTrade("Momentum", "LONG", 100.0, 99.0, 103.0, 50, relative_volume=1.0)


def test_live_account_state_does_not_touch_the_paper_engines_peak_equity():
    from quant_intelligence.execution.engine import ENGINE
    from quant_intelligence.ui import state

    before = ENGINE.peak_equity
    state.get_live_account_state(_FakeLiveBroker([], balance=123_456.0))
    assert ENGINE.peak_equity == before


def test_live_daily_pnl_comes_from_the_brokers_positions_not_the_paper_engine():
    from quant_intelligence.ui import state

    loss = (SETTINGS.risk.max_daily_loss_pct + 0.5) / 100 * 1_000_000  # half a percent past the configured limit
    rows = [{"netQty": 0, "realizedProfit": -0.6 * loss, "unrealizedProfit": 0.0},
            {"netQty": 65, "realizedProfit": 0.0, "unrealizedProfit": -0.4 * loss, "buyAvg": 100.0, "sellAvg": 0.0}]
    acct = state.get_live_account_state(_FakeLiveBroker(rows))
    assert acct.daily_pnl == pytest.approx(-loss) and acct.daily_pnl_known is True
    veto = evaluate_trade(acct, _trade_for_risk())
    assert not veto.approved and "Daily loss" in veto.reason


def test_missing_live_pnl_figures_fail_closed_instead_of_counting_as_zero():
    from quant_intelligence.ui import state

    rows = [{"netQty": 65, "buyAvg": 100.0, "sellAvg": 0.0}]  # broker did not report profit fields
    acct = state.get_live_account_state(_FakeLiveBroker(rows))
    assert acct.daily_pnl_known is False
    decision = evaluate_trade(acct, _trade_for_risk())
    assert not decision.approved and "unavailable" in decision.reason and decision.checks["daily_pnl_known"] is False


# ------------------------------------------------- the order must be the trade that was approved
@pytest.mark.parametrize("field,value", [
    ("instrument", "BANKNIFTY"), ("direction", "SHORT"), ("quantity", 130), ("price", 55.0),
    ("stop_price", 40.0), ("target_price", 70.0),
])
def test_an_order_that_differs_from_the_approved_trade_is_rejected_and_the_approval_is_spent(monkeypatch, field, value):
    broker = _live_broker(monkeypatch)
    decision_id = _approved_decision()
    order = _order(decision_id)
    setattr(order, field, value)
    ack = broker.place_order(order)
    assert ack.status == "REJECTED" and "exact order" in ack.reject_reason
    broker.client.place_order.assert_not_called()
    assert broker.place_order(_order(decision_id)).status == "REJECTED"  # the genuine order cannot reuse the spent approval


def test_the_exact_approved_order_is_sent(monkeypatch):
    broker = _live_broker(monkeypatch)
    assert broker.place_order(_order(_approved_decision())).status == "PENDING"
    assert broker.client.place_order.call_count == 1


def test_two_positions_on_one_contract_are_both_checked_for_exit():
    from quant_intelligence.execution import position_monitor

    broker = PaperBroker(starting_capital=1_000_000)
    for _ in range(2):
        broker.place_order(OrderRequest(strategy_name="X", instrument="NIFTY", direction="LONG", quantity=65, price=100.0,
                                        stop_price=80.0, target_price=130.0, option_type="CE", transaction_type="BUY",
                                        security_id="555"), market_price=100.0)
    assert len(broker.get_open_positions()) == 2
    original = position_monitor._fetch_quotes
    position_monitor._fetch_quotes = lambda *a, **k: {"555": {"last_price": 135.0}}  # past the target
    try:
        events = position_monitor.monitor_option_positions(broker, api_client=object())
    finally:
        position_monitor._fetch_quotes = original
    assert len(events) == 2 and not broker.get_open_positions()
