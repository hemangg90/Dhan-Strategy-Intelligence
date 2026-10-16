"""The order lifecycle: nothing is a fill unless the broker positively said so."""
import dataclasses
from unittest.mock import MagicMock

import pytest
from hypothesis import given, strategies as st

from quant_intelligence.brokers import dhan_broker
from quant_intelligence.brokers.base_broker import OrderAck, OrderRequest
from quant_intelligence.brokers.order_state import OrderState, TERMINAL, can_transition, from_broker_status, may_submit_again
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.risk.risk_engine import _note_approval

CONFIRM = "YES_I_UNDERSTAND_THE_RISK"


@pytest.mark.parametrize("status,expected", [
    ("TRANSIT", OrderState.SUBMISSION_ATTEMPTED), ("CONFIRM", OrderState.ACKNOWLEDGED), ("PENDING", OrderState.OPEN),
    ("PART_TRADED", OrderState.PARTIALLY_FILLED), ("TRADED", OrderState.FILLED), ("FILLED", OrderState.FILLED),
    ("REJECTED", OrderState.REJECTED), ("CANCELLED", OrderState.CANCELLED), ("EXPIRED", OrderState.CANCELLED),
    ("traded", OrderState.FILLED), (" Rejected ", OrderState.REJECTED),
])
def test_known_broker_statuses_map_to_states(status, expected):
    assert from_broker_status(status) is expected


@pytest.mark.parametrize("status", [None, "", "   ", "UNKNOWN", "SOMETHING_NEW", "SUCCESS", "OK", "ACCEPTED", "true", "0"])
def test_missing_or_unrecognised_statuses_are_unknown_never_filled(status):
    assert from_broker_status(status) is OrderState.SUBMISSION_UNKNOWN


@given(st.text(max_size=20))
def test_property_only_the_two_recognised_fill_words_can_ever_mean_filled(text):
    if from_broker_status(text) is OrderState.FILLED:
        assert text.strip().upper() in ("TRADED", "FILLED")


def test_an_unknown_outcome_never_permits_a_second_submission():
    assert not may_submit_again(OrderState.SUBMISSION_UNKNOWN)
    assert not any(may_submit_again(s) for s in (OrderState.OPEN, OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderState.FILLED))
    assert may_submit_again(OrderState.REJECTED) and may_submit_again(OrderState.CANCELLED)


def test_terminal_states_never_change_and_unknown_can_resolve():
    for state in TERMINAL:
        assert not any(can_transition(state, other) for other in OrderState)
    assert can_transition(OrderState.SUBMISSION_UNKNOWN, OrderState.FILLED) and can_transition(OrderState.SUBMISSION_UNKNOWN, OrderState.REJECTED)
    assert not can_transition(OrderState.OPEN, OrderState.CREATED) and can_transition(OrderState.OPEN, OrderState.FILLED)


def test_an_ack_exposes_its_state():
    assert OrderAck("1", "PENDING").state is OrderState.OPEN and OrderAck("1", "WEIRD").state is OrderState.SUBMISSION_UNKNOWN


def _live(monkeypatch, reply):
    live = dataclasses.replace(SETTINGS, trading_mode="LIVE", trading_live_confirm=CONFIRM, demo_lockdown=False,
                               dhan_client_id="1000000001", dhan_access_token="tok")
    monkeypatch.setattr(dhan_broker, "SETTINGS", live)
    broker = dhan_broker.DhanBroker()
    broker.client = MagicMock()
    broker.client.place_order.return_value = reply
    return broker


def _order(decision_id="RISK-state00000001"):
    _note_approval(decision_id)
    return OrderRequest(strategy_name="X", instrument="NIFTY", direction="LONG", quantity=65, price=50.0, decision_id=decision_id,
                        security_id="777", exchange_segment="NSE_FNO", transaction_type="BUY")


def test_a_clean_reply_is_open_not_filled(monkeypatch):
    broker = _live(monkeypatch, {"orderId": "9", "orderStatus": "PENDING"})
    ack = broker.place_order(_order())
    assert ack.state is OrderState.OPEN and ack.state is not OrderState.FILLED and broker.unreconciled_order is None


def test_a_reply_without_a_status_defaults_to_open_but_never_to_filled(monkeypatch):
    ack = _live(monkeypatch, {"orderId": "9"}).place_order(_order())
    assert ack.state is OrderState.OPEN


@pytest.mark.parametrize("reply", [{}, {"orderStatus": "PENDING"}, {"orderId": "9", "orderStatus": "SOMETHING_NEW"},
                                   {"orderId": "", "orderStatus": "TRADED"}])
def test_a_reply_we_cannot_interpret_is_an_unknown_outcome_and_blocks_further_orders(monkeypatch, reply):
    broker = _live(monkeypatch, reply)
    ack = broker.place_order(_order("RISK-state00000002"))
    assert ack.status == "UNKNOWN" and ack.state is OrderState.SUBMISSION_UNKNOWN and broker.unreconciled_order
    again = broker.place_order(_order("RISK-state00000003"))
    assert again.status == "REJECTED" and "UNKNOWN outcome" in again.reject_reason
    assert broker.client.place_order.call_count == 1
