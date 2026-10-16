"""DEMO_LOCKDOWN must close the live-order gate even when the env says LIVE and confirmed."""
import dataclasses
from unittest.mock import MagicMock

from quant_intelligence.brokers import dhan_broker
from quant_intelligence.brokers.base_broker import OrderRequest
from quant_intelligence.config.settings import SETTINGS

CONFIRM = "YES_I_UNDERSTAND_THE_RISK"


def _live(**kw):
    return dataclasses.replace(SETTINGS, trading_mode="LIVE", trading_live_confirm=CONFIRM, **kw)


def test_live_gate_opens_only_without_lockdown():
    assert _live(demo_lockdown=False).live_mode_fully_authorized
    locked = _live(demo_lockdown=True)
    assert not locked.live_mode_fully_authorized and not locked.is_live_mode


def test_lockdown_blocks_place_order_before_any_network_call(monkeypatch):
    monkeypatch.setattr(dhan_broker, "SETTINGS", _live(demo_lockdown=True))
    broker = dhan_broker.DhanBroker()
    broker.client = MagicMock()
    ack = broker.place_order(OrderRequest(strategy_name="Momentum", instrument="NIFTY", direction="LONG",
                                          quantity=65, price=50.0, security_id="777"))
    assert ack.status == "REJECTED"
    broker.client.place_order.assert_not_called()
