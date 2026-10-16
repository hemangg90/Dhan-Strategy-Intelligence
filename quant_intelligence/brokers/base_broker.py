"""Broker-independent execution interface.

The research/risk/execution engine never talks to Dhan (or any broker)
directly - it only talks to this interface. PaperBroker and DhanBroker both
implement it identically from the caller's perspective.
"""
from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass


@dataclass
class OrderRequest:
    strategy_name: str
    instrument: str
    direction: str  # LONG / SHORT
    quantity: int
    order_type: str = "MARKET"
    price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    decision_id: str | None = None

    # Options fields (all optional - absent means "trade the underlying directly",
    # same as before options support existed). See quant_intelligence/options/.
    security_id: str | None = None
    exchange_segment: str = "NSE_EQ"  # NSE_EQ / NSE_FNO / IDX_I / ...
    product_type: str = "INTRADAY"  # INTRADAY / MARGIN / CNC
    transaction_type: str | None = None  # BUY / SELL (the option transaction; None = use `direction`)
    option_type: str | None = None  # CE / PE
    strike: float | None = None
    expiry: str | None = None
    lot_size: int = 1
    tag: str | None = None  # e.g. "TIE-BREAK": the order came from a tied ranking, sized down
    # What the ranker expected when it chose this strategy (kept on the position for realised-vs-expected analysis).
    expected_r: float | None = None
    confidence: str | None = None
    regime: str | None = None


@dataclass
class OrderAck:
    order_id: str
    status: str  # PENDING / FILLED / REJECTED
    reject_reason: str | None = None
    fill_price: float | None = None
    timestamp: dt.datetime | None = None

    @property
    def state(self):
        """The normalised lifecycle state (see brokers/order_state.py). Unrecognised statuses are SUBMISSION_UNKNOWN."""
        from quant_intelligence.brokers.order_state import from_broker_status

        return from_broker_status(self.status)


class BaseBroker(abc.ABC):
    mode: str = "PAPER"
    name: str = "base"

    @abc.abstractmethod
    def place_order(self, order: OrderRequest) -> OrderAck:
        raise NotImplementedError

    @abc.abstractmethod
    def get_positions(self) -> list[dict]:
        raise NotImplementedError

    @abc.abstractmethod
    def is_connected(self) -> bool:
        raise NotImplementedError
