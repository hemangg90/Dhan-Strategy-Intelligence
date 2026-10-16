"""Order lifecycle states, and the one rule that matters: only a broker status we positively recognise as a fill is a fill.

States: CREATED -> SUBMISSION_ATTEMPTED -> ACKNOWLEDGED -> OPEN -> PARTIALLY_FILLED -> FILLED, or REJECTED / CANCELLED,
or SUBMISSION_UNKNOWN when the outcome cannot be established (timeout, missing or unrecognised status). A submission
function returning without an exception proves nothing about the order, so nothing here ever infers FILLED from silence.

The mapping from broker status strings uses the names Dhan documents (TRANSIT, PENDING, CONFIRM, PART_TRADED, TRADED,
REJECTED, CANCELLED, EXPIRED) and the paper broker's own FILLED / REJECTED. It has NOT been exercised against a live
account; anything unrecognised becomes SUBMISSION_UNKNOWN, the safe reading.
"""
from __future__ import annotations

from enum import Enum


class OrderState(str, Enum):
    CREATED = "CREATED"
    SUBMISSION_ATTEMPTED = "SUBMISSION_ATTEMPTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    SUBMISSION_UNKNOWN = "SUBMISSION_UNKNOWN"


TERMINAL = frozenset({OrderState.FILLED, OrderState.REJECTED, OrderState.CANCELLED})

_FROM_BROKER = {
    "CREATED": OrderState.CREATED,
    "TRANSIT": OrderState.SUBMISSION_ATTEMPTED,  # on its way to the exchange: not yet acknowledged
    "CONFIRM": OrderState.ACKNOWLEDGED,
    "PENDING": OrderState.OPEN,
    "OPEN": OrderState.OPEN,
    "PART_TRADED": OrderState.PARTIALLY_FILLED,
    "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
    "TRADED": OrderState.FILLED,
    "FILLED": OrderState.FILLED,
    "REJECTED": OrderState.REJECTED,
    "CANCELLED": OrderState.CANCELLED,
    "EXPIRED": OrderState.CANCELLED,
}

# Which states may follow which. SUBMISSION_UNKNOWN may resolve to anything once a human (or a real status lookup)
# establishes the truth; the terminal states never change.
_NEXT = {
    OrderState.CREATED: {OrderState.SUBMISSION_ATTEMPTED, OrderState.REJECTED},
    OrderState.SUBMISSION_ATTEMPTED: {OrderState.ACKNOWLEDGED, OrderState.OPEN, OrderState.REJECTED, OrderState.SUBMISSION_UNKNOWN,
                                      OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED},
    OrderState.ACKNOWLEDGED: {OrderState.OPEN, OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.REJECTED, OrderState.CANCELLED},
    OrderState.OPEN: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED},
    OrderState.PARTIALLY_FILLED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED},
    OrderState.SUBMISSION_UNKNOWN: set(OrderState) - {OrderState.CREATED, OrderState.SUBMISSION_UNKNOWN},
    OrderState.FILLED: set(),
    OrderState.REJECTED: set(),
    OrderState.CANCELLED: set(),
}


def from_broker_status(status: str | None) -> OrderState:
    """Normalise a broker status string. Missing, empty or unrecognised input is SUBMISSION_UNKNOWN, never a fill."""
    if not status:
        return OrderState.SUBMISSION_UNKNOWN
    return _FROM_BROKER.get(str(status).strip().upper(), OrderState.SUBMISSION_UNKNOWN)


def can_transition(current: OrderState, new: OrderState) -> bool:
    return new in _NEXT[current]


def may_submit_again(state: OrderState) -> bool:
    """Only after the broker has definitively said the earlier order is dead and nothing was executed. An unknown
    outcome, an open order or a fill never allows a blind second submission."""
    return state in (OrderState.REJECTED, OrderState.CANCELLED)
