"""User-triggered paper-trading actions: exit one position, or square off a market.

Everything goes through the shared paper broker under its lock, so it is safe alongside the
background auto-traders. Closing uses the latest live premium (shared quote cache) - never a made-up
price: with no live quote the position stays open and the user is told why.
"""
from __future__ import annotations

from quant_intelligence.brokers.dhan_api_client import DhanApiClient
from quant_intelligence.execution.engine import ENGINE, MCX_ENGINE
from quant_intelligence.execution.position_monitor import _fetch_quotes
from quant_intelligence.execution.square_off import EOD_REASON  # noqa: F401  (re-exported for callers)

MANUAL_REASON = "MANUAL"


def exit_position(position_id: str, reason: str = MANUAL_REASON) -> tuple[bool, str]:
    """Close one open paper position at its latest traded premium. Returns (closed, message)."""
    engine = next((e for e in (ENGINE, MCX_ENGINE) if position_id in e.broker.positions), None)
    if engine is None:
        return False, "That position is not open any more."
    broker = engine.broker
    with engine.lock:
        pos = broker.positions.get(position_id)
        if pos is None or pos["status"] != "OPEN":
            return False, "That position is not open any more."
        snapshot = dict(pos)
    client = DhanApiClient()
    if not client.is_configured():
        return False, "Set your Dhan credentials to get a live price - the position is left open."
    if not snapshot.get("security_id"):
        return False, "This position has no security id, so it cannot be priced - left open."
    try:
        quotes = _fetch_quotes(client, [snapshot], max_age=1.0)
    except Exception as e:  # rate limit, expired token, network
        return False, f"No live price available ({e}) - the position is left open."
    price = (quotes.get(str(snapshot["security_id"])) or {}).get("last_price")
    if price is None:
        return False, "No live price for this contract right now - the position is left open."
    with engine.lock:
        closed = broker.close_position(position_id, float(price), reason)
        if closed is None:
            return False, "That position was already closed."
        engine.add_pnl(closed["net_pnl"])
    return True, f"Closed at ₹{price:,.2f}  (P&L {closed['net_pnl']:+,.0f})"


def square_off_all() -> list[dict]:
    """Close every open paper position in both markets at their latest prices."""
    closed = list(ENGINE.square_off_now())
    closed += MCX_ENGINE.square_off_now()
    return closed
