"""Capital accounting for the paper account.

PaperBroker only moves `cash` when a position CLOSES (realised P&L); it does not deduct the
premium when an option is bought. So funds already committed to open positions must be
derived from the positions themselves - that is what `capital_summary` does.

For a bought option (or long stock) the outlay is entry price x quantity. For a SOLD option
only the premium value is known here; the exchange margin requirement is NOT modelled, so
`sell_premium_value` is informational and is not counted as used capital.
"""
from __future__ import annotations


def is_buy_position(pos: dict) -> bool:
    return (pos.get("transaction") or pos["direction"]) in ("BUY", "LONG")


def capital_required(entry_price: float, quantity: int) -> float:
    """Outlay to BUY `quantity` units at `entry_price` (premium x quantity for options)."""
    return float(entry_price) * int(quantity)


def capital_summary(broker) -> dict:
    open_positions = broker.get_open_positions()
    used = sum(capital_required(p["entry_price"], p["quantity"]) for p in open_positions if is_buy_position(p))
    sell_premium = sum(
        capital_required(p["entry_price"], p["quantity"]) for p in open_positions if not is_buy_position(p)
    )
    return {
        "starting_capital": broker.capital,
        "account_value": broker.cash,  # starting capital + realised P&L
        "capital_used": used,
        "available": broker.cash - used,
        "sell_premium_value": sell_premium,
        "open_positions": len(open_positions),
    }


def combined_capital_summary(*brokers) -> dict:
    """The capital summaries of several separate accounts added together (NSE + MCX), for a combined view."""
    parts = [capital_summary(b) for b in brokers]
    return {key: sum(p[key] for p in parts) for key in parts[0]}
