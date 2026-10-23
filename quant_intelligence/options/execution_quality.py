"""
Execution quality of one option contract: can it be traded at a believable price, and what does trading it cost?

Eligibility (hard rejections - each one is a stated reason, never silently skipped):
  * the quote is invalid: a bid or ask missing, non-positive, or crossed (ask < bid)
  * the quote is stale: the snapshot is older than `max_quote_age_sec` (the chain gives no per-contract quote time)
  * the spread is wider than `max_spread_pct` of the mid
  * the premium is under `min_premium`
  * the quoted size on the side we would take is below `min_top_qty_lots` lots
  * no volume today, or no open interest
Fill price assumptions are conservative and never the last price or the mid:
  BUY fills at the ask plus `slippage_ticks` ticks; SELL at the bid minus the same. If the order is bigger than the
  quoted top-of-book size, a further impact of half the spread per extra multiple of the top size (at most 3x) is added -
  a heuristic, labelled as such, not a market-impact model.
Costs use the application's own contract-note model (`execution/charges.py`) on those fill prices, round trip, plus the
slippage on both legs; `breakeven_move_pct` is how far the premium must rise (buy) for the round trip to net zero if the
market does not move.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.charges import option_trade_charges
from quant_intelligence.options.contracts import OptionQuote
from quant_intelligence.utils.market_profile import profile_for


@dataclass(frozen=True)
class ExecutionLimits:
    max_spread_pct: float
    max_quote_age_sec: float
    min_premium: float
    min_top_qty_lots: float
    require_volume: bool = True

    @staticmethod
    def from_settings() -> "ExecutionLimits":
        return ExecutionLimits(SETTINGS.option_max_spread_pct, SETTINGS.option_max_quote_age_sec, SETTINGS.option_min_premium,
                               SETTINGS.option_min_top_qty_lots)


@dataclass(frozen=True)
class ExecutionQuality:
    eligible: bool
    reasons: tuple[str, ...] = ()  # hard rejections
    warnings: tuple[str, ...] = ()
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None
    abs_spread: float | None = None
    rel_spread_pct: float | None = None
    bid_qty: float | None = None
    ask_qty: float | None = None
    volume: float | None = None
    oi: float | None = None
    quote_age_sec: float | None = None
    entry_fill: float | None = None  # conservative fill for the intended transaction
    exit_fill: float | None = None  # conservative fill when closing at an unchanged market
    slippage_per_unit: float | None = None  # entry slippage vs the mid
    round_trip_costs: float | None = None  # charges + slippage on both legs, rupees for `quantity`
    breakeven_move_pct: float | None = None
    quantity: int | None = None
    assumptions: tuple[str, ...] = field(default_factory=tuple)


def assess_execution_quality(
    quote: OptionQuote, now: dt.datetime, transaction: str = "BUY", quantity: int | None = None,
    limits: ExecutionLimits | None = None,
) -> ExecutionQuality:
    limits = limits or ExecutionLimits.from_settings()
    reasons: list[str] = []
    warnings: list[str] = []
    bid, ask = quote.bid, quote.ask
    qty_lots = None

    valid = True
    if bid is None or ask is None:
        reasons.append("no two-sided quote (bid or ask missing)")
        valid = False
    elif not (math.isfinite(bid) and math.isfinite(ask)) or bid <= 0 or ask <= 0:
        reasons.append("bid or ask is not a positive number" + (" (no bid)" if bid is not None and bid <= 0 else ""))
        valid = False
    elif ask < bid:
        reasons.append(f"crossed quote (bid {bid:g} above ask {ask:g})")
        valid = False

    age = None
    if quote.quote_time is None:
        reasons.append("no quote time: freshness cannot be checked")
    else:
        age = (now - quote.quote_time).total_seconds()
        if age < -5:
            reasons.append(f"quote time is {-age:.0f}s in the future")
        elif age > limits.max_quote_age_sec:
            reasons.append(f"quote is stale ({age:.0f}s old, limit {limits.max_quote_age_sec:g}s)")

    mid = abs_spread = rel = None
    if valid:
        mid = (bid + ask) / 2.0
        abs_spread = ask - bid
        rel = abs_spread / mid * 100.0
        if rel > limits.max_spread_pct:
            reasons.append(f"spread {rel:.1f}% of mid exceeds {limits.max_spread_pct:g}%")
        if mid < limits.min_premium:
            reasons.append(f"premium {mid:g} below the minimum {limits.min_premium:g}")

    side_qty = quote.ask_qty if transaction == "BUY" else quote.bid_qty
    lot = quote.lot_size or 1
    if side_qty is None:
        warnings.append("quoted size is not reported")
    else:
        qty_lots = side_qty / lot
        if qty_lots < limits.min_top_qty_lots:
            reasons.append(f"only {side_qty:g} units ({qty_lots:.1f} lots) quoted on the {'ask' if transaction == 'BUY' else 'bid'}")
    if limits.require_volume:
        if quote.volume is not None and quote.volume <= 0:
            reasons.append("no volume traded today")
        elif quote.volume is None:
            warnings.append("volume is not reported")
    if quote.oi is not None and quote.oi <= 0:
        reasons.append("no open interest")
    elif quote.oi is None:
        warnings.append("open interest is not reported")

    entry = exit_ = slip = costs = breakeven = None
    assumptions = (f"BUY fills at the ask + {SETTINGS.costs.slippage_ticks} tick(s); SELL at the bid - {SETTINGS.costs.slippage_ticks} tick(s); "
                   "never the last price or mid",
                   "size beyond the quoted top of book adds half the spread per extra multiple (max 3x): a heuristic",
                   "charges from the application's contract-note model")
    if valid:
        tick = SETTINGS.costs.tick_size
        slip_ticks = SETTINGS.costs.slippage_ticks * tick
        impact = 0.0
        if quantity and side_qty and quantity > side_qty:
            impact = min(quantity / side_qty - 1.0, 3.0) * abs_spread / 2.0
            warnings.append(f"order of {quantity} exceeds the {side_qty:g} quoted: extra impact {impact:.2f} per unit assumed")
        if transaction == "BUY":
            entry = ask + slip_ticks + impact
            exit_ = max(bid - slip_ticks, 0.0)
        else:
            entry = max(bid - slip_ticks - impact, 0.0)
            exit_ = ask + slip_ticks
        slip = abs(entry - mid)
        if quantity and entry and entry > 0:
            market = profile_for(quote.underlying).name
            charges = option_trade_charges(entry, exit_, quantity, transaction, market).total
            slippage_cost = (abs(entry - mid) + abs(exit_ - mid)) * quantity
            costs = charges + slippage_cost
            breakeven = costs / (entry * quantity) * 100.0
    return ExecutionQuality(
        eligible=not reasons, reasons=tuple(reasons), warnings=tuple(warnings), bid=bid, ask=ask, mid=mid, abs_spread=abs_spread,
        rel_spread_pct=rel, bid_qty=quote.bid_qty, ask_qty=quote.ask_qty, volume=quote.volume, oi=quote.oi, quote_age_sec=age,
        entry_fill=entry, exit_fill=exit_, slippage_per_unit=slip, round_trip_costs=costs, breakeven_move_pct=breakeven,
        quantity=quantity, assumptions=assumptions,
    )
