"""
Premium translation: underlying index-point stop/target -> option-premium stop/target.

An option's premium does not move 1:1 with its underlying - it moves
approximately `delta` points of premium per point of underlying, and that
relationship itself decays (gamma) as the underlying moves further, and
decays over time (theta) regardless of underlying movement. Delta-scaling is
a standard, transparent first-order approximation: it is NOT exact,
especially close to expiry (high gamma) or for far-OTM strikes (delta drifts
fast). Treat the resulting stop/target as an estimate to monitor against
live premium, not a guarantee - the actual exit is always driven by the
live/paper LTP crossing the computed level, never by the underlying's price.

Risk sizing reuses the existing risk_engine.evaluate_trade unchanged: this
module's whole job is to produce a premium-denominated (entry_price,
stop_price) pair that flows into the same ProposedTrade fields that already
work for index-point strategies.
"""
from __future__ import annotations

from dataclasses import dataclass

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.options.option_selector import OptionContract
from quant_intelligence.strategies.base_strategy import Setup
from quant_intelligence.utils.logging_utils import log_event

MIN_PREMIUM_TICK = 0.05
DEFAULT_DELTA_WHEN_MISSING = 0.5  # an assumption, not data: translate_setup logs it and flags the setup


class PremiumSizingError(ValueError):
    pass


@dataclass
class PremiumSetup:
    """Setup, but denominated in option premium rather than underlying index points."""

    direction: str  # BUY / SELL (the option transaction, not the underlying LONG/SHORT)
    entry_price: float
    stop_price: float
    target_price: float
    contract: OptionContract
    delta_assumed: bool = False  # True when the contract had no delta and DEFAULT_DELTA_WHEN_MISSING was used


def translate_setup(underlying_setup: Setup, contract: OptionContract) -> PremiumSetup:
    """Scale the underlying setup's stop/target distance into premium terms using the
    contract's live delta, then apply BUY/SELL-specific floors:
      - BUY: max loss is capped at the premium paid by construction, so the stop is
        floored at MIN_PREMIUM_TICK (an option can't go below zero).
      - SELL: writing is unbounded risk unless a stop is enforced, so a stop above the
        entry premium is REQUIRED here - `PremiumSizingError` if the delta-scaled stop
        would come out non-positive or absent.
    """
    delta_assumed = contract.delta is None
    if delta_assumed:
        # A default for missing market data. Behaviour is unchanged (owner decision pending), but it is never silent.
        log_event("premium_model", f"No delta for {getattr(contract, 'trading_symbol', 'contract')}: assuming "
                  f"{DEFAULT_DELTA_WHEN_MISSING} to scale the premium stop and target", level="WARNING")
    delta = abs(contract.delta) if contract.delta is not None else DEFAULT_DELTA_WHEN_MISSING
    index_move_to_stop = abs(underlying_setup.entry_price - underlying_setup.stop_price)
    index_move_to_target = abs(underlying_setup.entry_price - underlying_setup.target_price)

    premium_stop_distance = max(delta * index_move_to_stop, MIN_PREMIUM_TICK)
    premium_target_distance = max(delta * index_move_to_target, MIN_PREMIUM_TICK)

    entry = contract.premium

    if contract.transaction == "BUY":
        stop = max(entry - premium_stop_distance, MIN_PREMIUM_TICK)
        target = entry + premium_target_distance
    elif contract.transaction == "SELL":
        stop = entry + premium_stop_distance
        target = max(entry - premium_target_distance, MIN_PREMIUM_TICK)
        if stop <= entry:
            raise PremiumSizingError(
                "SELL trades require a stop above the entry premium (unbounded-risk writes are not allowed); "
                f"computed stop {stop} <= entry {entry}"
            )
    else:
        raise PremiumSizingError(f"Unknown transaction type: {contract.transaction}")

    return PremiumSetup(
        direction=contract.transaction,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        contract=contract,
        delta_assumed=delta_assumed,
    )


def max_loss_per_lot(premium_setup: PremiumSetup) -> float:
    """Worst-case loss per lot before the risk engine's own per-trade % check applies.
    For BUY this is bounded by construction (premium paid); for SELL it is the
    enforced premium stop distance from `translate_setup`."""
    return abs(premium_setup.entry_price - premium_setup.stop_price) * premium_setup.contract.lot_size


def max_lots_within_risk_budget(premium_setup: PremiumSetup, equity: float) -> int:
    """How many lots keep total premium-at-risk within SETTINGS.risk.max_risk_per_trade_pct."""
    per_lot_risk = max_loss_per_lot(premium_setup)
    if per_lot_risk <= 0 or equity <= 0:
        return 0
    budget = equity * SETTINGS.risk.max_risk_per_trade_pct / 100.0
    return int(budget // per_lot_risk)
