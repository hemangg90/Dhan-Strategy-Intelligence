"""Fully-automatic paper-trade execution.

Extracts the manual "fetch chain -> resolve setup -> size -> risk-check ->
place order" flow that `pages/07_Paper_Trading.py` walks a human through one
button click at a time, into functions any caller (a Streamlit auto-refresh
fragment, a multi-stock loop) can invoke on a timer. No new order-placement
path is introduced: this calls exactly the same `detect_setup`/
`detect_chain_setup`, `select_contract`, `translate_setup`, `evaluate_trade`,
and `broker.place_order` functions the manual page uses, so every safety
check (risk engine veto, mandatory-stop-on-SELL) applies identically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from quant_intelligence.brokers.base_broker import OrderAck, OrderRequest
from quant_intelligence.brokers.dhan_api_client import DhanApiClient
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.setup_detector import (
    SETUP_TRIGGERED,
    WAITING_FOR_SETUP,
    detect_chain_setup,
    detect_setup,
)
from quant_intelligence.options.chain_analytics import ChainSnapshot
from quant_intelligence.options.option_selector import (
    OptionSelectionError,
    get_underlying_info,
    option_segment_for,
    select_contract,
)
from quant_intelligence.options.premium_model import PremiumSizingError, translate_setup
from quant_intelligence.research.pipeline import PipelineOutput
from quant_intelligence.execution.capital import capital_required, capital_summary
from quant_intelligence.ranking.ranking_engine import TIE_BREAK_TAG, tradable
from quant_intelligence.risk.risk_engine import AccountState, ProposedTrade, evaluate_trade, risk_multiplier
from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES, get_strategy


@dataclass
class AutoCycleResult:
    instrument: str
    strategy_name: str | None = None
    setup_status: str | None = None  # WAITING_FOR_SETUP / SETUP_TRIGGERED / None
    order: OrderAck | None = None
    risk_decision_id: str | None = None
    reason: str = ""
    capital_required: float | None = None  # premium x quantity to BUY the resolved contract
    capital_used: float = 0.0  # capital actually committed by a filled BUY order
    tag: str | None = None  # "TIE-BREAK" when the trade came from a tied ranking


def size_position(
    account: AccountState, entry_price: float, stop_price: float, lot_size: int = 1, size_factor: float = 1.0
) -> int:
    """Quantity such that the risk-engine's per-trade risk check passes by construction.

    Uses 90% of `max_risk_per_trade_pct` as headroom against rounding, since
    the final `quantity` is rounded down to a whole number of lots. The result is then capped so
    entry_price x quantity never exceeds `max_capital_per_trade_pct` of equity: a tight stop makes
    the risk-based quantity huge (e.g. 8 lots of natural gas), which the risk budget alone allows.
    The budget is scaled by `risk_multiplier` (drawdown / daily loss / losing streak), exactly as the
    risk engine's check is, so size shrinks in a bad stretch.
    """
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0 or account.equity <= 0:
        return 0
    multiplier, _ = risk_multiplier(account)
    budget = account.equity * (SETTINGS.risk.max_risk_per_trade_pct * 0.9 * multiplier) / 100.0
    raw_qty = budget / stop_distance
    lots = max(int(raw_qty // lot_size), 0)
    cap_pct = SETTINGS.risk.max_capital_per_trade_pct
    if cap_pct > 0 and entry_price > 0:
        capital_budget = account.equity * cap_pct / 100.0
        lots = min(lots, int(capital_budget // (entry_price * lot_size)))
    if size_factor < 1.0 and lots > 0:
        lots = max(1, int(lots * size_factor))  # a tie-break trade is sized down, but never to zero lots
    return lots * lot_size


NO_SIZE_PREFIX = "NO SIZE:"  # recognised by the decision log (reports/decision_log.py)


def risk_budget(account: AccountState) -> tuple[float, float, str]:
    """(rupees one stopped trade may lose right now, the risk multiplier, why the multiplier is below 1)."""
    multiplier, why = risk_multiplier(account)
    return account.equity * (SETTINGS.risk.max_risk_per_trade_pct * 0.9 * multiplier) / 100.0, multiplier, why


def _zero_size_reason(account: AccountState, entry_price: float | None = None, stop_price: float | None = None,
                      lot_size: int = 1) -> str:
    """Why the quantity came out zero - specific when the trade's prices are known."""
    budget, multiplier, why = risk_budget(account)
    cut = f" (budget cut to {multiplier:.0%}: {why})" if multiplier < 1 else ""
    if entry_price is not None and stop_price is not None:
        one_lot_risk = abs(entry_price - stop_price) * lot_size
        if one_lot_risk > budget > 0:
            return (f"{NO_SIZE_PREFIX} one lot ({lot_size}) would lose Rs {one_lot_risk:,.0f} at its stop, above the Rs {budget:,.0f} "
                    f"allowed per trade{cut} - not traded (a tighter stop or a larger account would fit it).")
        cap_pct = SETTINGS.risk.max_capital_per_trade_pct
        if cap_pct > 0 and entry_price > 0 and entry_price * lot_size > account.equity * cap_pct / 100.0:
            return (f"{NO_SIZE_PREFIX} one lot costs Rs {entry_price * lot_size:,.0f} in premium, above the per-trade capital cap "
                    f"of Rs {account.equity * cap_pct / 100.0:,.0f} ({cap_pct:g}% of equity).")
    return f"{NO_SIZE_PREFIX} position sizing produced zero quantity (budget Rs {budget:,.0f}{cut}; stop too wide for one lot, or equity too small)."


DIRECTION_FILTER_PREFIX = "Direction filtered"


def direction_blocked(setup_direction: str, transaction: str, allowed_direction: str | None) -> bool:
    """True when the heatmap selection only allows the opposite direction. Applies to BOUGHT options only: a written
    (SELL) option has its own payoff and is outside what 'follow the move' means, so it is never filtered."""
    return bool(allowed_direction) and transaction == "BUY" and setup_direction != allowed_direction


def direction_filtered_reason(underlying: str, setup_direction: str, allowed_direction: str | None, move_note: str = "") -> str:
    wanted = "bullish (call)" if allowed_direction == "LONG" else "bearish (put)"
    got = "bullish" if setup_direction == "LONG" else "bearish"
    return (f"{DIRECTION_FILTER_PREFIX}: {underlying}{' ' + move_note if move_note else ''} - the heatmap allows only {wanted} "
            f"trades, but this setup is {got}.")


def chain_needed(output: PipelineOutput, underlying: str, broker: PaperBroker, allowed_direction: str | None = None) -> bool:
    """Would `run_auto_option_cycle` need the option chain for this instrument right now?

    Cheap and network-free: True only if a strategy was selected, there is no open position on the
    underlying, and either it is a chain-aware strategy or its entry setup has triggered on the latest
    bar. Lets callers fetch the (rate-limited) chain only when it will actually be used, and outside any
    lock - mirroring the early exits of `run_auto_option_cycle`."""
    if not tradable(output.ranking, "PAPER")[0]:
        return False
    if any(pos["instrument"] == underlying for pos in broker.get_open_positions()):
        return False
    strategy_name = output.ranking.selected_strategy
    if strategy_name in CHAIN_AWARE_STRATEGY_NAMES:
        return True
    strategy = get_strategy(strategy_name)
    status = detect_setup(strategy, output.market_state, output.ohlcv.tail(5))
    if status.status != SETUP_TRIGGERED:
        return False
    if allowed_direction and status.setup is not None:  # a setup the heatmap will refuse needs no (rate-limited) chain
        return not direction_blocked(status.setup.direction, status.setup.meta.get("transaction", "BUY"), allowed_direction)
    return True


def _expectation(output: PipelineOutput) -> dict:
    """What the ranker expected from the strategy it selected (stored on the order and position)."""
    ranking = output.ranking
    score = next((s for s in getattr(ranking, "ranked", []) if s.strategy_name == ranking.selected_strategy), None)
    return {"expected_r": getattr(score, "expected_r", None), "confidence": getattr(score, "confidence_label", None),
            "regime": getattr(output, "regime_label", None)}


def run_auto_option_cycle(
    output: PipelineOutput,
    underlying: str,
    broker: PaperBroker,
    client: DhanApiClient,
    chain: "ChainSnapshot | Callable[[], ChainSnapshot | None] | None",
    account: AccountState,
    allowed_direction: str | None = None,
    move_note: str = "",
) -> AutoCycleResult:
    """One automatic cycle (see `_run_auto_option_cycle`); the outcome is also written to the decision log.

    `allowed_direction` ("LONG"/"SHORT") comes from the heatmap mover selection: a BOUGHT option whose setup points the
    other way is skipped (reason starts "Direction filtered"). None = no restriction."""
    result = _run_auto_option_cycle(output, underlying, broker, client, chain, account, allowed_direction, move_note)
    from quant_intelligence.reports.decision_log import log_cycle

    log_cycle(output, result)
    return result


def _run_auto_option_cycle(
    output: PipelineOutput,
    underlying: str,
    broker: PaperBroker,
    client: DhanApiClient,
    chain: "ChainSnapshot | Callable[[], ChainSnapshot | None] | None",
    account: AccountState,
    allowed_direction: str | None = None,
    move_note: str = "",
) -> AutoCycleResult:
    """One automatic cycle for the index/options paper-trading flow (07_Paper_Trading).

    `chain` is a ready ChainSnapshot, None, or a zero-argument PROVIDER. A provider is called only
    when the chain is actually needed - a chain-aware strategy was selected, or a setup triggered and
    a contract must be resolved. Most instruments most cycles are NO TRADE or still waiting, so this
    avoids fetching ~25 option chains per scan (Dhan allows one every 3 seconds per account).
    """
    result = AutoCycleResult(instrument=underlying)
    chain_holder = {"chain": chain}

    def get_chain() -> "ChainSnapshot | None":
        value = chain_holder["chain"]
        if callable(value):
            value = value()  # may raise (rate limit, network); handled by the callers below
            chain_holder["chain"] = value
        return value

    ok, why_not = tradable(output.ranking, "PAPER")
    if not ok:
        result.reason = f"NO TRADE: {why_not}"
        return result

    if any(pos["instrument"] == underlying for pos in broker.get_open_positions()):
        result.reason = f"Skipped: already have an open position on {underlying}."
        return result

    tie_break = output.ranking.tie_break
    result.tag = TIE_BREAK_TAG if tie_break else None
    size_factor = SETTINGS.tie_break_size_factor if tie_break else 1.0

    strategy_name = output.ranking.selected_strategy
    result.strategy_name = strategy_name
    strategy = get_strategy(strategy_name)
    recent_ohlcv = output.ohlcv.tail(5)
    is_chain_strategy = strategy_name in CHAIN_AWARE_STRATEGY_NAMES

    if is_chain_strategy:
        try:
            chain_now = get_chain()
        except Exception as e:
            result.setup_status = "CHAIN_ERROR"
            result.reason = f"Option chain unavailable: {e}"
            return result
        if chain_now is None:
            result.reason = f"{strategy_name} needs a live option chain - none fetched."
            return result
        setup_status = detect_chain_setup(strategy, output.market_state, recent_ohlcv, chain_now)
    else:
        setup_status = detect_setup(strategy, output.market_state, recent_ohlcv)
    result.setup_status = setup_status.status

    if setup_status.status != SETUP_TRIGGERED:
        result.reason = "Waiting for setup." if setup_status.status == WAITING_FOR_SETUP else "No setup."
        return result

    setup = setup_status.setup
    transaction = setup.meta.get("transaction", "BUY")

    if direction_blocked(setup.direction, transaction, allowed_direction):
        result.reason = direction_filtered_reason(underlying, setup.direction, allowed_direction, move_note)
        return result

    try:
        chain = get_chain()  # first (and only) fetch for a non-chain strategy: the setup has triggered
    except Exception as e:
        result.setup_status = "CHAIN_ERROR"
        result.reason = f"Setup triggered but the option chain could not be fetched: {e}"
        return result
    if chain is None or chain.underlying != underlying:
        result.reason = "Setup triggered but no live option chain available to resolve a contract."
        return result

    try:
        lot_size = get_underlying_info(underlying)["lot_size"]
        contract = select_contract(chain, setup.direction, transaction, SETTINGS.option_moneyness_offset, lot_size)
        premium_setup = translate_setup(setup, contract)
    except (OptionSelectionError, PremiumSizingError) as e:
        result.reason = f"Could not resolve a tradeable option: {e}"
        return result

    quantity = size_position(account, premium_setup.entry_price, premium_setup.stop_price, contract.lot_size, size_factor)
    if quantity <= 0:
        result.reason = _zero_size_reason(account, premium_setup.entry_price, premium_setup.stop_price, contract.lot_size)
        result.setup_status = result.setup_status or SETUP_TRIGGERED
        return result

    result.capital_required = capital_required(premium_setup.entry_price, quantity)
    if transaction == "BUY":
        available = capital_summary(broker)["available"]
        if result.capital_required > available:
            result.reason = (
                f"Insufficient funds: needs Rs {result.capital_required:,.0f}, "
                f"Rs {available:,.0f} available."
            )
            return result

    proposed = ProposedTrade(
        strategy_name=strategy_name,
        direction=setup.direction,
        entry_price=premium_setup.entry_price,
        stop_price=premium_setup.stop_price,
        target_price=premium_setup.target_price,
        quantity=quantity,
        relative_volume=output.market_state.get("relative_volume"),
        data_quality_status=output.data_quality_status,
        quote_time=getattr(output, "timestamp", None), long_premium=True,
        instrument=underlying,
    )
    decision = evaluate_trade(account, proposed)
    result.risk_decision_id = decision.decision_id

    if not decision.approved:
        result.reason = f"RISK ENGINE VETO ({decision.decision_id}): {decision.reason}"
        return result

    order = OrderRequest(
        strategy_name=strategy_name,
        instrument=underlying,
        direction=setup.direction,
        quantity=quantity,
        order_type="MARKET",
        price=premium_setup.entry_price,
        stop_price=premium_setup.stop_price,
        target_price=premium_setup.target_price,
        decision_id=decision.decision_id,
        security_id=contract.security_id,
        exchange_segment=option_segment_for(underlying),
        product_type=SETTINGS.option_product_type,
        transaction_type=contract.transaction,
        option_type=contract.option_type,
        strike=contract.strike,
        expiry=contract.expiry,
        lot_size=contract.lot_size,
        tag=result.tag,
        **_expectation(output),
    )
    result.order = broker.place_order(order, market_price=premium_setup.entry_price)
    result.reason = f"Order {result.order.status}" + (f" ({TIE_BREAK_TAG}, size x{size_factor:g})" if tie_break else "")
    if result.order.status == "FILLED" and transaction == "BUY":
        result.capital_used = result.capital_required
    return result


def run_auto_equity_cycle(output: PipelineOutput, broker: PaperBroker, account: AccountState) -> AutoCycleResult:
    """One automatic equity cycle (see `_run_auto_equity_cycle`); the outcome is also written to the decision log."""
    result = _run_auto_equity_cycle(output, broker, account)
    from quant_intelligence.reports.decision_log import log_cycle

    log_cycle(output, result)
    return result


def _run_auto_equity_cycle(output: PipelineOutput, broker: PaperBroker, account: AccountState) -> AutoCycleResult:
    """One automatic cycle for a direct equity (cash-market) paper trade - no options.

    Used for per-stock trading where there is no live option chain: only the
    13 price-action strategies apply (chain-aware strategies are skipped, see
    `CHAIN_AWARE_STRATEGY_NAMES`).
    """
    instrument = output.instrument
    result = AutoCycleResult(instrument=instrument)

    if output.ranking.is_no_trade:
        result.reason = f"NO TRADE: {output.ranking.reason}"
        return result

    if any(pos["instrument"] == instrument for pos in broker.get_open_positions()):
        result.reason = f"Skipped: already have an open position on {instrument}."
        return result

    strategy_name = output.ranking.selected_strategy
    result.strategy_name = strategy_name

    if strategy_name in CHAIN_AWARE_STRATEGY_NAMES:
        result.reason = f"{strategy_name} needs an option chain - not supported for per-stock equity trading."
        return result

    strategy = get_strategy(strategy_name)
    recent_ohlcv = output.ohlcv.tail(5)
    setup_status = detect_setup(strategy, output.market_state, recent_ohlcv)
    result.setup_status = setup_status.status

    if setup_status.status != SETUP_TRIGGERED:
        result.reason = "Waiting for setup." if setup_status.status == WAITING_FOR_SETUP else "No setup."
        return result

    setup = setup_status.setup
    quantity = size_position(account, setup.entry_price, setup.stop_price)
    if quantity <= 0:
        result.reason = _zero_size_reason(account, setup.entry_price, setup.stop_price)
        return result

    proposed = ProposedTrade(
        strategy_name=strategy_name,
        direction=setup.direction,
        entry_price=setup.entry_price,
        stop_price=setup.stop_price,
        target_price=setup.target_price,
        quantity=quantity,
        relative_volume=output.market_state.get("relative_volume"),
        data_quality_status=output.data_quality_status,
        quote_time=getattr(output, "timestamp", None),
        instrument=instrument,
    )
    decision = evaluate_trade(account, proposed)
    result.risk_decision_id = decision.decision_id

    if not decision.approved:
        result.reason = f"RISK ENGINE VETO ({decision.decision_id}): {decision.reason}"
        return result

    order = OrderRequest(
        strategy_name=strategy_name,
        instrument=instrument,
        direction=setup.direction,
        quantity=quantity,
        order_type="MARKET",
        price=setup.entry_price,
        stop_price=setup.stop_price,
        target_price=setup.target_price,
        decision_id=decision.decision_id,
        exchange_segment="NSE_EQ",
        product_type="CNC",
        **_expectation(output),
    )
    result.order = broker.place_order(order, market_price=setup.entry_price)
    result.reason = f"Order {result.order.status}"
    return result
