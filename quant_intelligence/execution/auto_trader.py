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

import dataclasses
import math
from dataclasses import dataclass, field
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
from quant_intelligence.brokers.reconciliation import reconcile_paper
from quant_intelligence.execution.setup_validation import PREFIX as VALIDATION_PREFIX, validate_setup
from quant_intelligence.features.levels import assess_trade, find_levels
from quant_intelligence.intelligence.market_context import context_from_pipeline
from quant_intelligence.intelligence.options_context import build_options_context
from quant_intelligence.options.contracts import quotes_from_chain
from quant_intelligence.options.execution_quality import ExecutionLimits, assess_execution_quality
from quant_intelligence.options.greeks import compute_greeks
from quant_intelligence.options.scenario import ScenarioInputs
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeutil import now_ist
from quant_intelligence.options.chain_analytics import ChainSnapshot
from quant_intelligence.options.option_selector import (
    OptionSelectionError,
    get_underlying_info,
    choose_contract,
    option_segment_for,
)
from quant_intelligence.options.contract_scoring import select_best_contract as select_scored_contract
from quant_intelligence.options.premium_model import PremiumSizingError, translate_setup
from quant_intelligence.research.pipeline import PipelineOutput
from quant_intelligence.execution.capital import capital_required, capital_summary
from quant_intelligence.ranking.ranking_engine import TIE_BREAK_TAG, tradable
from quant_intelligence.risk.risk_engine import AccountState, ProposedTrade, evaluate_trade, limits_for, risk_multiplier
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
    level_review: str = ""  # the support/resistance and liquidity-level reading for this entry
    validation_warnings: list = field(default_factory=list)  # advisory findings of the setup-validation stage
    contract_review: str = ""  # why this strike: the chain-scored pick (OI build-up, spread, delta, walls)


def _sizable_inputs(account: AccountState, entry_price, stop_price, lot_size) -> bool:
    """Equity and both prices finite numbers, lot size a positive whole number."""
    numbers_ok = all(isinstance(x, (int, float)) and math.isfinite(x) for x in (account.equity, entry_price, stop_price))
    return numbers_ok and isinstance(lot_size, int) and lot_size >= 1


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
    if not _sizable_inputs(account, entry_price, stop_price, lot_size):
        return 0  # an invalid input can never become a quantity (NaN would otherwise crash `int()` below)
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0 or account.equity <= 0:
        return 0
    multiplier, _ = risk_multiplier(account)
    budget = account.equity * (limits_for(account.market).max_risk_per_trade_pct * 0.9 * multiplier) / 100.0
    raw_qty = budget / stop_distance
    lots = max(int(raw_qty // lot_size), 0)
    cap_pct = limits_for(account.market).max_capital_per_trade_pct
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
    return account.equity * (limits_for(account.market).max_risk_per_trade_pct * 0.9 * multiplier) / 100.0, multiplier, why


def _zero_size_reason(account: AccountState, entry_price: float | None = None, stop_price: float | None = None,
                      lot_size: int = 1) -> str:
    """Why the quantity came out zero - specific when the trade's prices are known."""
    if not _sizable_inputs(account, entry_price if entry_price is not None else 1.0, stop_price if stop_price is not None else 0.5, lot_size):
        return f"{NO_SIZE_PREFIX} an input to sizing is not a valid number (account equity, entry, stop or lot size) - not traded."
    budget, multiplier, why = risk_budget(account)
    cut = f" (budget cut to {multiplier:.0%}: {why})" if multiplier < 1 else ""
    if entry_price is not None and stop_price is not None:
        one_lot_risk = abs(entry_price - stop_price) * lot_size
        if one_lot_risk > budget > 0:
            return (f"{NO_SIZE_PREFIX} one lot ({lot_size}) would lose Rs {one_lot_risk:,.0f} at its stop, above the Rs {budget:,.0f} "
                    f"allowed per trade{cut} - not traded (a tighter stop or a larger account would fit it).")
        cap_pct = limits_for(account.market).max_capital_per_trade_pct
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


LEVEL_NATIVE_STRATEGIES = {"Camarilla Pivot Reversal"}  # these trade the pivot levels themselves: stop and target are already levels
LEVEL_VETO_PREFIX = "LEVEL CHECK:"


def apply_level_check(output: PipelineOutput, setup, strategy_name: str | None, chain=None):
    """Check a triggered setup against support/resistance and liquidity levels (features/levels.py).

    Returns (setup, review, veto_reason): the setup with its stop/target moved to the levels (or unchanged), a one-line
    reading for the log, and a reason when the trade must not go ahead (the next level leaves no room). Fail-open on
    missing data: with no usable price history or ATR there are no levels, so the setup passes unchanged."""
    if not SETTINGS.levels_enabled or strategy_name in LEVEL_NATIVE_STRATEGIES:
        return setup, "", None
    try:
        atr = output.market_state.get("atr_14")
        context = None
        if chain is not None and hasattr(chain, "strikes"):
            from quant_intelligence.options.contract_scoring import chain_context

            context = chain_context(chain)
        levels = find_levels(output.ohlcv, atr, context)
        if not levels:
            return setup, "", None
        check = assess_trade(setup.direction, setup.entry_price, setup.stop_price, setup.target_price, levels, atr,
                             SETTINGS.levels_min_room_mult, SETTINGS.levels_buffer_atr, output.ohlcv)
    except Exception as e:  # a level-reading fault must never stop the trading cycle or place an unchecked order silently
        from quant_intelligence.utils.logging_utils import log_event

        log_event("levels", f"Level check failed, setup left unchanged: {e}", level="WARNING")
        return setup, "", None
    if not check.ok:
        return setup, check.summary, f"{LEVEL_VETO_PREFIX} {check.reason}."
    if check.stop_changed or check.target_capped:
        setup = dataclasses.replace(setup, stop_price=check.stop, target_price=check.target,
                                    meta={**setup.meta, "levels": check.summary})
    return setup, check.summary, None


def _expectation(output: PipelineOutput) -> dict:
    """What the ranker expected from the strategy it selected (stored on the order and position)."""
    ranking = output.ranking
    score = next((s for s in getattr(ranking, "ranked", []) if s.strategy_name == ranking.selected_strategy), None)
    return {"expected_r": getattr(score, "expected_r", None), "confidence": getattr(score, "confidence_label", None),
            "regime": getattr(output, "regime_label", None)}


def _scenario_inputs(output: PipelineOutput, setup, market, chain: ChainSnapshot, now, underlying: str) -> ScenarioInputs | None:
    """What the contract scenarios need: the underlying's ATR, a holding horizon, a move scale, and the strategy's own record
    (target, stop, share of positive outcomes and the effective sample behind it). None when the ATR is unknown."""
    if market is None or not market.atr:
        return None
    ranking = output.ranking
    cond = next((si.conditional_metrics for si in getattr(output, "strategy_intel", []) if si.strategy_name == ranking.selected_strategy), {})
    from quant_intelligence.utils.timeframe import parse_timeframe_minutes

    bar_minutes = float(parse_timeframe_minutes(market.timeframe))
    horizon = SETTINGS.scenario_horizon_bars
    sigma_move = None
    try:  # prefer the volatility model's stored forecast over ATR scaling when a fresh one exists
        from quant_intelligence.volatility.store import latest_forecast

        fc = latest_forecast(underlying, max_age_days=5.0)
        if fc and fc.get("sigma_1d"):
            sigma_move = chain.spot_price * float(fc["sigma_1d"]) * (horizon * bar_minutes / profile_for(underlying).session_minutes) ** 0.5
    except Exception:
        sigma_move = None
    return ScenarioInputs(
        spot=chain.spot_price, atr=market.atr, now=now, horizon_bars=horizon, bar_minutes=bar_minutes, sigma_move=sigma_move,
        direction=setup.direction, target_price=setup.target_price, stop_price=setup.stop_price,
        win_prob=cond.get("prob_positive_return"), effective_n=cond.get("n_effective") or cond.get("sample_size"))


def _option_risk_info(chain: ChainSnapshot, contract, quantity: int, transaction: str, now) -> tuple[dict, object | None]:
    """The contract's quote, Greeks and time to expiry as the risk engine needs them. Anything unknown stays None so the
    risk engine can refuse it; nothing is defaulted."""
    quote = next((q for q in quotes_from_chain(chain) if q.strike == contract.strike and q.option_type == contract.option_type), None)
    greeks = compute_greeks(quote, chain.spot_price, now) if quote is not None else None
    info = {
        "underlying": contract.underlying, "transaction": transaction,
        "greeks": ({"delta": greeks.delta, "gamma": greeks.gamma, "vega": greeks.vega, "theta": greeks.theta, "spot": chain.spot_price}
                   if greeks is not None and greeks.available else None),
        "trading_days_to_expiry": greeks.trading_days if greeks is not None else None,
    }
    return info, quote


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
    for note in (result.level_review, result.contract_review):
        if note and result.reason:
            result.reason = f"{result.reason} | {note}"
    if result.validation_warnings and result.reason:
        result.reason = f"{result.reason} | warnings: {'; '.join(result.validation_warnings)}"
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

    now = now_ist()
    market_ctx = None
    try:
        lot_size = get_underlying_info(underlying)["lot_size"]
    except OptionSelectionError as e:
        result.reason = f"Could not resolve a tradeable option: {e}"
        return result

    if SETTINGS.setup_validation:
        # The staged validation: data, session, freshness, an ACTIVE setup, eligible contracts and room to run. Any failure to
        # build the contexts stops the trade (fail closed) rather than trading on unchecked data.
        try:
            market_ctx = context_from_pipeline(output, chain)
            options_ctx = build_options_context(chain, now=now, realized_vol=output.market_state.get("realized_vol_20"),
                                                atr=market_ctx.atr, expected_lot_size=lot_size,
                                                limits=ExecutionLimits.from_settings())
            verdict = validate_setup(ranking=output.ranking, setup_status=setup_status, market=market_ctx, options=options_ctx,
                                     now=now, strategy_name=strategy_name, transaction=transaction)
        except Exception as e:
            result.reason = f"{VALIDATION_PREFIX} [CONTEXT_ERROR]: could not build the market and option context ({e})"
            return result
        result.level_review = verdict.level_review
        result.validation_warnings = [w.message for w in verdict.warnings]
        if not verdict.valid:
            result.reason = verdict.message
            return result
        setup = verdict.setup
    elif transaction == "BUY":  # a written option has its own payoff; 'room to run' is about the bought direction
        setup, result.level_review, veto = apply_level_check(output, setup, strategy_name, chain)
        if veto:
            result.reason = veto
            return result

    try:
        cap_pct = limits_for(account.market).max_capital_per_trade_pct if account is not None else 0.0
        max_outlay = account.equity * cap_pct / 100.0 if cap_pct > 0 else None  # skips strikes one lot of which is over the cap
        if SETTINGS.option_selection == "SCORED":
            scenario = _scenario_inputs(output, setup, market_ctx, chain, now, underlying) if market_ctx is not None else None
            choice = select_scored_contract(chain, setup.direction, transaction, lot_size, max_outlay, scenario=scenario, now=now)
            contract = choice.contract
            result.contract_review = choice.summary
        else:
            contract = choose_contract(chain, setup.direction, transaction, SETTINGS.option_moneyness_offset, lot_size,
                                       max_outlay=max_outlay)
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

    if SETTINGS.reconcile_before_order:
        # The book in memory must match what is stored before another order is added to it (a swallowed save error would
        # otherwise go unnoticed and a restart would rebuild a different book).
        recon = reconcile_paper(broker)
        if not recon.ok:
            result.reason = f"RECONCILIATION FAILED - no new orders until it is resolved: {recon.reason}"
            return result

    option_info, quote = (_option_risk_info(chain, contract, quantity, transaction, now) if SETTINGS.setup_validation else (None, None))
    proposed = ProposedTrade(
        strategy_name=strategy_name,
        direction=setup.direction,
        entry_price=premium_setup.entry_price,
        stop_price=premium_setup.stop_price,
        target_price=premium_setup.target_price,
        quantity=quantity,
        relative_volume=output.market_state.get("relative_volume"),
        data_quality_status=output.data_quality_status,
        quote_time=(chain.fetched_at if option_info is not None else getattr(output, "timestamp", None)), long_premium=True,
        instrument=underlying,
        bid=quote.bid if quote is not None else None, ask=quote.ask if quote is not None else None,
        lot_size=contract.lot_size if option_info is not None else None, option=option_info,
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
        entry_greeks=(option_info or {}).get("greeks"),
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
    if result.level_review and result.reason:
        result.reason = f"{result.reason} | {result.level_review}"
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
    setup, result.level_review, veto = apply_level_check(output, setup, strategy_name)
    if veto:
        result.reason = veto
        return result
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
