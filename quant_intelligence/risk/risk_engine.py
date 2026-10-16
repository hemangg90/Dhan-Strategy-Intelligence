"""
Deterministic Risk Engine.

This module has ZERO dependency on the research/AI layers. It only looks at:
account state (equity, today's P&L, open positions), the proposed trade, and
hard-coded limits from config/settings.py. It has VETO authority - nothing
downstream can override a rejection, and every rejection records the exact
reason to the risk_events table for audit.

Checks implemented:
  - max risk per trade (position size vs. stop distance vs. equity %)
  - max daily loss (kill switch for the rest of the day)
  - max strategy exposure (concentration in one strategy)
  - max portfolio exposure (total capital at risk)
  - max drawdown (from peak equity, carried across days)
  - max trade count per day
  - max open positions
  - correlated-group exposure (same-direction count and open risk per group)
  - losing-streak cooldown, per-strategy daily loss stop, profit giveback lock
  - liquidity check (minimum relative volume)
  - data quality check
  - broker connectivity check
  - emergency kill switch (manual override, always wins)

Size throttle: `risk_multiplier` scales the per-trade risk budget down as drawdown, the day's loss
or a losing streak builds. The per-trade check and `auto_trader.size_position` both use it, so sizes
shrink in a bad stretch instead of staying full until the hard drawdown veto stops all trading.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import numbers
import threading
import time
from dataclasses import dataclass, field

from quant_intelligence.config.risk_groups import group_for
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.logging_utils import log_event, new_decision_id
from quant_intelligence.utils.timeutil import now_ist


@dataclass
class AccountState:
    equity: float  # marked to market: cash + unrealised P&L of open positions
    peak_equity: float
    daily_pnl: float  # realised today + unrealised
    open_positions_count: int
    trades_today: int
    exposure_by_strategy: dict = field(default_factory=dict)  # strategy_name -> notional at risk
    total_exposure: float = 0.0
    broker_connected: bool = True
    kill_switch_engaged: bool = False
    unrealised_pnl: float = 0.0
    open_positions: list = field(default_factory=list)  # [{"instrument", "direction", "risk"}]
    closed_today: list = field(default_factory=list)  # [{"strategy", "net_pnl", "closed_at"}], oldest first
    day_peak_pnl: float = 0.0  # highest daily_pnl reached today
    # False when today's P&L could not be established (e.g. a live broker that did not report it). The daily-loss limit
    # cannot be checked then, so the engine fails closed instead of treating the unknown as zero.
    daily_pnl_known: bool = True


@dataclass
class ProposedTrade:
    strategy_name: str
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    quantity: int
    relative_volume: float | None = None
    data_quality_status: str = "OK"
    instrument: str = ""
    # Optional market snapshot at order time. None means "not supplied" and skips that check, except that a
    # supplied quote with no usable bid/ask is never assumed to be fine.
    quote_time: dt.datetime | None = None  # when the price used for entry was observed
    bid: float | None = None
    ask: float | None = None
    lot_size: int | None = None  # quantity must be a whole multiple of this when supplied
    # True when the prices are an option PREMIUM that is being bought (a SHORT underlying view buys a put),
    # so stop/target geometry is validated as long even though `direction` describes the underlying.
    long_premium: bool = False


# Deterministic order-time limits, independent of the ranker's own data-quality verdict.
MAX_QUOTE_AGE_SEC = 15 * 60  # a 5-minute bar plus generous latency
MAX_SPREAD_PCT = 2.0  # (ask - bid) / mid, in percent
MAX_SLIPPAGE_STOP_FRACTION = 0.5  # half-spread (expected slippage) vs distance from entry to stop


# Approvals issued by `evaluate_trade`, so a broker can insist that an order was actually vetted. A live order must
# present a decision id that this process approved moments ago; each approval can be spent once (no replays).
APPROVAL_TTL_SEC = 120.0
_approvals: dict[str, tuple[float, dict | None]] = {}
_approvals_lock = threading.Lock()


def _snapshot(trade: ProposedTrade) -> dict:
    return {"instrument": (trade.instrument or "").upper(), "direction": trade.direction, "quantity": int(trade.quantity),
            "entry": float(trade.entry_price), "stop": float(trade.stop_price), "target": float(trade.target_price)}


def _note_approval(decision_id: str, snapshot: dict | None = None) -> None:
    with _approvals_lock:
        now = time.monotonic()
        for k in [k for k, (t, _) in _approvals.items() if now - t > APPROVAL_TTL_SEC]:
            del _approvals[k]
        _approvals[decision_id] = (now, snapshot)


def _order_matches(snapshot: dict, order) -> str | None:
    """None if the submitted order is the trade that was approved; otherwise what differs."""
    def same(a, b):
        return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-6)

    if (getattr(order, "instrument", "") or "").upper() != snapshot["instrument"]:
        return "instrument"
    if getattr(order, "direction", None) != snapshot["direction"]:
        return "direction"
    if int(getattr(order, "quantity", -1)) != snapshot["quantity"]:
        return "quantity"
    if not same(getattr(order, "price", None), snapshot["entry"]):
        return "price"
    for field, key in (("stop_price", "stop"), ("target_price", "target")):
        value = getattr(order, field, None)
        if value is not None and not same(value, snapshot[key]):
            return field
    return None


def consume_approval(decision_id: str | None, order=None, max_age_sec: float = APPROVAL_TTL_SEC) -> bool:
    """True once for a decision id that `evaluate_trade` approved within `max_age_sec`, and (when `order` is given) only
    if the order is the approved trade: same instrument, direction, quantity and prices. False for anything else
    (unknown id, expired, already used, no id, or a different order), so a missing or mismatched approval fails closed.
    An approval is spent by the attempt even when the order does not match."""
    if not decision_id:
        return False
    with _approvals_lock:
        entry = _approvals.pop(decision_id, None)
    if entry is None or time.monotonic() - entry[0] > max_age_sec:
        return False
    snapshot = entry[1]
    if order is not None and snapshot is not None:
        difference = _order_matches(snapshot, order)
        if difference:
            log_event("risk_engine", f"Order differs from the approved trade ({difference}): rejected", level="ERROR", decision_id=decision_id)
            return False
    return True


@dataclass
class RiskDecision:
    decision_id: str
    approved: bool
    reason: str
    checks: dict = field(default_factory=dict)


def _pct(amount: float, equity: float) -> float:
    return amount / equity * 100 if equity > 0 else 0.0


def drawdown_pct(account: AccountState) -> float:
    return _pct(account.peak_equity - account.equity, account.peak_equity) if account.peak_equity > 0 else 0.0


def loss_streak(account: AccountState) -> int:
    """Consecutive losing trades at the end of today's closed trades (a non-loss ends the streak)."""
    n = 0
    for t in reversed(account.closed_today):
        if (t.get("net_pnl") or 0.0) < 0:
            n += 1
        else:
            break
    return n


def risk_multiplier(account: AccountState) -> tuple[float, str]:
    """Fraction (0-1] of the normal per-trade risk budget allowed right now, and why.

    The smallest of: the drawdown throttle (1.0 below DD_THROTTLE_START_PCT, falling linearly to
    DD_THROTTLE_FLOOR at MAX_DRAWDOWN_PCT), 0.5 once half the daily loss limit is used, and 0.5 after
    LOSS_STREAK_THROTTLE_AFTER consecutive losers today."""
    limits = SETTINGS.risk
    factors: list[tuple[float, str]] = [(1.0, "full size")]

    dd = drawdown_pct(account)
    start, end = limits.dd_throttle_start_pct, limits.max_drawdown_pct
    if end > start and dd > start:
        floor = min(max(limits.dd_throttle_floor, 0.0), 1.0)
        frac = min((dd - start) / (end - start), 1.0)
        factors.append((1.0 - (1.0 - floor) * frac, f"drawdown {dd:.1f}% (throttle from {start:g}%)"))

    daily_loss = _pct(-account.daily_pnl, account.equity)
    if limits.max_daily_loss_pct > 0 and daily_loss >= limits.max_daily_loss_pct / 2:
        factors.append((0.5, f"day's loss {daily_loss:.1f}% is over half the {limits.max_daily_loss_pct:g}% limit"))

    streak = loss_streak(account)
    if limits.loss_streak_throttle_after > 0 and streak >= limits.loss_streak_throttle_after:
        factors.append((0.5, f"{streak} losing trades in a row"))

    return min(factors, key=lambda f: f[0])


def _invalid_account_reason(account: AccountState) -> str | None:
    """Mandatory account inputs must be finite and sensible; otherwise no limit can be checked and nothing is approved."""
    for name in ("equity", "peak_equity", "daily_pnl", "total_exposure"):
        value = getattr(account, name)
        if not (isinstance(value, numbers.Real) and math.isfinite(float(value))):
            return f"account {name} {value!r} is not a finite number"
    if account.equity <= 0:
        return f"account equity {account.equity:g} is not positive"
    if account.peak_equity < 0:
        return "account peak equity is negative"
    for name in ("open_positions_count", "trades_today"):
        value = getattr(account, name)
        if not isinstance(value, numbers.Integral) or value < 0:
            return f"account {name} {value!r} is not a non-negative whole number"
    return None


def evaluate_trade(account: AccountState, trade: ProposedTrade, now: dt.datetime | None = None) -> RiskDecision:
    """Approve or veto a proposed trade. Fails closed: any unexpected error while evaluating is a veto, and an
    approval is only returned once its audit record has been persisted."""
    try:
        return _evaluate_trade(account, trade, now)
    except Exception as exc:
        decision_id = new_decision_id("RISK")
        log_event("risk_engine", f"Risk evaluation raised {type(exc).__name__}: trade blocked", level="ERROR", decision_id=decision_id)
        return _veto(decision_id, f"Risk evaluation failed ({type(exc).__name__}): trade blocked", {"evaluation_error": True})


def _evaluate_trade(account: AccountState, trade: ProposedTrade, now: dt.datetime | None = None) -> RiskDecision:
    decision_id = new_decision_id("RISK")
    checks: dict[str, bool] = {}
    limits = SETTINGS.risk
    now = now or now_ist()

    invalid_account = _invalid_account_reason(account)
    checks["account_state_valid"] = invalid_account is None
    if invalid_account:
        return _veto(decision_id, f"Invalid account state: {invalid_account}", checks)

    if account.kill_switch_engaged:
        return _veto(decision_id, "Emergency kill switch is engaged", checks)

    if not account.broker_connected:
        return _veto(decision_id, "Broker is disconnected - cannot execute", checks)

    invalid = _invalid_order_reason(trade)
    checks["order_parameters_valid"] = invalid is None
    if invalid:
        return _veto(decision_id, f"Invalid order: {invalid}", checks)

    if trade.data_quality_status != "OK":
        return _veto(decision_id, f"Data quality is {trade.data_quality_status} - trade blocked", checks)

    if trade.quote_time is not None:
        age = _age_seconds(trade.quote_time, now)
        checks["quote_fresh"] = age is not None and 0 <= age <= MAX_QUOTE_AGE_SEC
        if not checks["quote_fresh"]:
            shown = "unknown" if age is None else f"{age / 60:.1f} min"
            return _veto(decision_id, f"Price is stale (age {shown}, max {MAX_QUOTE_AGE_SEC // 60} min)", checks)

    if trade.bid is not None or trade.ask is not None:
        spread_problem = _spread_slippage_reason(trade)
        checks["spread_and_slippage_acceptable"] = spread_problem is None
        if spread_problem:
            return _veto(decision_id, spread_problem, checks)

    checks["daily_pnl_known"] = account.daily_pnl_known
    if not account.daily_pnl_known:
        return _veto(decision_id, "Today's P&L is unavailable from the broker: the daily loss limit cannot be verified", checks)

    daily_loss_pct = -account.daily_pnl / account.equity * 100 if account.equity > 0 else 0
    checks["daily_loss_within_limit"] = daily_loss_pct < limits.max_daily_loss_pct
    if not checks["daily_loss_within_limit"]:
        return _veto(decision_id, f"Daily loss {daily_loss_pct:.2f}% exceeds limit {limits.max_daily_loss_pct}%", checks)

    dd = drawdown_pct(account)
    checks["drawdown_within_limit"] = dd < limits.max_drawdown_pct
    if not checks["drawdown_within_limit"]:
        return _veto(decision_id, f"Drawdown {dd:.2f}% exceeds limit {limits.max_drawdown_pct}%", checks)

    day_peak_pct = _pct(account.day_peak_pnl, account.equity)
    checks["profit_lock_not_triggered"] = not (
        limits.profit_lock_trigger_pct > 0
        and day_peak_pct >= limits.profit_lock_trigger_pct
        and account.daily_pnl <= account.day_peak_pnl * (1 - limits.profit_lock_giveback)
    )
    if not checks["profit_lock_not_triggered"]:
        return _veto(
            decision_id,
            f"Profit lock: today's P&L peaked at {day_peak_pct:.2f}% and has given back "
            f"{limits.profit_lock_giveback:.0%} or more - no new trades today",
            checks,
        )

    streak = loss_streak(account)
    if limits.loss_streak_pause > 0 and streak >= limits.loss_streak_pause:
        last_closed = account.closed_today[-1].get("closed_at")
        resume_at = last_closed + dt.timedelta(minutes=limits.loss_streak_pause_min) if last_closed else None
        checks["loss_streak_cooldown_over"] = resume_at is not None and now >= resume_at
        if not checks["loss_streak_cooldown_over"]:
            when = f" until {resume_at:%H:%M}" if resume_at else ""
            return _veto(decision_id, f"{streak} losing trades in a row - new entries paused{when}", checks)

    checks["trade_count_within_limit"] = account.trades_today < limits.max_trades_per_day
    if not checks["trade_count_within_limit"]:
        return _veto(decision_id, f"Max trades per day ({limits.max_trades_per_day}) already reached", checks)

    checks["open_positions_within_limit"] = (
        limits.max_open_positions <= 0 or account.open_positions_count < limits.max_open_positions
    )
    if not checks["open_positions_within_limit"]:
        return _veto(decision_id, f"{account.open_positions_count} positions already open (max {limits.max_open_positions})", checks)

    strategy_losses = sum(
        1 for t in account.closed_today if t.get("strategy") == trade.strategy_name and (t.get("net_pnl") or 0.0) < 0
    )
    checks["strategy_losses_within_limit"] = (
        limits.strategy_max_losses_per_day <= 0 or strategy_losses < limits.strategy_max_losses_per_day
    )
    if not checks["strategy_losses_within_limit"]:
        return _veto(
            decision_id,
            f"{trade.strategy_name} has lost {strategy_losses} trades today (max {limits.strategy_max_losses_per_day}) - done for the day",
            checks,
        )

    multiplier, why = risk_multiplier(account)
    checks["risk_multiplier"] = round(multiplier, 3)
    allowed_risk_pct = limits.max_risk_per_trade_pct * multiplier
    risk_amount = abs(trade.entry_price - trade.stop_price) * trade.quantity
    risk_pct = risk_amount / account.equity * 100 if account.equity > 0 else 100
    checks["risk_per_trade_within_limit"] = risk_pct <= allowed_risk_pct
    if not checks["risk_per_trade_within_limit"]:
        scaled = f" (reduced to {multiplier:.0%}: {why})" if multiplier < 1 else ""
        return _veto(decision_id, f"Trade risk {risk_pct:.2f}% exceeds max risk per trade {allowed_risk_pct:.2f}%{scaled}", checks)

    strategy_exposure = account.exposure_by_strategy.get(trade.strategy_name, 0.0) + risk_amount
    strategy_exposure_pct = strategy_exposure / account.equity * 100 if account.equity > 0 else 100
    checks["strategy_exposure_within_limit"] = strategy_exposure_pct <= limits.max_strategy_exposure_pct
    if not checks["strategy_exposure_within_limit"]:
        return _veto(
            decision_id,
            f"Strategy exposure {strategy_exposure_pct:.2f}% exceeds limit {limits.max_strategy_exposure_pct}%",
            checks,
        )

    total_exposure_pct = (account.total_exposure + risk_amount) / account.equity * 100 if account.equity > 0 else 100
    checks["portfolio_exposure_within_limit"] = total_exposure_pct <= limits.max_portfolio_exposure_pct
    if not checks["portfolio_exposure_within_limit"]:
        return _veto(
            decision_id,
            f"Portfolio exposure {total_exposure_pct:.2f}% exceeds limit {limits.max_portfolio_exposure_pct}%",
            checks,
        )

    if trade.instrument:
        group = group_for(trade.instrument)
        in_group = [p for p in account.open_positions if group_for(p.get("instrument")) == group]
        same_dir = sum(1 for p in in_group if p.get("direction") == trade.direction)
        checks["group_positions_within_limit"] = (
            limits.max_group_positions <= 0 or same_dir + 1 <= limits.max_group_positions
        )
        if not checks["group_positions_within_limit"]:
            return _veto(
                decision_id,
                f"{same_dir} positions already open {trade.direction} in group {group} (max {limits.max_group_positions})",
                checks,
            )
        group_risk_pct = _pct(sum(p.get("risk") or 0.0 for p in in_group) + risk_amount, account.equity)
        checks["group_risk_within_limit"] = limits.max_group_risk_pct <= 0 or group_risk_pct <= limits.max_group_risk_pct
        if not checks["group_risk_within_limit"]:
            return _veto(
                decision_id,
                f"Open risk in group {group} would be {group_risk_pct:.2f}% of equity (max {limits.max_group_risk_pct}%)",
                checks,
            )

    min_liquidity = 0.3
    checks["liquidity_sufficient"] = trade.relative_volume is None or trade.relative_volume >= min_liquidity
    if not checks["liquidity_sufficient"]:
        return _veto(decision_id, f"Relative volume {trade.relative_volume:.2f} below minimum liquidity threshold {min_liquidity}", checks)

    if not _persist_risk_event(decision_id, "APPROVED", "All risk checks passed", checks):
        # An approval that cannot be audited is not an approval.
        return _veto(decision_id, "Risk audit trail unavailable: the approval could not be recorded, trade blocked", checks)
    log_event("risk_engine", "Trade approved by risk engine", level="INFO", decision_id=decision_id, strategy=trade.strategy_name)
    _note_approval(decision_id, _snapshot(trade))
    return RiskDecision(decision_id, True, "All risk checks passed", checks)


def _finite_positive(x) -> bool:
    return isinstance(x, numbers.Real) and float(x) == float(x) and 0 < float(x) < float("inf")


def _invalid_order_reason(trade: ProposedTrade) -> str | None:
    """Structural validity of the order: prices, stop side, target side, quantity, lot size. None if valid."""
    if trade.direction not in ("LONG", "SHORT"):
        return f"direction '{trade.direction}' is not LONG or SHORT"
    for name, price in (("entry", trade.entry_price), ("stop", trade.stop_price), ("target", trade.target_price)):
        if not _finite_positive(price):
            return f"{name} price {price!r} is not a positive number"
    if trade.entry_price == trade.stop_price:
        return "stop equals entry (zero stop distance)"
    long = trade.direction == "LONG" or trade.long_premium
    if long and trade.stop_price >= trade.entry_price:
        return f"LONG stop {trade.stop_price:g} is not below entry {trade.entry_price:g}"
    if not long and trade.stop_price <= trade.entry_price:
        return f"SHORT stop {trade.stop_price:g} is not above entry {trade.entry_price:g}"
    if long and trade.target_price <= trade.entry_price:
        return f"LONG target {trade.target_price:g} is not above entry {trade.entry_price:g}"
    if not long and trade.target_price >= trade.entry_price:
        return f"SHORT target {trade.target_price:g} is not below entry {trade.entry_price:g}"
    if not isinstance(trade.quantity, numbers.Integral) or isinstance(trade.quantity, bool) or trade.quantity <= 0:
        return f"quantity {trade.quantity!r} is not a positive whole number"
    if trade.lot_size and trade.quantity % trade.lot_size != 0:
        return f"quantity {trade.quantity} is not a multiple of lot size {trade.lot_size}"
    return None


def _age_seconds(observed: dt.datetime, now: dt.datetime) -> float | None:
    try:
        if (observed.tzinfo is None) != (now.tzinfo is None):
            observed, now = observed.replace(tzinfo=None), now.replace(tzinfo=None)
        return (now - observed).total_seconds()
    except Exception:
        return None


def _spread_slippage_reason(trade: ProposedTrade) -> str | None:
    """Veto reason if the quote is unusable, the spread is too wide, or expected slippage eats the stop."""
    if not (_finite_positive(trade.bid) and _finite_positive(trade.ask)):
        return "Spread unknown: bid/ask missing or non-positive - cannot verify execution cost"
    bid, ask = float(trade.bid), float(trade.ask)  # type: ignore[arg-type]
    if ask < bid:
        return f"Crossed quote (bid {bid:g} > ask {ask:g}) - cannot trust price"
    spread = ask - bid
    spread_pct = spread / ((ask + bid) / 2) * 100
    if spread_pct > MAX_SPREAD_PCT:
        return f"Spread {spread_pct:.2f}% exceeds max {MAX_SPREAD_PCT:g}%"
    stop_distance = abs(trade.entry_price - trade.stop_price)
    if spread / 2 > MAX_SLIPPAGE_STOP_FRACTION * stop_distance:
        return (
            f"Expected slippage {spread / 2:.2f} is over {MAX_SLIPPAGE_STOP_FRACTION:.0%} of the "
            f"stop distance {stop_distance:.2f}"
        )
    return None


def _veto(decision_id: str, reason: str, checks: dict) -> RiskDecision:
    log_event("risk_engine", f"Trade VETOED: {reason}", level="WARNING", decision_id=decision_id)
    _persist_risk_event(decision_id, "VETO", reason, checks)
    return RiskDecision(decision_id, False, reason, checks)


def _json_safe(details: dict) -> dict:
    """Plain-Python copy of the check results: numpy booleans / floats (from numpy inputs) are not JSON serialisable,
    and an unserialisable audit row would otherwise block an approval that is perfectly valid."""
    def default(value):
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    return json.loads(json.dumps(details, default=default))


def _persist_risk_event(decision_id: str, event_type: str, reason: str, details: dict) -> bool:
    """Write the audit row. Returns whether it was stored (a failure is logged, never silent)."""
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import RiskEvent

        with get_session() as session:
            session.add(RiskEvent(decision_id=decision_id, event_type=event_type, reason=reason, details=_json_safe(details)))
        return True
    except Exception as exc:  # a veto stands without its row, but a lost audit row must not be invisible
        log_event("risk_engine", f"Could not persist risk event {decision_id}: {type(exc).__name__}", level="ERROR")
        return False
