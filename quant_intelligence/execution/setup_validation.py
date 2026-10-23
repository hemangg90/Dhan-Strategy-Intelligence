"""
Setup validation: the stage between "a strategy was ranked first" and "pick a contract".

A ranking is not an entry signal, an analogue is not an entry signal and a high options score is not an entry signal.
Only a setup that is ACTIVE NOW, on fresh data, in a tradeable session, with something tradeable to buy, with room to run,
proceeds. The result is one of:
  VALID     every blocking check passed (warnings may remain)
  WAIT      the setup is not active: waiting for its entry conditions, or it fired too many bars ago
  NO_TRADE  something blocks trading: no tradeable ranking, bad or stale data, closed session, no options data, no
            eligible contract, or no room before the next significant level
Blocking checks (hard): ranking tradable; market and options data status; session open; last bar fresh; setup active and
not expired; at least one eligible contract near the money; room to the next significant level (BUY only).
Advisory (warnings, never block): structure trend opposing a trend strategy, an opposing liquidity event, extreme
volatility, near-expiry gamma/theta, data weaknesses. They are shown, because making them blocks would change what the
strategies mean without evidence that it helps.
Pure logic over the contexts; no network.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from quant_intelligence.config.holidays import is_trading_day
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.features.levels import assess_trade
from quant_intelligence.intelligence.market_context import MarketContext
from quant_intelligence.intelligence.options_context import OptionsContext
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeframe import parse_timeframe_minutes

VALID, WAIT, NO_TRADE = "VALID", "WAIT", "NO_TRADE"
PREFIX = "SETUP VALIDATION"
TREND_STRATEGIES = {"Momentum", "Donchian Channel Breakout", "Supertrend Trend Following", "EMA 50/200 Crossover",
                    "MACD Signal Crossover", "Opening Range Breakout", "CPR Breakout", "Inside Bar Breakout"}
LEVEL_NATIVE_STRATEGIES = {"Camarilla Pivot Reversal"}  # trade the pivot levels themselves; stop and target already are levels


@dataclass(frozen=True)
class Reason:
    code: str
    message: str
    component: str
    blocking: bool = True


@dataclass
class ValidationResult:
    status: str
    reasons: list[Reason] = field(default_factory=list)  # why it is not VALID
    warnings: list[Reason] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    setup: object | None = None  # the setup, with stop/target moved to the levels when that applied
    level_review: str = ""
    bar_age_minutes: float | None = None

    @property
    def valid(self) -> bool:
        return self.status == VALID

    @property
    def code(self) -> str | None:
        return self.reasons[0].code if self.reasons else None

    @property
    def message(self) -> str:
        """One line for the decision log; a no-room verdict keeps its established 'LEVEL CHECK:' wording."""
        if not self.reasons:
            return ""
        first = self.reasons[0]
        return first.message if first.code == "NO_ROOM" else f"{PREFIX} [{first.code}]: {first.message}"


def _block(result: ValidationResult, status: str, code: str, message: str, component: str) -> ValidationResult:
    result.status = status
    result.reasons.append(Reason(code, message, component))
    return result


def validate_setup(
    *,
    ranking,
    setup_status,
    market: MarketContext,
    options: OptionsContext | None,
    now: dt.datetime,
    strategy_name: str | None,
    transaction: str = "BUY",
    mode: str = "PAPER",
    require_options: bool = True,
) -> ValidationResult:
    from quant_intelligence.execution.setup_detector import SETUP_TRIGGERED, WAITING_FOR_SETUP
    from quant_intelligence.ranking.ranking_engine import tradable

    res = ValidationResult(VALID, setup=getattr(setup_status, "setup", None))

    ok, why = tradable(ranking, mode)
    res.checks["ranking_tradable"] = ok
    if not ok:
        return _block(res, NO_TRADE, "RANKING", why or "the ranking is NO TRADE", "ranking")

    res.checks["market_data_ok"] = market.status.ok
    if not market.status.ok:
        return _block(res, NO_TRADE, "DATA_QUALITY", f"market data is not usable ({market.status.reason})", "market_context")
    if market.status.level != "OK":
        res.warnings.append(Reason("DATA_WARNING", market.status.reason, "market_context", False))

    profile = profile_for(market.underlying)
    in_session = is_trading_day(now.date(), profile.name) and profile.open <= now.time() < profile.close
    res.checks["session_open"] = in_session
    if not in_session:
        return _block(res, NO_TRADE, "SESSION", f"the {profile.name} session is closed at {now:%H:%M} ({profile.label})", "session")

    bar_minutes = float(parse_timeframe_minutes(market.timeframe))
    max_age = max(SETTINGS.setup_max_bar_age_min, 2.0 * bar_minutes + 5.0)  # a bar is stamped when it opens: allow a bar and some latency
    age = (now - market.timestamp.to_pydatetime().replace(tzinfo=None)).total_seconds() / 60.0
    res.bar_age_minutes = age
    res.checks["bar_fresh"] = 0 <= age <= max_age
    if not res.checks["bar_fresh"]:
        return _block(res, NO_TRADE, "STALE_DATA", f"the last bar is {age:.0f} minutes old (limit {max_age:g})", "market_context")

    status = getattr(setup_status, "status", None)
    res.checks["setup_active"] = status == SETUP_TRIGGERED
    if status != SETUP_TRIGGERED:
        return _block(res, WAIT if status == WAITING_FOR_SETUP else NO_TRADE, "SETUP_NOT_ACTIVE",
                      "the entry conditions are not met on the latest bar" if status == WAITING_FOR_SETUP else "no setup on the latest bar",
                      "strategy")
    setup = res.setup
    setup_age_bars = (market.timestamp - setup.timestamp).total_seconds() / 60.0 / bar_minutes
    res.checks["setup_not_expired"] = setup_age_bars <= SETTINGS.setup_valid_bars
    if not res.checks["setup_not_expired"]:
        return _block(res, WAIT, "SETUP_EXPIRED", f"the setup fired {setup_age_bars:.0f} bars ago (valid for {SETTINGS.setup_valid_bars})", "strategy")

    if require_options:
        if options is None:
            return _block(res, NO_TRADE, "OPTIONS_DATA", "no option chain context is available", "options_context")
        res.checks["options_data_ok"] = options.status.ok
        if not options.status.ok:
            return _block(res, NO_TRADE, "OPTIONS_DATA", f"option data is not usable ({options.status.reason})", "options_context")
        eligible = options.liquidity.get("eligible", 0)
        res.checks["eligible_contracts"] = bool(eligible)
        if not eligible:
            return _block(res, NO_TRADE, "NO_ELIGIBLE_CONTRACT",
                          "no contract near the money has a fresh two-sided quote, an acceptable spread and quoted size", "options_context")
        if options.status.level != "OK":
            res.warnings.append(Reason("OPTIONS_WARNING", options.status.reason, "options_context", False))
        if "NEAR_EXPIRY" in options.expiry_info.flags:
            res.warnings.append(Reason("NEAR_EXPIRY", f"{options.expiry_info.risk_band.lower()} gamma/theta: "
                                                     f"{options.expiry_info.trading_days:.2f} trading days left", "expiry", False))

    # room to the next significant level (a bought option only: a written option has its own payoff)
    if SETTINGS.levels_enabled and transaction == "BUY" and strategy_name not in LEVEL_NATIVE_STRATEGIES and market.atr:
        check = assess_trade(setup.direction, setup.entry_price, setup.stop_price, setup.target_price, list(market.levels), market.atr,
                             SETTINGS.levels_min_room_mult, SETTINGS.levels_buffer_atr, None)
        res.level_review = check.summary
        res.checks["room_to_run"] = check.ok
        if not check.ok:
            return _block(res, NO_TRADE, "NO_ROOM", f"LEVEL CHECK: {check.reason}.", "levels")
        if check.stop_changed or check.target_capped:
            import dataclasses

            res.setup = dataclasses.replace(setup, stop_price=check.stop, target_price=check.target,
                                            meta={**setup.meta, "levels": check.summary})

    # advisory reading
    trend = market.structure.trend
    if strategy_name in TREND_STRATEGIES and ((setup.direction == "LONG" and trend == "DOWN") or (setup.direction == "SHORT" and trend == "UP")):
        res.warnings.append(Reason("STRUCTURE_OPPOSES", f"the market structure trend is {trend.lower()} against this {setup.direction.lower()} "
                                                        "trend-following setup", "market_structure", False))
    against = "SELL_SIDE" if setup.direction == "LONG" else "BUY_SIDE"
    for ev in market.liquidity.active_events():
        if ev.kind == "BREAKOUT_ACCEPTED" and ev.side == against:
            res.warnings.append(Reason("LIQUIDITY_OPPOSES", f"price was accepted beyond {ev.source.replace('_', ' ').lower()} {ev.level:g} "
                                                            "against this direction", "liquidity", False))
            break
    pctl = market.volatility.get("atr_percentile_100")
    if pctl is not None and pctl >= 0.95:
        res.warnings.append(Reason("EXTREME_VOLATILITY", f"ATR is in its top {100 - pctl * 100:.0f}% of recent history", "volatility", False))
    res.checks["all_blocking_checks_passed"] = True
    return res
