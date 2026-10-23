"""
Expiry intelligence: how much trading time is left on a contract and what that means for its risk.

Time comes from the exchange calendar and session hours the application already holds (`config/holidays.py`,
`utils/market_profile.py`, `config/expiry_rules.py`), never from a hard-coded expiry weekday: the expiry date is whatever
the instrument's own chain says. The only convention used is "the contract's last moment is the close of its expiry day".

Risk bands from trading days left (thresholds are module constants, adjustable in one place):
  EXTREME  at most 0.25 trading day (about the last session quarter): premium is almost all gamma and theta
  HIGH     at most 1 trading day
  ELEVATED at most 3 trading days
  NORMAL   more
These describe how fast delta and decay change, not whether a trade is good.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from quant_intelligence.config.expiry_rules import time_to_expiry_years, trading_days_between
from quant_intelligence.config.holidays import is_trading_day
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.market_profile import profile_for

EXTREME_DAYS, HIGH_DAYS, ELEVATED_DAYS = 0.25, 1.0, 3.0


@dataclass(frozen=True)
class ExpiryInfo:
    available: bool
    reason: str = ""
    expiry: dt.date | None = None
    trading_days: float | None = None  # fractional, includes the rest of today's session
    years: float | None = None  # trading years, the clock Greeks use
    sessions_remaining: int | None = None  # sessions that still have time left (today counts while it is open)
    is_expiry_day: bool = False
    minutes_to_close: float | None = None
    risk_band: str = ""  # EXTREME / HIGH / ELEVATED / NORMAL
    is_monthly: bool | None = None  # last expiry of its month among the listed ones (None if the list is unknown)
    flags: tuple[str, ...] = ()


def expiry_info(underlying: str, expiry: str | dt.date, now: dt.datetime, listed_expiries: list[str] | None = None) -> ExpiryInfo:
    try:
        day = expiry if isinstance(expiry, dt.date) and not isinstance(expiry, dt.datetime) else dt.date.fromisoformat(str(expiry)[:10])
    except ValueError:
        return ExpiryInfo(False, f"the expiry {expiry!r} is not a valid date")
    profile = profile_for(underlying)
    market = profile.name
    flags: list[str] = []
    if not is_trading_day(day, market):
        flags.append("EXPIRY_ON_NON_TRADING_DAY")
        return ExpiryInfo(False, f"the expiry {day} is not a trading day on the {market} calendar - the contract metadata disagrees "
                                 "with the exchange calendar", day, flags=tuple(flags))
    today = now.date()
    if day < today or (day == today and now >= dt.datetime.combine(today, profile.close)):
        return ExpiryInfo(False, "the contract has expired", day, 0.0, 0.0, 0, flags=("EXPIRED",))
    years = time_to_expiry_years(now, day, profile, SETTINGS.trading_days_per_year)
    days = years * SETTINGS.trading_days_per_year
    open_today = is_trading_day(today, market) and now < dt.datetime.combine(today, profile.close)
    sessions = trading_days_between(today, day, market) + (1 if open_today else 0)
    minutes_to_close = None
    if open_today and day == today:
        minutes_to_close = (dt.datetime.combine(today, profile.close) - now).total_seconds() / 60.0
        flags.append("EXPIRY_DAY")
    band = "EXTREME" if days <= EXTREME_DAYS else "HIGH" if days <= HIGH_DAYS else "ELEVATED" if days <= ELEVATED_DAYS else "NORMAL"
    if band in ("EXTREME", "HIGH"):
        flags.append("NEAR_EXPIRY")
    monthly = None
    if listed_expiries:
        same_month = sorted(e[:10] for e in listed_expiries if e[:7] == day.isoformat()[:7])
        monthly = bool(same_month) and same_month[-1] == day.isoformat()
    return ExpiryInfo(True, "", day, days, years, sessions, day == today, minutes_to_close, band, monthly, tuple(flags))


def decay_and_gamma_profile(premium: float, spot: float, gamma: float, theta_per_day: float) -> dict:
    """What the expiry clock does to ONE unit of an option: daily decay as a share of its price, and how much delta changes
    for a 1% move in the underlying. Inputs are per unit, from `options/greeks.py`."""
    if not (premium > 0 and spot > 0):
        return {"theta_pct_of_premium_per_day": None, "delta_change_per_1pct_move": None}
    return {"theta_pct_of_premium_per_day": theta_per_day / premium * 100.0,
            "delta_change_per_1pct_move": gamma * spot * 0.01}
