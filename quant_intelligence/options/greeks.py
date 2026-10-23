"""
Option Greeks for a quoted contract, computed by the application (not just copied from the broker).

Model
  * European exercise, one flat volatility, constant rates (Black-Scholes-Merton; `options/black_scholes.py`).
    NSE/BSE index and stock options and MCX commodity options are European, so this is the appropriate model; it would
    not be for an American-style contract, which this module does not support.
  * NSE / BSE: continuous dividend yield q = SETTINGS.dividend_yield (carry on the index or stock).
    MCX options are on futures: Black-76 is BSM with q = r (the forward equals the futures price S).
  * r = SETTINGS.risk_free_rate (continuous, annual).
  * T = remaining TRADING time to the expiry-day close in years of trading time (`config.expiry_rules.time_to_expiry_years`,
    holidays excluded), the same clock the volatility is annualised on. No overnight or weekend decay is counted.
  * Volatility = the implied volatility SOLVED from the contract's own price (mid of a two-sided quote, else last traded),
    so price and Greeks are consistent. The broker's IV is reported beside it; if the solver cannot find one and the
    broker gave an IV, that is used and `iv_source` says so. With neither, the Greeks are UNAVAILABLE - never zero.
Units
  delta, gamma   per 1 unit of the underlying
  vega           rupees per 1 volatility point (1.00 = 1%) of change in implied volatility, per unit
  theta          rupees per TRADING day, per unit
  rho            rupees per 1% change in the risk-free rate, per unit
Limits: a flat-vol model ignores the smile, so Greeks of far wings are approximate; very close to expiry (under
`MIN_T_TRADING_DAYS`) the IV solve is unstable and the Greeks are reported unavailable instead.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

from quant_intelligence.config.expiry_rules import time_to_expiry_years
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.options import black_scholes as bs
from quant_intelligence.options.contracts import OptionQuote
from quant_intelligence.utils.market_profile import profile_for

MIN_T_TRADING_DAYS = 0.02  # about 7 minutes of an NSE session
PROVIDER_DELTA_TOLERANCE = 0.05


@dataclass(frozen=True)
class GreekSet:
    available: bool
    reason: str = ""
    model: str = ""  # BSM / BLACK76
    kind: str = ""  # CALL / PUT
    iv: float | None = None  # decimal, annualised (0.18 = 18%)
    iv_source: str = ""  # solved_from_price / provider
    price_basis: str = ""  # mid / ltp
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None  # per 1 vol point, per unit
    theta: float | None = None  # per trading day, per unit
    rho: float | None = None  # per 1% rate, per unit
    t_years: float | None = None
    trading_days: float | None = None
    provider_iv: float | None = None  # decimal
    provider_delta: float | None = None
    provider_delta_gap: float | None = None
    provider_disagrees: bool = False

    def scaled(self, quantity: float) -> dict:
        """Position exposure for `quantity` units (negative for short). Rupee Greeks for vega, theta, rho."""
        if not self.available:
            raise ValueError("Greeks are unavailable: " + self.reason)
        return {"delta": self.delta * quantity, "gamma": self.gamma * quantity, "vega": self.vega * quantity,
                "theta": self.theta * quantity, "rho": self.rho * quantity}


def rates_for(underlying: str) -> tuple[float, float, str]:
    """(r, q, model) for the instrument's market."""
    r = SETTINGS.risk_free_rate
    if profile_for(underlying).name == "MCX":
        return r, r, "BLACK76"
    return r, SETTINGS.dividend_yield, "BSM"


def _unavailable(reason: str, **kw) -> GreekSet:
    return GreekSet(False, reason, **kw)


def compute_greeks(quote: OptionQuote, spot: float, now: dt.datetime) -> GreekSet:
    """Greeks of one contract at `spot` and `now` (naive IST). See the module docstring."""
    if quote.option_type not in ("CE", "PE"):
        return _unavailable(f"unknown option type {quote.option_type!r}")
    kind = bs.CALL if quote.option_type == "CE" else bs.PUT
    if not (isinstance(spot, (int, float)) and math.isfinite(spot) and spot > 0):
        return _unavailable("the underlying price is not a positive number", kind=kind)
    if not (isinstance(quote.strike, (int, float)) and quote.strike > 0):
        return _unavailable("the strike is not a positive number", kind=kind)
    try:
        expiry = dt.date.fromisoformat(str(quote.expiry)[:10])
    except ValueError:
        return _unavailable(f"the expiry {quote.expiry!r} is not a valid date", kind=kind)
    profile = profile_for(quote.underlying)
    r, q, model = rates_for(quote.underlying)
    t = time_to_expiry_years(now, expiry, profile, SETTINGS.trading_days_per_year)
    days = t * SETTINGS.trading_days_per_year
    if t <= 0:
        return _unavailable("the contract has expired", model=model, kind=kind, t_years=t, trading_days=days)
    if days < MIN_T_TRADING_DAYS:
        return _unavailable(f"too close to expiry ({days:.3f} trading days) for a stable implied volatility", model=model,
                            kind=kind, t_years=t, trading_days=days)

    mid = quote.mid
    if mid is not None:
        price, basis = mid, "mid"
    elif quote.ltp is not None and quote.ltp > 0:
        price, basis = quote.ltp, "ltp"
    else:
        return _unavailable("no usable price (no two-sided quote and no last traded price)", model=model, kind=kind,
                            t_years=t, trading_days=days)

    provider_iv = quote.iv / 100.0 if quote.iv is not None and quote.iv > 0 else None
    sigma, why = bs.implied_vol_with_reason(price, spot, quote.strike, t, r, q, kind)
    source = "solved_from_price"
    if not math.isfinite(sigma):
        if provider_iv is None:
            return _unavailable(f"implied volatility could not be solved ({why}) and the broker gave none", model=model,
                                kind=kind, t_years=t, trading_days=days)
        sigma, source = provider_iv, "provider"

    g = bs.greeks(spot, quote.strike, t, r, q, sigma, kind)
    rho = bs.rho(spot, quote.strike, t, r, q, sigma, kind)
    gap, disagrees = None, False
    if quote.delta is not None and math.isfinite(quote.delta):
        gap = abs(g["delta"] - quote.delta)
        disagrees = gap > PROVIDER_DELTA_TOLERANCE
    return GreekSet(
        True, "", model, kind, sigma, source, basis, g["delta"], g["gamma"], g["vega"] / 100.0,
        g["theta"] / SETTINGS.trading_days_per_year, rho / 100.0, t, days, provider_iv, quote.delta, gap, disagrees,
    )
