"""Black-Scholes-Merton pricing, greeks and implied volatility (European options, continuous dividend yield q).

Time `T` is in YEARS of trading time (trading days / trading_days_per_year), not calendar time. For
commodity options on futures (MCX) pass q = r (Black-76: the forward equals spot). Known limits: one flat
volatility (no skew/smile), European exercise, constant rates.

Greeks: delta, gamma per unit of underlying; vega per 1.00 (100 vol points) change in sigma; theta per YEAR
of trading time (divide by trading days for per-day).
"""
from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm

CALL, PUT = "CALL", "PUT"


def _check_kind(kind: str) -> str:
    kind = kind.upper()
    if kind not in (CALL, PUT):
        raise ValueError(f"kind must be CALL or PUT, got {kind!r}")
    return kind


def intrinsic(S: float, K: float, kind: str) -> float:
    return max(S - K, 0.0) if _check_kind(kind) == CALL else max(K - S, 0.0)


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    vol_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_t
    return d1, d1 - vol_t


def price(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str) -> float:
    """Option value. At expiry (T<=0) or with no volatility it is the discounted-forward intrinsic value."""
    kind = _check_kind(kind)
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0 or sigma <= 0:
        t = max(T, 0.0)
        fwd = S * math.exp(-q * t) - K * math.exp(-r * t)
        return max(fwd, 0.0) if kind == CALL else max(-fwd, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if kind == CALL:
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str) -> dict:
    """{"delta","gamma","vega","theta"} (see module docstring for units). Gamma/vega/theta are 0 at T<=0."""
    kind = _check_kind(kind)
    if T <= 0 or sigma <= 0:
        itm = (S > K) if kind == CALL else (S < K)
        delta = (1.0 if kind == CALL else -1.0) if itm else 0.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
    pdf = norm.pdf(d1)
    gamma = disc_q * pdf / (S * sigma * math.sqrt(T))
    vega = S * disc_q * pdf * math.sqrt(T)
    common = -S * disc_q * pdf * sigma / (2.0 * math.sqrt(T))
    if kind == CALL:
        delta = disc_q * norm.cdf(d1)
        theta = common - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = common + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def rho(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str) -> float:
    """Change in value per 1.00 (100 percentage points) change in the risk-free rate. 0 at T<=0 or sigma<=0."""
    kind = _check_kind(kind)
    if T <= 0 or sigma <= 0:
        return 0.0
    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r = math.exp(-r * T)
    return K * T * disc_r * norm.cdf(d2) if kind == CALL else -K * T * disc_r * norm.cdf(-d2)


def implied_vol_with_reason(
    market_price: float, S: float, K: float, T: float, r: float, q: float, kind: str,
    lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-8,
) -> tuple[float, str | None]:
    """Solve price(sigma) = market_price with Brent's method. Returns (iv, None) or (nan, reason)."""
    kind = _check_kind(kind)
    nan = float("nan")
    if not all(map(math.isfinite, (market_price, S, K, T, r, q))):
        return nan, "non-finite input"
    if market_price <= 0 or S <= 0 or K <= 0:
        return nan, "price, spot and strike must be positive"
    if T <= 0:
        return nan, "no time value at expiry"
    floor = price(S, K, T, r, q, lo, kind)
    ceiling = price(S, K, T, r, q, hi, kind)
    if market_price < floor - tol:
        return nan, "price below the no-arbitrage lower bound (intrinsic)"
    if market_price > ceiling + tol:
        return nan, f"price needs volatility above {hi:.0%}"
    if abs(market_price - floor) <= tol:
        return lo, None
    try:
        iv = brentq(lambda s: price(S, K, T, r, q, s, kind) - market_price, lo, hi, xtol=tol, maxiter=200)
    except (ValueError, RuntimeError) as e:
        return nan, f"solver failed: {e}"
    return float(iv), None


def implied_vol(market_price: float, S: float, K: float, T: float, r: float, q: float, kind: str, **kw) -> float:
    return implied_vol_with_reason(market_price, S, K, T, r, q, kind, **kw)[0]
