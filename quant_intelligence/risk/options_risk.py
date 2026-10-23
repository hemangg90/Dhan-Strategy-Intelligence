"""
Option-specific risk controls, called by the risk engine for any trade that carries option information.

This module only MEASURES and COMPARES. It knows nothing about strategies, rankings or contract selection, and it cannot be
overridden by them: the risk engine calls it and a veto stops the trade (fail closed).

Exposure definitions (summed over open option positions plus the proposed trade; BUY = +1, SELL = -1; Greeks are per unit,
signed as the option's own delta is, taken at the position's ENTRY - they are not re-marked, which is a stated limitation):
  premium at risk     premium paid x quantity on bought options: the most they can lose (written options are not counted here)
  net delta notional  sum(delta x quantity x spot): rupees of underlying exposure
  gamma (1% move)     sum(gamma x quantity x spot^2 x 0.01): rupees of delta-notional that change for a 1% move
  vega                sum(vega x quantity): rupees per 1 volatility point
  theta               sum(theta x quantity): rupees per trading day
Limits are percentages of equity (0 disables a limit) and are read from `RiskLimits`; the absolute post-trade value is compared.
If a limit is enabled and the Greeks of the proposed trade, or of an open position, are unknown, the trade is vetoed: an
exposure that cannot be measured cannot be shown to be within its limit.
"""
from __future__ import annotations

import math


def _sign(transaction: str | None) -> float:
    return -1.0 if (transaction or "BUY").upper() == "SELL" else 1.0


def _finite(*xs) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def position_exposure(pos: dict) -> dict | None:
    """Exposure of one open OPTION position from the Greeks stored at entry, or None when they are not stored."""
    g = pos.get("entry_greeks") or {}
    qty = pos.get("quantity")
    spot = g.get("spot")
    if not (_finite(qty, spot, g.get("delta"), g.get("gamma"), g.get("vega"), g.get("theta"))):
        return None
    s = _sign(pos.get("transaction"))
    return {
        "delta_notional": s * g["delta"] * qty * spot,
        "gamma_1pct": s * g["gamma"] * qty * spot * spot * 0.01,
        "vega": s * g["vega"] * qty,
        "theta": s * g["theta"] * qty,
    }


def exposure_from_positions(open_positions: list[dict]) -> dict:
    """The account's option exposure: totals, per-underlying premium, and how many open option positions had no stored Greeks."""
    out = {"premium_at_risk": 0.0, "by_underlying": {}, "delta_notional": 0.0, "gamma_1pct": 0.0, "vega": 0.0, "theta": 0.0,
           "n_option_positions": 0, "unknown_positions": 0}
    for p in open_positions:
        if not p.get("option_type"):
            continue
        out["n_option_positions"] += 1
        if (p.get("transaction") or "BUY") == "BUY" and _finite(p.get("entry_price"), p.get("quantity")):
            prem = p["entry_price"] * p["quantity"]
            out["premium_at_risk"] += prem
            key = p.get("underlying") or p.get("instrument") or "?"
            out["by_underlying"][key] = out["by_underlying"].get(key, 0.0) + prem
        e = position_exposure(p)
        if e is None:
            out["unknown_positions"] += 1
            continue
        for k in ("delta_notional", "gamma_1pct", "vega", "theta"):
            out[k] += e[k]
    return out


def trade_exposure(option: dict, quantity: int, transaction: str) -> dict | None:
    return position_exposure({"entry_greeks": option.get("greeks"), "quantity": quantity, "transaction": transaction})


def check_option_trade(account_exposure: dict, option: dict, quantity: int, entry_price: float, transaction: str,
                       equity: float, limits, lot_size: int | None = None) -> tuple[dict, str | None]:
    """(checks, veto reason). `option` carries underlying, greeks (per unit + spot), trading_days_to_expiry. Fails closed."""
    checks: dict = {}
    if equity <= 0:
        return checks, "Equity is not positive: option exposure limits cannot be evaluated"
    ex = account_exposure or {}
    pct = lambda amount: abs(amount) / equity * 100.0  # noqa: E731

    lots = quantity / lot_size if lot_size else None
    max_lots = getattr(limits, "max_order_lots", 0)
    checks["order_size_within_limit"] = not (max_lots and lots is not None and lots > max_lots)
    if not checks["order_size_within_limit"]:
        return checks, f"Order of {lots:g} lots exceeds the maximum order size of {max_lots} lots"

    days = option.get("trading_days_to_expiry")
    min_days = getattr(limits, "min_trading_days_to_expiry", 0.0)
    if min_days > 0 and transaction == "BUY":
        checks["time_to_expiry_known"] = _finite(days)
        if not checks["time_to_expiry_known"]:
            return checks, "Time to expiry is unknown: the expiry constraint cannot be verified"
        checks["expiry_not_too_close"] = days >= min_days
        if not checks["expiry_not_too_close"]:
            return checks, f"Only {days:.2f} trading days to expiry (minimum {min_days:g} for a new long option)"

    premium = entry_price * quantity if transaction == "BUY" else 0.0
    limit = getattr(limits, "max_premium_at_risk_pct", 0.0)
    if limit > 0 and transaction == "BUY":
        total = ex.get("premium_at_risk", 0.0) + premium
        checks["premium_at_risk_within_limit"] = pct(total) <= limit
        if not checks["premium_at_risk_within_limit"]:
            return checks, f"Premium at risk would be {pct(total):.2f}% of equity (max {limit:g}%)"
    limit = getattr(limits, "max_underlying_premium_pct", 0.0)
    if limit > 0 and transaction == "BUY":
        key = option.get("underlying") or "?"
        total = ex.get("by_underlying", {}).get(key, 0.0) + premium
        checks["underlying_premium_within_limit"] = pct(total) <= limit
        if not checks["underlying_premium_within_limit"]:
            return checks, f"Premium at risk in {key} would be {pct(total):.2f}% of equity (max {limit:g}%)"

    greek_limits = {k: getattr(limits, f"max_{k}_pct", 0.0) for k in ("net_delta", "gamma", "vega", "theta")}
    if any(v > 0 for v in greek_limits.values()):
        mine = trade_exposure(option, quantity, transaction)
        checks["trade_greeks_known"] = mine is not None
        if mine is None:
            return checks, "Greeks of the proposed option are unavailable: the Greek exposure limits cannot be verified"
        checks["open_greeks_known"] = ex.get("unknown_positions", 0) == 0
        if not checks["open_greeks_known"]:
            return checks, (f"{ex['unknown_positions']} open option position(s) have no stored Greeks: the portfolio Greek exposure "
                            "cannot be verified")
        for name, key, label in (("net_delta", "delta_notional", "Net delta"), ("gamma", "gamma_1pct", "Gamma"),
                                 ("vega", "vega", "Vega"), ("theta", "theta", "Theta")):
            limit = greek_limits[name]
            if limit <= 0:
                continue
            total = ex.get(key, 0.0) + mine[key]
            checks[f"{name}_within_limit"] = pct(total) <= limit
            if not checks[f"{name}_within_limit"]:
                return checks, f"{label} exposure would be {pct(total):.2f}% of equity (max {limit:g}%)"
    return checks, None
