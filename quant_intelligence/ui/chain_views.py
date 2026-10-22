"""Option-chain views for the trading UI (pure functions over a `ChainSnapshot`) and the order-ticket
arithmetic: what a trade costs, risks and could earn after charges."""
from __future__ import annotations

import pandas as pd

from quant_intelligence.execution.charges import option_trade_charges


def ladder_frame(chain, window: int = 10) -> pd.DataFrame:
    """Strikes around the money laid out like a broker's chain: calls on the left, strike in the middle,
    puts on the right. `window` strikes either side of ATM (all strikes if window <= 0)."""
    rows = sorted(chain.strikes, key=lambda r: r.strike)
    if not rows:
        return pd.DataFrame()
    atm = min(rows, key=lambda r: abs(r.strike - chain.spot_price)).strike
    if window > 0:
        i = next(k for k, r in enumerate(rows) if r.strike == atm)
        rows = rows[max(0, i - window): i + window + 1]
    return pd.DataFrame(
        [
            {
                "ce_oi": r.ce_oi, "ce_oi_change": r.ce_oi_change, "ce_iv": r.ce_iv, "ce_ltp": r.ce_ltp,
                "strike": r.strike,
                "pe_ltp": r.pe_ltp, "pe_iv": r.pe_iv, "pe_oi_change": r.pe_oi_change, "pe_oi": r.pe_oi,
                "is_atm": r.strike == atm,
            }
            for r in rows
        ]
    )


def max_pain(chain) -> float | None:
    """Strike at which option buyers' total payoff at expiry is smallest (writers' pain is least)."""
    rows = [r for r in chain.strikes if (r.ce_oi or r.pe_oi)]
    if not rows:
        return None
    best, best_pain = None, None
    for candidate in rows:
        p = candidate.strike
        pain = sum((r.ce_oi or 0.0) * max(0.0, p - r.strike) + (r.pe_oi or 0.0) * max(0.0, r.strike - p) for r in rows)
        if best_pain is None or pain < best_pain:
            best, best_pain = p, pain
    return best


def chain_stats(chain) -> dict:
    rows = chain.strikes
    ce_total = sum(r.ce_oi or 0.0 for r in rows)
    pe_total = sum(r.pe_oi or 0.0 for r in rows)
    top_call = max(rows, key=lambda r: r.ce_oi or 0.0, default=None)
    top_put = max(rows, key=lambda r: r.pe_oi or 0.0, default=None)
    return {
        "spot": chain.spot_price,
        "atm": chain.atm_strike,
        "expiry": chain.expiry,
        "pcr": chain.total_pcr,
        "atm_iv": chain.atm_iv,
        "max_pain": max_pain(chain),
        "call_oi": ce_total,
        "put_oi": pe_total,
        "resistance": top_call.strike if top_call and (top_call.ce_oi or 0) > 0 else None,  # heaviest call OI
        "support": top_put.strike if top_put and (top_put.pe_oi or 0) > 0 else None,  # heaviest put OI
    }


def ticket_math(
    entry: float, stop: float, target: float, quantity: int, side: str = "BUY", market: str = "NSE",
    available: float | None = None,
) -> dict:
    """Cost, risk and reward of an option order before it is placed. All amounts in rupees.

    BUY: `amount_needed` is the premium paid. SELL: premium is received, so it is shown as the premium
    value only (exchange margin is not modelled). Charges are estimated for exiting at the stop and at the
    target; `net_loss` / `net_profit` include them.
    """
    max_loss = abs(entry - stop) * quantity
    max_profit = abs(target - entry) * quantity
    at_stop = option_trade_charges(entry, stop, quantity, side, market).total
    at_target = option_trade_charges(entry, target, quantity, side, market).total
    amount = entry * quantity
    funds_ok = True if side == "SELL" or available is None else amount <= available
    return {
        "amount_needed": amount,
        "max_loss": max_loss,
        "max_profit": max_profit,
        "risk_reward": (max_profit / max_loss) if max_loss else None,
        "charges_at_stop": at_stop,
        "charges_at_target": at_target,
        "net_loss": max_loss + at_stop,
        "net_profit": max_profit - at_target,
        "funds_ok": funds_ok,
        "utilisation_pct": (amount / available * 100.0) if available and side == "BUY" else None,
    }


def ranked_frame(ranked) -> pd.DataFrame:
    """The strike ranking (options/strike_ranker.RankedStrike list, best first) as a table: score, the four score
    parts as 0-100, the raw numbers behind them, and why a strike was rejected or what data it lacked."""
    return pd.DataFrame(
        [
            {
                "rank": i + 1 if r.eligible else None,
                "strike": r.strike, "type": r.option_type, "score": round(r.score, 1) if r.eligible else None,
                "premium": r.premium, "delta": r.delta, "iv": r.iv, "spread_%": r.spread_pct, "theta_%/day": r.theta_pct,
                "oi": r.oi, "volume": r.volume,
                **{f"{k}_fit": round(v * 100) for k, v in r.parts.items()},
                "notes": "; ".join(r.notes),
            }
            for i, r in enumerate(ranked)
        ]
    )
