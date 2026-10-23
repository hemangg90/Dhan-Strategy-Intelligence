"""
OptionsContext: one expiry's option chain at one moment, as a typed, validated, reproducible object.

Built from a `ChainSnapshot` (the broker's live chain). Everything the broker did not provide is reported as
unavailable (in `status.unavailable` and on the sub-objects), never filled. `info_cutoff` is the snapshot time: IV rank
and percentile use only sessions before the snapshot's date, and nothing newer than the snapshot is read.

Blocks
  chain / quotes      the normalised contracts (`options/contracts.py`)
  atm                 ATM strike, ATM implied volatility (percent), ATM call and put Greeks computed by the application
  iv                  rank, percentile and changes from the recorded IV history (`options/iv_history.py`)
  iv_vs_realized      ATM IV against the underlying's realized volatility (a comparison, not a signal)
  skew / term         strike-wise IV around the money; expiry-wise ATM IV when other expiries were supplied
  oi                  open interest, change in OI, PCR, concentration (`options/oi_features.py`) - context only
  liquidity           spread and size near the money (`options/execution_quality.py`)
  expiry              trading time left and the risk band (`options/expiry.py`)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.intelligence.market_context import _json_safe
from quant_intelligence.intelligence.status import DataStatus, critical, good, warning, worst
from quant_intelligence.options.chain_analytics import ChainSnapshot
from quant_intelligence.options.contracts import OptionQuote, quotes_from_chain, validate_chain
from quant_intelligence.options.execution_quality import ExecutionLimits, assess_execution_quality
from quant_intelligence.options.expiry import ExpiryInfo, expiry_info
from quant_intelligence.options.greeks import GreekSet, compute_greeks
from quant_intelligence.options.iv_history import IvStats, iv_stats
from quant_intelligence.options.oi_features import OiFeatures, oi_features


@dataclass(frozen=True)
class OptionsContext:
    timestamp: dt.datetime  # the snapshot time
    underlying: str
    expiry: str
    spot: float
    info_cutoff: dt.datetime
    chain: ChainSnapshot
    quotes: tuple[OptionQuote, ...]
    atm_strike: float | None
    atm_iv: float | None  # percent
    atm_greeks: dict  # {"CE": GreekSet, "PE": GreekSet}
    iv: IvStats
    iv_vs_realized: dict
    skew: dict
    term_structure: tuple[tuple[str, float], ...]
    oi: OiFeatures
    liquidity: dict
    expiry_info: ExpiryInfo
    status: DataStatus
    chain_hash: str = ""
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.underlying:
            raise ValueError("an OptionsContext needs an underlying")

    def quote(self, strike: float, option_type: str) -> OptionQuote | None:
        for q in self.quotes:
            if q.strike == strike and q.option_type == option_type:
                return q
        return None

    def as_record(self) -> dict:
        import dataclasses

        return _json_safe({
            "timestamp": self.timestamp, "underlying": self.underlying, "expiry": self.expiry, "spot": self.spot,
            "info_cutoff": self.info_cutoff, "atm_strike": self.atm_strike, "atm_iv": self.atm_iv, "chain_hash": self.chain_hash,
            "atm_greeks": {k: dataclasses.asdict(v) for k, v in self.atm_greeks.items()},
            "iv": dataclasses.asdict(self.iv), "iv_vs_realized": self.iv_vs_realized, "skew": self.skew,
            "term_structure": [list(t) for t in self.term_structure], "oi": dataclasses.asdict(self.oi),
            "liquidity": self.liquidity, "expiry_info": dataclasses.asdict(self.expiry_info), "status": self.status.to_dict(),
            "parameters": self.parameters, "n_contracts": len(self.quotes),
        })


def _hash_chain(chain: ChainSnapshot) -> str:
    h = hashlib.sha1()
    h.update(f"{chain.underlying}|{chain.expiry}|{chain.spot_price}".encode())
    for r in chain.strikes:
        h.update(repr((r.strike, r.ce_ltp, r.ce_oi, r.ce_iv, r.ce_bid, r.ce_ask, r.pe_ltp, r.pe_oi, r.pe_iv, r.pe_bid, r.pe_ask)).encode())
    return h.hexdigest()[:16]


def _skew_block(chain: ChainSnapshot, n: int = 5) -> dict:
    atm = chain.atm_strike
    rows = sorted(chain.strikes, key=lambda r: r.strike)
    if atm is None or not rows:
        return {"available": False, "reason": "no strikes", "points": []}
    idx = min(range(len(rows)), key=lambda i: abs(rows[i].strike - atm))
    near = rows[max(0, idx - n): idx + n + 1]
    points = [{"strike": r.strike, "moneyness_pct": (r.strike / chain.spot_price - 1) * 100.0, "ce_iv": r.ce_iv, "pe_iv": r.pe_iv}
              for r in near]
    return {"available": any(p["ce_iv"] or p["pe_iv"] for p in points), "reason": "", "points": points,
            "put_minus_call_iv_one_step": chain.iv_skew}


def _liquidity_block(quotes: list[OptionQuote], atm: float | None, now: dt.datetime, limits: ExecutionLimits, window: int = 5) -> dict:
    if atm is None:
        return {"available": False, "reason": "no ATM strike"}
    strikes = sorted({q.strike for q in quotes})
    idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - atm))
    near = set(strikes[max(0, idx - window): idx + window + 1])
    scored = [(q, assess_execution_quality(q, now, "BUY", None, limits)) for q in quotes if q.strike in near]
    spreads = [a.rel_spread_pct for _, a in scored if a.rel_spread_pct is not None]
    eligible = sum(1 for _, a in scored if a.eligible)
    return {"available": bool(scored), "contracts_considered": len(scored), "eligible": eligible,
            "eligible_share": eligible / len(scored) if scored else None,
            "median_spread_pct": float(np.median(spreads)) if spreads else None,
            "max_spread_pct": float(max(spreads)) if spreads else None,
            "limits": {"max_spread_pct": limits.max_spread_pct, "max_quote_age_sec": limits.max_quote_age_sec}}


def build_options_context(
    chain: ChainSnapshot,
    *,
    now: dt.datetime | None = None,
    realized_vol: float | None = None,
    atr: float | None = None,
    expected_lot_size: int | None = None,
    expected_strike_step: float | None = None,
    listed_expiries: list[str] | None = None,
    other_expiries: dict[str, ChainSnapshot] | None = None,
    limits: ExecutionLimits | None = None,
    oi_window: int = 5,
) -> OptionsContext:
    """Assemble the context for `chain`. When the chain fails validation the context is returned with a CRITICAL status and
    empty analytics - callers must check `status.ok`."""
    now = now or chain.fetched_at or dt.datetime.now()
    limits = limits or ExecutionLimits.from_settings()
    params = {"oi_window": oi_window, "max_spread_pct": limits.max_spread_pct, "max_quote_age_sec": limits.max_quote_age_sec,
              "iv_min_observations": 30, "risk_free_rate": SETTINGS.risk_free_rate, "dividend_yield": SETTINGS.dividend_yield}
    check = validate_chain(chain, now=now, max_age_sec=limits.max_quote_age_sec, expected_lot_size=expected_lot_size,
                           expected_strike_step=expected_strike_step)
    ts = chain.fetched_at or now
    empty_oi = oi_features(ChainSnapshot(chain.underlying, chain.expiry, chain.spot_price, []))
    empty_greeks = {"CE": GreekSet(False, "chain failed validation"), "PE": GreekSet(False, "chain failed validation")}
    if not check.ok:
        return OptionsContext(ts, chain.underlying, chain.expiry, chain.spot_price, ts, chain, (), None, None, empty_greeks,
                              IvStats(False, "chain failed validation"), {"available": False}, {"available": False}, (), empty_oi,
                              {"available": False}, ExpiryInfo(False, "chain failed validation"), check, "", params)

    quotes = quotes_from_chain(chain)
    atm = chain.atm_strike
    atm_iv = chain.atm_iv
    info = expiry_info(chain.underlying, chain.expiry, now, listed_expiries)
    atm_greeks = {}
    for kind in ("CE", "PE"):
        q = next((x for x in quotes if x.strike == atm and x.option_type == kind), None)
        atm_greeks[kind] = compute_greeks(q, chain.spot_price, now) if q else GreekSet(False, "no ATM contract")
    stats = iv_stats(chain.underlying, atm_iv, now, info.trading_days if info.available else None)

    iv_rv = {"available": False, "reason": "realized volatility not supplied"}
    if atm_iv and realized_vol is not None and math.isfinite(realized_vol) and realized_vol > 0:
        iv_dec = atm_iv / 100.0
        iv_rv = {"available": True, "atm_iv": iv_dec, "realized_vol": realized_vol, "iv_minus_rv": iv_dec - realized_vol,
                 "iv_over_rv": iv_dec / realized_vol,
                 "note": "a comparison of two different measures (forward-looking IV vs trailing realized); not a trading signal"}
    term = tuple(sorted((e, s.atm_iv) for e, s in (other_expiries or {}).items() if s.atm_iv))
    if atm_iv:
        term = tuple(sorted(term + ((chain.expiry, atm_iv),)))

    statuses = [check]
    if not info.available:
        statuses.append(critical("expiry", info.reason))
    elif "NEAR_EXPIRY" in info.flags:
        statuses.append(warning("expiry", f"{info.risk_band.lower()} gamma/theta: {info.trading_days:.2f} trading days left"))
    if atm_iv is None:
        statuses.append(warning("implied_volatility", "the chain reports no ATM implied volatility", ("atm_iv",)))
    if not stats.available:
        statuses.append(warning("iv_history", stats.reason, ("iv_rank", "iv_percentile")))
    status = worst("options_context", statuses) if len(statuses) > 1 else statuses[0]

    return OptionsContext(
        timestamp=ts, underlying=chain.underlying, expiry=chain.expiry, spot=chain.spot_price, info_cutoff=ts, chain=chain,
        quotes=tuple(quotes), atm_strike=atm, atm_iv=atm_iv, atm_greeks=atm_greeks, iv=stats, iv_vs_realized=iv_rv,
        skew=_skew_block(chain), term_structure=term, oi=oi_features(chain, window=oi_window, atr=atr),
        liquidity=_liquidity_block(quotes, atm, now, limits), expiry_info=info, status=status, chain_hash=_hash_chain(chain),
        parameters=params,
    )
