"""
Normalised option contracts and chain-snapshot validation.

`quotes_from_chain` flattens a `ChainSnapshot` (one row per strike) into one `OptionQuote` per contract with a uniform
schema. A field the broker did not provide is None and is named in `unavailable`; it is never filled with a default.
A quote of 0 is data (Dhan reports 0 when nothing is quoted), and the liquidity layer treats a zero bid as no bid.

`validate_chain` is the gate before any analytics use a snapshot. It returns a `DataStatus`:
  CRITICAL  no strikes, bad spot, bad or past expiry, duplicated strikes, lot size or strike step that disagrees with
            the instrument master, or a snapshot older than the limit
  WARNING   gaps in the strike ladder (an incomplete chain), a large share of contracts with no quote, or no snapshot
            time recorded (age cannot be checked)
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from quant_intelligence.intelligence.status import DataStatus, critical, good, warning, worst
from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow

_FIELDS = ("bid", "ask", "bid_qty", "ask_qty", "ltp", "volume", "oi", "oi_change", "iv", "delta", "gamma", "theta", "vega", "prev_close")


@dataclass(frozen=True)
class OptionQuote:
    underlying: str
    exchange_segment: str | None
    security_id: str | None
    option_type: str  # CE / PE
    strike: float
    expiry: str  # YYYY-MM-DD
    lot_size: int | None
    bid: float | None
    ask: float | None
    bid_qty: float | None
    ask_qty: float | None
    ltp: float | None
    volume: float | None
    oi: float | None
    oi_change: float | None
    iv: float | None  # percent, as the broker reports it
    delta: float | None  # provider Greeks, in the provider's units
    gamma: float | None
    theta: float | None
    vega: float | None
    prev_close: float | None
    quote_time: dt.datetime | None  # the SNAPSHOT time; the chain has no per-contract quote time
    source: str
    unavailable: tuple[str, ...]

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None or self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            return None
        return (self.bid + self.ask) / 2.0

    @property
    def has_two_sided_quote(self) -> bool:
        return self.mid is not None


def _leg(chain: ChainSnapshot, row: StrikeRow, kind: str) -> OptionQuote:
    p = "ce_" if kind == "CE" else "pe_"
    g = lambda name: getattr(row, p + name)  # noqa: E731
    values = {"bid": g("bid"), "ask": g("ask"), "bid_qty": g("bid_qty"), "ask_qty": g("ask_qty"), "ltp": g("ltp"),
              "volume": g("volume"), "oi": g("oi"), "oi_change": g("oi_change"), "iv": g("iv"), "delta": g("delta"),
              "gamma": g("gamma"), "theta": g("theta"), "vega": g("vega"), "prev_close": g("prev_close")}
    return OptionQuote(
        underlying=chain.underlying, exchange_segment=chain.exchange_segment, security_id=g("security_id"), option_type=kind,
        strike=row.strike, expiry=chain.expiry, lot_size=chain.lot_size, quote_time=chain.fetched_at, source=chain.source,
        unavailable=tuple(k for k in _FIELDS if values[k] is None), **values,
    )


def quotes_from_chain(chain: ChainSnapshot) -> list[OptionQuote]:
    out: list[OptionQuote] = []
    for row in chain.strikes:
        out.append(_leg(chain, row, "CE"))
        out.append(_leg(chain, row, "PE"))
    return out


def _parse_date(text: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(text)[:10])
    except ValueError:
        return None


def _step(strikes: list[float]) -> float | None:
    diffs = sorted({round(b - a, 6) for a, b in zip(strikes, strikes[1:]) if b > a})
    if not diffs:
        return None
    counts = {d: sum(1 for a, b in zip(strikes, strikes[1:]) if round(b - a, 6) == d) for d in diffs}
    return max(counts, key=lambda d: (counts[d], -d))


def validate_chain(
    chain: ChainSnapshot,
    *,
    now: dt.datetime | None = None,
    max_age_sec: float | None = 120.0,
    expected_lot_size: int | None = None,
    expected_strike_step: float | None = None,
    min_quoted_share: float = 0.5,
) -> DataStatus:
    comp = "option_chain"
    if chain is None or not getattr(chain, "strikes", None):
        return critical(comp, "the chain has no strikes")
    if not (isinstance(chain.spot_price, (int, float)) and chain.spot_price > 0):
        return critical(comp, "the underlying price in the chain is missing or not positive")
    expiry = _parse_date(chain.expiry)
    if expiry is None:
        return critical(comp, f"the expiry {chain.expiry!r} is not a valid date")
    today = (now or dt.datetime.now()).date()
    if expiry < today:
        return critical(comp, f"the expiry {expiry} has already passed")
    strikes = [r.strike for r in chain.strikes]
    if len(set(strikes)) != len(strikes):
        return critical(comp, "duplicate strikes in the chain")
    if any(not (isinstance(s, (int, float)) and s > 0) for s in strikes):
        return critical(comp, "a strike is not a positive number")
    step = _step(sorted(strikes))
    if expected_strike_step is not None and step is not None and abs(step - expected_strike_step) > 1e-6:
        return critical(comp, f"the chain's strike step {step:g} differs from the instrument master's {expected_strike_step:g}")
    if expected_lot_size is not None:
        if chain.lot_size is None:
            return critical(comp, "the chain carries no lot size to check against the instrument master", ("lot_size",))
        if int(chain.lot_size) != int(expected_lot_size):
            return critical(comp, f"lot size {chain.lot_size} differs from the instrument master's {expected_lot_size}")

    statuses = [good(comp)]
    if max_age_sec is not None:
        if chain.fetched_at is None:
            statuses.append(warning(comp, "no snapshot time recorded, so the age of the quotes cannot be checked", ("quote_time",)))
        else:
            age = ((now or dt.datetime.now()) - chain.fetched_at).total_seconds()
            if age < -5:
                return critical(comp, f"the snapshot time is {-age:.0f}s in the future")
            if age > max_age_sec:
                return critical(comp, f"the snapshot is {age:.0f}s old (limit {max_age_sec:g}s)")
    ordered = sorted(strikes)
    if step:
        gaps = [(a, b) for a, b in zip(ordered, ordered[1:]) if b - a > 1.5 * step]
        if gaps:
            statuses.append(warning(comp, f"the strike ladder has {len(gaps)} gap(s), e.g. {gaps[0][0]:g} to {gaps[0][1]:g} (incomplete chain)"))
    quotes = quotes_from_chain(chain)
    quoted = sum(1 for q in quotes if q.has_two_sided_quote)
    if quoted / len(quotes) < min_quoted_share:
        statuses.append(warning(comp, f"only {quoted} of {len(quotes)} contracts have a two-sided quote"))
    return statuses[0] if len(statuses) == 1 else worst(comp, statuses)
