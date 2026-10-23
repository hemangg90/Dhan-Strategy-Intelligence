"""
Open-interest and positioning features of an option chain.

These are CONTEXT, not conclusions. Large call open interest at a strike is not "resistance" and large put open
interest is not "support": open interest says contracts exist, not who is long or short, nor whether they are hedged.
The peaks are reported by name (`call_oi_peak`, `put_oi_peak`) and nothing here turns them into a direction.

Definitions
  PCR_OI      total put OI / total call OI                    (None when call OI is zero or missing)
  PCR_VOLUME  total put volume / total call volume             (None when call volume is zero or missing)
  window PCR  PCR_OI over the strikes within `window` steps of ATM (a configurable strike universe)
  concentration  share of a side's OI held by its `top_n` strikes; HHI = sum of squared strike shares (1 = one strike)
  oi_to_volume   day volume / OI for a side: how much of the open interest traded today (turnover of the position)
  concentration change  today's top-n share minus the share implied by previous OI (OI - change); needs previous OI
A field that cannot be computed is None and listed in `unavailable` with the reason in `notes`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow


@dataclass(frozen=True)
class OiFeatures:
    total_call_oi: float | None
    total_put_oi: float | None
    call_oi_change: float | None
    put_oi_change: float | None
    oi_change_coverage: float  # share of contracts that reported a change in OI (0-1)
    pcr_oi: float | None
    pcr_volume: float | None
    pcr_oi_window: float | None
    window_strikes: int
    call_oi_peak: dict | None  # {"strike", "oi", "share", "distance_pct", "distance_atr"}
    put_oi_peak: dict | None
    call_concentration: float | None  # top-n share
    put_concentration: float | None
    call_hhi: float | None
    put_hhi: float | None
    call_concentration_change: float | None
    put_concentration_change: float | None
    call_oi_to_volume: float | None  # day volume / OI
    put_oi_to_volume: float | None
    unavailable: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


def _vals(rows: list[StrikeRow], attr: str) -> list[float | None]:
    return [getattr(r, attr) for r in rows]


def _sum(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(sum(clean)) if clean else None


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return num / den


def _top_share(by_strike: dict[float, float], top_n: int) -> float | None:
    total = sum(by_strike.values())
    if total <= 0:
        return None
    return sum(sorted(by_strike.values(), reverse=True)[:top_n]) / total


def _hhi(by_strike: dict[float, float]) -> float | None:
    total = sum(by_strike.values())
    if total <= 0:
        return None
    return sum((v / total) ** 2 for v in by_strike.values())


def _peak(by_strike: dict[float, float], spot: float, atr: float | None) -> dict | None:
    total = sum(by_strike.values())
    if total <= 0:
        return None
    strike = max(by_strike, key=by_strike.get)
    dist = strike - spot
    return {"strike": strike, "oi": by_strike[strike], "share": by_strike[strike] / total,
            "distance_pct": dist / spot * 100.0, "distance_atr": (dist / atr) if atr and atr > 0 else None}


def oi_features(chain: ChainSnapshot, *, window: int = 5, top_n: int = 3, atr: float | None = None) -> OiFeatures:
    rows = sorted(chain.strikes, key=lambda r: r.strike)
    unavailable: list[str] = []
    notes: list[str] = []
    if not rows:
        return OiFeatures(None, None, None, None, 0.0, None, None, None, window, None, None, None, None, None, None, None, None,
                          None, None, ("all",), ("the chain has no strikes",))

    call_oi = _sum(_vals(rows, "ce_oi"))
    put_oi = _sum(_vals(rows, "pe_oi"))
    call_vol = _sum(_vals(rows, "ce_volume"))
    put_vol = _sum(_vals(rows, "pe_volume"))
    call_chg = _sum(_vals(rows, "ce_oi_change"))
    put_chg = _sum(_vals(rows, "pe_oi_change"))
    n_legs = 2 * len(rows)
    reported = sum(1 for r in rows for v in (r.ce_oi_change, r.pe_oi_change) if v is not None)
    coverage = reported / n_legs

    pcr_oi = _ratio(put_oi, call_oi)
    if pcr_oi is None:
        unavailable.append("pcr_oi")
        notes.append("PCR (OI) is undefined: call open interest is zero or missing")
    pcr_vol = _ratio(put_vol, call_vol)
    if pcr_vol is None:
        unavailable.append("pcr_volume")
        notes.append("PCR (volume) is undefined: call volume is zero or missing")

    atm = chain.atm_strike
    pcr_window = None
    if atm is not None:
        idx = min(range(len(rows)), key=lambda i: abs(rows[i].strike - atm))
        near = rows[max(0, idx - window): idx + window + 1]
        pcr_window = _ratio(_sum(_vals(near, "pe_oi")), _sum(_vals(near, "ce_oi")))
    if pcr_window is None:
        unavailable.append("pcr_oi_window")

    call_map = {r.strike: float(r.ce_oi) for r in rows if r.ce_oi}
    put_map = {r.strike: float(r.pe_oi) for r in rows if r.pe_oi}
    prev_call = {r.strike: float(r.ce_oi - r.ce_oi_change) for r in rows if r.ce_oi is not None and r.ce_oi_change is not None
                 and r.ce_oi - r.ce_oi_change > 0}
    prev_put = {r.strike: float(r.pe_oi - r.pe_oi_change) for r in rows if r.pe_oi is not None and r.pe_oi_change is not None
                and r.pe_oi - r.pe_oi_change > 0}

    def change(today: dict, prev: dict) -> float | None:
        a, b = _top_share(today, top_n), _top_share(prev, top_n)
        # only comparable when the previous book covers (nearly) the same strikes as today's
        if a is None or b is None or len(prev) < 0.8 * len(today):
            return None
        return a - b

    call_conc_chg, put_conc_chg = change(call_map, prev_call), change(put_map, prev_put)
    if call_conc_chg is None or put_conc_chg is None:
        unavailable.append("concentration_change")
        notes.append("the change in OI concentration needs previous open interest for the same strikes")

    c_oi_vol, p_oi_vol = _ratio(call_vol, call_oi), _ratio(put_vol, put_oi)
    if c_oi_vol is None or p_oi_vol is None:
        unavailable.append("oi_to_volume")
    if call_chg is None or put_chg is None or coverage < 0.8:
        unavailable.append("oi_change_totals")
        notes.append(f"only {coverage:.0%} of contracts reported a change in OI; totals of the change are incomplete")

    return OiFeatures(
        total_call_oi=call_oi, total_put_oi=put_oi, call_oi_change=call_chg, put_oi_change=put_chg, oi_change_coverage=coverage,
        pcr_oi=pcr_oi, pcr_volume=pcr_vol, pcr_oi_window=pcr_window, window_strikes=window,
        call_oi_peak=_peak(call_map, chain.spot_price, atr), put_oi_peak=_peak(put_map, chain.spot_price, atr),
        call_concentration=_top_share(call_map, top_n), put_concentration=_top_share(put_map, top_n),
        call_hhi=_hhi(call_map), put_hhi=_hhi(put_map),
        call_concentration_change=call_conc_chg, put_concentration_change=put_conc_chg,
        call_oi_to_volume=c_oi_vol, put_oi_to_volume=p_oi_vol, unavailable=tuple(dict.fromkeys(unavailable)), notes=tuple(notes),
    )
