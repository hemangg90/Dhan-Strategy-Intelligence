"""
Best-strike ranking.

Pure logic, no network calls - scores every strike of an already-fetched ChainSnapshot for one side (CE for a
LONG setup, PE for a SHORT one) and returns them best first. `select_best_contract` turns the top eligible
strike into an OptionContract, so it is a drop-in for `option_selector.select_contract` (switched on with
OPTION_SELECTION=BEST).

A strike is first FILTERED (it must be tradeable at all), then SCORED 0-100 from four parts:

  delta fit   how close |delta| is to the profile's target (a bought option wants some sensitivity to the move
              without paying for deep in-the-money; a written one wants distance from spot)
  spread      bid-ask spread as % of the mid price: what the round trip gives away before the trade can work
  liquidity   open interest and volume relative to the most active strike on the same side (log scale)
  decay       |theta| / premium per day: a bought option should lose little of its price to time, a written one
              should collect a lot of it

The weights and targets are heuristics, not a fitted model: nothing here has been shown to improve results over
ATM (see docs/SELECTOR_EVALUATION.md for how the strategy selector itself fared). Every score part is returned,
so the Option Chain page can show why a strike ranked where it did. Missing data is never guessed: a part with
no input scores a neutral 0.5 and the strike is flagged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow
from quant_intelligence.options.option_selector import OptionContract, OptionSelectionError, build_contract


@dataclass(frozen=True)
class Profile:
    delta_target: float
    delta_band: tuple[float, float]  # |delta| outside this is not eligible
    max_spread_pct: float            # spread above this is not eligible
    spread_good_pct: float           # a spread this tight scores full marks
    decay_scale: float               # |theta|/premium per day at which the decay part reaches 0 (BUY) or 1 (SELL)
    weights: dict[str, float]
    sell: bool = False


BUY_PROFILE = Profile(
    delta_target=0.45, delta_band=(0.30, 0.60), max_spread_pct=10.0, spread_good_pct=0.5, decay_scale=0.15,
    weights={"delta": 0.35, "spread": 0.25, "liquidity": 0.25, "decay": 0.15},
)
SELL_PROFILE = Profile(
    delta_target=0.20, delta_band=(0.10, 0.35), max_spread_pct=10.0, spread_good_pct=0.5, decay_scale=0.15,
    weights={"delta": 0.35, "spread": 0.25, "liquidity": 0.25, "decay": 0.15}, sell=True,
)
NEUTRAL = 0.5


@dataclass
class RankedStrike:
    strike: float
    option_type: str
    security_id: str | None
    premium: float | None
    delta: float | None
    iv: float | None
    oi: float | None
    volume: float | None
    spread_pct: float | None
    theta_pct: float | None  # |theta| as % of premium, per day
    score: float             # 0-100; 0 for a strike that is not eligible
    parts: dict[str, float] = field(default_factory=dict)  # each part 0-1
    eligible: bool = True
    notes: list[str] = field(default_factory=list)  # why it was rejected, or what data was missing


def _side(row: StrikeRow, option_type: str) -> dict:
    p = "ce" if option_type == "CE" else "pe"
    return {"security_id": getattr(row, f"{p}_security_id"), "premium": getattr(row, f"{p}_ltp"),
            "delta": getattr(row, f"{p}_delta"), "iv": getattr(row, f"{p}_iv"), "oi": getattr(row, f"{p}_oi"),
            "volume": getattr(row, f"{p}_volume"), "bid": getattr(row, f"{p}_bid"), "ask": getattr(row, f"{p}_ask"),
            "theta": getattr(row, f"{p}_theta")}


def _number(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def _log_share(value: float | None, top: float) -> float | None:
    if value is None or top <= 0:
        return None
    return min(1.0, math.log1p(max(value, 0.0)) / math.log1p(top))


def rank_strikes(
    chain: ChainSnapshot, direction: str, transaction: str, lot_size: int = 1, max_outlay: float | None = None,
    profile: Profile | None = None,
) -> list[RankedStrike]:
    """Every strike of the side `direction` maps to (LONG -> CE, SHORT -> PE), eligible ones first, best first.

    max_outlay: the most one lot may cost (premium x lot_size, Rs); a dearer strike is not eligible. Used for BUY only
    (a written option's premium is received, not paid)."""
    if direction not in ("LONG", "SHORT"):
        raise OptionSelectionError(f"Unknown direction: {direction}")
    if transaction not in ("BUY", "SELL"):
        raise OptionSelectionError(f"Unknown transaction: {transaction}")
    if not chain.strikes:
        raise OptionSelectionError("Option chain has no strikes")
    profile = profile or (SELL_PROFILE if transaction == "SELL" else BUY_PROFILE)
    option_type = "CE" if direction == "LONG" else "PE"
    sides = [(row, _side(row, option_type)) for row in chain.strikes]
    top_oi = max((_number(s["oi"]) or 0.0 for _, s in sides), default=0.0)
    top_vol = max((_number(s["volume"]) or 0.0 for _, s in sides), default=0.0)
    lo, hi = profile.delta_band

    ranked: list[RankedStrike] = []
    for row, s in sides:
        premium, delta = _number(s["premium"]), _number(s["delta"])
        bid, ask, theta = _number(s["bid"]), _number(s["ask"]), _number(s["theta"])
        oi, volume = _number(s["oi"]), _number(s["volume"])
        spread_pct = None
        if bid is not None and ask is not None and ask >= bid > 0:
            spread_pct = (ask - bid) / ((ask + bid) / 2) * 100.0
        theta_pct = abs(theta) / premium * 100.0 if theta is not None and premium else None
        out = RankedStrike(strike=row.strike, option_type=option_type, security_id=s["security_id"], premium=premium,
                           delta=delta, iv=_number(s["iv"]), oi=oi, volume=volume, spread_pct=spread_pct,
                           theta_pct=theta_pct, score=0.0)
        ranked.append(out)

        if s["security_id"] is None or premium is None or premium <= 0:
            out.eligible, out.notes = False, ["no quote"]
            continue
        if not oi or not volume:
            out.notes.append("no open interest" if not oi else "no volume today")
            out.eligible = False
        if spread_pct is not None and spread_pct > profile.max_spread_pct:
            out.notes.append(f"spread {spread_pct:.1f}% too wide")
            out.eligible = False
        if delta is not None and not lo <= abs(delta) <= hi:
            out.notes.append(f"|delta| {abs(delta):.2f} outside {lo:.2f}-{hi:.2f}")
            out.eligible = False
        if transaction == "BUY" and max_outlay is not None and premium * lot_size > max_outlay:
            out.notes.append(f"one lot costs Rs {premium * lot_size:,.0f}, over the Rs {max_outlay:,.0f} cap")
            out.eligible = False

        half_width = max(profile.delta_target - lo, hi - profile.delta_target)
        delta_part = None if delta is None else max(0.0, 1.0 - abs(abs(delta) - profile.delta_target) / half_width)
        spread_part = None if spread_pct is None else max(
            0.0, min(1.0, 1.0 - (spread_pct - profile.spread_good_pct) / (profile.max_spread_pct - profile.spread_good_pct)))
        oi_share, vol_share = _log_share(oi, top_oi), _log_share(volume, top_vol)
        shares = [x for x in (oi_share, vol_share) if x is not None]
        liquidity_part = sum(shares) / len(shares) if shares else None
        decay_part = None
        if theta_pct is not None:
            burden = min(1.0, theta_pct / (profile.decay_scale * 100.0))
            decay_part = burden if profile.sell else 1.0 - burden

        parts = {}
        for name, value in (("delta", delta_part), ("spread", spread_part), ("liquidity", liquidity_part),
                            ("decay", decay_part)):
            if value is None:
                out.notes.append(f"no {name} data")
            parts[name] = NEUTRAL if value is None else value
        out.parts = parts
        if out.eligible:
            out.score = 100.0 * sum(profile.weights[k] * v for k, v in parts.items())

    ranked.sort(key=lambda r: (not r.eligible, -r.score, abs(r.strike - chain.spot_price)))
    return ranked


def select_best_contract(
    chain: ChainSnapshot, direction: str, transaction: str, lot_size: int = 1, max_outlay: float | None = None,
) -> OptionContract:
    """The top-ranked eligible strike as a tradeable contract; OptionSelectionError (with the reasons) if none is."""
    ranked = rank_strikes(chain, direction, transaction, lot_size, max_outlay)
    best = ranked[0]
    if not best.eligible:
        near = sorted(ranked, key=lambda r: abs(r.strike - chain.spot_price))[:3]
        why = "; ".join(f"{r.strike:g}: {', '.join(r.notes)}" for r in near)
        raise OptionSelectionError(f"No {best.option_type} strike passes the liquidity / spread / delta filters ({why})")
    row = next(r for r in chain.strikes if r.strike == best.strike)
    return build_contract(chain, row, best.option_type, transaction, lot_size)
