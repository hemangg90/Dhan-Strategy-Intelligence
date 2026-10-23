"""
Chain-scored option contract selection.

`option_selector.select_contract` takes a fixed strike (ATM, or N strikes out). This module instead looks at the
live option chain and picks the strike that is best to trade, then explains the choice.

Candidates are the strikes within `window` steps of ATM on the side the setup needs (calls for LONG, puts for
SHORT). A candidate is rejected outright when it is not tradeable:
  * no quote / no security id / premium below the minimum
  * no open interest, or open interest far below the chain median (thin)
  * no volume today (when volume is reported)
  * no bid, or a bid/ask spread wider than the limit (when bid/ask are reported)
  * delta outside the band for the trade (BUY 0.25-0.75, SELL 0.10-0.45) when delta is reported
Those that survive get a 0-100 score from six parts:
  liquidity (OI and volume rank within the chain)            30%
  spread (tighter is better)                                 20%
  delta fit (BUY: near 0.50, SELL: near 0.25)                20%
  OI build-up of the contract itself (see `classify_buildup`) 15%
  OI peak position (contextual only)                          5%
  IV value (BUY: cheaper than ATM, SELL: richer than ATM)     10%
The OI peak part is small on purpose: heavy open interest at a strike is context, not proof of support or resistance.
If no candidate survives, selection fails with "NO SUITABLE CONTRACT" and the reason counts, and nothing is traded (fail closed).

Staged selection
  1. Eligibility    structural filters above, plus - when `OPTION_REQUIRE_QUOTE` is on (the default) - the execution-quality
                    gate (`options/execution_quality.py`): a fresh two-sided quote, spread, quoted size, volume, open interest.
  2. Quality score  the 0-100 score above (a description of the contract's liquidity, cost and positioning).
  3. Scenarios      when scenario inputs are given, each survivor is re-priced across spot / volatility / time scenarios after
                    realistic fills and costs (`options/scenario.py`).
  4. Ranking        by the scenario objective (lower-bound expected net return on premium less a tail-loss term) when every
                    survivor has one; otherwise by the quality score. The basis used is reported.
The contract's `premium` is the executable price (the ask for a buy, the bid for a sell) when the quote is two-sided; the last
traded price is kept in `last_price`.

Pure logic on an already-fetched ChainSnapshot, no network calls.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from statistics import median

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow
from quant_intelligence.options.contracts import OptionQuote, quotes_from_chain
from quant_intelligence.options.execution_quality import ExecutionLimits, ExecutionQuality, assess_execution_quality
from quant_intelligence.options.greeks import GreekSet, compute_greeks
from quant_intelligence.options.option_selector import OptionContract, OptionSelectionError
from quant_intelligence.options.scenario import ScenarioInputs, ScenarioResult, evaluate_scenarios
from quant_intelligence.utils.timeutil import now_ist

WEIGHTS = {"liquidity": 0.30, "spread": 0.20, "delta": 0.20, "buildup": 0.15, "oi_peak": 0.05, "iv": 0.10}

LONG_BUILDUP = "LONG_BUILDUP"  # price up, open interest up: new buyers
SHORT_BUILDUP = "SHORT_BUILDUP"  # price down, open interest up: new writers
SHORT_COVERING = "SHORT_COVERING"  # price up, open interest down: writers buying back
LONG_UNWINDING = "LONG_UNWINDING"  # price down, open interest down: buyers leaving
NEUTRAL = "NEUTRAL"
UNKNOWN = "UNKNOWN"

MIN_OI_MOVE = 0.01  # an OI change below 1% of the prior OI is treated as no change

# How good each reading is for the side taking the trade (the option's own price and OI against yesterday).
_BUY_BUILDUP = {LONG_BUILDUP: 1.0, SHORT_COVERING: 0.75, NEUTRAL: 0.45, UNKNOWN: 0.4, LONG_UNWINDING: 0.15, SHORT_BUILDUP: 0.0}
_SELL_BUILDUP = {SHORT_BUILDUP: 1.0, LONG_UNWINDING: 0.75, NEUTRAL: 0.45, UNKNOWN: 0.4, SHORT_COVERING: 0.15, LONG_BUILDUP: 0.0}


def classify_buildup(ltp: float | None, prev_close: float | None, oi: float | None, oi_change: float | None) -> str:
    """The classic F&O reading of one option: what its price did against the prior close, and what its OI did."""
    if ltp is None or prev_close is None or oi is None or oi_change is None:
        return UNKNOWN
    prior_oi = oi - oi_change
    if prior_oi <= 0:
        return UNKNOWN
    oi_move = oi_change / prior_oi
    price_move = ltp - prev_close
    if abs(oi_move) < MIN_OI_MOVE or price_move == 0:
        return NEUTRAL
    if oi_move > 0:
        return LONG_BUILDUP if price_move > 0 else SHORT_BUILDUP
    return SHORT_COVERING if price_move > 0 else LONG_UNWINDING


@dataclass
class _Leg:
    """One side (CE or PE) of a strike row, flattened."""

    strike: float
    ltp: float | None
    oi: float | None
    oi_change: float | None
    volume: float | None
    iv: float | None
    delta: float | None
    security_id: str | None
    bid: float | None
    ask: float | None
    prev_close: float | None


def _leg(row: StrikeRow, option_type: str) -> _Leg:
    if option_type == "CE":
        return _Leg(row.strike, row.ce_ltp, row.ce_oi, row.ce_oi_change, row.ce_volume, row.ce_iv, row.ce_delta,
                    row.ce_security_id, row.ce_bid, row.ce_ask, row.ce_prev_close)
    return _Leg(row.strike, row.pe_ltp, row.pe_oi, row.pe_oi_change, row.pe_volume, row.pe_iv, row.pe_delta,
                row.pe_security_id, row.pe_bid, row.pe_ask, row.pe_prev_close)


@dataclass
class ScoredStrike:
    strike: float
    score: float  # 0-100
    parts: dict[str, float]  # each part 0-1
    buildup: str
    spread_pct: float | None
    delta: float | None
    oi: float | None
    volume: float | None
    quote: OptionQuote | None = None
    execution: ExecutionQuality | None = None
    greeks: GreekSet | None = None
    scenario: ScenarioResult | None = None

    @property
    def objective(self) -> float | None:
        return self.scenario.objective if self.scenario is not None and self.scenario.available else None


@dataclass
class ContractChoice:
    contract: OptionContract
    scored: ScoredStrike
    considered: int
    rejected: list[tuple[float, str]] = field(default_factory=list)
    ranking: list[ScoredStrike] = field(default_factory=list)  # every passing strike, best first
    context: dict = field(default_factory=dict)  # chain-level reading (PCR, OI peaks, max pain)
    basis: str = "quality score"  # what the ranking used: "scenario objective" or "quality score"
    status: str = "SELECTED"

    @property
    def summary(self) -> str:
        s = self.scored
        bits = [f"{self.contract.strike:g} {self.contract.option_type}", f"score {s.score:.0f}/100", s.buildup.replace("_", " ").lower()]
        if s.spread_pct is not None:
            bits.append(f"spread {s.spread_pct:.1f}%")
        if s.delta is not None:
            bits.append(f"delta {abs(s.delta):.2f}")
        ctx = self.context
        if ctx.get("pcr") is not None:
            bits.append(f"PCR {ctx['pcr']:.2f}")
        if ctx.get("put_oi_peak") is not None and self.contract.option_type == "PE":
            bits.append(f"put OI peak {ctx['put_oi_peak']:g}")
        if ctx.get("call_oi_peak") is not None and self.contract.option_type == "CE":
            bits.append(f"call OI peak {ctx['call_oi_peak']:g}")
        sc = s.scenario
        if sc is not None and sc.available and sc.expected_strategy_lower is not None:
            bits.append(f"expected net {sc.expected_strategy_lower / sc.entry * 100:+.0f}% of premium (lower bound), tail loss "
                        f"{sc.cvar10 / sc.entry * 100:.0f}%")
        return "Strike pick: " + ", ".join(bits) + f" (best of {len(self.ranking)} of {self.considered} considered, ranked by {self.basis})"


def chain_context(chain: ChainSnapshot) -> dict:
    """Chain-level reading: PCR, the strike with the most call OI at/above spot and the most put OI at/below spot (context,
    not support or resistance), max pain and ATM IV."""
    spot = chain.spot_price
    calls = [r for r in chain.strikes if r.strike >= spot and r.ce_oi]
    puts = [r for r in chain.strikes if r.strike <= spot and r.pe_oi]
    return {
        "pcr": chain.total_pcr,
        "call_oi_peak": max(calls, key=lambda r: r.ce_oi or 0.0).strike if calls else None,
        "put_oi_peak": max(puts, key=lambda r: r.pe_oi or 0.0).strike if puts else None,
        "max_pain": chain.max_pain_strike,
        "atm_iv": chain.atm_iv,
    }


def _percentile(value: float, population: list[float]) -> float:
    """Share of `population` at or below `value` (0-1)."""
    if not population:
        return 0.0
    return sum(1 for x in population if x <= value) / len(population)


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _reject_reason(leg: _Leg, tx: str, median_oi: float | None, max_spread: float, min_premium: float, min_oi_share: float) -> str | None:
    if leg.security_id is None or leg.ltp is None:
        return "no quote"
    if leg.ltp < min_premium:
        return f"premium too low ({leg.ltp:g})"
    if not leg.oi:
        return "no open interest"
    if median_oi and leg.oi < median_oi * min_oi_share:
        return "thin open interest"
    if leg.volume is not None and leg.volume <= 0:
        return "no volume today"
    if leg.bid is not None and leg.ask is not None:
        if leg.bid <= 0:
            return "no bid"
        if leg.ask < leg.bid:
            return "crossed quote"
        mid = (leg.bid + leg.ask) / 2.0
        if mid > 0 and (leg.ask - leg.bid) / mid * 100.0 > max_spread:
            return f"spread too wide ({(leg.ask - leg.bid) / mid * 100.0:.1f}%)"
    if leg.delta is not None:
        low, high = (0.25, 0.75) if tx == "BUY" else (0.10, 0.45)
        if not low <= abs(leg.delta) <= high:
            return f"delta out of band ({abs(leg.delta):.2f})"
    return None


def _quote_index(chain: ChainSnapshot) -> dict[tuple[float, str], OptionQuote]:
    return {(q.strike, q.option_type): q for q in quotes_from_chain(chain)}


def select_best_contract(chain: ChainSnapshot, direction: str, transaction: str, lot_size: int = 1,
                         max_outlay: float | None = None, *, scenario: ScenarioInputs | None = None,
                         now: dt.datetime | None = None, require_quote: bool | None = None,
                         limits: ExecutionLimits | None = None) -> ContractChoice:
    """Pick the best tradeable contract on the chain for a directional setup (see the module docstring).

    LONG -> calls, SHORT -> puts, whatever `transaction` is (same convention as `select_contract`).
    `max_outlay` (rupees) rejects strikes where one lot's premium costs more than that (the per-trade capital cap).
    `scenario` adds the scenario analysis and ranks by its objective. `require_quote` (default SETTINGS.option_require_quote)
    applies the execution-quality gate. Raises OptionSelectionError ("NO SUITABLE CONTRACT ...") when the chain is empty or no
    strike in the window is tradeable."""
    if direction not in ("LONG", "SHORT"):
        raise OptionSelectionError(f"Unknown direction: {direction}")
    if transaction not in ("BUY", "SELL"):
        raise OptionSelectionError(f"Unknown transaction: {transaction}")
    if not chain.strikes:
        raise OptionSelectionError("Option chain has no strikes")

    now = now or now_ist()
    require_quote = SETTINGS.option_require_quote if require_quote is None else require_quote
    limits = limits or ExecutionLimits.from_settings()
    option_type = "CE" if direction == "LONG" else "PE"
    rows = sorted(chain.strikes, key=lambda r: r.strike)
    atm = chain.atm_strike or 0.0
    atm_idx = min(range(len(rows)), key=lambda i: abs(rows[i].strike - atm))
    window = max(int(SETTINGS.option_window_strikes), 0)
    legs_all = [_leg(r, option_type) for r in rows]
    legs = legs_all[max(atm_idx - window, 0): atm_idx + window + 1]
    quotes = _quote_index(chain)

    positive_oi = [leg.oi for leg in legs_all if leg.oi]
    median_oi = median(positive_oi) if positive_oi else None
    oi_pop = positive_oi
    vol_pop = [leg.volume for leg in legs_all if leg.volume is not None]
    context = chain_context(chain)
    atm_iv = context["atm_iv"]
    put_peak, call_peak = context["put_oi_peak"], context["call_oi_peak"]

    target_delta = 0.50 if transaction == "BUY" else 0.25
    buildup_map = _BUY_BUILDUP if transaction == "BUY" else _SELL_BUILDUP

    rejected: list[tuple[float, str]] = []
    scored: list[tuple[ScoredStrike, _Leg]] = []
    for leg in legs:
        quote = quotes.get((leg.strike, option_type))
        why = _reject_reason(leg, transaction, median_oi, SETTINGS.option_max_spread_pct, SETTINGS.option_min_premium,
                             SETTINGS.option_min_oi_vs_median)
        if not why and max_outlay is not None and leg.ltp is not None and leg.ltp * lot_size > max_outlay:
            why = f"one lot costs {leg.ltp * lot_size:,.0f}, over the per-trade cap of {max_outlay:,.0f}"
        eq = None
        if not why and require_quote and quote is not None:
            eq = assess_execution_quality(quote, now, transaction, lot_size, limits)
            if not eq.eligible:
                why = eq.reasons[0]
        greeks = None
        if not why and scenario is not None and quote is not None:
            greeks = compute_greeks(quote, chain.spot_price, now)
            # Do not propose what the risk engine is certain to refuse: if half the spread is more than its allowed share of the
            # premium stop distance (delta x the underlying stop move), the contract's cost eats the trade's stop.
            if quote.has_two_sided_quote and scenario.stop_price and (greeks.available or leg.delta is not None):
                from quant_intelligence.risk.risk_engine import MAX_SLIPPAGE_STOP_FRACTION

                delta_abs = abs(greeks.delta) if greeks.available else abs(leg.delta)
                premium_stop = delta_abs * abs(scenario.spot - scenario.stop_price)
                half_spread = (quote.ask - quote.bid) / 2.0
                if premium_stop > 0 and half_spread > MAX_SLIPPAGE_STOP_FRACTION * premium_stop:
                    why = f"spread eats the stop (half-spread {half_spread:.2f} vs premium stop {premium_stop:.2f})"
        if why:
            rejected.append((leg.strike, why))
            continue

        liquidity = _percentile(leg.oi or 0.0, oi_pop)
        if leg.volume is not None and vol_pop:
            liquidity = 0.6 * liquidity + 0.4 * _percentile(leg.volume, vol_pop)

        spread_pct = None
        if leg.bid is not None and leg.ask is not None:
            mid = (leg.bid + leg.ask) / 2.0
            spread_pct = (leg.ask - leg.bid) / mid * 100.0 if mid > 0 else None
        spread = _clip(1.0 - spread_pct / SETTINGS.option_max_spread_pct) if spread_pct is not None else 0.4

        delta = _clip(1.0 - abs(abs(leg.delta) - target_delta) / 0.25) if leg.delta is not None else 0.4

        build = classify_buildup(leg.ltp, leg.prev_close, leg.oi, leg.oi_change)
        buildup = buildup_map[build]

        if option_type == "CE":
            wall = None if call_peak is None else (leg.strike <= call_peak if transaction == "BUY" else leg.strike >= call_peak)
        else:
            wall = None if put_peak is None else (leg.strike >= put_peak if transaction == "BUY" else leg.strike <= put_peak)
        wall_score = 0.5 if wall is None else (1.0 if wall else (0.2 if transaction == "BUY" else 0.4))

        if leg.iv is not None and atm_iv:
            ratio = leg.iv / atm_iv
            iv_score = _clip(1.0 - (ratio - 0.9) / 0.4) if transaction == "BUY" else _clip((ratio - 0.8) / 0.4)
        else:
            iv_score = 0.5

        parts = {"liquidity": liquidity, "spread": spread, "delta": delta, "buildup": buildup, "oi_peak": wall_score, "iv": iv_score}
        total = 100.0 * sum(WEIGHTS[k] * v for k, v in parts.items())
        entry = ScoredStrike(leg.strike, total, parts, build, spread_pct, leg.delta, leg.oi, leg.volume, quote, eq)
        if scenario is not None and quote is not None:
            entry.greeks = greeks if greeks is not None else compute_greeks(quote, chain.spot_price, now)
            entry.execution = eq or assess_execution_quality(quote, now, transaction, lot_size, limits)
            entry.scenario = evaluate_scenarios(quote, entry.greeks, scenario, entry.execution, transaction, lot_size)
        scored.append((entry, leg))

    considered = len(legs)
    if not scored:
        counts: dict[str, int] = {}
        for _, why in rejected:
            key = why.split(" (")[0]
            counts[key] = counts.get(key, 0) + 1
        detail = "; ".join(f"{n} x {k}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        raise OptionSelectionError(
            f"NO SUITABLE CONTRACT: no tradeable {option_type} among the {considered} strikes around ATM ({detail or 'none in window'}) - not traded.")

    spot = chain.spot_price
    by_objective = scenario is not None and all(e.objective is not None for e, _ in scored)
    if by_objective:
        scored.sort(key=lambda p: (-p[0].objective, -p[0].score, abs(p[0].strike - spot)))
    else:
        scored.sort(key=lambda p: (-p[0].score, abs(p[0].strike - spot)))
    best, best_leg = scored[0]
    quote = best.quote
    executable = None
    if quote is not None and quote.has_two_sided_quote:
        executable = quote.ask if transaction == "BUY" else quote.bid
    contract = OptionContract(
        security_id=str(best_leg.security_id),
        trading_symbol=f"{chain.underlying} {chain.expiry} {int(best_leg.strike)} {option_type}",
        underlying=chain.underlying,
        strike=best_leg.strike,
        option_type=option_type,
        expiry=chain.expiry,
        lot_size=lot_size,
        transaction=transaction,
        premium=float(executable if executable is not None else (best_leg.ltp or 0.0)),
        delta=best_leg.delta,
        iv=best_leg.iv,
        last_price=best_leg.ltp,
    )
    return ContractChoice(contract=contract, scored=best, considered=considered, rejected=rejected,
                          ranking=[s for s, _ in scored], context=context,
                          basis="scenario objective" if by_objective else "quality score")
