"""
Standard strategy interface.

Every strategy in the library implements this same interface so the ranking
engine, backtester, and setup detector can treat all strategies uniformly.
Strategies NEVER place broker orders directly - they only produce setups /
signals / entry-stop-target proposals that flow through the risk engine.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Setup:
    """A single historical or live occurrence of a strategy's entry conditions being met."""

    timestamp: pd.Timestamp
    direction: str  # LONG / SHORT
    entry_price: float
    stop_price: float
    target_price: float
    meta: dict = field(default_factory=dict)


def setup_geometry_ok(setup: Setup) -> bool:
    """A tradable setup has a positive entry, a stop on the losing side and a target on the winning side.
    Some strategies target a level (band midline, pivot, prior close) that price may already have passed, which would
    otherwise be simulated as an instant "target hit" at a loss."""
    e, s, t = setup.entry_price, setup.stop_price, setup.target_price
    if not all(isinstance(x, (int, float)) and x == x and x > 0 for x in (e, s, t)):
        return False
    return (s < e < t) if setup.direction == "LONG" else (t < e < s)


@dataclass
class TradeResult:
    setup: Setup
    exit_timestamp: pd.Timestamp | None
    exit_price: float | None
    r_multiple: float | None
    holding_period_bars: int
    mfe: float
    mae: float
    gross_pnl: float
    outcome: str  # WIN / LOSS / SCRATCH / OPEN
    # Why the trade ended: STOP, STOP_GAP (a bar opened beyond the stop, filled at the worse open), TARGET, TIME
    # (max holding reached) or DATA_END (the data ran out first: the trade was still OPEN, so its result is NOT final).
    exit_reason: str | None = None


class BaseStrategy(abc.ABC):
    name: str = "base"
    description: str = ""
    required_features: list[str] = []
    default_parameters: dict = {}

    # Chain-aware strategies (see quant_intelligence/options/) set this to the
    # ChainSnapshot-derived fields they read (e.g. "total_pcr", "atm_iv") and
    # implement check_chain_setup below. Pure price-action strategies leave
    # this empty and are never asked for a chain-based setup.
    required_chain_features: list[str] = []

    def check_chain_setup(self, market_state, recent_ohlcv, chain) -> Setup | None:
        """Chain-aware strategies override this to combine option-chain analytics
        (a ChainSnapshot from quant_intelligence.options.chain_analytics) with price
        action. Default: no chain-based setup (pure price-action strategy)."""
        return None

    # Parameters that set how far the stop and target sit from the entry; STOP_ATR_SCALE multiplies their DEFAULTS.
    SCALED_PARAMETERS = ("stop_atr_mult", "target_atr_mult")

    def __init__(self, parameters: dict | None = None):
        from quant_intelligence.config.settings import SETTINGS

        defaults = dict(self.default_parameters)
        scale = SETTINGS.stop_atr_scale
        if scale != 1.0:
            for key in self.SCALED_PARAMETERS:
                if key in defaults:
                    defaults[key] = round(defaults[key] * scale, 4)
        # Explicit parameters (a user's own values) always win and are never scaled.
        self.parameters = {**defaults, **(parameters or {})}
        self.validate_parameters(self.parameters)

    def validate_parameters(self, parameters: dict) -> None:
        """Override to raise ValueError on invalid parameter combinations."""
        return None

    @abc.abstractmethod
    def generate_historical_setups(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> list[Setup]:
        """Scan historical OHLCV+features and return every historical setup occurrence.

        `ohlcv` and `features` must be aligned (same length/index, features computed
        with no look-ahead). Implementations must only use data up to bar i when
        deciding whether a setup occurs at bar i.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def check_setup(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        """Check whether the strategy's entry conditions are met RIGHT NOW, live/paper context."""
        raise NotImplementedError

    def generate_signal(self, market_state, recent_ohlcv: pd.DataFrame) -> Setup | None:
        """Default: a signal is just a live setup. Strategies can override for extra filtering."""
        return self.check_setup(market_state, recent_ohlcv)

    def calculate_entry_stop_target(self, direction: str, reference_price: float, atr: float) -> tuple[float, float, float]:
        """Default ATR-based stop/target; strategies may override with their own logic."""
        stop_mult = self.parameters.get("stop_atr_mult", 1.0)
        target_mult = self.parameters.get("target_atr_mult", 2.0)
        if direction == "LONG":
            stop = reference_price - stop_mult * atr
            target = reference_price + target_mult * atr
        else:
            stop = reference_price + stop_mult * atr
            target = reference_price - target_mult * atr
        return reference_price, stop, target

    def simulate_trade(self, setup: Setup, ohlcv: pd.DataFrame, entry_idx: int, max_holding_bars: int = 75) -> TradeResult:
        """Bar-by-bar forward simulation from entry_idx: whichever of stop/target is hit first wins.
        If neither is hit within max_holding_bars, exit at the close of the last bar (time-based exit).
        This is intentionally simple and conservative (no intra-bar look-ahead: within a bar we
        check stop before target for adverse conservatism on longs, and vice versa for shorts).
        """
        direction = setup.direction
        n = len(ohlcv)
        # Entry price is the CLOSE of entry_idx, so monitoring must start at the NEXT
        # bar - checking entry_idx's own high/low against a stop derived from its close
        # would double-count that bar's intrabar range and produce spuriously high
        # stop-out rates.
        monitor_start_idx = entry_idx + 1
        end_idx = min(entry_idx + max_holding_bars, n - 1)

        mfe = 0.0
        mae = 0.0
        exit_price = None
        exit_ts = None
        outcome = "OPEN"
        exit_reason = None

        risk_per_unit = abs(setup.entry_price - setup.stop_price)
        if risk_per_unit == 0:
            risk_per_unit = 1e-6

        if monitor_start_idx > end_idx:
            last_bar = ohlcv.iloc[entry_idx]
            pnl_dir = 1 if direction == "LONG" else -1
            gross_pnl = pnl_dir * (last_bar["close"] - setup.entry_price)
            return TradeResult(
                setup=setup,
                exit_timestamp=last_bar["timestamp"],
                exit_price=last_bar["close"],
                r_multiple=gross_pnl / risk_per_unit,
                holding_period_bars=0,
                mfe=0.0,
                mae=0.0,
                gross_pnl=gross_pnl,
                outcome="WIN" if gross_pnl > 0 else ("LOSS" if gross_pnl < 0 else "SCRATCH"),
                exit_reason="DATA_END" if entry_idx >= n - 1 else "TIME",
            )

        for idx in range(monitor_start_idx, end_idx + 1):
            bar = ohlcv.iloc[idx]
            if direction == "LONG":
                favorable = bar["high"] - setup.entry_price
                adverse = setup.entry_price - bar["low"]
            else:
                favorable = setup.entry_price - bar["low"]
                adverse = bar["high"] - setup.entry_price

            mfe = max(mfe, favorable)
            mae = max(mae, adverse)

            if direction == "LONG":
                hit_stop = bar["low"] <= setup.stop_price
                hit_target = bar["high"] >= setup.target_price
            else:
                hit_stop = bar["high"] >= setup.stop_price
                hit_target = bar["low"] <= setup.target_price

            if hit_stop:
                # A bar that OPENS beyond the stop gapped through it: the stop cannot fill at its price, only at the
                # (worse) open. If both stop and target fall inside one bar the order of events is unknowable from OHLC,
                # so the stop is assumed first (conservative).
                bar_open = float(bar["open"])
                stop_fill = min(setup.stop_price, bar_open) if direction == "LONG" else max(setup.stop_price, bar_open)
                exit_price = stop_fill
                exit_reason = "STOP_GAP" if stop_fill != setup.stop_price else "STOP"
                outcome = "LOSS"
                exit_ts = bar["timestamp"]
                break
            if hit_target:
                # Filled at the target even if the bar opened beyond it (the better open is ignored: conservative).
                exit_price = setup.target_price
                exit_reason = "TARGET"
                outcome = "WIN"
                exit_ts = bar["timestamp"]
                break

        if exit_price is None:
            # Time-based exit at the close of the last bar in the window.
            last_bar = ohlcv.iloc[end_idx]
            exit_price = last_bar["close"]
            exit_ts = last_bar["timestamp"]
            pnl_dir = 1 if direction == "LONG" else -1
            gross_pnl_check = pnl_dir * (exit_price - setup.entry_price)
            exit_reason = "DATA_END" if entry_idx + max_holding_bars > n - 1 else "TIME"
            outcome = "WIN" if gross_pnl_check > 0 else ("LOSS" if gross_pnl_check < 0 else "SCRATCH")

        pnl_dir = 1 if direction == "LONG" else -1
        gross_pnl = pnl_dir * (exit_price - setup.entry_price)
        r_multiple = gross_pnl / risk_per_unit

        return TradeResult(
            setup=setup,
            exit_timestamp=exit_ts,
            exit_price=exit_price,
            r_multiple=r_multiple,
            holding_period_bars=end_idx - entry_idx,
            mfe=mfe,
            mae=mae,
            gross_pnl=gross_pnl,
            outcome=outcome,
            exit_reason=exit_reason,
        )
