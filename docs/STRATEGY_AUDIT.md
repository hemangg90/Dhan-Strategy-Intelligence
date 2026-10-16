# Strategy audit

Scope: the 17 strategies in `quant_intelligence/strategies/registry.py` (`STRATEGY_CLASSES`, verified `len == 17`).
Status labels follow the audit scale: **WORKING** (verified by test or direct execution), **NEEDS FIXING**,
**INCOMPLETE**, **NOT IMPLEMENTED**, **UNVERIFIED**. "Empirically validated" is a separate, higher bar: **no strategy
here is empirically validated as profitable** (see `docs/SELECTOR_EVALUATION.md`).

## Common behaviour (all price-action strategies)

| Aspect | What the code does | Assessment |
|---|---|---|
| Signal timing | Evaluated on the **completed** bar; `entry = row["close"]` of the signal bar. | The fill is assumed at the very close that generated the signal: no latency, no next-bar-open. **Optimistic, documented, not changed** (changing it would change every result). |
| Look-ahead | Indicators are backward-looking. Verified by `tests/test_chronology.py`: every feature column, and every strategy's setup list, is identical when future bars are removed (cut points early / mid / end of a session). | **WORKING** after the opening-range fix below. The test shows no leak under that tampering; it does not prove every conceivable leak is absent. |
| Stop / target | `BaseStrategy.calculate_entry_stop_target` = ATR multiples (scaled by `STOP_ATR_SCALE`); some strategies use a level (VWAP, band midline, pivot, prior close, supertrend, inside-bar extreme). | Level-based targets can land on the wrong side of the entry: measured on real NIFTY data, 7/155 Bollinger and 15/288 Camarilla setups. **Fixed**: `setup_geometry_ok` + `run_backtest` skip them and count them in `metrics["n_invalid_setups_skipped"]`. |
| Trade simulation | `simulate_trade`: monitor from the bar **after** entry; stop-first if both levels are inside one bar; time exit at `max_holding_bars` (75). | **Fixed**: a bar that opens through the stop now fills at the (worse) open (`exit_reason="STOP_GAP"`); trades still open when the data ends are flagged `DATA_END` and excluded from statistics instead of being booked as closed. |
| Costs | Net R uses an *option-premium* cost model (`backtesting/engine._apply_costs`): brokerage, STT, exchange, SEBI, stamp, GST, 1-tick slippage each side, on an assumed premium (1.5 % of the underlying) and delta 0.5. | **Modelled, not observed.** Real option fills are not in the data. Never present these as actual historical fills. |
| Overlap | One strategy never has two overlapping trades in a backtest (`next_available_idx`). | WORKING. |
| Missing data | A NaN in any required feature skips the bar. | WORKING. |

## Per-strategy table

| # | Strategy (file) | Entry rule (as implemented) | Stop / target | Status | Notes |
|---|---|---|---|---|---|
| 1 | Opening Range Breakout (`orb_strategy.py`) | Close crosses the 30-minute opening range, from minute 31; once per side per session | 0.75 / 1.5 ATR | WORKING | **Fixed:** `opening_range_high/low` used a whole-session max, so bars inside the first 30 minutes saw the completed range (look-ahead). Now a running max/min, fixed after the window. ORB's minute-31 guard meant no strategy traded on it, but the feature was wrong. 14/56 setups have a stop under 0.05 % of price. |
| 2 | Momentum (`momentum_strategy.py`) | `momentum_20` beyond ±0.3 % with agreeing trend slope and relative volume ≥ 1.1; 10-bar cooldown | 1.0 / 2.0 ATR | WORKING | Needs volume; skipped by the pipeline for volume-less series (`research/pipeline._uses_volume`). |
| 3 | VWAP Mean Reversion (`vwap_mean_reversion_strategy.py`) | Stretch ≥ 0.35 % from session VWAP with |slope| ≤ 0.0006 and a reversal bar | stop 1.0 ATR, target = VWAP | WORKING | Needs volume. |
| 4 | RSI(2) Mean Reversion (`rsi2_mean_reversion_strategy.py`) | RSI(2) ≤ 10 above EMA200 (long) / ≥ 90 below (short) | 1.5 / 1.5 ATR | WORKING | EMA200 needs 200 bars of warm-up. |
| 5 | Bollinger Band Mean Reversion (`bollinger_band_reversion_strategy.py`) | Previous close outside the 20-bar band, now back inside, low trend slope | stop 1.0 ATR, target = band midline | NEEDS FIXING → fixed in the backtest | 4.5 % of setups had the target on the wrong side (price already past the midline); now skipped and counted. 16/155 tiny stops. |
| 6 | Supertrend Trend Following (`supertrend_strategy.py`) | Supertrend(10, 3) direction flip | stop = supertrend line, target 3 ATR | WORKING | Stop distance is whatever the line is, so risk per trade varies widely. |
| 7 | EMA 50/200 Crossover (`ema_crossover_strategy.py`) | EMA50 crosses EMA200 | 2 / 6 ATR | WORKING | Rare (20 setups in ~54 sessions of NIFTY): statistically thin. |
| 8 | Donchian Channel Breakout (`donchian_breakout_strategy.py`) | Close beyond the prior 20-bar high/low (shifted by one bar: causal) | 2 / 4 ATR | WORKING | 15-bar cooldown. |
| 9 | MACD Signal Crossover (`macd_crossover_strategy.py`) | MACD/signal cross filtered by EMA200 side | 1.5 / 2.5 ATR | WORKING | |
| 10 | CPR Breakout (`cpr_breakout_strategy.py`) | Close crosses the prior session's CPR (TC/BC), from minute 16 | 1.0 / 2.0 ATR | WORKING | Levels verified per-session and shifted by one session (not by one bar). 9/46 tiny stops. |
| 11 | Camarilla Pivot Reversal (`camarilla_reversal_strategy.py`) | Prior bar touched R3/S3 and the close fades back, from minute 16 | stop beyond the touch, target = pivot | NEEDS FIXING → fixed in the backtest | 5.2 % wrong-side targets (skipped, counted). **77/288 (27 %) have a stop under 0.05 % of price**: tiny risk makes net R dominated by fixed costs and hugely negative; this is an economic fact of the design, not hidden. |
| 12 | Inside Bar Breakout (`inside_bar_breakout_strategy.py`) | Close beyond the previous inside bar's range | stop = opposite extreme, target 2 ATR | WORKING | 48/275 tiny stops. |
| 13 | Gap Fill Reversion (`gap_fill_reversion_strategy.py`) | Opening gap 0.3 %-1.5 % faded in the first 20 minutes; once per session | stop 1.0 ATR, target = prior close | WORKING | Prior close is reconstructed from the gap (known at the open: no leak). The `cooldown_bars` default (100) is declared but unused. 14 setups in ~54 sessions: very thin. |
| 14 | PCR Extreme Contrarian (`pcr_contrarian_strategy.py`) | Live only (`check_chain_setup`), needs a live option chain | | INCOMPLETE | `generate_historical_setups` returns `[]`: **no backtest, never ranked**, cannot be selected. |
| 15 | OI Buildup Confirmation (`oi_buildup_strategy.py`) | as above | | INCOMPLETE | as above |
| 16 | IV Crush Premium Writer (`iv_crush_writer_strategy.py`) | as above; *sells* options | | INCOMPLETE | as above. Writing options needs margin; margin is not modelled anywhere. |
| 17 | Unusual Volume Fade (`unusual_volume_fade_strategy.py`) | as above | | INCOMPLETE | as above |

The Decision Center and website label 14-17 "needs a live option chain: no historical setups, cannot be ranked".

## Tests

* `tests/test_chronology.py` - 18 tests: all feature columns causal; each of the 17 strategies' setups unchanged when
  future bars are removed (the four chain-aware ones pass trivially: they return no setups).
* `tests/test_backtest_integrity.py` - 12 tests: stop / target / same-bar / gap / entry-bar / time-exit / data-end /
  invalid-geometry / daily Sharpe behaviour of the simulator, on hand-built bars.
* `tests/test_strategies.py`, `tests/test_backtester.py` (pre-existing): smoke-level (each strategy runs on synthetic data).
  **They do not check entry correctness per strategy**; that remains an open gap (see Remaining work).

## Not verified

* Whether any strategy has a real, persistent edge. Not established; see the selector evaluation.
* Behaviour on 1/3/15-minute timeframes (indicator parameters are bar counts, not durations).
* Intraday behaviour around exchange holidays / half-days beyond what `utils/market_calendar.py` handles.
