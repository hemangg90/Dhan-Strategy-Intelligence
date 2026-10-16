# Calculation inventory

Every important formula, where it lives, how it is defined, how it behaves on bad input, and how it is tested.
"Test" means an independent known-answer or property test (the expected value is computed separately, never by the code
under test) in `quant_intelligence/tests/test_calculations.py` unless another file is named.
**Fixed** = a defect found in this audit and corrected with a regression test. **Convention** = a documented choice that
differs from the textbook definition and was deliberately left unchanged because changing it would alter trading signals.

## 1. Indicators and features (`features/feature_engine.py`)

| Quantity | Implemented definition | Edge cases | Test | Status |
|---|---|---|---|---|
| `return_n` | `log(close).diff(n)` (log return) | first n bars NaN; non-positive close is rejected upstream by the quality gate | yes | OK |
| `momentum_20` | `close.pct_change(20)` (simple return) | first 20 NaN | yes | OK |
| `trend_slope` | OLS slope of `close` over 20 bars, divided by the current close (fraction of price per bar) | NaN until 20 bars | yes (straight line) | OK |
| `market_structure` | heuristic 10-bar higher-high / lower-low flag in {-1, 0, 1} | n/a | no | **UNVERIFIED**: not a standard indicator; documented, not independently defined |
| `atr_14` | **simple** mean of True Range over 14 bars; TR = max(H-L, \|H-prevC\|, \|L-prevC\|); the first bar's TR is H-L | NaN until 14 TRs | yes | Convention: Wilder's ATR uses recursive smoothing and differs slightly |
| `atr_percentile_100` | percentile rank of the current ATR in the trailing 100 values, including itself (min 20) | NaN until 20 | indirect | OK |
| `realized_vol_20` | sample std (ddof 1) of 1-bar log returns over 20 bars x sqrt(252 x session_minutes / bar_minutes) | NaN until 20 returns | yes (5 and 15 minute) | **Fixed**: the bar length was hard-coded to 5 minutes, mis-scaling every other timeframe; it is now inferred from the data (`infer_bar_minutes`) |
| `vol_expansion` | `realized_vol_20` / its trailing-100 mean (unit-free) | NaN until 20 | indirect | OK |
| `relative_volume` | volume / mean volume of the trailing 20 bars **including the current bar** | zero-volume series -> 0/0 = NaN (not 0) | yes | Convention (a common alternative excludes the current bar) |
| `vwap` | per-session cumulative sum(typical x volume) / sum(volume), typical = (H+L+C)/3; resets every session | zero volume -> NaN | yes | OK |
| `vwap_distance_pct` | (close - vwap) / vwap x 100 (percent) | NaN if vwap NaN | indirect | OK |
| `opening_range_high/low` | running max/min of the first 30 minutes of the session, then held | | yes (+ chronology test) | **Fixed**: used the whole-session max, so bars inside the window saw the completed range (look-ahead) |
| `gap_pct` | (session open - previous session close) / previous close x 100, at the first bar, held for the session | first session NaN | yes | OK |
| `rsi_n` | Wilder smoothing as an EWM with alpha = 1/n, seeded with the first average (not an SMA seed), min_periods = n | all gains -> 100; all losses -> 0; **no movement -> NaN** | yes (independent recursion, bounds property) | **Fixed**: 0/0 (flat prices) returned 100, "overbought", which would trigger a false RSI(2) SHORT. Convention: the seed differs from Wilder's SMA seed; values converge after ~3n bars |
| Bollinger bands | SMA20 +/- 2 x **sample** std (ddof 1) | NaN until 20 | yes (pins the convention) | Convention: classic Bollinger uses the population std (ddof 0): bands are sqrt(20/19) = 2.6 % wider here. Changing it alters Bollinger-strategy signals, so it is left for an owner decision |
| EMA(n) | `ewm(span=n, adjust=False)`, alpha = 2/(n+1), seeded with the first close, min_periods = n | NaN until n | yes | OK |
| MACD | EMA12 - EMA26; signal EMA9 of MACD; histogram = difference | | yes | OK |
| Supertrend(10, 3) | bands = (H+L)/2 +/- 3 x ATR10 (simple-mean ATR); standard ratcheting final bands; direction flips on close beyond the opposite band | NaN until ATR exists | property test (trends) | Partially verified; the description string says "computed on close" but hl2 is used (wording only) |
| Donchian(20) | highest high / lowest low of the previous 20 bars, **excluding** the current bar | NaN until 21 | yes | OK |
| CPR / Camarilla | previous **session's** H, L, C: pivot (H+L+C)/3, BC (H+L)/2, TC 2P-BC, R/S3 = C +/- 1.1 x range / 4, R/S4 = C +/- 1.1 x range / 2 | first session NaN | yes (hand numbers) | OK |
| `expiry_proximity_days` | calendar days to the next Thursday | | no | **UNVERIFIED / possibly outdated**: it is a weekday heuristic, named "trading days" in the docstring but counts calendar days and ignores holidays; the exchange's current weekly-expiry weekday should be confirmed |

## 2. Trades, costs, R (`backtesting/engine.py`, `strategies/base_strategy.py`)

| Quantity | Definition | Edge cases | Test | Status |
|---|---|---|---|---|
| Gross R | direction x (exit - entry) / \|entry - stop\| in **underlying points** | zero stop distance is replaced by 1e-6 in the simulator; the risk engine and `setup_geometry_ok` reject zero-distance setups first | yes | OK |
| Stop fill | worse of the stop and the bar open (gap); target fills at the target | | yes | **Fixed** (was always at the stop) |
| Option economics | premium = 1.5 % of the underlying (or a Black-Scholes style model when enabled), delta 0.5 constant; option P&L = exit premium - entry premium, floored at the 0.05 tick | a bought option cannot lose more than its premium | yes | **Fixed**: the P&L ignored the floor, so a large adverse move could "lose" more than the premium. **Modelled, not observed** |
| Costs | brokerage 2 x Rs 20; STT on sell premium; exchange charge and SEBI fee on turnover; stamp on buy premium; GST 18 % on brokerage + exchange + SEBI; slippage 1 tick each side | | yes (independent calculation) | Rates from config; verify against a contract note |
| Net R | option P&L after costs / (delta x \|entry - stop\| x quantity) | zero risk -> 0 | yes | OK |
| Equity / drawdown | cumulative net P&L (or R); drawdown = min over time of (value - running peak), **with the starting value counted as a peak** | empty -> 0 | yes (+ exhaustive-search property) | **Fixed**: five separate copies measured from the first trade, under-reporting drawdowns of strategies that start with losses; now one shared function (`analytics/metrics.max_drawdown`) |
| Win rate | global: share of trades with gross R > 0; conditional (analogues): share with net R > 0 | | indirect | Documented inconsistency (gross vs net); both are labelled in the UI |
| Profit factor | gross profit / gross loss over net P&L; **None when there is no losing trade** | | yes | Added (was not computed) |
| Payoff ratio | mean win / \|mean loss\| over strict winners and losers; None if either is missing | | yes | **Fixed**: break-even trades were counted as losses |
| Expectancy | mean per-trade R or P&L | | yes | OK |
| Sharpe / Sortino | **daily** net R (sum of R of trades closed that day); Sharpe = mean / sample std x sqrt(252); Sortino divides by sqrt(mean(min(x,0)^2)) (downside deviation, target 0); None below 10 days | | yes | **Fixed**: per-trade R was annualised by sqrt(252) |

## 3. Analogues, ranking, statistics (`analogues/`, `ranking/`, `research/`)

| Quantity | Definition | Test | Status |
|---|---|---|---|
| Analogue distance | Euclidean in z-scored space over 6 features (mean/std of the observations supplied); NaN replaced by the column mean; top-30 nearest | `test_chronology` / selector tests | Limits: no minimum similarity (poor matches are still returned), similarity = 1 - d / d_max is relative, not calibrated |
| Conditional expectancy | mean net R of the analogues | yes | OK |
| Standard error | cluster-robust by trading day: sqrt(g/(g-1) x sum of squared day-sums of residuals) / n; NaN with one day | yes (hand numbers) | OK |
| Confidence | n < 10 insufficient; days >= 15 HIGH, >= 6 MEDIUM, else LOW | yes (`test_ranking*`) | Heuristic thresholds |
| Shrinkage | normal-normal posterior toward the strategy's own mean, tau = 0.35 R | yes | OK |
| Score | 0.5 x (shrunk edge - 1 SE) + 0.3 x (P(win) - 0.5) + 0.3 x confidence weight - 0.2 x drawdown penalty | yes | Heuristic weights; no significance test or multiple-comparison correction |
| Eligibility | shrunk edge >= 0.05 R, confidence >= MEDIUM, edge positive in >= 50 % of >= 3 time folds | yes | OK |
| NO TRADE / tie | data not OK, nothing eligible, or top two statistically tied (then a tagged half-size trade if enabled) | yes | OK |

## 4. Risk and sizing (`risk/risk_engine.py`, `execution/auto_trader.py`)

| Quantity | Definition | Test |
|---|---|---|
| Trade risk % | \|entry - stop\| x quantity / equity x 100 (premium terms for options) | yes |
| Drawdown % | (peak - equity) / peak x 100; daily loss % = -daily P&L / equity x 100 | yes |
| Position size | `floor(budget / stop_distance / lot) x lot`, budget = equity x risk% x 0.9 x risk multiplier, then capped at `max_capital_per_trade_pct` of equity | yes (known answer + properties) |
| Validity | finite positive prices; stop and target on the correct side; whole-lot positive quantity; account state finite and positive | yes (properties) |

## 5. Options (`options/`)

| Quantity | Definition | Status |
|---|---|---|
| Premium stop/target | `entry -/+ delta x underlying stop/target distance`, floored at the 0.05 tick (first-order, documented as an estimate) | Estimate; no test of accuracy |
| **Missing delta** | `translate_setup` substitutes **0.5** when the contract's delta is missing | **Open P2**: a default for missing mandatory input. Now logged as a WARNING and flagged (`PremiumSetup.delta_assumed`) but still applied; refusing the trade instead is an owner decision because it changes live behaviour. Tested (`test_premium_translation.py`) |
| Put-call ratio, max pain, IV skew, ATM IV, unusual volume | total PCR = sum(PE OI) / sum(CE OI), None without call OI; max pain = the expiry price minimising total option-holder payout (ties to the lowest strike); skew = put IV one strike below ATM minus call IV one strike above; unusual = volume/OI more than 2 population standard deviations above the mean | Verified against hand-computed answers (`test_chain_formulas.py`) |
| Greeks / IV | taken from the broker's chain, not recomputed; the Black-Scholes style backtest model (`volatility/`, off by default) is not independently verified | UNVERIFIED |

## 6. Duplicated formulas found

* Max drawdown: 5 copies -> consolidated (`analytics/metrics.max_drawdown`), plus two correct ones (workbench, risk engine).
* Option charges: 3 copies (backtest, paper broker, Positions page) -> the paper broker and Positions page now share
  `execution/charges.option_trade_charges`; the backtest keeps its own because it works on an *assumed* premium.
* Sharpe: the backtest metrics and the workbench now both call `analytics/metrics.annualised_sharpe`.
