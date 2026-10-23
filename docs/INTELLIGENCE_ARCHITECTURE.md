# Market + options intelligence: architecture, implementation map and status

Status: **done** = implemented and tested; **partial** = works with a stated gap; **missing** = not built, with the reason.
This document is the honest ledger. Passing tests do not make the system production-ready: no part of it has been shown to make
money, and the options-aware parts cannot be back-tested for lack of historical option data.

## The pipeline

```
MARKET DATA ──┬─> UNDERLYING INTELLIGENCE ─> MarketContext ──┐
              └─> OPTIONS INTELLIGENCE ────> OptionsContext ─┤
                                                             v
     ANALOGUE ENGINE -> STRATEGY RANKING -> SETUP VALIDATION -> OPTIONS SELECTOR (eligibility, score, scenarios)
                                                             v
                  RISK ENGINE (independent, fail-closed) -> APPROVE / REJECT -> paper EXECUTION
```

Where it runs: `execution/auto_trader.py::_run_auto_option_cycle` (the order of the stages above), with the staged pieces in their
own modules so each can be tested alone.

## Implementation map

| Area | Where | Status |
|---|---|---|
| Data providers, cache, provenance, quality gate | `data/`, `data_adapters/` (existing) | done (reused) |
| Instrument master, lot sizes, holidays, expiry rules | `data_adapters/dhan_instrument_master.py`, `config/` (existing) | done (reused) |
| Common data-status object (OK / WARNING / CRITICAL, component, reason, unavailable fields) | `intelligence/status.py` | done |
| **MarketContext** (typed, validated, reproducible, JSON-safe) | `intelligence/market_context.py` | done |
| Regime, trend, ATR, volatility percentile, relative volume | `regimes/`, `features/feature_engine.py` (existing) | done (reused); the regime scores stay *heuristic*, labelled as such - no calibration was built |
| Swing points with confirmation time, BOS / MSS, displacement, fair value gaps (mitigation, invalidation, expiry) | `features/market_structure.py` | done; look-ahead tested |
| Liquidity zones (swings, equal highs/lows, prior day/week, opening range) and events (sweep, accepted/failed breakout, unresolved) with tolerance, confirmation, expiry, invalidation | `features/liquidity.py` | done; zones are *structural hypotheses*, not known stop locations |
| Support/resistance levels with weight, reactions (independent excursions), recency, behaviour, higher-timeframe flag, confluence, consolidation edges; trade room check; structure stop | `features/levels.py` | done |
| Session VWAP, slope, crossings, rejections; approximate volume profile (POC/VAH/VAL, flagged as approximation) | `features/vwap_volume.py` | done; unavailable (named) for series without volume (e.g. indices) |
| Normalised option contract with named missing fields; chain validation (lot size, strike step, expiry, snapshot age, ladder gaps) | `options/contracts.py` | done |
| Greeks (delta, gamma, vega, theta, rho), BSM / Black-76, trading-time clock, IV solved from price, provider cross-check, near-expiry guard | `options/greeks.py`, `options/black_scholes.py` | done; validated against textbook values and finite differences |
| IV history, IV rank, IV percentile, changes, tenor-matched figures | `options/iv_history.py`, `options/iv_recorder.py` | done, but **empty until about 30 sessions are recorded** (the broker offers no IV history); reported unavailable until then |
| Open interest, change in OI, PCR (OI, volume, windowed), concentration, OI-to-volume | `options/oi_features.py` | done; interpretation deliberately withheld (no support/resistance claim) |
| Contract liquidity, spread, size, quote freshness, conservative fills, slippage, charges, break-even | `options/execution_quality.py` | done |
| Expiry intelligence (trading days, sessions left, risk band, monthly flag) from the exchange calendar | `options/expiry.py` | done |
| **OptionsContext** (typed, validated, reproducible, chain hash) | `intelligence/options_context.py` | done |
| Analogue engine: outcome-availability times, no-look-ahead `as_of`, structure/options features, redundancy control, like-for-like filters, explanations | `analogues/analogue_engine.py`, `research/ranking_core.py`, `intelligence/context_features.py` | done; the `context` method is opt-in (`ANALOGUE_METHOD=context`) |
| Options features inside historical analogues | - | **missing**: no historical option chains exist; they are used only when present and are named as not compared |
| Strategy ranking evidence (sample size, CI, profit factor, tail loss, drawdown, costs, regime record, NO TRADE reasons) | `ranking/ranking_report.py` over the existing ranking | done; "options-specific performance" is reported unavailable |
| Selector comparison A (existing) vs B (+ structure); C, D | `ranking/selector_comparison.py` | A vs B done (paired, day-clustered interval, honest verdict); C and D **missing** (need option history) |
| Setup validation stage (ranking tradable, data, session, freshness, active and unexpired setup, options data, eligible contracts, room to run; advisory warnings) | `execution/setup_validation.py` | done |
| Options selector (eligibility -> score -> scenarios -> ranking; NO SUITABLE CONTRACT; rejected alternatives with reasons) | `options/contract_scoring.py`, `options/scenario.py` | done for single-leg long/short options; multi-leg structures are **not supported** (no execution or risk model for them) |
| Risk engine option controls (premium at risk, per-underlying concentration, delta, gamma, vega, theta, expiry, order size, quote freshness, spread) | `risk/options_risk.py`, `risk/risk_engine.py` | done; Greeks are measured at entry, not re-marked |
| Event-risk controls | - | **missing**: the application has no event calendar |
| Margin-aware short option risk | - | **missing**: margin is not modelled; short options are limited by Greeks and the mandatory stop |
| Duplicate-order prevention, single-use approvals, order-state machine | `brokers/` (existing) | done (reused) |
| Position reconciliation (paper: memory vs database, blocks new orders; live: primitive only) | `brokers/reconciliation.py` | partial: live order-book reconciliation needs an orders endpoint the client does not have |
| Partial fills, timeouts, cancel failures on a live broker | `brokers/order_state.py` (existing) | partial: the paper broker fills fully; live behaviour is as before and live is locked |
| UI: Levels & Structure, Chain / Selector / Risk, ranking evidence, risk exposure and reconciliation | `pages/20_*`, `pages/21_*`, `pages/02_*`, `ui/risk_control.py` | done |

## What the data cannot give (and how the system behaves)

* **No historical option chains.** Analogues and rankings are built from the underlying. Options features are used live and
  recorded going forward. Selectors C and D are reported UNAVAILABLE, never approximated.
* **IV rank and percentile** appear only after enough recorded sessions. Until then the page and the context say so.
* **Quote time** is the snapshot's time; the chain has no per-contract timestamp.
* **Stop locations** are never known from OHLCV; liquidity zones are structural hypotheses.
* **Open interest** says where contracts exist, not who holds them, so it is never turned into support or resistance.

## Chronological integrity

Swings are known only `swing_bars` after they form; every structure and liquidity answer at bar t is computed from bars <= t and
is unchanged when later bars are altered (tests mutate the future). Analogues exclude any observation whose outcome had not become
known by the decision time (`outcome_available_at`), refuse an `as_of` they cannot verify, and report the information cutoff. The
walk-forward evaluator only lets the ranker see trades that had closed. IV statistics use only earlier sessions.

## Paper and live

Paper trading is the default. Live trading remains locked behind `DEMO_LOCKDOWN` / `TRADING_MODE` / the live confirmation, and
nothing in this work enables it. Hard risk vetoes are enforced in the order path (`auto_trader` -> `evaluate_trade` -> broker), not
only displayed.

## Validation results (what the data showed)

* **Level check** (`LEVELS_ENABLED`, replay of every historical setup of 12 strategies on cached NIFTY and BANKNIFTY 5-minute candles,
  levels built only from earlier bars, 1.5 stop-distances of room required): 2,329 setups, 1,918 vetoed (82%). Mean net R was -0.055
  for all setups, -0.090 for those kept and -0.047 for those vetoed; moving stops and targets to levels changed the kept trades from
  -0.095 to -0.119 R. Two months of data and tiny per-strategy samples: this is not evidence that levels are useless, but it is no
  evidence they help, and a filter that removes four setups in five is not a sensible default. It therefore ships **off**.
* **Selector A vs B** (existing analogue selector vs the same with market-structure features), paired by bar, cached 5-minute
  candles, day-clustered 95% intervals: NIFTY +0.048 R/bar (-0.002, +0.094), BANKNIFTY -0.027 (-0.074, +0.027), RELIANCE -0.003
  (-0.040, +0.038), ICICIBANK -0.025 (-0.054, +0.002), CRUDEOIL -0.044 (-0.086, +0.001); 200-290 common bars each. Every verdict is
  **not distinguishable**: structure features did not measurably change the selector's results on this short history. C and D
  cannot be evaluated (no option history).
