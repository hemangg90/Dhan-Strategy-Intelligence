# Regime analysis and historical analogues

What changed, how it works, and what the evidence does and does not show. Code: `regimes/`, `analogues/analogue_engine.py`,
`research/ranking_core.py`. Tests: `tests/test_regime_analogue_upgrade.py` (plus the existing `test_regime.py`,
`test_ranking_accuracy.py`).

## Regime

Same ten labels and the same `(label, probabilities)` call. Still heuristic scores through a softmax: **not calibrated
probabilities**, display-only, not an input to ranking or orders.

| Change | Why |
|---|---|
| `trend_z`, `momentum_z` features: slope and momentum divided by their own rolling standard deviation (500 bars, past only) | A fixed `tanh(slope*500)` called a quiet index and a volatile commodity by the same yardstick. "Strong" now means strong for that instrument. Prices rescaled x250 give identical labels (test). |
| Two axes beside the label: direction (UP/DOWN/NEUTRAL), volatility (LOW/NORMAL/HIGH) and its trend (EXPANDING/STEADY/CONTRACTING) | A market can be trending and high-volatility at once; the ten overlapping labels hid that. |
| `clarity` (0-1): margin between the top two scores + agreement of the label with the axes | The old "confidence" (largest softmax share, about 0.2) said little. Clarity is also a heuristic, not a probability. |
| Overlays only where meaningful: EVENT_DRIVEN only for instruments with the weekly-expiry rule (`OPTION_UNDERLYINGS`); LIQUIDITY_STRESS only where volume exists | The next-Thursday rule meant nothing for stocks and MCX; indices have no volume. |
| `classify_series`: EMA of scores (alpha 0.3) and hysteresis (challenger must lead by 0.3 for 4 bars), `age_bars` | Bar-by-bar labels flickered. On NIFTY (about 3,100 bars) the label changed 1,123 times raw and 174 times smoothed. Causal: appending bars never changes earlier labels (test). |
| `regimes/universe.py`: regime for all 25 cached instruments from local Parquet only; `/api/regimes`; "All instruments" tab | The regime used to exist only for the one selected instrument. Instruments failing the quality gate show their status and no regime. |
| `regimes/validation.py`: per label, the next hour's realised volatility and return | Checks whether a label separates future behaviour at all. |

What the validation shows today (NIFTY, 53 days, next hour, final settings): bars labelled TREND_UP, HIGH_VOL and TREND_DOWN
were followed by more volatile hours (1.26x, 1.17x, 1.11x the overall level) and LOW_VOL by calmer ones (0.92x), the direction
one would hope for. COMPRESSION (0.97x), VOL_EXPANSION (0.94x) and EVENT_DRIVEN (0.84x, only 2 days) did not behave as their
names suggest. Return direction is not reliably separated (most intervals include zero), and several labels cover under 15
days. This is descriptive evidence from 2.5 months, not proof; the labels are display-only.

Persistence parameters (0.3 / 0.3 / 4) were chosen only to reduce flicker, by looking at the number of label changes, not at
any trading result.

## Analogues

`ANALOGUE_METHOD=legacy` (default) keeps the previous behaviour exactly (6 features, z-score, nearest 30; test pins it).
`ANALOGUE_METHOD=robust` turns on:

* nine features (adds instrument-relative trend/momentum, gap, position in the opening range and in the previous day's range);
* median/MAD scaling clipped at +-4, so one extreme value cannot dominate;
* a distance cutoff from the observations' own pairwise distances (30th percentile): "no good match" returns few analogues;
* at most 2 analogues per trading day, and a similar time-of-day window when at least 20 remain;
* kernel weights `exp(-(d/h)^2)`; weighted mean, win rate, day-clustered SE, day-block bootstrap interval, effective sample size;
* only trades that had closed (`exit_timestamp <= as_of`) when an `as_of` time is given;
* a **conditioning test**: for 60 random reference trades (own day excluded) the same analogue procedure is run, and the live
  set's distance from the overall mean is compared with theirs. If the live set is not unusual (p > 0.20) the analogues add
  no detectable information, the page says so, and the estimate falls back to the strategy's overall mean.

## Evidence (read this before switching the default)

Walk-forward comparison, NIFTY, 122 decision points (`evaluate_ranker`, same causal protocol for both):

| | trades taken | no-trade points | mean net R | 95% CI |
|---|---|---|---|---|
| legacy | 11 | 31 | -0.17 | -1.08 ... +0.65 |
| robust | 6 | 75 | +1.13 | -0.34 ... +1.89 |

Neither result is distinguishable from zero or from the other. What the numbers do show is that the robust method **trades far
less** (it abstains when the analogues are few or uninformative), which is the intended conservatism. They do **not** show an
edge, and six trades cannot. For that reason the default stays `legacy`; switching is an owner decision once there is more
history (see below).

## Limits

* The cache holds about 1.5-2.5 months of 5-minute candles per instrument. Every statistic above is limited by that. The one
  real fix is a longer history (Dhan intraday history needs a valid token; not done here).
* Analogue outcomes are strategy-trade R in modelled option economics, not forward index returns.
* Regime history depends on the 500-bar warm-up for `trend_z`; the first bars of a series use the legacy scaling.
* Persistence is path dependent: a very short window can start in a different label than a long one.
