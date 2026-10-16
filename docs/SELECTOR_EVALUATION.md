# Selector evaluation

**Bottom line: this evaluation does not show that the strategy selector beats a random pick, a fixed strategy or a
regime-based rule. It also cannot show that it does not: the data are too short.** Treat every number below as
descriptive. Raw output: `docs/results/selector_evaluation_2026-10-16.json` / `.txt`.

## What was evaluated

`quant_intelligence/ranking/evaluate_ranker.py` walks forward through each instrument's cached 5-minute candles. At every
bar where at least one strategy has a setup, the ranker (the same code as the live pipeline, via
`research/ranking_core.py`) chooses a strategy using **only trades that had already closed** (`exit_idx <= t`,
`entry_idx < t`). The outcome it is judged on is the net-of-cost R of that strategy's trade at bar `t`.

| Item | Value |
|---|---|
| Instruments / data | NIFTY (54 sessions, 2026-07-31 to 2026-10-16) and BANKNIFTY (42 sessions, to 2026-09-29): real candles from the local cache, no network |
| Warm-up | first 40 % of bars (at least 400): used only to build history; **no decision is scored in it** |
| Decision points scored | 1,078 (610 NIFTY, 468 BANKNIFTY) over 37 distinct trading days |
| Costs | modelled option-premium costs (brokerage, STT, exchange, SEBI, stamp, GST, slippage) on an *assumed* premium of 1.5 % of the underlying and delta 0.5. **These are not observed option fills.** |
| Tie-break trades | **excluded** (evaluator default). The paper trader's default does take tie-break picks; those were not evaluated here. |
| Code state | after this session's fixes (gap-aware stops, unresolved trades dropped, wrong-side targets skipped, causal opening range). Earlier results in `data_cache/eval/` predate them and are not comparable. |

Reproduce: `python -m quant_intelligence.ranking.evaluate_ranker NIFTY BANKNIFTY --json out.json` (about 8 minutes).

## Results (net R per trade, after modelled costs)

95 % intervals resample whole trading days (trades on one day are correlated). They are approximate: decision points
at consecutive bars can describe overlapping trades, so they likely understate the true uncertainty.

| Policy | Trades | Mean net R | 95 % CI | Win rate | Profit factor | Max drawdown (R) |
|---|---|---|---|---|---|---|
| **Selector (existing)** | **55** | **+0.011** | **-0.292 to +0.339** | 40.0 % | 1.01 | -11.95 |
| Random pick on the bars the selector traded | 55 | +0.009 | n/a | n/a | n/a | n/a |
| Fixed strategy, best on warm-up only | 110 | -0.095 | -0.404 to +0.270 | 36.4 % | 0.88 | -21.86 |
| Regime-based selector | 234 | -0.168 | -0.388 to +0.092 | 38.5 % | 0.80 | -41.44 |
| Take a random strategy at every signal bar | 1,078 | -0.256 | -0.437 to -0.096 | 35.7 % | 0.70 | -279.34 |

Selector minus random pick on the same bars: **+0.001 R (95 % CI -0.113 to +0.125)**. Per instrument: NIFTY 41 trades,
mean +0.134 (CI -0.361 to +0.627), lift -0.006; BANKNIFTY 14 trades, mean -0.352 (CI -1.121 to +0.228), lift +0.022.

### How to read this

* **Strategy choice adds nothing detectable.** On the bars where the selector traded, a random pick earned the same
  (+0.009 vs +0.011). The selector's confidence interval comfortably includes zero and every baseline's mean.
* **Trading at every signal lost money, significantly** (CI excludes zero). The selector's positive mean therefore comes
  from *when* it chose to trade (only when some strategy cleared its edge/confidence bars: 5.1 % of decision points),
  not from *which* strategy it picked. That is a hypothesis worth testing on more data, not a demonstrated edge: with 55
  trades the difference from zero is not statistically established.
* **Baselines are not clearly worse than the selector** either (their intervals overlap it). No ordering is established.
* Mean net R is far from the headline: at a profit factor of 1.01 the selector is, within noise, break-even.

### Breakdowns (exploratory, small samples, no multiple-comparison correction)

By strategy (selector trades): Momentum 40 trades, +0.165 R; CPR Breakout 8, -0.393; Inside Bar 6, -0.298; Gap Fill 1,
-1.101. Momentum accounts for 73 % of trades. By regime label at entry: HIGH_VOL 22 trades +0.507; RANGE 17 trades
-0.513; VOL_EXPANSION 11 trades -0.445; COMPRESSION 3 trades +0.831; LOW_VOL 2 trades +0.275. These slices were not
planned in advance and are too small to act on; the regime label itself is a heuristic score (see PROJECT_AUDIT).

## Why the evidence is weak

1. **Length:** 37 scored trading days and 55 selector trades from two highly correlated indices.
2. **Dependence:** overlapping trades, one volatility regime, one market phase.
3. **Multiple testing:** 17 strategies, several baselines and many slices; the best-looking cell will look good by chance.
4. **Modelled costs and premiums:** the economics rest on an assumed option premium and delta.
5. **Selection among 17 strategies** ("winner's curse") is only partly handled by the ranker's shrinkage.

## Not done (disclosed, not hidden)

* **Buy-and-hold / instrument benchmark:** not implemented; the policies trade short option positions on signals and are
  not comparable with holding the index without further assumptions.
* **Sensitivity to parameter changes** (`top_k`, thresholds, shrinkage): not run. No broad parameter search was done on
  purpose, to avoid manufacturing a good-looking result.
* **Longer history:** the cache holds about 2.5 months. A multi-year evaluation needs more data.
* **Tie-break trades** and the commodity / stock instruments: not evaluated here.
* **Out-of-sample parameter freezing:** no parameters were tuned on this data; the "out-of-sample" part is everything
  after the warm-up, but with so little data it should not be read as validation.

## Chronology tests supporting the evaluation

`tests/test_chronology.py` (features and strategy setups are unchanged when future bars are removed) and
`tests/test_selector_chronology.py` (rewriting all data after a cut-off does not change any earlier decision, and the
fixed baseline is chosen from the warm-up only). They show these specific tamperings do not change earlier decisions;
they do not prove the absence of every possible leak.
