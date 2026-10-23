# Choosing the option contract

Code: `options/contract_scoring.py` (selector), `options/scenario.py` (scenario P&L), `options/execution_quality.py` (fills and
costs), `options/greeks.py`. Used by the auto-trader (`execution/auto_trader.py`) and the Paper Trading ticket. Tests:
`test_contract_scoring.py`, `test_scenario.py`, `test_option_pipeline.py`.

Select the mode with `OPTION_SELECTION`: `SCORED` (default; this document), `BEST` (the separate delta/spread/liquidity/decay ranking
in `options/strike_ranker.py`) or `OFFSET` (plain ATM plus `OPTION_MONEYNESS_OFFSET`). None of the three has been shown to beat the
others; the scorer's weights and the scenario assumptions are judgement, not fitted to results.

## Stages

1. **Eligibility** - candidates are the strikes within `OPTION_WINDOW_STRIKES` (default 4) of ATM on the side the setup needs (LONG =
   calls, SHORT = puts). A contract is rejected, with the reason recorded, if it has no quote or security id, a premium under
   `OPTION_MIN_PREMIUM`, no open interest or under `OPTION_MIN_OI_VS_MEDIAN` of the chain median, no volume, a delta outside 0.25-0.75
   (buying) / 0.10-0.45 (writing), or one lot costs more than the per-trade capital cap. With `OPTION_REQUIRE_QUOTE` (default on)
   the execution-quality gate also applies: a fresh two-sided quote (snapshot no older than `OPTION_MAX_QUOTE_AGE_SEC`), spread
   within `OPTION_MAX_SPREAD_PCT`, quoted size of at least `OPTION_MIN_TOP_QTY_LOTS` lots. And when a stop is known, a contract
   whose half-spread exceeds half its premium stop distance is rejected ("spread eats the stop"): the risk engine would refuse it,
   so it is not proposed.
2. **Quality score (0-100)** - liquidity 30%, spread 20%, delta fit 20%, OI build-up 15%, IV value 10%, OI-peak position 5%.
   OI build-up reads the contract's own price move against the prior close with its OI change (long build-up is best for a buyer,
   short build-up worst; reversed for a writer). The OI-peak part is contextual and small: heavy open interest at a strike does not
   say who holds it, so it is never treated as support or resistance.
3. **Scenarios** - each survivor is re-priced across spot, volatility and time scenarios after conservative fills and the
   contract-note costs (`scenario.py`; the assumptions are listed with every result): the neutral distribution (what the contract
   earns with no view - for a bought option normally negative) and the strategy path (target with probability p, stop with 1-p, p
   being the comparable observations' share of positive outcomes, lower-bounded for sample size).
4. **Ranking** - by the scenario objective (lower-bound expected net return on premium less 0.25 x the 10% tail loss as a share of
   premium) when every survivor has one, otherwise by the quality score. The basis is reported.
5. **Result** - the chosen contract, priced at the **executable** side (ask for a buy, bid for a sell; the last price is kept as
   `last_price`), the ranking, every rejected alternative with its reason, the scenario table and the assumptions. If nothing
   survives the result is **NO SUITABLE CONTRACT** and nothing is traded.

Contracts are not "good" or "bad" in isolation: a deep in-the-money call can rank first because it decays less per rupee of
premium, and it also needs more capital per lot. The capital cap and the risk engine decide whether that is affordable.

## Where you see it

* Decision log and the order's reason: `Strike pick: 24500 CE, score 78/100, long buildup, spread 0.3%, delta 0.52, PCR 1.10, call OI
  peak 24700, expected net -3% of premium (lower bound), tail loss -17% (best of 7 of 9 considered, ranked by scenario objective)`.
* **Options Intelligence** page: the candidates, the rejected alternatives, the scenario table and assumptions, the proposed position
  and the risk engine's verdict.
* Paper Trading ticket: a strike-choice switch (scored / ranked / fixed offset) with the table of every strike scored.

## Limits

* The feed gives only today against the previous close, so OI build-up is a one-day reading, not an intraday trend.
* The chain has no per-contract quote time: freshness is the age of the whole snapshot.
* Flat-volatility European pricing (Black-Scholes; Black-76 for MCX): far wings are approximate.
* Scenario weights (normal moves, +/-15% IV shifts) and the objective's tail term are judgement. The expected net figures are
  estimates under those assumptions, not forecasts.
* The live Dhan page (12) uses `choose_contract`, so `SCORED` there picks by the quality score without scenarios.
