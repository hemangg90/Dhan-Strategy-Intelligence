# Candidate workflow, provenance and audit trail

How a user follows one candidate from data to outcome, what is shown at each step, and what none of it proves. Code:
`ui/candidate.py`, `pages/00_Candidate.py`, `data/provenance.py`, `research/evidence.py`, `research/sensitivity.py`,
`risk/risk_engine.preview_trade`, `reports/trace.py`. Tests: `test_candidate.py`, `test_candidate_evidence.py`,
`test_risk_preview_sensitivity.py`, `test_trace.py`.

## The path (page: Candidate)

Data -> Market -> Evidence -> Setup -> Risk engine -> Paper order -> Outcome. Each stage shows PASS / WARN / FAIL / WAIT / SKIP
and, when it stops the candidate, names the condition that failed. The banner reads CANDIDATE READY, WAITING FOR SETUP or
NO TRADE, plus the stage that stopped it. A ranker NO TRADE and a risk veto now appear in the same view.

Three kinds of setup are kept apart: **HISTORICAL_SETUP** (a past occurrence in the strategy's backtest), **CURRENT_CANDIDATE**
(a setup that triggers on the latest closed bar) and **PAPER_TRADE** (a simulated order that was placed; see Trace).

The risk stage is a **preview** (`preview_trade`): the same checks in the same order with the same fail-closed behaviour, but it
writes no `RiskEvent` and issues no approval, so a broker can never spend it. The candidate page previews in the
underlying's price terms with one lot; the real order is checked on option premiums, so per-trade risk there is indicative.

## Data provenance

Every result carries `provenance`: source (cache / cache+tail / dhan / csv), last bar, age, quality, and a currency label.

| Label | Meaning |
|---|---|
| LIVE | session open and the latest bar closed within 1.5 bar intervals |
| DELAYED | session open but the latest bar is older (feed lag, failed refresh) |
| HISTORICAL | session closed: the last completed session, not the live market |
| SIMULATED | paper outcomes and synthetic data |

Currency is separate from quality (OK / DEGRADED / FAIL); quality still forces NO TRADE. It is shown on Home, Decision Center,
Candidate, Settings & Data Status and the website.

## Ranking evidence (display only; selection is unchanged)

Per strategy: profit factor on net R, trades per day, cost drag (gross minus net R), in-sample vs holdout (the last 30% of
trades), net R by regime at entry (causal labels), the edge gap to the winner in standard errors. Beside the ranking: how many
strategies were compared and the edge needed to stand out after that many tries (Sidak). **Parameter sensitivity** (Backtest
Workbench) re-runs a strategy with each setting at 0.8x and 1.2x and reports whether the sign survives.

## Audit trail (Positions & Orders -> Trace, Rejected & skipped)

`trade_trace` joins `ScanDecision -> RiskEvent -> Order -> Fill -> Position` by `decision_id` into one timeline in IST (risk
events are stored in UTC and converted), with expected R at entry against realised R, and lists missing links instead of hiding
them (trades older than the decision log have no scan decision). Legacy rows are shown as gross. `orphans` lists approvals
with no position, filled orders with no position and positions with no approval. "Rejected & skipped" is the first reader of
`ScanDecision` rows that did not become trades.

## Reports and onboarding

The Daily Report adds setups that did not become trades, data-quality findings from that day, a methodology section, labels
(trade results SIMULATED, expected R HISTORICAL), and a printable HTML export (open in a browser, print to PDF). The Guide page
has the reading order, the stages, a searchable glossary, a walkthrough on the current data (labelled with its currency), and
what the product cannot tell you.

## What none of this proves

* The selector's edge is still unestablished (about 2.5 months of data; see `SELECTOR_EVALUATION.md`).
* Holdout, regime breakdown and sensitivity are descriptive, use the same small sample, and cannot show an edge is real.
* The multiple-testing note is informational; it does not change which strategy is selected.
* Backtests fill at the signal bar's close with modelled option economics; paper fills are simulated.
* The live Dhan path has never run against a funded account.
