# Reliability audit

Scope: correctness of calculations, data integrity, risk fail-closed behaviour, backtest validity, external-API and
persistence resilience, application stability. Companion documents (not repeated here): architecture and module map in
`PROJECT_AUDIT.md`; formulas in `CALCULATION_INVENTORY.md`; data rules in `DATA_QUALITY.md`; strategy detail in
`STRATEGY_AUDIT.md`; execution safety in `RISK_AND_EXECUTION_AUDIT.md`; test results in `TESTING.md` and `TEST_BASELINE.md`.
Severity: **P0** critical (wrong decisions, bypassable risk, unsafe orders, invalid data acted on) · **P1** high ·
**P2** medium · **P3** low. "Fixed" always means a regression test exists.

## 1. Critical execution paths and their guards

```
candles -> validate_ohlcv (finite prices, schema, timestamps, freshness)      [FAIL/DEGRADED => NO TRADE]
        -> features (causal) -> regime -> backtests -> analogues -> ranking     [insufficient evidence => NO TRADE]
        -> setup -> size_position (rejects NaN/inf/zero)                         [invalid input => quantity 0]
        -> evaluate_trade  (account valid? order valid? fresh? spread? limits?)  [any error or unaudited approval => VETO]
        -> approval registered (single use, 120 s, bound to instrument/direction/quantity/prices)
        -> PaperBroker.place_order (one position per approved decision)  |  DhanBroker.place_order (live; locked by default)
        -> persistence (logged on failure) -> monitor -> close (gross, costs, net) -> reports
```
Every arrow was exercised by `tests/test_end_to_end.py`, which uses the real quality gate, features, ranking, risk
engine, paper broker and database, and forces only the moment a setup triggers.

## 2. Defect register

### Found and fixed in this engagement

| Sev | Defect | Fix | Regression test |
|---|---|---|---|
| **P0** | A candle with a NaN, infinite or non-numeric price was rated **OK** by the data gate (NaN compares False with every check); a missing column raised `KeyError` | finite-price, schema and timestamp checks; verdict `INVALID` | `test_data_quality_gate.py` (incl. property test) |
| **P0** | Risk engine had no check of order validity, stale price, spread/slippage, or account state; an evaluation error could surface as an exception | validity, freshness, spread vetoes; `_invalid_account_reason`; any exception becomes a veto | `test_risk_engine.py`, `test_execution_safety.py` |
| **P0** | Live broker did not require a risk approval, nor that the order matched the approved trade; approvals could be replayed | single-use approval bound to instrument, direction, quantity, entry/stop/target | `test_execution_safety.py` |
| **P0** | A live order timeout raised an unhandled exception while the order might have been placed | outcome `UNKNOWN`, all further live orders blocked until a human checks | `test_execution_safety.py`, `test_api_resilience.py` |
| **P0** | Live daily-loss limit read the paper engine's P&L; missing broker P&L counted as zero; live peak equity overwrote the paper engine's | broker-derived P&L, fail closed when absent, separate peak | `test_execution_safety.py` |
| **P0** | An approval could be returned although its audit row failed to save (and an unserialisable numpy value in the row would have made every approval fail once that was enforced) | approval only after a successful audit write; audit details made JSON-safe | `test_risk_engine.py` |
| **P1** | RSI returned 100 ("maximally overbought") for flat prices (0/0): false RSI(2) SHORT signals | undefined (NaN) | `test_calculations.py` |
| **P1** | Max drawdown measured from the first trade, not the start, in five places: early losses under-reported, which also feeds the ranker's drawdown penalty | one shared function counting the start as a peak | `test_calculations.py` |
| **P1** | Look-ahead in `opening_range_*` (whole-session aggregate) | running range | `test_chronology.py`, `test_features.py` |
| **P1** | Stops filled at the stop price through gaps (backtest and both monitors) | worse of stop and open | `test_backtest_integrity.py`, `test_execution_safety.py` |
| **P1** | Paper `net_pnl` carried no costs | gross, costs, net booked together; new columns, backward compatible | `test_execution_safety.py`, `test_end_to_end.py` |
| **P1** | **Regression introduced earlier in this engagement, found by code reading (not by a failing test) and then pinned by one:** after `net_pnl` became net, the Positions page still treated it as gross and subtracted charges again, double-counting costs on every newly closed trade | `costs IS NOT NULL` marks current-convention rows; legacy rows unchanged; CSV importer follows the same rule | `test_end_to_end.py` (stored row -> frame -> stats) |
| **P1** | Logging wrote caller-supplied details unscrubbed to the log file and the database | `redact()` for JWTs and token/secret/password/PIN fields | `test_diagnostics.py` |
| **P1** | Restoring trade history from an exported file converted legacy rows into the new P&L convention, so a restore no longer matched what had been stored (caught by an existing test) | each row is restored in its own convention (`charges_booked` column) | `test_history_tools.py` |
| **P1** | The Daily Report page crashed (`KeyError`) on a stored report with missing fields (older version or fixture): stored reports outlive code versions | `eod.is_complete`; an incomplete report is rebuilt with a visible notice | `test_pages_degrade.py` |
| **P2** | A cache file copied or renamed into another instrument's slot would have been served as that instrument | files carry an identity stamp; a mismatch is refused (legacy files still readable) | `test_persistence_failures.py` |
| **P2** | A reply from Dhan without a usable order id or a recognised status was reported as a normal acknowledgement | explicit order-state model (`brokers/order_state.py`): anything unrecognised is `SUBMISSION_UNKNOWN`, never a fill, and blocks further live orders | `test_order_state.py` |
| **P2** | The option chain formulas (put-call ratio, max pain, IV skew, unusual volume) had no independent tests | hand-computed reference tests (formulas were correct) | `test_chain_formulas.py` |
| **P2** | `size_position` crashed (`int(NaN)`) on a NaN account equity | refuses invalid inputs, clear reason | `test_calculations.py`, `test_end_to_end.py` |
| **P2** | Dhan client: raw `requests` exceptions escaped, no retry for 5xx, no jitter, a 200 with an HTML body was read as empty data | bounded jittered exponential retry for reads; orders never retried; non-JSON is an error | `test_api_resilience.py` |
| **P2** | Open trades at the end of the data booked as closed; wrong-side targets simulated as wins; Sharpe/Sortino annualised per trade; realised vol hard-coded to 5-minute bars | excluded / skipped / daily basis / inferred bar length | `test_backtest_integrity.py`, `test_calculations.py` |
| **P2** | Windows `PermissionError` on cache replace/read while another thread had the file open | bounded retry | `test_data_fixes.py` |
| **P2** | Persistence failures swallowed (`except: pass`); the same approved decision could open two paper positions | logged at ERROR; duplicate decision refused | `test_persistence_failures.py` |
| **P3** | Payoff ratio counted break-even trades as losses; option P&L ignored the premium floor | corrected | `test_calculations.py` |

### Open

| Sev | Issue | Why not fixed |
|---|---|---|
| **P1** | The selector's edge is **unestablished**; only ~2.5 months of data (`SELECTOR_EVALUATION.md`) | needs longer history, not code |
| **P1** | Backtests fill at the signal bar's close and model option premium and costs; no real option prices | needs data; documented in every run's `conventions` |
| **P1** | Live path never run against a funded account; **no order-status lookup / reconciliation against Dhan and no broker idempotency key is implemented**: an unknown outcome blocks further live orders and a human must check the order book; no broker-side stops. The lifecycle model and its safe defaults exist, but the status strings follow Dhan's documentation and were not exercised live | cannot be verified without a live account; calling unverified endpoints would be speculation |
| **P2** | `translate_setup` assumes delta 0.5 when a contract's delta is missing (now logged as a WARNING and flagged on the setup as `delta_assumed`, but still applied) | refusing the trade instead alters live behaviour: owner decision |
| **P2** | Bollinger bands use the sample std (textbook: population) | changing it alters signals: owner decision (pinned by a test) |
| **P2** | `expiry_proximity_days` is a weekday heuristic (calendar days, "next Thursday") | needs confirmation of the exchange's current expiry day |
| **P2** | Margin for written options, SPAN, and equity statutory charges are not modelled | scope |
| **P3** | statsmodels 0.15 reports a rank-deficient regression (`SingularMatrixWarning`) in the optional HAR-RV volatility fallback (`volatility/har_rv.py`); those parameters are not uniquely determined. Volatility models are off by default | needs a model decision (regularise or drop the proxy path) |
| **P2** | `ENGINE` / `COMMODITY_RUNNER` singleton thread safety not audited; the Black-Scholes premium model (off by default) has no independent test | scope |

## 3. Application stability

All 19 page entries (router, dashboard, 00-16) execute headlessly with no exception (`scripts/ui_smoke_test.py`), and
**also with the research pipeline forced offline** (`tests/test_pages_degrade.py`): no page raises, and each page that
needs the pipeline tells the user why there is no data. Every non-test module imports without error (134 modules). A diagnostics section on **Performance & Reports > System Health**
reports dependencies, execution mode, credential state, API (only on demand), market-data freshness, candle cache,
database, unresolved live order and recent warnings, each as OK / WARN / FAIL / NOT_TESTED.

## 4. Environmental notes

pandas 3.0.2 and numpy 2.5.2 are far newer than the declared minimums (`pandas>=2.1`); the suite was run on those. Older
versions were **not** tested. Hypothesis is a development-only dependency (`requirements-dev.txt`).
