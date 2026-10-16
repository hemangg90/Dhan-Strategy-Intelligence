# Testing

How to set up, run and read the tests. History of results and the list of changed tests: `TEST_BASELINE.md`. What the
tests prove and do not prove is stated per area below.

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # Windows (use a SHORT path: see the note below)
pip install -r requirements-dev.txt
```
`requirements.txt` has the runtime packages; `requirements-dev.txt` adds `hypothesis` (property-based tests). Python 3.12 was
used; the README states 3.11+ but **3.11 was not tested**. Nothing needs credentials or network access: tests use seeded
synthetic data, a temporary database and log directory (`tests/conftest.py`), and mocked Dhan responses.

* **Windows long paths:** creating the virtualenv under a very deep folder fails during `pip install` (`OSError ... No such file
  or directory`, with a long-path hint). Put it near the drive root, e.g. `C:\nivenv`.
* **Memory:** the suite imports pandas, streamlit and statsmodels and runs page tests in-process; keep about 1.5 GB free.
  On a busy machine run it in groups (below) rather than all at once.

## Commands

| Purpose | Command | Time |
|---|---|---|
| Everything | `python -m pytest -q` | about 5-6 min |
| Fast core (no UI, no slow research) | `python -m pytest -q --ignore=quant_intelligence/tests/test_pages_degrade.py --ignore=quant_intelligence/tests/test_selector_chronology.py --ignore=quant_intelligence/tests/test_chronology.py` | not timed |
| Risk and execution safety | `python -m pytest -q quant_intelligence/tests/test_risk_engine.py quant_intelligence/tests/test_execution_safety.py quant_intelligence/tests/test_order_state.py quant_intelligence/tests/test_risk_drawdown.py quant_intelligence/tests/test_risk_budget.py` | seconds |
| Calculations | `python -m pytest -q quant_intelligence/tests/test_calculations.py quant_intelligence/tests/test_chain_formulas.py quant_intelligence/tests/test_premium_translation.py` | seconds |
| Look-ahead (chronology) | `python -m pytest -q quant_intelligence/tests/test_chronology.py quant_intelligence/tests/test_selector_chronology.py` | about 3 min |
| Every page, normal and with the pipeline down | `python scripts/ui_smoke_test.py` and `python -m pytest -q quant_intelligence/tests/test_pages_degrade.py` | about 2-3 min each |
| Pre-demo environment checks | `python scripts/demo_preflight.py` | seconds |

## Categories

| Category | Files | What it verifies | Limits |
|---|---|---|---|
| Known-answer calculations | `test_calculations`, `test_chain_formulas`, `test_premium_translation`, `test_backtest_integrity`, `test_features` | Indicators, drawdown, Sharpe, costs, R, sizing, option-chain formulas against **independently computed** references (plain loops or hand arithmetic written in the test) | The reference for the cost model shares the configured rates; the rates themselves are not verified against a contract note |
| Properties (Hypothesis) | `test_calculations`, `test_data_quality_gate`, `test_order_state`, `test_api_resilience` | Drawdown equals an exhaustive search and is never positive; RSI stays within 0-100; invalid prices are never rated OK; non-positive quantities and invalid prices are never approved; sizing is whole lots inside the risk budget and never shrinks with equity; nothing but the two fill words can mean "filled"; backoff stays within its bounds | Properties cover the generated ranges only |
| Look-ahead | `test_chronology`, `test_selector_chronology` | Every feature column and every strategy's setups are unchanged when future bars are removed; rewriting all data after a cut-off does not change earlier selector decisions | Detects these tamperings; does not prove no leak can exist |
| Data validation | `test_data_quality_gate`, `test_market_calendar_quality`, `test_data_accuracy`, `test_data_fixes` | Schema, NaN/inf/non-numeric prices, timestamps, gaps, duplicates, staleness, timezone, units | Instrument identity is by file stamp, not by price content |
| Risk engine | `test_risk_engine`, `test_risk_drawdown`, `test_risk_budget` | Normal approval and rejection, every limit, invalid orders and account state, exceptions and audit failure become vetoes | |
| Execution safety | `test_execution_safety`, `test_order_state`, `test_demo_lockdown`, `test_tie_break` | A live order needs a single-use approval matching the exact order; timeouts and unreadable replies leave the outcome unknown and block further orders; lockdown closes the live gate; unrecognised statuses are never fills | Mocked broker only: **real Dhan behaviour is not tested** |
| Failure injection | `test_api_resilience`, `test_persistence_failures`, `test_pages_degrade`, `test_diagnostics` | Timeouts, 5xx, non-JSON bodies, corrupt/empty/mislabelled cache files, interrupted writes, failed database writes, duplicate decisions, pipeline outage on every page, secrets in logs | |
| Integration / end to end | `test_end_to_end`, `test_integration_pipeline` | Quality gate -> features -> regime -> ranking -> setup -> risk -> paper execution -> persistence -> restart recovery -> reporting, with real components; the moment a setup triggers is forced | Synthetic candles; no broker |
| Application | `scripts/ui_smoke_test.py`, `test_pages_degrade`, `test_web_service`, `test_ui_logic`, `test_decision_view`, `test_dashboard_ui` | Every page renders without an exception; the website is read-only | Layout and visual quality are checked by eye, not by tests |

## Dependency notes

* The suite was run on **pandas 3.0, numpy 2.5, streamlit 1.61 / 1.66, SQLAlchemy 2.0 / 2.1**; older versions inside the declared
  ranges (`pandas>=2.1`, `streamlit>=1.35`) were not tested.
* The website tests use `fastapi.testclient`, which needs an HTTP client: `httpx` for Starlette 1.0.x, `httpx2` for newer
  Starlette. Both are in `requirements-dev.txt`; without the right one, collection of `test_web_service.py` fails with a
  clear message (this is what a first clean install hit).
* `arch`, `openpyxl`, `fastapi` and `hypothesis` are optional at runtime; a missing one is reported by the Diagnostics
  section (System Health) as a WARN, and the features that need it fail visibly.

## What a green run does not mean

It does not show that any strategy is profitable, that the selector beats chance (see `SELECTOR_EVALUATION.md`), or that the
live Dhan path works: no live account, order endpoint or real option chain is touched by any test. No real order has
ever been placed by the test suite or by this audit.
