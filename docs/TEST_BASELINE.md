# Test baseline and results

## Environment

Windows 10 Pro, Python **3.12.10**, streamlit 1.61.1, fastapi 0.135.3, uvicorn 0.44.0. The README states Python 3.11+;
**3.11 was not tested**. No network and no broker are used by any test.

## Commands

```
python -m compileall -q quant_intelligence scripts      # syntax
python -m pytest -q                                      # whole suite
python scripts/ui_smoke_test.py                          # every page headlessly (Streamlit AppTest)
python scripts/demo_preflight.py                         # pre-demo environment checks
```
Import check: every non-test module (134) imported with no failure (run ad hoc with `pkgutil.walk_packages`).

## Baseline (before any change)

`python -m pytest -q` -> **532 passed, 0 failed, 0 skipped, 1 warning, 150.9 s**. `ui_smoke_test.py`: all 17 pages rendered
without exceptions. Compilation: clean.

The one warning was **not benign**: `PytestUnhandledThreadExceptionWarning: PermissionError [WinError 5]` raised inside the
writer thread of `test_data_fixes.py::test_a_reader_never_sees_a_torn_file_while_a_writer_replaces_it`. Classification:
**application defect (Windows concurrency)**: `data_manager._write_cache` used a bare `os.replace`, which Windows refuses
while another thread holds the file open; the test only watched for torn reads and so passed regardless. Fixed (bounded
retry on write and read) and the test now asserts the writer survives and the production reader never reports "unreadable".

## Final

`python -m pytest -q` -> **615 passed, 0 failed, 0 skipped, 2 warnings, 373.8 s** (83 more tests than the baseline).
`ui_smoke_test.py`: all 19 entries (router, dashboard, 00-16) rendered without exceptions.

The 2 warnings were FastAPI's `on_event` deprecation notice in the website's startup hook. They were removed afterwards by
switching to a lifespan handler; the website tests (7) were re-run and pass and the server was started to confirm, but
**the full suite was not re-run after that one-line change**.

### Added (83 tests)

| File | Tests | What it protects |
|---|---|---|
| `test_chronology.py` | 18 | Every feature column and every strategy's setups are unchanged when future bars are removed |
| `test_selector_chronology.py` | 3 | Rewriting all data after a cut-off does not change earlier selector decisions; the fixed baseline uses warm-up data only; summary reports samples and baselines |
| `test_backtest_integrity.py` | 12 | Stop / target / same-bar / gap / entry-bar / time-exit / data-end / invalid-geometry handling; daily Sharpe |
| `test_execution_safety.py` | 13 | Live order needs a fresh single-use risk approval; timeouts leave the outcome unknown and block more orders; paper P&L net of costs; gap-aware monitor exits; live account state (no paper contamination, fail-closed P&L) |
| `test_risk_engine.py` | +19 | Each new veto (invalid order, stale quote, spread, slippage, crossed or missing quote) and that the veto is final |
| `test_web_service.py` | 7 | JSON cleaning, stale-while-revalidate cache, no duplicate computation during warm-up, read-only website |
| `test_decision_view.py`, `test_dashboard_ui.py`, `test_demo_lockdown.py`, `test_regime.py`, `test_tie_break.py` | 4 / 2 / 2 / +2 / +1 | Explanation rows and "n/a" handling, bar rendering, lockdown closes the live gate, regime scores, live tie-break default |

## Existing tests that were changed, and why

Tests were never deleted, skipped or loosened to obtain a green run. Each change is either a strengthening or a
legitimate change of intended behaviour:

| Test | Change | Reason |
|---|---|---|
| `test_features.py::test_opening_range_is_consistent_within_session` | Replaced | It asserted that the opening range is constant for the whole session from bar 0, i.e. it **pinned a look-ahead leak**. The new test requires a running range inside the 30-minute window, fixed afterwards and equal to the true range. (`test_features_have_no_lookahead` already existed but did not catch this: its coverage was too narrow.) |
| `test_risk_drawdown.py::test_third_same_direction_index_position...` | SHORT fixture given a SHORT-shaped stop/target | The old fixture was an invalid order (stop below a short entry) that the new order-validity check correctly rejects. |
| `test_mover_selection.py` (fixture) | `timestamp=now_ist()` instead of 2031 | The freshness check correctly vetoed a bar timestamped five years in the future. |
| `test_tie_break.py` (live order test) | Registers a risk approval before each live order | Live orders now require one (new safety rule). |
| `test_tie_break.py::test_tradable_honours_the_per_mode_switches` | Defaults: paper on, live **off** | The default for `TIE_BREAK_LIVE` was deliberately changed to off. Each switch is still tested both ways. |
| `test_data_fixes.py::test_a_reader_never_sees_a_torn_file...` | Asserts writer errors == [] and uses the production reader; reader sleeps 1 ms between reads | Strengthened (see above). A reader that never releases the file starves any Windows rename, which is not how the app reads. Stable 8/8 repeated runs. |
| `test_web_service.py` (stale-cache test) | Waits for the refreshed value instead of an exact label | The first version was racy (my own test bug). |

## Failures seen during development (all resolved, none ignored)

| Failure | Category |
|---|---|
| order-validity check rejected a SHORT fixture with LONG-shaped prices | incorrect test expectation (fixed in the test) |
| option orders failed direction/stop geometry check | **application defect in my first version** (a SHORT view buys a put, so premium prices are long): fixed with `long_premium` |
| two `test_mover_selection` tests, stale 2031 timestamp | test fixture issue |
| `test_tie_break` fake pipeline output lacked `timestamp` | test infrastructure (the code now tolerates it) |
| live-order test without risk approval | intended behaviour change |
| live P&L test hard-coded a 1.5 % limit; the test environment uses another | incorrect test expectation (now derives from settings) |

## Not run / not covered

* Real Dhan connectivity, real order placement, rate limits against Dhan, real option-chain responses.
* Real market data in tests (synthetic seeded data; the real-data selector evaluation is a script, see `SELECTOR_EVALUATION.md`).
* Browser automation: pages were inspected manually via headless Chrome screenshots; there are no automated browser tests.
* Python 3.11, macOS/Linux. Streamlit Community Cloud deployment.
* Per-strategy entry-correctness tests (only smoke tests, plus the new chronology and simulator tests).
