# Data quality

Every candle dataset passes `data/quality.validate_ohlcv` before it feeds a trading calculation. The verdict has two
parts, and both are stored with the data (`market_data_metadata`) and shown in the app:

* **`status`** (what gates trading): `OK`, `DEGRADED`, `FAIL`. Only `OK` may drive a new decision; anything else forces
  **NO TRADE** in the ranker (`ranking_engine.rank_and_select`) and a veto in the risk engine.
* **`category`** (why): `VALID`, `VALID_WITH_WARNINGS`, `INCOMPLETE`, `STALE`, `INVALID`, `UNAVAILABLE`.

## Rules

| Check | Result | Category |
|---|---|---|
| No rows / no data source reachable | FAIL | UNAVAILABLE (`DataUnavailableError`: there is **no synthetic fallback** at runtime) |
| A required column is missing (`timestamp, open, high, low, close, volume`) | FAIL, with the column named | INVALID |
| Timestamp missing or unparseable | FAIL | INVALID |
| Price missing (NaN), non-numeric or infinite in open/high/low/close | FAIL | INVALID. *Previously rated OK: NaN compares False with everything.* |
| Price zero or negative | FAIL | INVALID |
| High below low; open or close outside [low, high] | FAIL | INVALID (an inconsistent **first bar of a session** is dropped by `clean_ohlcv`, a known exchange opening-call artefact; the same fault elsewhere fails) |
| Fewer than 5 rows | FAIL | INVALID |
| More than 2 % of intraday bars outside the exchange session (wrong timezone / clock) | FAIL | INVALID |
| Duplicate timestamps | DEGRADED (cleaned by keeping the last) | INCOMPLETE |
| Intra-session gap larger than 1.5 x the bar interval (overnight, weekend and holiday gaps are expected and **not** flagged) | DEGRADED | INCOMPLETE |
| Negative, missing or non-finite volume | DEGRADED | INCOMPLETE |
| **Stale**: last bar older than 30 minutes behind the latest bar the exchange calendar says should exist (`utils/market_calendar`) | DEGRADED | STALE |
| Rows out of chronological order | warning (the data is sorted before use); status stays OK | VALID_WITH_WARNINGS |
| Steps shorter than half the bar interval inside a session (data not at the requested timeframe) | warning | VALID_WITH_WARNINGS |

Staleness is relative to the **market clock**, not wall-clock time, so Saturday's data is not "stale" if Friday's last
bar is present. Freshness is judged again at order time by the risk engine (`quote_time`, 15-minute limit), independently of
this gate: a last bar of 15:25 is research-fresh on Saturday but too old to place an order on.

## What happens to data that is not OK

* **Never silently substituted.** No data is interpolated, filled forward, or replaced by another instrument's data.
* A not-OK pipeline result is shown with its issues (Dashboard, Decision Center, website) and the decision is NO TRADE,
  with the issues appended to the reason.
* Cached candles are reused only after validation; an unreadable cache file is treated as missing and re-fetched
  (`data_manager._read_cache`), never read as partial data. Cache writes are atomic (temp file then replace, with a short
  retry for Windows file locks).

## Not covered (stated plainly)

* **Instrument identity of files written before stamping existed.** Cache files now carry an identity stamp (their own
  `instrument_timeframe` name) and a file whose contents say otherwise (copied or renamed) is refused and re-fetched.
  Older files have no stamp and are accepted; they are stamped the next time they are written.
* **Corporate actions and price scale changes** (splits, contract rollovers) are not detected.
* **Cross-source reconciliation** (comparing two providers) is not performed.
* **Volume** is unavailable for some index series; volume-based strategies are skipped for those, not fed zeros.

Tests: `tests/test_data_quality_gate.py` (37 cases including a property test that no invalid price can be rated OK),
`test_market_calendar_quality.py`, `test_data_accuracy.py`, `test_data_fixes.py`, `test_persistence_failures.py`.
