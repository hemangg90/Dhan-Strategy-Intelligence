# Reliability changelog

Newest first. "Compat" notes anything that changes behaviour or stored data. Test counts are in `TESTING.md`.

## Calculations
* `analytics/metrics.py` (new): one definition each of `max_drawdown` (the start counts as a peak), `profit_factor`,
  `payoff_ratio`, `expectancy`, `annualised_sharpe`; undefined cases return `None`. Used by the backtest engine, the
  evaluator, the Positions page, the workbench and the daily-series report. *Compat:* drawdown figures can be larger (more
  conservative) for strategies that open with losses; payoff ratio no longer counts break-evens as losses.
* `features/feature_engine.py`: RSI is `NaN` (not 100) when there is no price movement; realised volatility annualises by
  the bar length inferred from the data (`infer_bar_minutes`; identical for 5-minute data); opening range is causal.
* `backtesting/engine.py`: stops that gap fill at the worse open; trades open at the end of the data are excluded
  (`n_unresolved_excluded`); wrong-side targets are skipped (`n_invalid_setups_skipped`); Sharpe/Sortino from daily net R;
  option P&L respects the premium floor; `profit_factor` added; every run records its `conventions`.
* `execution/auto_trader.size_position`: refuses non-finite or non-positive inputs (it used to crash on NaN).

## Data
* `data/quality.validate_ohlcv`: schema, timestamp, finite-price and interval checks; `category` alongside `status`; a
  missing column is a verdict, not a `KeyError`. *Compat:* data with NaN/inf prices is now `FAIL` (it was `OK`).
* `data/data_manager`: bounded retry on `PermissionError` when replacing or reading cache files on Windows; cache files
  carry an identity stamp and a mismatching file is refused.
* `options/premium_model`: a missing delta is logged and flagged (`delta_assumed`) instead of silently defaulted.

## Risk and execution
* `risk/risk_engine`: order-validity, stale-price, spread/slippage and account-state vetoes; any exception is a veto;
  approval only after the audit row is stored (JSON-safe); approvals are single use, expire after 120 s and are bound to
  the approved trade; `daily_pnl_known` fails closed.
* `brokers/dhan_broker`: requires a matching approval; transport failure returns `UNKNOWN` and blocks further orders.
* `brokers/dhan_api_client`: jittered exponential backoff, bounded retries for reads on timeout/connection/5xx, never for
  orders; non-JSON success bodies are errors.
* `brokers/paper_broker` + `paper_costs.py` (new, reuses `execution/charges.py`): closes book gross, costs and net;
  one position per approved decision; persistence failures are logged.
* `execution/position_monitor`: gap-aware stop exits.
* `ui/state.get_live_account_state`: separate live peak equity; P&L from the broker, never the paper engine.
* `config/settings`: `DEMO_LOCKDOWN`; `TIE_BREAK_LIVE` defaults to off.

## Order lifecycle
* `brokers/order_state.py` (new): `OrderState` (created, submission attempted, acknowledged, open, partially filled, filled,
  rejected, cancelled, **submission unknown**), `from_broker_status`, `can_transition`, `may_submit_again`; `OrderAck.state`.
  A missing or unrecognised status is *unknown*, never a fill. `DhanBroker` treats a reply without an order id or a
  recognised status as an unknown outcome and blocks further live orders.

## Persistence and schema
* `database/models.Position`: new nullable columns `gross_pnl`, `costs`, `costs_model`, added automatically at start-up
  (`_sync_schema`, `ALTER TABLE ADD COLUMN`). *Compat:* existing rows are untouched. **Convention:** `costs IS NOT NULL` =
  current rows (`net_pnl` is net); `costs IS NULL` = legacy rows (`net_pnl` holds the gross). `ui/position_views` and
  `reports/history_io` honour both.

## Observability
* `utils/logging_utils.redact`: JWTs and token/secret/password/PIN fields are removed from log lines and stored events.
* `diagnostics.py` + a Diagnostics section on System Health: OK / WARN / FAIL / NOT_TESTED per component.

## Application
* `app.py` is a navigation router (five workspaces); the dashboard is `home_page.py`; `research/shared_cache.py` shares
  pipeline results across sessions (stale-while-revalidate, no duplicate computation during warm-up).
* `web/`: optional read-only FastAPI website (lifespan start-up; no order endpoints).

## Reports
* `reports/eod.is_complete` and `pages/15_Daily_Report.py`: a stored daily report missing fields (saved by an older version)
  is detected and rebuilt with a visible notice instead of raising `KeyError`.
* `ui/position_views.positions_frame` has a `charges_booked` column, and `reports/history_io` restores each exported row in
  its own P&L convention (booked or legacy), so a restore reproduces what was stored.

## Tests and tooling
* New: `test_calculations`, `test_chronology`, `test_selector_chronology`, `test_backtest_integrity`,
  `test_execution_safety`, `test_api_resilience`, `test_data_quality_gate`, `test_persistence_failures`,
  `test_diagnostics`, `test_end_to_end`, `test_web_service`, `test_decision_view`, `test_dashboard_ui`,
  `test_demo_lockdown`; extended `test_risk_engine`, `test_regime`, `test_tie_break`.
* `requirements-dev.txt` (new): adds Hypothesis for property tests and the HTTP client the website tests need (`httpx`,
  `httpx2`); runtime requirements gain `fastapi`, `uvicorn`.
* New this round: `test_order_state`, `test_chain_formulas`, `test_premium_translation`, `test_pages_degrade`.

## Regime and analogue upgrade (2026-10-19)
* Regime: instrument-relative trend inputs, direction/volatility axes, clarity, persistence (`classify_series`), overlays only
  where meaningful, all-instrument scan (`regimes/universe.py`, `/api/regimes`), forward-behaviour check (`regimes/validation.py`).
* Analogues: `ANALOGUE_METHOD=robust` (default stays `legacy`): nine features, robust scaling, distance cutoff, per-day cap,
  kernel weights, bootstrap interval, informativeness test, `as_of` filter. Evidence and limits: `REGIME_AND_ANALOGUES.md`.
* Page 06 confidence text now matches the day-based rule. Evaluator takes `--analogue legacy|robust`.
* Tests: 820 -> 846 (26 new in `test_regime_analogue_upgrade.py`). Full run: 845 passed, 1 failed (Regime page timed out
  scanning 25 instruments inline); fixed by a non-blocking scan, affected tests and the page smoke test re-run green.

## Candidate workflow, provenance, audit trail (2026-10-20)
* `data/provenance.py`: LIVE / DELAYED / HISTORICAL / SIMULATED label on every result (Home, Decision Center, Candidate, website).
* `research/evidence.py`, `research/sensitivity.py`: profit factor, frequency, cost drag, holdout, regime breakdown, versus-winner,
  Sidak note; parameter sensitivity in the Backtest Workbench. Display only; selection unchanged.
* `risk_engine.preview_trade`: the real checks with no audit row and no approval (context variable, reset on error).
* `pages/00_Candidate.py` + `ui/candidate.py`: one page from data to the risk verdict, naming the failing condition.
* `reports/trace.py`: decision -> risk -> order -> fill -> position trace, orphans, rejected-decision reader; new tabs in Positions & Orders.
* Settings & Data Status and Guide pages; Daily Report gains rejected setups, data-quality findings, methodology, labels, printable HTML.
* Details and limits: `CANDIDATE_WORKFLOW.md`.
