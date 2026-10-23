# Editable risk controls

Code: `risk/limit_overrides.py`, `ui/risk_control.py`, pages `09_Risk_Control.py` (NSE) and `09_Risk_Control_MCX.py` (MCX).
Tests: `test_limit_overrides.py`, `test_market_accounts.py`.

## NSE and MCX are separate accounts

Each market has its own paper account: its own `TradingEngine` (`ENGINE` for NSE, `MCX_ENGINE` for MCX, also exported as
`COMMODITY_RUNNER`), with its own broker and cash, daily counters, drawdown peak, profit-lock peak, kill switch and risk
limits (`SETTINGS.risk` for NSE, `SETTINGS.risk_mcx` for MCX). Nothing is shared.

* `AccountState.market` ("NSE" or "MCX") selects the limit set. The risk checks, `risk_multiplier`, `size_position`,
  `risk_budget` and the Candidate preview all read `limits_for(account.market)`, so sizing and the veto always use the
  same limits.
* A trade is checked against the account of its own market: `engine_for(instrument)` (commodity underlyings are MCX).
* Starting capital: `PAPER_STARTING_CAPITAL` (NSE) and `PAPER_STARTING_CAPITAL_MCX` (defaults to the same value).
  `risk_mcx` starts with the same values as `risk`; the MCX environment values are not separate variables.
* Restart recovery filters the database by market: each account rebuilds only its own market's positions, trade count,
  daily P&L, cash and drawdown peak. The drawdown-peak reset is stored per market (`risk_peak_reset.json` for NSE,
  `risk_peak_reset_mcx.json` for MCX).
* Risk audit rows record the market (`details["market"]`); rows written before the split count as NSE.
* Pages that show both markets together (Positions & Orders, the open-positions block) combine the two accounts.
* History: before this change both markets shared one account. After it, each account starts from its own market's past
  closed trades, so the starting point of each is its capital plus that market's realised P&L.

The risk limits start from environment variables. Each market's Risk Control page can change ten of them for that
market while the app runs. Every read site (`risk_engine`, `auto_trader.size_position`, `risk_budget`) reads `SETTINGS.risk` on each
call, so a new value applies to the next evaluation. Open positions and approvals already issued (120 s, single use) are
not re-judged.

| Limit | Allowed range (0 = off where marked) |
|---|---|
| Max risk per trade | 0.05 - 2 % |
| Max capital per trade | 0 (off) or 0.5 - 25 % |
| Max daily loss | 0.25 - 5 % |
| Max drawdown | 1 - 20 % |
| Max trades per day | 1 - 30 |
| Max open positions | 0 (off) or 1 - 15 |
| Strategy exposure | 1 - 50 % |
| Portfolio exposure | 1 - 100 % (not below strategy exposure) |
| Loss-streak pause after | 0 (off) or 1 - 10 losses |
| Loss-streak pause length | 5 - 480 min |

The throttle, correlated-group, strategy loss cap and profit-lock limits, and the order-time checks (quote age, spread,
slippage), are not editable from the app. The ranges are the hard ceilings: the app refuses anything outside them.

## Rules
* **Tightening** applies at once.
* **Loosening** (a larger cap, a shorter pause, switching a check off) needs the confirmation tick and a reason, and is
  refused while the kill switch is engaged.
* Nothing can be changed under `DEMO_LOCKDOWN`.
* A change is saved (table `risk_limit_overrides`) and audited (a `LIMIT_CHANGE` risk event with old value, new value,
  direction and reason) in one transaction. If that fails the change is not applied.
* Saved values are reloaded once when the app starts (`load_persisted`, called from the engine's restore and the website).
  A saved value that is no longer valid is skipped and logged.
* "Reset to defaults" returns to the environment/default values and forgets the saved ones.

## Limits of this feature
* The app is one shared process with no user login: anyone who can open the page can change these limits. The ceilings
  and the audit trail are the safeguards. Keep `DEMO_LOCKDOWN=1` on a public app.
* With SQLite on a host with a temporary disk (Streamlit Cloud) saved changes are lost when the app restarts. Set
  `DATABASE_URL` to a hosted database, or put the values in Secrets, to keep them.

## Option-specific controls

Code: `risk/options_risk.py`, called from `risk/risk_engine.py` for any trade that carries option details
(`ProposedTrade.option`). The risk engine stays independent of ranking, validation and contract selection: it only measures and
vetoes, and a veto stops the order before it reaches a broker. Tests: `test_options_risk.py`, `test_option_pipeline.py`.

For an option trade the engine also requires, and fails closed without:

* a **quote time** and a snapshot no older than `OPTION_MAX_QUOTE_AGE_SEC` (default 120 s; ordinary prices still use 15 minutes),
* a **two-sided bid/ask** whose spread is within `OPTION_MAX_SPREAD_PCT` (default 8%; was 2% for every instrument but only checked
  when a quote happened to be supplied) and whose half-spread is at most half of the stop distance.

Limits (percent of equity, `0` = off; set in the environment; both the NSE and MCX accounts start from these values):

| Limit | Variable | Default | Measures |
|---|---|---|---|
| Premium at risk | `MAX_PREMIUM_AT_RISK_PCT` | 15 | premium paid on all open bought options plus this one |
| Premium per underlying | `MAX_UNDERLYING_PREMIUM_PCT` | 8 | the same for one underlying (concentration) |
| Net delta | `MAX_NET_DELTA_PCT` | 800 | abs(sum of delta x quantity x spot) - option notional is large, so this is a large number |
| Gamma | `MAX_GAMMA_PCT` | 0 (off) | delta-notional that changes for a 1% move |
| Vega | `MAX_VEGA_PCT` | 2 | net rupees per volatility point |
| Theta | `MAX_THETA_PCT` | 1 | net rupees of decay per trading day |
| Time to expiry | `MIN_TRADING_DAYS_TO_EXPIRY` | 0.25 | no NEW long option closer to expiry than this |
| Order size | `MAX_ORDER_LOTS` | 100 | largest single order in lots |

The defaults are starting points for a ~Rs 10 lakh paper account, not recommendations; tune them to the account.

**Greek exposure is measured at entry.** Each option position stores its per-unit Greeks and the spot when it was opened
(`Position.entry_greeks`); the portfolio totals add those up. They are not re-marked as the market moves, so a position that has
gone deep in the money still counts at its entry delta. If an enabled Greek limit cannot be checked - the proposed option's
Greeks are unavailable, or an open option position has none stored (for example one opened before this feature) - the trade is
vetoed. Short options add their Greeks with the opposite sign; margin is not modelled, so the limits are not margin limits.

**Reconciliation.** Before each paper order the engine's book is compared with the saved positions
(`brokers/reconciliation.py`); a mismatch blocks new orders (`RECONCILE_BEFORE_ORDER`). For live trading the comparison primitives
exist (`reconcile_live`) but the broker's order book is not available through the application's client, so order-level
reconciliation is not done; the live broker's lock on an order with an unknown outcome remains the guard.

**Duplicate orders.** One approved decision opens one position (the paper broker refuses a repeat of a decision id) and an
approval is single-use (the live broker only accepts a decision id this process approved moments ago).
