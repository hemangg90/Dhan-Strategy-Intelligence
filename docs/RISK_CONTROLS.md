# Editable risk controls

Code: `risk/limit_overrides.py`, `pages/09_Risk_Control.py`. Tests: `test_limit_overrides.py`.

The risk limits (`SETTINGS.risk`) start from environment variables. The Risk Control page can now change ten of them
while the app runs. Every read site (`risk_engine`, `auto_trader.size_position`, `risk_budget`) reads `SETTINGS.risk` on each
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
