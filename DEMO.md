# Demo guide: Nifty Intelligence (Dhan Dev)

All results shown are **historical statistics, not predictions**. The walk-forward evaluation in this repo has not
shown the selector beating a random pick; say so if asked. Orders are paper-only; live placement is locked.

## Before going on stage

1. Generate a fresh Dhan access token (tokens last about 24 h) and enter it in `.env` (`DHAN_ACCESS_TOKEN`).
   **The token in `.env` expired on 30 Sep**; until it is renewed the app shows 16-day-old cached bars and NO TRADE.
2. Run the checklist from the project root:

   ```
   python scripts/demo_preflight.py
   ```

   It must end with `PREFLIGHT OK`. It checks the lockdown flag, the closed live-order gate, token validity,
   database access, and how old the cached NIFTY / BANKNIFTY candles are. It never prints credentials.
3. Warm the data: open the app once (below) about 10 minutes early so the candle cache refreshes from Dhan.
4. Close other windows and hide the browser's saved-password and bookmarks bars.

## Start command

PowerShell, from the project root:

```
$env:DEMO_LOCKDOWN = "1"
streamlit run quant_intelligence/app.py
```

Open http://localhost:8501. `DEMO_LOCKDOWN=1` forces paper mode regardless of `.env`, blocks `DhanBroker.place_order`,
and hides the client ID, token and credential form.

## Website (recommended for the stage)

A read-only website (top navigation: Dashboard, Strategies, Backtests, Risk, Method) runs on the same research core.
It has no order endpoints at all.

```
$env:DEMO_LOCKDOWN = "1"
python -m uvicorn quant_intelligence.web.app:app --port 8000
```

Open http://localhost:8000. The default view is computed in the background at startup (about 30-40 seconds), so start
it a few minutes early; later refreshes never block the page. The Streamlit app above remains as a fallback.

## Five-minute sequence

| Min | Page | Say / show |
|---|---|---|
| 0:00 | Status bar (top) | Market open/closed, Dhan connection, newest bar time. "Paper mode, live orders locked." |
| 0:30 | **Decision Center**, section 1 | Instrument, last bar, data age, data quality. If data is stale the app says so and refuses to trade. |
| 1:15 | Decision Center, section 2 | The decision. NO TRADE is a first-class outcome and it shows its reason. |
| 2:00 | Decision Center, section 3 | Why each of the 17 strategies is selected, eligible or rejected: net expectancy, standard error, win rate, sample size, distinct days, drawdown. Chain-aware strategies are marked as unrankable (no historical setups). |
| 3:00 | Strategy Intelligence | Pick one strategy: global (gross) vs conditional (net) statistics. Historical analogues page for the matched past states. |
| 3:45 | Decision Center, section 4, then Risk Control | The independent risk engine: it never sees the ranking. Show a recent veto reason from Risk Control. |
| 4:30 | Research Reports / Backtest Lab | In-sample vs out-of-sample split and the honest "no proven edge versus random" result. |

To show a risk veto live without trading, use Paper Trading and size a trade far above the per-trade risk limit; the
engine vetoes it with the exact reason.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Everything says NO TRADE, "data is stale" | Token expired or market closed. Renew the token, press **Refresh pipeline** in the sidebar. |
| "Your Dhan access token expired" | Generate a new token in Dhan, update `.env`, restart the app. |
| Rate-limit banner in the sidebar | Dhan paused requests; wait a minute and press Refresh pipeline. The cache keeps working. |
| Port 8501 busy | `streamlit run quant_intelligence/app.py --server.port 8502` |
| `ModuleNotFoundError` | Run from the project root with the virtualenv active; `pip install -r requirements.txt`. |
| Page shows "Pipeline could not run" | No cached data and Dhan unreachable. Fix the network or token; there is intentionally no synthetic fallback. |

## Verification record (this build)

- Baseline before changes: 532 tests passed; all 17 pages rendered without exceptions (`scripts/ui_smoke_test.py`).
- The suite and smoke test are re-run after changes; see the final report for the current counts.
- No live trades are ever placed by the test suite; `DhanBroker` is mocked.

## Known limitations

- Live-Dhan-only: if the token or network fails, the demo shows NO TRADE on stale data (by design).
- No demonstrated edge over a random pick; tie-break picks carry no evidence of an edge.
- Option premiums in backtests are modelled (fixed percentage / delta or Black-Scholes), not real option history.
- PCR, OI-buildup, IV-crush and unusual-volume strategies have no historical setups and are never ranked.
- Regime probabilities are not calibrated.
- The live Dhan order path has not been exercised against a funded account.
