# Demo runbook: Nifty Intelligence

Every figure shown is a **historical statistic, not a prediction**. Paper trading only; live orders are locked.
If asked: the selector has **not** been shown to beat a random pick (see `docs/SELECTOR_EVALUATION.md`: 55 trades, mean
+0.011 R, 95 % CI -0.29 to +0.34). Say so; it is the honest result and the strongest thing you can say about the project.

## 1. The day before and the morning of

1. **Renew the Dhan token** (they last about 24 h; the check below tells you when it expires) and put it in `.env`
   as `DHAN_ACCESS_TOKEN`. If you renew the evening before, it can expire around the time you present: **renew again on the
   morning of the demo**.
2. From the project root:

   ```
   python scripts/demo_preflight.py
   ```

   It must end with `PREFLIGHT OK`. It checks the lockdown flag, the closed live-order gate, token validity and time to
   expiry, database access and how old the cached NIFTY / BANKNIFTY candles are. It never prints credentials.
3. **Market hours:** NSE is closed on Saturdays. On a closed day the app correctly shows the **last session's** data
   ("last bar ... N hours old"). It is still a valid demo of research, ranking, risk and reports, but there are no new
   signals and no auto-trading. Do not describe closed-market data as live.
4. Close other windows. Hide the browser bookmarks bar and saved-password prompts.

## 2. Start (PowerShell, project root)

```
$env:DEMO_LOCKDOWN = "1"
streamlit run quant_intelligence/app.py
```

Open http://localhost:8501. **Open it once about two minutes early**: the research pipeline (about 40 s) is computed
once in the background at startup and then shared, so later refreshes and new tabs load immediately.

`DEMO_LOCKDOWN=1` forces paper mode regardless of `.env`, blocks `DhanBroker.place_order`, and hides the client ID, the
token (not even its last characters) and the credential form.

**Alternative: the read-only website** (top navigation: Dashboard, Strategies, Backtests, Risk, Method):

```
$env:DEMO_LOCKDOWN = "1"
python -m uvicorn quant_intelligence.web.app:app --port 8000
```

It has no order endpoints at all. Open http://localhost:8000.

## 3. Five-minute sequence (Streamlit top bar: five workspaces)

| Min | Where | Say / show |
|---|---|---|
| 0:00 | **Market Intelligence > Dashboard** | Chips: data quality, PAPER MODE, LIVE ORDERS LOCKED. Last bar and its age. "The system first asks: is the data valid and current?" |
| 0:45 | Dashboard, decision banner | NO TRADE / SELECTED / **TIE-BREAK** with the actual reason. If TIE-BREAK: "the top two are statistically tied, so this is flagged and sized down; it carries no demonstrated edge." A decision is *not* an order. |
| 1:30 | **Strategy Intelligence > Decision Center** | Why each of the 17 strategies is selected, eligible or rejected: net expectancy, standard error, win rate, sample size, distinct days. The four option-chain strategies say "needs a live option chain". |
| 2:30 | **Research Lab > Historical Analogues**, then **Backtest Workbench** | "How did this strategy behave in similar past states?"; stored runs, equity curve, drawdown, daily returns. |
| 3:30 | **Risk & Execution > Risk Control** | The independent risk engine: it never sees the ranking; any failed mandatory check means NO TRADE. Show a recent verdict and reason. |
| 4:15 | **Performance & Reports > Daily Report** / Research Reports | Real stored results, traceable. |
| 4:45 | Close | Limits: no proven edge, modelled option costs, paper only. |

To show a risk veto without trading: **Risk & Execution > Paper Trading**, size a trade far above the per-trade risk
limit; the engine vetoes it with the exact reason.

## 4. If something fails (fallbacks)

| Symptom | Cause / fix |
|---|---|
| "Data DEGRADED", "stale", NO TRADE for data reasons | Token expired or feed down. Renew the token, press **Refresh pipeline** (sidebar). This is also a good demonstration that the system refuses to trade on bad data: say so. |
| "Your Dhan access token expired" | Generate a new token, update `.env`, restart the app. |
| Rate-limit banner | Dhan paused requests; wait a minute, press **Refresh pipeline**. Cached candles keep working. |
| Page stuck on "Running initial pipeline" > 90 s | The first computation is still running or the network is slow. Wait; or switch to the website (`:8000`), which serves the last computed result. |
| Port busy | Add `--server.port 8502` (Streamlit) or `--port 8001` (website). |
| `ModuleNotFoundError` | Run from the project root with the virtualenv active: `pip install -r requirements.txt`. |
| No internet at the venue | The app and website work from cached candles (the website serves Plotly from the installed package). Data will be as old as the cache and labelled so. |
| Complete failure | Show `docs/SELECTOR_EVALUATION.md` and `docs/PROJECT_AUDIT.md` on screen: real, traceable results and a candid account of what is and is not validated. |

## 5. What not to claim

* Not "validated", not "profitable", not "predicts the market". The selector's advantage over chance is **not established**.
* Not "live trading ready": the live path has never run against a funded account.
* Backtest option costs and premiums are **modelled**, not real fills.
* Regime figures are heuristic scores, not calibrated probabilities.

## 6. Known limitations

See `README.md` ("Known limitations") and `docs/PROJECT_AUDIT.md` section 5 for the complete, current list.
