# UI redesign: plan and record

Framework unchanged: **Streamlit** (`streamlit 1.61`). No new frontend framework was introduced. A read-only website
(FastAPI + one HTML page) exists as an *optional addition* at the owner's request; it reuses the same research code and
is not a replacement.

## 1. Page map (before: a flat sidebar list of 17 pages)

| Page | Real purpose | Workspace now |
|---|---|---|
| `home_page.py` (was `app.py`) | Current decision, ranking, regime scores | Market Intelligence |
| 01 Market State | Latest feature vector, data quality | Market Intelligence |
| 05 Regime Analysis | Regime scores over time | Market Intelligence |
| 16 NSE Heatmap | Movers, treemap, selection for auto traders | Market Intelligence |
| **00 Decision Center** (new) | Why each strategy is selected / eligible / rejected; risk checks | Strategy Intelligence |
| 02 Strategy Intelligence | Global vs conditional statistics | Strategy Intelligence |
| 03 Strategy Library | Descriptions, parameters | Strategy Intelligence |
| 04 Backtest Lab | Backtests, IS/OOS, walk-forward (now a workbench) | Research Lab |
| 06 Historical Analogues | The past setups behind today's scores | Research Lab |
| 10 Research Reports | Reports from stored data | Research Lab |
| 09 Risk Control | Limits, kill switch, risk events | Risk & Execution |
| 07 Paper Trading | Manual paper terminal | Risk & Execution |
| 13 / 14 Auto Trading | Background paper traders (NSE / MCX) | Risk & Execution |
| 12 Live Orders | Manual live ticket, double-gated | Risk & Execution |
| 08 Positions & Orders | Positions, orders, analytics | Performance & Reports |
| 15 Daily Report | Day-by-day performance | Performance & Reports |
| 11 System Health | Connectivity, logs, stale data | Performance & Reports |

Navigation is a `st.navigation(..., position="top")` router in `app.py`; each page file keeps its own content and
`apply_theme()` call, so no page was rewritten.

## 2. Design system

* Tokens live in `ui/theme.py` (surface `#f7f8fa`, card `#ffffff`, border `#e5e7eb`, ink `#111827`; status green `#059669`,
  red `#dc2626`, amber `#d97706`, blue `#2563eb`); `.streamlit/config.toml` sets the light base theme.
* Shared components: `ui/components.py` (chips, status bar, KPI tiles, P&L text, empty states, position cards) and the
  new `ui/dashboard.py` (hero header, KPI cards, decision banner, probability bars, ranked table with expectancy bars).
* Colour is never the only signal: every status chip carries an icon and text; P&L carries sign and arrow.
* Readability for a projector: base font 17 px, monospace figures, one-line strategy names, horizontally scrollable tables.
* Dense, restrained, no animation beyond Streamlit's own.

## 3. What the main surfaces now show

* **Dashboard / Decision Center**: instrument, last bar and its age, data quality and any issues, regime (labelled as a
  heuristic score), and a banner that distinguishes **NO TRADE**, **SELECTED** and **TIE-BREAK**. Under it the ranking
  table states each strategy's status with the actual reason (ineligibility text from the ranker; "needs a live option
  chain" for the four unrankable strategies). Missing values are `n/a`, never zero.
* A banner states that a selection is **not an order**: it still needs a triggered setup and an independent risk approval.
  (Candidate → validated setup → risk-approved trade → executed order remains visible across Decision Center, Paper
  Trading, Risk Control and Positions; a single combined pipeline view is **not** built yet.)
* **NO TRADE** is presented as a decision with its real reason (stale data, nothing clears the edge/confidence bars, ...),
  not as an error; data-quality problems render as warnings above it.
* **Charts**: light styling, hairline grids, labelled axes. The backtest workbench charts are computed from stored trades
  (equity from net P&L, drawdown, daily returns); the buy-and-hold benchmark is drawn **only** when cached prices cover
  the run and otherwise the page says it is absent.

## 4. Preserved

Every existing page, control, background engine, database table and test. Layouts of pages other than the Dashboard,
Decision Center and Backtest Lab are unchanged apart from the light theme.

## 5. Verified

`scripts/ui_smoke_test.py` executes every page headlessly with no exceptions. The Dashboard and Decision Center were also
rendered in a real browser (Chrome over DevTools) against live data; the website's four main pages likewise.

## 6. Not done

* A single end-to-end "candidate → approved → executed" timeline page.
* Visual review of every secondary page at projector resolution (only the main pages were inspected).
* Mobile layout for Streamlit pages (the website is responsive; Streamlit tables scroll horizontally).
* Per-chart review of the legacy charts (Market State, Regime Analysis) beyond restyling and relabelling.
