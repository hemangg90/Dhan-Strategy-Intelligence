# Risk and execution audit

Nothing in this audit placed, or could place, a real order: tests use `PaperBroker` or a mocked `DhanApiClient`; no
live account was contacted. **Live trading has never been verified against a funded account** and is not claimed to work.

## 1. The real execution flow (traced in code)

```
research/pipeline.run_pipeline -> RankingDecision (selected | NO TRADE)
        |
        |  execution/auto_trader.run_auto_option_cycle / run_auto_equity_cycle   (automatic, PAPER only)
        |  pages/07_Paper_Trading.py                                            (manual, PAPER)
        |  pages/12_Live_Options_Trading.py                                     (manual, LIVE, double-gated)
        v
 tradable(decision, mode)  ->  detect_setup  ->  size_position  ->  ProposedTrade
        v
 risk/risk_engine.evaluate_trade(account, trade)   <- sees NO ranking input
        |  veto -> reason logged to risk_events, flow stops
        |  approve -> decision_id registered (single use, 120 s)
        v
 broker.place_order(OrderRequest(decision_id=...))
        PaperBroker  (brokers/paper_broker.py)   - simulated fill, persisted
        DhanBroker   (brokers/dhan_broker.py)    - LIVE: gates below, then POST /orders (never auto-retried)
        v
 execution/position_monitor (stop / target), execution/square_off (end of day), reports/*
```

**Order-submission paths (complete list, from `grep place_order`):** `execution/auto_trader.py` x2 and
`pages/07_Paper_Trading.py` (PaperBroker only); `pages/12_Live_Options_Trading.py` (DhanBroker). The automatic traders are
constructed with a `PaperBroker` in `execution/engine.py` and nothing else: **they cannot reach a live broker**
(verified: `DhanBroker` is imported only by page 12). The FastAPI website (`web/`) has no order endpoint
(`tests/test_web_service.py` asserts POST/PUT/DELETE are rejected and no route contains "order").

## 2. Mandatory risk controls

Status of every control in your list. "Enforced" means every order path above passes through it.

| Control | Status | Where | Evidence |
|---|---|---|---|
| Maximum risk per trade | WORKING, enforced | `evaluate_trade`; sizing in `auto_trader.size_position` | `test_risk_engine`, `test_risk_budget` |
| Daily loss limit | WORKING (paper). **Live fixed this session** | `evaluate_trade`; live P&L from `ui/state.get_live_account_state` | live path previously used the *paper* engine's P&L and overwrote its peak equity; now uses the broker's position rows and fails closed if absent (`daily_pnl_known`); `test_execution_safety` |
| Portfolio / strategy / group exposure | WORKING | `evaluate_trade` | `test_risk_drawdown`. Exposure is *stop-risk* based, not notional/margin; live positions carry risk 0 (only the count limit applies) |
| Drawdown limit | WORKING (paper). Live: peak is now tracked separately from the paper engine | `evaluate_trade`, `risk_multiplier` | `test_risk_drawdown` |
| Losing-streak / profit-lock / per-strategy loss | WORKING | `evaluate_trade` | `test_risk_drawdown` |
| Liquidity (relative volume) | WORKING, **fails open when volume is unknown** | `evaluate_trade` | a missing `relative_volume` passes (documented limitation; index candles) |
| **Data quality / freshness** | **Added this session** | `evaluate_trade` (quote age <= 15 min, quality `OK`) | `test_risk_engine`; independent of the ranker's own data veto |
| **Valid entry / stop / target / quantity / lot** | **Added this session** | `_invalid_order_reason` | one test per failure mode |
| **Spread and slippage** | **Added this session**, enforced **only when bid/ask is supplied**; a supplied but unusable quote fails closed | `_spread_slippage_reason` | callers do not yet pass bid/ask, so in practice this check is dormant: **INCOMPLETE** |
| Kill switch, broker connectivity | WORKING | `evaluate_trade` | tests |
| Duplicate-order prevention | **Added this session (live)**: a risk approval is single-use | `risk_engine.consume_approval` | `test_execution_safety` |
| Broker-state reconciliation | **NOT IMPLEMENTED** | | after an unknown order outcome the broker blocks further orders until a human checks the order book; there is no automatic reconciliation against Dhan's order book |

**Risk veto cannot be overridden by the selector.** `evaluate_trade` takes only `AccountState` and `ProposedTrade`
(no ranking input; `test_the_veto_is_final_whatever_the_ranker_said`). Every caller returns on a veto before building an
order. **New defence in depth:** `DhanBroker.place_order` now refuses any order that does not carry a fresh, unused
approval issued by `evaluate_trade` in the same process (`consume_approval`), so skipping the engine or replaying an
approval is impossible by construction on the live path. `PaperBroker` does not require an approval (many tests and the
manual paper page place orders directly); every real caller still runs the engine first.

## 3. Position sizing (units checked)

`size_position`: `budget = equity x max_risk_pct x 0.9 x risk_multiplier`; `quantity = budget / stop_distance`
(**premium** stop distance for options), floored to whole lots, then capped by `max_capital_per_trade_pct` of equity
(premium x lot). Zero or non-positive stop distance or equity returns 0 (`NO SIZE:` reason in the decision log).
Verified by `test_risk_budget`. **Not modelled:** margin for *written* options, SPAN, contract-specific tick rounding
beyond one `tick_size`, partial fills.

## 4. Paper trading

| Item | Finding |
|---|---|
| Fills | One adverse tick of slippage on entry; **was**: none on exit and **no costs**, yet the result was stored as `net_pnl`. **Fixed**: `close_position` now subtracts brokerage, STT/exchange/SEBI/stamp/GST (option positions) or brokerage + GST (others) and one exit tick; stores `gross_pnl`, `costs`, `costs_model`. Equity/futures statutory charges are not modelled (`costs_model = "BROKERAGE_ONLY"`). |
| Stops / gaps | **Fixed**: both monitors recorded the exit at the stop price even if price had already gone through it (bar open, or a sampled premium). They now exit at the worse of the two. Targets still fill at the target (conservative). |
| Persistence | `except Exception: pass` around order/fill/position/risk-event writes hid failures. **Fixed**: now logged at ERROR; the in-memory simulation still continues. |
| Restart recovery | `tests/test_square_off_restore.py` (pre-existing) covers it. WORKING. |
| Display vs records | Reports read `Position.net_pnl`; now gross-minus-costs consistently. Previously stored positions are unchanged (gross-labelled-net). |

## 5. Live-trading safeguards

Gates, in order, in `DhanBroker.place_order` (every one tested without network):

1. `Settings.live_mode_fully_authorized`: `TRADING_MODE=LIVE` **and** `TRADING_LIVE_CONFIRM=YES_I_UNDERSTAND_THE_RISK` **and not** `DEMO_LOCKDOWN`.
2. No earlier order left in an unknown state (`unreconciled_order`).
3. `security_id` present.
4. A fresh, unused risk approval for `decision_id`.
5. POST `/orders` with `retry=False` (`DhanApiClient.place_order`): never auto-retried.
6. A transport failure (`requests.RequestException`) returns status `UNKNOWN` with an instruction to check Dhan's order book, **and blocks every later order** until the app is restarted. Previously such an exception propagated unhandled out of the page while the order may have reached Dhan.

Page-level: warning banner, an explicit "I have validated this in Paper Trading" checkbox, a manual submit button,
`tradable(..., "LIVE")`, and the risk engine. **`TIE_BREAK_LIVE` now defaults to off** (it was on, contradicting the
README's own advice; a tie-break pick has no demonstrated edge). A rejected/blocked order no longer consumes a daily
trade slot.

**Machine-specific risk (not a code defect, but real):** this machine's `.env` has `TRADING_LIVE_CONFIRM` already set, so
flipping the single variable `TRADING_MODE=LIVE` would open gate 1. For any demo or screen-share set `DEMO_LOCKDOWN=1`;
consider removing the confirmation phrase from `.env` and typing it only when you intend to go live.

## 6. Unverified or open

* The live Dhan path has never run against a funded account; response field names used for live P&L
  (`realizedProfit`, `unrealizedProfit`) and balance (`availabelBalance`) are taken from Dhan's documented API, **not
  exercised live**. If the fields are absent the engine now vetoes (fail closed) rather than assuming zero.
* No bracket / stop-loss order is placed at the broker; live exits are not monitored by this software (page 12 places a
  market entry only). A live position has **no automatic stop**.
* `ENGINE` / `COMMODITY_RUNNER` are process-wide singletons used from background threads and the UI; their internal
  locking was not audited.
* Margin, SPAN, and charges for equity/futures are not modelled.
