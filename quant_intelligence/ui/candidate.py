"""Follow one candidate through every stage, from the data it rests on to the risk engine's verdict.

Pure logic (no Streamlit), so it can be tested. The same stages the product already runs are chained here and each one
reports PASS / WARN / FAIL / WAIT / SKIP and, when it stops the candidate, names the condition that failed:

  Data -> Market -> Evidence (ranking) -> Setup -> Risk (preview) -> Paper order -> Outcome

Three kinds of setup are kept apart: a HISTORICAL_SETUP (a past occurrence in the strategy's backtest), the
CURRENT_CANDIDATE (a setup that triggers on the latest closed bar) and a PAPER_TRADE (a simulated order that was placed).
The risk stage is a PREVIEW: the real checks, but nothing is recorded and no approval is issued.
"""
from __future__ import annotations

from quant_intelligence.execution.setup_detector import SETUP_TRIGGERED, WAITING_FOR_SETUP, detect_setup
from quant_intelligence.ranking.ranking_engine import tradable
from quant_intelligence.risk.risk_engine import MAX_QUOTE_AGE_SEC, ProposedTrade, preview_trade
from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES, get_strategy

PASS, WARN, FAIL, WAIT, SKIP, INFO = "PASS", "WARN", "FAIL", "WAIT", "SKIP", "INFO"
HISTORICAL_SETUP, CURRENT_CANDIDATE, PAPER_TRADE = "HISTORICAL_SETUP", "CURRENT_CANDIDATE", "PAPER_TRADE"

CHECK_LABELS = {
    "account_state_valid": "Account state is valid",
    "order_parameters_valid": "Order parameters valid (prices, side, whole lots)",
    "quote_fresh": f"Price no older than {MAX_QUOTE_AGE_SEC // 60} minutes",
    "spread_and_slippage_acceptable": "Spread and slippage acceptable",
    "daily_pnl_known": "Today's P&L known",
    "daily_loss_within_limit": "Daily loss within limit",
    "drawdown_within_limit": "Drawdown within limit",
    "profit_lock_not_triggered": "Profit lock not triggered",
    "loss_streak_cooldown_over": "Loss-streak cooldown over",
    "trade_count_within_limit": "Trades today within limit",
    "open_positions_within_limit": "Open positions within limit",
    "strategy_losses_within_limit": "Strategy's losses today within limit",
    "risk_per_trade_within_limit": "Risk per trade within limit",
    "strategy_exposure_within_limit": "Strategy exposure within limit",
    "portfolio_exposure_within_limit": "Portfolio exposure within limit",
    "group_positions_within_limit": "Correlated-group positions within limit",
    "group_risk_within_limit": "Correlated-group risk within limit",
    "liquidity_sufficient": "Liquidity sufficient",
}
LEVEL_FEATURES = {
    "VWAP": "vwap", "Opening range high": "opening_range_high", "Opening range low": "opening_range_low",
    "Previous day high": "prev_day_high", "Previous day low": "prev_day_low", "CPR pivot": "cpr_pivot",
}


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _stage(key: str, title: str, state: str, headline: str, detail: list[str] | None = None, failed: str | None = None) -> dict:
    return {"key": key, "title": title, "state": state, "headline": headline, "detail": detail or [], "failed_condition": failed}


def setup_geometry(setup) -> dict:
    """Entry / stop / target in underlying terms with the numbers a reader needs to judge them."""
    entry, stop, target = setup.entry_price, setup.stop_price, setup.target_price
    risk, reward = abs(entry - stop), abs(target - entry)
    return {
        "direction": setup.direction, "setup_time": setup.timestamp, "entry": entry, "stop": stop, "target": target,
        "risk_points": risk, "reward_points": reward, "risk_reward": (reward / risk) if risk > 0 else None,
        "invalidation": f"The idea is wrong if price trades {'below' if setup.direction == 'LONG' else 'above'} {stop:,.2f} (the stop).",
        "meta": dict(setup.meta or {}),
    }


def entry_assumptions(setup_time) -> list[str]:
    return [
        f"The signal is read on the closed bar of {setup_time:%d %b %H:%M}; the strategy decides only from data up to that bar.",
        "Backtests assume entry at that bar's close. A real order would fill later and at a different price.",
        "Option P&L in backtests is modelled from the underlying (assumed premium and delta), not taken from real option prices.",
        "Paper fills are simulated: modelled slippage and charges, not exchange fills.",
    ]


def historical_setups(intel, limit: int = 5) -> list[dict]:
    """The strategy's most recent PAST setups from its backtest, newest first, each labelled HISTORICAL_SETUP."""
    if intel is None:
        return []
    from quant_intelligence.backtesting.engine import market_for, net_r_multiple
    from quant_intelligence.research.pipeline import default_quantity

    rows = []
    market = market_for(intel.backtest.instrument)
    quantity = default_quantity(intel.backtest.instrument)
    trades = [t for t in intel.backtest.trades if t.r_multiple is not None]
    for t in sorted(trades, key=lambda t: t.setup.timestamp, reverse=True)[:limit]:
        try:
            net = net_r_multiple(t, quantity, market)
        except Exception:
            net = None
        g = setup_geometry(t.setup)
        rows.append({"status": HISTORICAL_SETUP, "time": t.setup.timestamp, "direction": g["direction"], "entry": g["entry"],
                     "stop": g["stop"], "target": g["target"], "risk_reward": g["risk_reward"], "outcome": t.outcome,
                     "exit_reason": t.exit_reason, "gross_r": t.r_multiple, "net_r": net})
    return rows


def risk_rows(decision) -> list[dict]:
    rows = [{"check": CHECK_LABELS.get(k, k.replace("_", " ")), "ok": bool(v)} for k, v in decision.checks.items() if isinstance(v, bool)]
    return rows


def build_candidate(output, account, recent_trades: list[dict] | None = None, detect=detect_setup, preview=preview_trade) -> dict:
    stages: list[dict] = []
    ranking = output.ranking
    prov = getattr(output, "provenance", None) or {}

    # 1. Data ---------------------------------------------------------------------------------------------------
    issues = [str(i) for i in (output.data_quality_issues or [])]
    if output.data_quality_status != "OK":
        stages.append(_stage("data", "Data", FAIL, f"Data quality is {output.data_quality_status}", issues,
                             failed=f"Data quality {output.data_quality_status}" + (f": {issues[0]}" if issues else "")))
    elif prov.get("mode") not in (None, "LIVE"):
        stages.append(_stage("data", "Data", WARN, f"Data is {prov['mode']}: fine for research, too old to place an order on",
                             [prov.get("mode_meaning", ""), f"Source: {prov.get('source')}; last bar {output.timestamp:%d %b %H:%M}"]))
    else:
        stages.append(_stage("data", "Data", PASS, "Data quality OK and current",
                             [f"Source: {prov.get('source', 'n/a')}; last bar {output.timestamp:%d %b %H:%M}"]))

    # 2. Market -------------------------------------------------------------------------------------------------
    axes = getattr(output, "regime_axes", None) or {}
    levels = {name: _num(output.market_state.get(feat)) for name, feat in LEVEL_FEATURES.items()}
    levels = {k: v for k, v in levels.items() if v is not None}
    stages.append(_stage("market", "Market", INFO, f"Regime: {str(output.regime_label).replace('_', ' ').title()}",
                         [f"Direction {axes.get('direction', 'n/a')}, volatility {axes.get('volatility', 'n/a')} "
                          f"({axes.get('volatility_trend', 'n/a')}); heuristic and display-only, not an input to the decision."]))

    # 3. Evidence -----------------------------------------------------------------------------------------------
    ok, why_not = tradable(ranking, "PAPER")
    selected = ranking.selected_strategy if ok else None
    top = [f"{s.strategy_name}: {'eligible' if s.eligible else 'rejected'}"
           + (f", conditional expectancy {s.expected_r:.2f} R" if s.expected_r is not None else "") for s in ranking.ranked[:3]]
    if selected:
        stages.append(_stage("evidence", "Evidence", WARN if ranking.tie_break else PASS,
                             f"{selected} ranks highest" + (" (tie-break: no clear winner)" if ranking.tie_break else ""),
                             [ranking.reason] + top))
    else:
        reason = why_not or ranking.reason
        stages.append(_stage("evidence", "Evidence", FAIL, "NO TRADE: no strategy qualifies", [reason] + top, failed=reason))

    # 4. Setup --------------------------------------------------------------------------------------------------
    setup = None
    intel = next((s for s in output.strategy_intel if s.strategy_name == selected), None) if selected else None
    if not selected:
        stages.append(_stage("setup", "Setup", SKIP, "No strategy selected, so there is nothing to look for"))
    elif selected in CHAIN_AWARE_STRATEGY_NAMES:
        stages.append(_stage("setup", "Setup", WAIT, f"{selected} needs a live option chain to look for a setup",
                             ["Load an option chain on the Paper Trading page."]))
    else:
        status = detect(get_strategy(selected), output.market_state, output.ohlcv.tail(5))
        if status.status == SETUP_TRIGGERED and status.setup is not None:
            setup = status.setup
            g = setup_geometry(setup)
            rr = f"{g['risk_reward']:.2f}" if g["risk_reward"] is not None else "n/a"
            stages.append(_stage("setup", "Setup", PASS, f"Setup triggered on the {setup.timestamp:%d %b %H:%M} bar ({CURRENT_CANDIDATE})",
                                 [f"{g['direction']} entry {g['entry']:,.2f}, stop {g['stop']:,.2f}, target {g['target']:,.2f}; "
                                  f"risk {g['risk_points']:,.2f} for reward {g['reward_points']:,.2f} (1 : {rr})", g["invalidation"]]))
        elif status.status == WAITING_FOR_SETUP:
            stages.append(_stage("setup", "Setup", WAIT, f"{selected} selected; its entry conditions are not met on the latest closed bar",
                                 [f"Latest closed bar: {output.timestamp:%d %b %H:%M}. A selection is not an order."]))
        else:
            stages.append(_stage("setup", "Setup", SKIP, "No setup status available"))

    # 5. Risk (preview) -----------------------------------------------------------------------------------------
    risk = None
    if setup is None:
        stages.append(_stage("risk", "Risk engine", SKIP, "Nothing to check until a setup triggers"))
    else:
        from quant_intelligence.research.pipeline import default_quantity

        proposed = ProposedTrade(
            strategy_name=selected, direction=setup.direction, entry_price=setup.entry_price, stop_price=setup.stop_price,
            target_price=setup.target_price, quantity=default_quantity(output.instrument),
            relative_volume=_num(output.market_state.get("relative_volume")), data_quality_status=output.data_quality_status,
            quote_time=getattr(output, "timestamp", None), instrument=output.instrument,
        )
        risk = preview(account, proposed)
        rows = risk_rows(risk)
        note = ("Preview in the underlying's price terms with one lot: the real order is checked on option premiums, so the "
                "per-trade risk shown here is only indicative.")
        if risk.approved:
            stages.append(_stage("risk", "Risk engine", PASS, "Would be approved right now",
                                 [f"{sum(r['ok'] for r in rows)} of {len(rows)} recorded checks passed.", note]))
        else:
            stages.append(_stage("risk", "Risk engine", FAIL, "Would be vetoed right now", [risk.reason, note], failed=risk.reason))

    # 6. Paper order / 7. Outcome -------------------------------------------------------------------------------
    if risk is not None and risk.approved:
        stages.append(_stage("order", "Paper order", PASS, "Ready: place the simulated order on the Paper Trading page",
                             ["The order will be checked again by the risk engine when submitted."]))
    else:
        stages.append(_stage("order", "Paper order", SKIP, "No order to place"))
    recent = recent_trades or []
    if recent:
        last = recent[0]
        stages.append(_stage("outcome", "Outcome", INFO, f"{len(recent)} earlier {PAPER_TRADE.lower().replace('_', ' ')}(s) for this strategy and instrument",
                             [f"Latest: {last.get('status')}, realised R {last['realised_r']:.2f}" if last.get("realised_r") is not None
                              else f"Latest: {last.get('status')}", "Simulated results; see the Trace tab for the full chain."]))
    else:
        stages.append(_stage("outcome", "Outcome", WAIT, "No paper trade yet for this strategy and instrument"))

    # Verdict ---------------------------------------------------------------------------------------------------
    blocking = next((s for s in stages if s["state"] == FAIL), None)
    if blocking is not None:
        verdict, reason = "NO TRADE", blocking["failed_condition"]
    elif setup is None and selected:
        blocking = next(s for s in stages if s["key"] == "setup")
        verdict, reason = "WAITING FOR SETUP", blocking["headline"]
    elif risk is not None and risk.approved:
        verdict, reason = "CANDIDATE READY", "Passed every stage; would be approved by the risk engine right now."
    else:
        verdict, reason = "NO TRADE", "No strategy qualifies."
    return {
        "stages": stages, "verdict": verdict, "reason": reason,
        "blocking_stage": blocking["title"] if blocking is not None else None,
        "status": CURRENT_CANDIDATE if setup is not None else None,
        "setup": setup_geometry(setup) if setup is not None else None,
        "assumptions": entry_assumptions(setup.timestamp) if setup is not None else [],
        "levels": levels, "risk_checks": risk_rows(risk) if risk is not None else [],
        "historical": historical_setups(intel), "strategy": selected,
    }
