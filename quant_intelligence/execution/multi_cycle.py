"""One paper-trading scan cycle across many instruments, ALL traded as options.

Indices (SETTINGS.option_underlyings) and stocks are each run through the research pipeline and,
when a setup triggers, resolved to an option contract from the live chain and paper-traded
via `run_auto_option_cycle`. Shared by 07_Paper_Trading ("All instruments together" mode) and
13_Auto_Multi_Stock_Trading. Streamlit-free: callers pass the account-state getter and
apply the returned P&L to their own session state.
"""
from __future__ import annotations

import contextlib
import datetime as dt
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeutil import now_ist
from typing import Callable

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.auto_trader import chain_needed, run_auto_option_cycle
from quant_intelligence.execution.position_monitor import monitor_option_positions
from quant_intelligence.options.option_selector import fetch_chain
from quant_intelligence.research.pipeline import run_pipeline


def run_multi_instrument_cycle(
    index_symbols: list[str],
    stock_symbols: list[str],
    timeframe: str,
    lookback_days: int,
    broker,
    client,
    get_account: Callable,
    broker_lock=None,
    on_fill: Callable | None = None,
    selection=None,
) -> tuple[list[dict], float]:
    """Returns (result rows, realised P&L from positions closed during the cycle).

    `selection` (execution.mover_selection.Selection) restricts the scan to the heatmap's movers and passes each
    one's allowed direction to the auto-trader; the symbol lists passed in are expected to be already filtered.

    `broker_lock` (an RLock shared by every runner using this broker) is held only around the
    steps that read cash/positions and place orders, so scans on different watchlists overlap
    while capital checks and fills stay atomic. `on_fill` is called once per filled order.
    """
    broker_lock = broker_lock if broker_lock is not None else contextlib.nullcontext()
    account = get_account()
    end = now_ist()
    start = end - dt.timedelta(days=lookback_days)
    rows: list[dict] = []
    pnl_delta = 0.0

    all_symbols = [(s, "index") for s in index_symbols] + [(s, "stock") for s in stock_symbols]

    def check_exits() -> None:
        """Stop/target check of every open option position against its live premium."""
        nonlocal pnl_delta
        if not client.is_configured():
            return
        try:
            with broker_lock:
                for e in monitor_option_positions(broker, client):
                    pnl_delta += e["position"]["net_pnl"]
        except Exception as e:
            rows.append({"symbol": "-", "type": "monitor", "strategy": "-", "status": "MONITOR_ERROR", "detail": str(e)})

    # Exits first: a slow scan of many instruments must not delay a stop or target that is already hit.
    check_exits()

    for symbol, kind in all_symbols:
        if profile_for(symbol).name == "MCX":
            kind = "commodity"
        try:
            output = run_pipeline(symbol, timeframe, start, end)
        except Exception as e:
            rows.append({"symbol": symbol, "type": kind, "strategy": "-", "status": "DATA_ERROR", "detail": str(e)})
            continue

        # The chain is fetched lazily, only if this instrument's strategy actually triggers (see
        # run_auto_option_cycle) - not for every instrument every cycle.
        allowed = selection.allowed_direction(symbol) if selection is not None else None
        move_note = (f"({'up' if allowed == 'LONG' else 'down'} {selection.change_pct[symbol]:+.2f}% today)"
                     if allowed else "")
        chain_provider = (
            (lambda sym=symbol: fetch_chain(client, sym, SETTINGS.option_expiry_preference))
            if client.is_configured()
            else None
        )

        # Not under broker_lock: the chain fetch can take seconds (rate-limited) and must not block
        # the other runner's fills. The lock is taken around the decision/fill below.
        triggered_chain = None
        if chain_provider is not None and chain_needed(output, symbol, broker, *([allowed] if allowed else [])):
            try:
                triggered_chain = chain_provider()
            except Exception as e:
                from quant_intelligence.reports.decision_log import log_chain_error

                log_chain_error(symbol, output, str(e))
                rows.append(
                    {
                        "symbol": symbol,
                        "type": kind,
                        "strategy": output.ranking.selected_strategy or "-",
                        "status": "CHAIN_ERROR",
                        "detail": str(e),
                    }
                )
                continue

        try:
            with broker_lock:
                account = get_account()  # fresh cash/exposure: another runner may have filled meanwhile
                cycle = run_auto_option_cycle(
                    output, symbol, broker, client, triggered_chain if triggered_chain is not None else chain_provider, account,
                    allowed, move_note,
                )
                if cycle.order is not None:
                    if on_fill is not None and cycle.order.status == "FILLED":
                        on_fill()
                    account = get_account()  # re-sync exposure/trade-count after a fill
        except Exception as e:  # one instrument's failure must not skip the other instruments or the exit check below
            rows.append({"symbol": symbol, "type": kind, "strategy": "-", "status": "CYCLE_ERROR",
                         "detail": f"{type(e).__name__}: {e}"})
            continue
        rows.append(
            {
                "symbol": symbol,
                "type": kind,
                "strategy": cycle.strategy_name or "-",
                "status": cycle.setup_status or ("NO_TRADE" if output.ranking.is_no_trade else "-"),
                "capital_required": round(cycle.capital_required, 2) if cycle.capital_required is not None else None,
                "capital_used": round(cycle.capital_used, 2),
                "tag": cycle.tag,
                "detail": cycle.reason,
            }
        )

    # Check every open option position's live premium against its stop/target once per cycle
    # (not only right after a new fill), so exits are never skipped on quiet cycles.
    check_exits()
    return rows, pnl_delta
