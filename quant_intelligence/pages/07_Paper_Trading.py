"""Paper Trading - a trading-terminal view of one instrument.

Quote header and price chart, the strategy leaderboard, the option-chain ladder with open interest, the
decision (regime, selected strategy, setup status) and an order ticket showing cost, risk and reward after
charges - then the risk engine and the paper broker. "All instruments" mode hands over to the shared
multi-instrument scanner.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from quant_intelligence.brokers.base_broker import OrderRequest
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.config.watchlist import WATCHLIST_STOCKS
from quant_intelligence.data_adapters.dhan_instrument_master import list_fno_stock_symbols
from quant_intelligence.execution.auto_trader import run_auto_option_cycle
from quant_intelligence.execution.capital import capital_summary
from quant_intelligence.execution.engine import ENGINE
from quant_intelligence.execution.position_monitor import monitor_option_positions, monitor_positions
from quant_intelligence.execution.setup_detector import SETUP_TRIGGERED, WAITING_FOR_SETUP, detect_chain_setup, detect_setup
from quant_intelligence.options.option_selector import (
    OptionSelectionError,
    fetch_chain,
    get_underlying_info,
    option_segment_for,
    select_contract,
)
from quant_intelligence.options.premium_model import PremiumSizingError, translate_setup
from quant_intelligence.ranking.ranking_engine import TIE_BREAK_TAG, tradable
from quant_intelligence.risk.risk_engine import ProposedTrade, evaluate_trade
from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES, get_strategy
from quant_intelligence.ui import format as F
from quant_intelligence.ui.chain_views import chain_stats, ladder_frame, ticket_math
from quant_intelligence.ui.charts import oi_chart, price_chart, regime_chart
from quant_intelligence.ui.components import chip, empty_state, kpi_row, page_header, quote_header
from quant_intelligence.ui.market_data import load_snapshot
from quant_intelligence.ui.multi_instrument_panel import render_multi_instrument_panel
from quant_intelligence.ui.open_positions import render_open_positions
from quant_intelligence.ui.state import get_account_state, init_session_state, run_pipeline_cached
from quant_intelligence.ui.tables import ladder_table, ranking_frame, ranking_table
from quant_intelligence.ui.theme import apply_theme
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeutil import now_ist

# The option chain is cached for SETTINGS.dhan.chain_cache_ttl_sec; polling faster only re-reads the cache.
_CHAIN_POLL_SEC = max(int(SETTINGS.ltp_refresh_seconds), int(SETTINGS.dhan.chain_cache_ttl_sec))
_CFG = {"displayModeBar": False}

apply_theme()
init_session_state()
page_header(
    "Paper Trading",
    "Select a strategy → wait for a valid setup → resolve an option contract → risk approval → execute (simulated).",
)

scope = st.segmented_control("Scope", ["Single instrument", "All instruments"], default="Single instrument",
                             key="pt_scope", label_visibility="collapsed") or "Single instrument"
if scope == "All instruments":
    st.caption(f"Scans {', '.join(SETTINGS.option_underlyings)} and the stock watchlist in one pass. One paper account and one risk engine are shared.")
    render_multi_instrument_panel()
    st.stop()

client = st.session_state["dhan_api_client"]
broker = ENGINE.broker

# ---------------------------------------------------------------------------------------------- instrument bar
watch = list(SETTINGS.option_underlyings) + [s["symbol"] for s in WATCHLIST_STOCKS]
choices = watch + [s for s in list_fno_stock_symbols() if s not in watch]
bar_pick, bar_quote, bar_auto = st.columns([2, 5, 2], vertical_alignment="top")
underlying = bar_pick.selectbox("Instrument", choices, key="option_underlying",
                                help="Index and stock options. The strategy pipeline runs on the instrument you pick.")
automate = bar_auto.toggle(
    "Auto-trade this instrument", value=False, key="pt_automate",
    help="Auto-fetch the chain, select the strategy and place the PAPER order when a setup triggers - no manual click. "
         "The same safety checks apply (risk-engine veto, mandatory stop on SELL). Stops if you leave this page; "
         "use the 'All instruments' scope for background auto-trading.",
)

# The pipeline (data -> regime -> strategy ranking) must run on the instrument being traded.
if st.session_state.get("pipeline_for") != underlying:
    st.session_state["instrument"] = underlying
    st.session_state["pipeline_output"] = None
with st.spinner(f"Running the research pipeline for {underlying}..."):
    output = run_pipeline_cached()
if output is None:
    st.error(st.session_state.get("pipeline_error", "Pipeline not available."))
    st.stop()
st.session_state["pipeline_for"] = underlying

with bar_quote:
    quote_header(underlying, load_snapshot(underlying, st.session_state.get("timeframe", SETTINGS.default_timeframe)),
                 tags=[profile_for(underlying).name, "INDEX" if underlying in SETTINGS.option_underlyings else "STOCK"])

# ---------------------------------------------------------------------------------------------- automation (unchanged logic)
if automate:

    @st.fragment(run_every=f"{SETTINGS.auto_trade_refresh_seconds}s")
    def _auto_fragment():
        provider = (lambda: fetch_chain(client, underlying, SETTINGS.option_expiry_preference)) if client.is_configured() else None
        fresh_output = run_pipeline_cached(force=True)
        if fresh_output is not None:
            account = get_account_state()
            cycle = run_auto_option_cycle(fresh_output, underlying, broker, client, provider, account)
            st.caption(f"Last automated cycle ({now_ist():%H:%M:%S} IST): {cycle.reason}")
            if cycle.order is not None:
                ENGINE.add_trade()
                events = monitor_positions(broker, fresh_output.ohlcv.iloc[-1])
                for e in events:
                    ENGINE.add_pnl(e["position"]["net_pnl"])
            if client.is_configured():
                for e in monitor_option_positions(broker, client):
                    ENGINE.add_pnl(e["position"]["net_pnl"])

    _auto_fragment()

# ---------------------------------------------------------------------------------------------- decision & setup
chain = st.session_state.get("option_chain")
if chain is not None and chain.underlying != underlying:
    chain = None  # the stored chain belongs to another instrument

ranking = output.ranking
ok_paper, why_not = tradable(ranking, "PAPER")
strategy_name = ranking.selected_strategy if ok_paper else None
setup_status = None
setup = None
needs_chain_msg = False
if strategy_name:
    strategy = get_strategy(strategy_name)
    recent = output.ohlcv.tail(5)
    if strategy_name in CHAIN_AWARE_STRATEGY_NAMES:
        if chain is None:
            needs_chain_msg = True
        else:
            setup_status = detect_chain_setup(strategy, output.market_state, recent, chain)
    else:
        setup_status = detect_setup(strategy, output.market_state, recent)
    if setup_status is not None and setup_status.status == SETUP_TRIGGERED:
        setup = setup_status.setup

main, side = st.columns([7, 4], gap="large")

# ---------------------------------------------------------------------------------------------- left: chart / chain / strategies
with main:
    tab_chart, tab_chain, tab_rank = st.tabs(["Chart", "Option chain", "Strategies"])

    with tab_chart:
        d = output.ohlcv
        days = sorted(pd.to_datetime(d["timestamp"]).dt.date.unique())[-3:]
        view = d[pd.to_datetime(d["timestamp"]).dt.date.isin(days)]
        levels = []
        if setup is not None:
            levels = [{"label": "Entry", "price": setup.entry_price, "tone": "info"},
                      {"label": "Stop", "price": setup.stop_price, "tone": "critical"},
                      {"label": "Target", "price": setup.target_price, "tone": "good"}]
        st.plotly_chart(price_chart(view, profile=profile_for(underlying), levels=levels, emas=(20, 50)),
                        width="stretch", config=_CFG)
        st.caption("Candles are closed bars from the pipeline's cache (refreshed once per closed bar). "
                   "Entry / stop / target lines are in the underlying's price terms.")

    with tab_chain:
        if not client.is_configured():
            empty_state("Set your Dhan credentials to load the option chain",
                        "Paste your Client ID and access token under API Keys in the sidebar.")
        else:
            t1, t2, t3 = st.columns([1, 1, 2])
            if t1.button("Fetch live option chain", key="pt_fetch_chain"):
                try:
                    st.session_state["option_chain"] = fetch_chain(client, underlying, SETTINGS.option_expiry_preference, max_age=0)
                    chain = st.session_state["option_chain"]
                except Exception as e:
                    st.session_state["option_chain"] = None
                    chain = None
                    st.error(f"Could not fetch option chain: {e}")
            live = t2.toggle("Live refresh", value=False, key="pt_live_chain",
                             help=f"Refresh every {_CHAIN_POLL_SEC}s. Responses are shared with every other tab and the "
                                  "auto-trader, so this adds no extra Dhan requests inside the cache window.")
            window = t3.slider("Strikes around ATM", 4, 25, 10, key="pt_ladder_window")

            def _render_chain() -> None:
                ch = st.session_state.get("option_chain")
                if ch is None or ch.underlying != underlying:
                    empty_state("No option chain loaded", "Click 'Fetch live option chain'.")
                    return
                s = chain_stats(ch)
                kpi_row([
                    {"label": "Spot", "value": F.num(s["spot"], 2)},
                    {"label": "ATM strike", "value": F.num(s["atm"], 0), "help": f"Expiry {F.short_date(s['expiry'])}"},
                    {"label": "PCR (OI)", "value": F.num(s["pcr"], 2),
                     "help": "Put OI / call OI across all strikes. Above ~1 means more puts written than calls"},
                    {"label": "ATM IV", "value": f"{s['atm_iv']:.1f}%" if s["atm_iv"] else "–"},
                    {"label": "Max pain", "value": F.num(s["max_pain"], 0)},
                    {"label": "Support / resistance", "value": f"{F.num(s['support'], 0)} / {F.num(s['resistance'], 0)}",
                     "help": "Strikes with the heaviest put OI (support) and call OI (resistance)"},
                ])
                st.plotly_chart(oi_chart(ch, window=min(window, 14)), width="stretch", config=_CFG)
                st.dataframe(ladder_table(ladder_frame(ch, window), ch.spot_price), width="stretch", hide_index=True,
                             height=min(35 * (2 * window + 2), 640))
                st.caption(f"Expiry {F.short_date(ch.expiry)} · shaded blue = calls in the money, orange = puts in the money "
                           f"· updated {now_ist():%H:%M:%S} IST")

            if live:

                @st.fragment(run_every=f"{_CHAIN_POLL_SEC}s")
                def _live_chain():
                    try:
                        st.session_state["option_chain"] = fetch_chain(client, underlying, SETTINGS.option_expiry_preference)
                    except Exception as e:
                        st.warning(f"Live refresh failed: {e}")
                    _render_chain()

                _live_chain()
            else:
                _render_chain()

    with tab_rank:
        st.caption("Every strategy's standing for today's market state. The leader is selected only if it clears the "
                   "edge, confidence and statistical-distinguishability bars; otherwise the system says NO TRADE.")
        st.dataframe(ranking_table(ranking_frame(output.strategy_intel, ranking.selected_strategy)),
                     width="stretch", hide_index=True)

# ---------------------------------------------------------------------------------------------- right: decision + ticket
with side:
    with st.container(border=True):
        quality = output.data_quality_status
        st.markdown(
            chip(f"Regime: {output.regime_label.replace('_', ' ').title()} · {output.regime_confidence * 100:.0f}%", "info")
            + " " + chip(f"Data {quality}", "good" if quality == "OK" else ("warning" if quality == "DEGRADED" else "critical")),
            unsafe_allow_html=True,
        )
        if strategy_name is None:
            st.markdown('<div class="qi-decision-title">NO TRADE</div>', unsafe_allow_html=True)
            st.caption(why_not or ranking.reason)
        else:
            if setup is not None:
                status_chip = chip("Setup triggered", "good")
            elif setup_status is not None and setup_status.status == WAITING_FOR_SETUP:
                status_chip = chip("Waiting for setup", "warning")
            else:
                status_chip = chip("Needs option chain", "warning")
            if ranking.tie_break:
                status_chip += " " + chip("TIE-BREAK", "warning")
            st.markdown(f'<div class="qi-decision-title">{strategy_name}</div>{status_chip}', unsafe_allow_html=True)
            st.caption(ranking.reason)
            if ranking.tie_break:
                st.warning(
                    f"No strategy is clearly best (runner-up: {ranking.tie_runner_up}). This trade is tagged TIE-BREAK and the "
                    f"auto-trader sizes it at x{SETTINGS.tie_break_size_factor:g} - consider a smaller lot here too. "
                    "Tie-break picks have no statistically proven edge."
                )
            if needs_chain_msg:
                st.info(f"{strategy_name} needs a live option chain - load one in the Option chain tab.")
            elif setup_status is not None and setup_status.status == WAITING_FOR_SETUP:
                st.caption("Entry conditions are not met on the latest closed bar yet.")
        st.markdown("**Regime probabilities**")
        st.plotly_chart(regime_chart(output.regime_probabilities), width="stretch", config=_CFG)

    if setup is not None:
        transaction = setup.meta.get("transaction", "BUY")
        with st.container(border=True):
            st.markdown(f'<div class="qi-decision-title">Order ticket</div>'
                        f'{chip(setup.direction, "info", icon="")} {chip(transaction, "info" if transaction == "BUY" else "serious", icon="")}',
                        unsafe_allow_html=True)
            if chain is None:
                st.warning("Load the option chain (Option chain tab) to resolve a tradeable contract.")
            else:
                moneyness = st.slider("Strike offset (0 = ATM, + = further OTM)", 0, 5, SETTINGS.option_moneyness_offset, key="pt_money")
                try:
                    lot_size = get_underlying_info(underlying)["lot_size"]
                    contract = select_contract(chain, setup.direction, transaction, moneyness, lot_size)
                    premium_setup = translate_setup(setup, contract)
                except (OptionSelectionError, PremiumSizingError) as e:
                    st.error(f"Could not resolve a tradeable option: {e}")
                else:
                    lots = st.number_input("Lots", min_value=1, value=1, step=1, key="pt_lots")
                    quantity = int(lots) * contract.lot_size
                    market = profile_for(underlying).name
                    available = capital_summary(broker)["available"]
                    t = ticket_math(premium_setup.entry_price, premium_setup.stop_price, premium_setup.target_price,
                                    quantity, contract.transaction, market, available)

                    st.markdown(
                        f"**{contract.trading_symbol}**  \n"
                        f"{contract.option_type} · strike {F.num(contract.strike, 0)} · expiry {F.short_date(contract.expiry)} "
                        f"· lot {contract.lot_size}"
                    )
                    kv = [
                        ("Premium (entry)", F.num(premium_setup.entry_price, 2)),
                        ("Stop-loss", F.num(premium_setup.stop_price, 2)),
                        ("Target", F.num(premium_setup.target_price, 2)),
                        ("Delta / IV", f"{contract.delta:.2f} / {contract.iv:.1f}%" if contract.delta is not None and contract.iv is not None else "–"),
                        ("Quantity", f"{lots} lot(s) × {contract.lot_size} = {quantity}"),
                        ("Amount needed" if contract.transaction == "BUY" else "Premium value (margin extra)", F.inr(t["amount_needed"])),
                        ("Max loss at stop (after charges)", F.inr(t["net_loss"])),
                        ("Max profit at target (after charges)", F.inr(t["net_profit"])),
                        ("Risk : reward", f"1 : {t['risk_reward']:.2f}" if t["risk_reward"] else "–"),
                        ("Est. charges (stop / target)", f"{F.inr(t['charges_at_stop'])} / {F.inr(t['charges_at_target'])}"),
                    ]
                    st.markdown('<div class="qi-kv">' + "".join(f"<span>{k}</span><span>{v}</span>" for k, v in kv) + "</div>",
                                unsafe_allow_html=True)
                    if t["utilisation_pct"] is not None:
                        used = min(t["utilisation_pct"], 100.0)
                        color = "var(--qi-good)" if t["funds_ok"] and used < 60 else ("var(--qi-warning)" if t["funds_ok"] else "var(--qi-critical)")
                        st.markdown(f'<div class="qi-meter"><div style="width:{used:.0f}%;background:{color}"></div></div>',
                                    unsafe_allow_html=True)
                        st.caption(f"Uses {t['utilisation_pct']:.1f}% of available funds ({F.inr(available)})")
                    if not t["funds_ok"]:
                        st.error("Amount needed exceeds available funds - this order would be refused.")

                    if st.button("Submit to risk engine → paper broker", type="primary", key="pt_submit", width="stretch"):
                        account = get_account_state()
                        proposed = ProposedTrade(
                            strategy_name=strategy_name, direction=setup.direction,
                            entry_price=premium_setup.entry_price, stop_price=premium_setup.stop_price,
                            target_price=premium_setup.target_price, quantity=quantity,
                            relative_volume=output.market_state.get("relative_volume"),
                            data_quality_status=output.data_quality_status, quote_time=getattr(output, "timestamp", None), long_premium=True, instrument=underlying,
                        )
                        decision = evaluate_trade(account, proposed)
                        if not decision.approved:
                            st.error(f"RISK ENGINE VETO ({decision.decision_id}): {decision.reason}")
                        else:
                            st.success(f"Risk approved ({decision.decision_id})")
                            order = OrderRequest(
                                strategy_name=strategy_name, instrument=underlying, direction=setup.direction,
                                quantity=quantity, order_type="MARKET", price=premium_setup.entry_price,
                                stop_price=premium_setup.stop_price, target_price=premium_setup.target_price,
                                decision_id=decision.decision_id, security_id=contract.security_id,
                                exchange_segment=option_segment_for(underlying),
                                product_type=SETTINGS.option_product_type, transaction_type=contract.transaction,
                                option_type=contract.option_type, strike=contract.strike, expiry=contract.expiry,
                                lot_size=contract.lot_size, tag=TIE_BREAK_TAG if ranking.tie_break else None,
                            )
                            ack = broker.place_order(order, market_price=premium_setup.entry_price)
                            ENGINE.add_trade()
                            if ack.status == "FILLED":
                                st.success(f"Paper order FILLED: {contract.transaction} {order.quantity} {contract.trading_symbol} "
                                           f"@ {ack.fill_price:.2f} (order {ack.order_id})")
                            else:
                                st.error(f"Order {ack.status}: {ack.reject_reason}")

# ---------------------------------------------------------------------------------------------- positions
st.markdown("### Open positions")
render_open_positions(key="p07")

with st.expander("Position monitor - check stop / target hits now"):
    mc1, mc2 = st.columns(2)
    if mc1.button("Check underlying stop/target hits", key="pt_mon_underlying"):
        events = monitor_positions(broker, output.ohlcv.iloc[-1])
        if events:
            for e in events:
                st.write(f"{e['type']}: {e['position']['position_id']} closed @ {e['position']['exit_price']}, "
                         f"net P&L {e['position']['net_pnl']:.2f}")
                ENGINE.add_pnl(e["position"]["net_pnl"])
        else:
            st.info("No stop/target hits on the latest bar.")
    if mc2.button("Check option premium stop/target hits (live LTP)", disabled=not client.is_configured(), key="pt_mon_option"):
        events = monitor_option_positions(broker, client)
        if events:
            for e in events:
                st.write(f"{e['type']}: {e['position']['position_id']} closed @ {e['position']['exit_price']}, "
                         f"net P&L {e['position']['net_pnl']:.2f}")
                ENGINE.add_pnl(e["position"]["net_pnl"])
        else:
            st.info("No option stop/target hits at current LTP.")
st.caption("Full trade history, order book, fills and performance analytics are on the Positions & Orders page.")
