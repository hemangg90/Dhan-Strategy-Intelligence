"""Options Intelligence: the live chain, IV, Greeks, open interest, liquidity, expiry, the contract selector and its risk check.

Read-only. It fetches the chain (one Dhan call, shared through the chain cache), builds the OptionsContext, runs the staged
contract selector with scenario analysis, and shows what the independent risk engine would say. Nothing here places an order.
Every block is labelled: MEASURED (as the broker reported it), CALCULATED (derived by the application), ESTIMATE (a stated
assumption), or UNAVAILABLE (the data source did not provide it - never filled in).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from quant_intelligence.brokers.reconciliation import reconcile_paper
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data_adapters.dhan_instrument_master import list_fno_stock_symbols
from quant_intelligence.execution.auto_trader import _option_risk_info, _scenario_inputs, size_position
from quant_intelligence.execution.engine import engine_for
from quant_intelligence.intelligence.market_context import context_from_pipeline
from quant_intelligence.intelligence.options_context import build_options_context
from quant_intelligence.options.contract_scoring import select_best_contract
from quant_intelligence.options.execution_quality import ExecutionLimits, assess_execution_quality
from quant_intelligence.options.option_selector import OptionSelectionError, fetch_chain, get_underlying_info
from quant_intelligence.options.premium_model import PremiumSizingError, translate_setup
from quant_intelligence.risk.risk_engine import ProposedTrade, limits_for, preview_trade
from quant_intelligence.strategies.base_strategy import Setup
from quant_intelligence.ui.charts import oi_chart
from quant_intelligence.ui.components import chip, empty_state, page_header
from quant_intelligence.ui.credentials_panel import render_credentials_panel
from quant_intelligence.ui.state import account_state_for, init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme
from quant_intelligence.utils.timeutil import now_ist

apply_theme()
init_session_state()
page_header(
    "Options Intelligence",
    "The live option chain, implied volatility, Greeks, open interest, liquidity and expiry - and the contract the selector would "
    "pick, with its scenarios and the risk engine's verdict. Nothing here places an order.",
)

client = st.session_state["dhan_api_client"]
if not client.is_configured():
    st.error("Dhan credentials are not set - the option chain needs them.")
    render_credentials_panel()
    st.stop()

c1, c2, c3 = st.columns([2, 1, 1])
underlying = c1.selectbox("Underlying", list(SETTINGS.option_underlyings) + list_fno_stock_symbols(), key="oi_underlying")
side = c2.selectbox("View", ["Bullish (calls)", "Bearish (puts)"], key="oi_side")
if c3.button("Fetch live chain", key="oi_fetch", width="stretch"):
    try:
        st.session_state["option_chain"] = fetch_chain(client, underlying, SETTINGS.option_expiry_preference, max_age=0)
    except Exception as e:
        st.session_state["option_chain"] = None
        st.error(f"Could not fetch the option chain: {e}")

chain = st.session_state.get("option_chain")
if chain is None or chain.underlying != underlying:
    empty_state("No chain loaded", " Pick an underlying and press Fetch live chain.")
    st.stop()

now = now_ist()
output = run_pipeline_cached() if st.session_state.get("instrument") == underlying else None
market = context_from_pipeline(output, chain) if output is not None else None
try:
    expected_lot = get_underlying_info(underlying)["lot_size"]
except OptionSelectionError:
    expected_lot = None
ctx = build_options_context(chain, now=now, realized_vol=(output.market_state.get("realized_vol_20") if output is not None else None),
                            atr=(market.atr if market is not None else None), expected_lot_size=expected_lot)
tone = {"OK": "good", "WARNING": "warning", "CRITICAL": "critical"}[ctx.status.level]
st.markdown(chip(f"Option data: {ctx.status.level}", tone, icon="") + " " + chip(f"{ctx.underlying} · expiry {ctx.expiry}", "info", icon="")
            + " " + chip(f"snapshot {ctx.timestamp:%H:%M:%S}", "muted", icon=""), unsafe_allow_html=True)
if ctx.status.reason:
    st.caption(ctx.status.reason)
if not ctx.status.ok:
    empty_state("Option context unavailable", ctx.status.reason)
    st.stop()

m = st.columns(6)
m[0].metric("Spot", f"{ctx.spot:,.2f}")
m[1].metric("ATM strike", f"{ctx.atm_strike:g}")
m[2].metric("ATM IV", f"{ctx.atm_iv:.1f}%" if ctx.atm_iv else "-")
iv = ctx.iv
m[3].metric("IV rank / percentile", f"{iv.iv_rank:.0f} / {iv.iv_percentile:.0f}" if iv.available and iv.iv_rank is not None else "unavailable",
            help=iv.reason or "Rank and percentile over the recorded daily IV history (options/iv_history.py).")
m[4].metric("PCR (OI)", f"{ctx.oi.pcr_oi:.2f}" if ctx.oi.pcr_oi is not None else "unavailable")
e = ctx.expiry_info
m[5].metric("Trading days left", f"{e.trading_days:.1f}" if e.available else "-", help=f"{e.risk_band.title()} gamma/theta band" if e.available else e.reason)
if not iv.available:
    st.info(f"IV rank and percentile are unavailable: {iv.reason}")

tabs = st.tabs(["Chain", "Greeks", "IV", "OI & PCR", "Liquidity", "Contract selector", "Risk & execution"])

with tabs[0]:
    st.caption("MEASURED: price, OI, change in OI, volume, bid/ask, provider IV and delta. CALCULATED: spread %, computed delta.")
    rows = []
    for r in sorted(chain.strikes, key=lambda x: x.strike):
        def spread(bid, ask):
            return None if bid is None or ask is None or bid <= 0 or ask < bid else round((ask - bid) / ((ask + bid) / 2) * 100, 2)
        rows.append({"Strike": r.strike, "CE OI": r.ce_oi, "CE ΔOI": r.ce_oi_change, "CE vol": r.ce_volume, "CE IV": r.ce_iv, "CE bid": r.ce_bid,
                     "CE ask": r.ce_ask, "CE spr%": spread(r.ce_bid, r.ce_ask), "CE LTP": r.ce_ltp, "PE LTP": r.pe_ltp, "PE spr%": spread(r.pe_bid, r.pe_ask),
                     "PE bid": r.pe_bid, "PE ask": r.pe_ask, "PE IV": r.pe_iv, "PE vol": r.pe_volume, "PE ΔOI": r.pe_oi_change, "PE OI": r.pe_oi})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=420)
    st.caption(f"Snapshot taken {ctx.timestamp:%H:%M:%S} ({(now - ctx.timestamp).total_seconds():.0f}s ago). The chain has no per-contract quote time, "
               "so this is the age of every quote shown.")

with tabs[1]:
    st.caption("CALCULATED by the application: " + ("Black-76" if ctx.atm_greeks["CE"].model == "BLACK76" else "Black-Scholes-Merton") +
               ", European exercise, one flat volatility solved from each contract's own price, r="
               f"{SETTINGS.risk_free_rate:.1%}, trading-time clock. Units: delta/gamma per unit; vega ₹ per volatility point; theta ₹ per trading day; rho ₹ per 1% rate.")
    rows = []
    for kind in ("CE", "PE"):
        g = ctx.atm_greeks[kind]
        rows.append({"Contract": f"{ctx.atm_strike:g} {kind}", "Available": g.available, "IV (solved)": g.iv, "IV source": g.iv_source or "-",
                     "Price basis": g.price_basis or "-", "Delta": g.delta, "Gamma": g.gamma, "Vega": g.vega, "Theta": g.theta, "Rho": g.rho,
                     "Broker delta": g.provider_delta, "Disagrees": g.provider_disagrees, "Why not": g.reason})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("A Greek the data cannot support is shown as unavailable with the reason, never as zero. Very close to expiry the solver is unstable "
               "and Greeks are withheld.")

with tabs[2]:
    a, b = st.columns(2)
    iv_rv = ctx.iv_vs_realized
    a.metric("ATM IV vs realized", f"{iv_rv['iv_minus_rv'] * 100:+.1f} pts" if iv_rv.get("available") else "unavailable",
             help=iv_rv.get("note") or iv_rv.get("reason"))
    b.metric("IV change since last session", f"{iv.change_1d:+.1f} pts" if iv.available and iv.change_1d is not None else "unavailable")
    st.markdown("**Strike-wise skew (MEASURED)**")
    if ctx.skew.get("available"):
        st.dataframe(pd.DataFrame(ctx.skew["points"]), hide_index=True, width="stretch")
    st.markdown("**Expiry-wise structure**")
    if len(ctx.term_structure) > 1:
        st.dataframe(pd.DataFrame(ctx.term_structure, columns=["Expiry", "ATM IV %"]), hide_index=True)
    else:
        st.info("Only the selected expiry was fetched (each extra expiry costs a rate-limited broker call), so no term structure is shown.")
    st.caption(f"IV history: {iv.n_obs} recorded session(s). " + (iv.reason if not iv.available else
               f"Window {iv.window_days} sessions; tenor-matched figures: {('rank %.0f / pct %.0f' % (iv.iv_rank_tenor, iv.iv_percentile_tenor)) if iv.iv_rank_tenor is not None else 'not enough matching sessions'}."))

with tabs[3]:
    o = ctx.oi
    st.caption("Open interest is context, not a conclusion: the peaks are where contracts exist, not support or resistance, and the data does not say who holds them.")
    a, b, c, d = st.columns(4)
    a.metric("Call OI", f"{o.total_call_oi:,.0f}" if o.total_call_oi is not None else "-", help=f"change {o.call_oi_change:+,.0f}" if o.call_oi_change is not None else None)
    b.metric("Put OI", f"{o.total_put_oi:,.0f}" if o.total_put_oi is not None else "-", help=f"change {o.put_oi_change:+,.0f}" if o.put_oi_change is not None else None)
    c.metric("PCR (volume)", f"{o.pcr_volume:.2f}" if o.pcr_volume is not None else "unavailable")
    d.metric(f"PCR (±{o.window_strikes} strikes)", f"{o.pcr_oi_window:.2f}" if o.pcr_oi_window is not None else "unavailable")
    st.plotly_chart(oi_chart(chain), width="stretch", config={"displayModeBar": False})
    peaks = []
    for label, p in (("Most call OI", o.call_oi_peak), ("Most put OI", o.put_oi_peak)):
        if p:
            peaks.append({"": label, "Strike": p["strike"], "OI": p["oi"], "Share of side": f"{p['share']:.0%}", "Distance": f"{p['distance_pct']:+.2f}%",
                          "ATR": None if p["distance_atr"] is None else round(p["distance_atr"], 2)})
    if peaks:
        st.dataframe(pd.DataFrame(peaks), hide_index=True)
    for n in o.notes:
        st.caption(n)

with tabs[4]:
    liq = ctx.liquidity
    st.caption(f"Contracts within ±5 strikes of ATM judged against the execution limits: spread ≤ {SETTINGS.option_max_spread_pct:g}%, quote ≤ "
               f"{SETTINGS.option_max_quote_age_sec:g}s old, quoted size ≥ {SETTINGS.option_min_top_qty_lots:g} lot, volume and OI above zero. Fills are "
               "assumed at the ask (buy) / bid (sell) plus slippage - an ESTIMATE, never the last price.")
    if liq.get("available"):
        a, b, c = st.columns(3)
        a.metric("Eligible contracts", f"{liq['eligible']} of {liq['contracts_considered']}")
        b.metric("Median spread", f"{liq['median_spread_pct']:.2f}%" if liq.get("median_spread_pct") is not None else "-")
        c.metric("Widest spread", f"{liq['max_spread_pct']:.2f}%" if liq.get("max_spread_pct") is not None else "-")
    lim = ExecutionLimits.from_settings()
    near = [q for q in ctx.quotes if abs(q.strike - ctx.atm_strike) <= 5 * (chain.strikes[1].strike - chain.strikes[0].strike)]
    table = []
    for q in sorted(near, key=lambda x: (x.strike, x.option_type)):
        eq = assess_execution_quality(q, now, "BUY", q.lot_size, lim)
        table.append({"Contract": f"{q.strike:g} {q.option_type}", "Eligible": eq.eligible, "Spread %": eq.rel_spread_pct, "Bid qty": q.bid_qty, "Ask qty": q.ask_qty,
                      "Quote age s": eq.quote_age_sec, "Entry fill (est.)": eq.entry_fill, "Round-trip cost ₹ (est.)": eq.round_trip_costs,
                      "Breakeven move % (est.)": eq.breakeven_move_pct, "Why not": "; ".join(eq.reasons)})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", height=380)

direction = "LONG" if side.startswith("Bullish") else "SHORT"
selected = None
with tabs[5]:
    st.caption("Staged selection: eligibility -> quality score -> scenarios -> ranking. Illustrative unless a live setup exists: the stop and target "
               "below are 1 and 2 ATR from the spot in the chosen direction.")
    atr = market.atr if market is not None else None
    if not atr:
        st.warning("The selector needs the underlying's ATR. Select this underlying on the dashboard so its analysis runs, then reload.")
    else:
        sign = 1 if direction == "LONG" else -1
        setup = Setup(pd.Timestamp(now), direction, chain.spot_price, chain.spot_price - sign * atr, chain.spot_price + sign * 2 * atr, {"transaction": "BUY"})
        scenario = _scenario_inputs(output, setup, market, chain, now, underlying)
        lot = expected_lot or 1
        limits = limits_for(engine_for(underlying).profile.name)
        account = account_state_for(underlying)
        cap = account.equity * limits.max_capital_per_trade_pct / 100.0 if limits.max_capital_per_trade_pct > 0 else None
        try:
            selected = select_best_contract(chain, direction, "BUY", lot, cap, scenario=scenario, now=now)
        except OptionSelectionError as err:
            st.error(str(err))
        else:
            s = selected.scored
            st.success(f"{selected.summary}")
            rows = []
            for r in selected.ranking:
                sc = r.scenario
                rows.append({"Strike": r.strike, "Quality score": round(r.score, 1), "Build-up": r.buildup.replace("_", " ").title(),
                             "Spread %": None if r.spread_pct is None else round(r.spread_pct, 2), "|Delta|": None if r.greeks is None or not r.greeks.available else round(abs(r.greeks.delta), 2),
                             "Entry fill ₹": None if r.execution is None else r.execution.entry_fill,
                             "Exp. net % of premium (lower bound)": None if sc is None or not sc.available or sc.expected_strategy_lower is None else round(sc.expected_strategy_lower / sc.entry * 100, 1),
                             "Tail loss % (CVaR10)": None if sc is None or not sc.available else round(sc.cvar10 / sc.entry * 100, 1),
                             "Objective": None if r.objective is None else round(r.objective, 3)})
            st.markdown(f"**Candidates, best first (ranked by {selected.basis})**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if selected.rejected:
                st.markdown("**Rejected alternatives**")
                st.dataframe(pd.DataFrame([{"Strike": k, "Reason": why} for k, why in selected.rejected]), hide_index=True)
            sc = s.scenario
            if sc is not None and sc.available:
                st.markdown("**Scenario analysis of the pick (per unit, net of fills and costs - ESTIMATES)**")
                a, b, c, d = st.columns(4)
                a.metric("Entry (conservative fill)", f"₹{sc.entry:,.2f}")
                b.metric("No-view expectation", f"{sc.expected_neutral_return * 100:+.1f}% of premium", help="A bought option normally loses to decay and costs without a move.")
                c.metric("Tail loss (CVaR10)", f"₹{sc.cvar10:,.2f}")
                d.metric("Win probability (lower bound)", f"{sc.win_prob_lower:.0%}" if sc.win_prob_lower is not None else "unavailable",
                         help=f"point estimate {sc.win_prob:.0%}" if sc.win_prob is not None else "no comparable observations")
                st.dataframe(pd.DataFrame(sc.rows)[["scenario", "spot", "pnl_per_unit", "weight"]], hide_index=True, width="stretch", height=300)
                with st.expander("Assumptions"):
                    for line in sc.assumptions:
                        st.write("- " + line)
            else:
                st.info("Scenario analysis unavailable: " + (sc.reason if sc is not None else "no scenario inputs"))

with tabs[6]:
    st.caption("The proposed trade is sized by the existing sizing rule and checked by the independent risk engine (a preview: no approval is issued).")
    if selected is None or market is None:
        st.info("Run the contract selector first (needs a pick and the underlying's analysis).")
    else:
        contract = selected.contract
        setup = Setup(pd.Timestamp(now), direction, chain.spot_price, chain.spot_price - (1 if direction == "LONG" else -1) * market.atr,
                      chain.spot_price + (2 if direction == "LONG" else -2) * market.atr, {"transaction": "BUY"})
        try:
            premium_setup = translate_setup(setup, contract)
        except PremiumSizingError as err:
            st.error(str(err))
        else:
            account = account_state_for(underlying)
            qty = size_position(account, premium_setup.entry_price, premium_setup.stop_price, contract.lot_size)
            info, quote = _option_risk_info(chain, contract, max(qty, contract.lot_size), "BUY", now)
            a, b, c, d = st.columns(4)
            a.metric("Contract", f"{contract.strike:g} {contract.option_type}")
            b.metric("Quantity", f"{qty} ({qty // contract.lot_size} lots)" if qty else "0")
            c.metric("Premium at risk", f"₹{premium_setup.entry_price * qty:,.0f}")
            d.metric("Max loss at stop", f"₹{abs(premium_setup.entry_price - premium_setup.stop_price) * qty:,.0f}")
            ex = account.option_exposure
            st.markdown("**Open option exposure (from Greeks stored at entry)**")
            st.dataframe(pd.DataFrame([{"Premium at risk ₹": ex.get("premium_at_risk"), "Net delta notional ₹": ex.get("delta_notional"), "Gamma (1% move) ₹": ex.get("gamma_1pct"),
                                        "Vega ₹/pt": ex.get("vega"), "Theta ₹/day": ex.get("theta"), "Open option positions": ex.get("n_option_positions"),
                                        "Greeks unknown": ex.get("unknown_positions")}]), hide_index=True)
            if qty > 0:
                trade = ProposedTrade(strategy_name="(preview)", direction=direction, entry_price=premium_setup.entry_price, stop_price=premium_setup.stop_price,
                                      target_price=premium_setup.target_price, quantity=qty, data_quality_status="OK", long_premium=True, instrument=underlying,
                                      quote_time=chain.fetched_at, bid=quote.bid if quote else None, ask=quote.ask if quote else None,
                                      lot_size=contract.lot_size, option=info)
                verdict = preview_trade(account, trade)
                (st.success if verdict.approved else st.error)(("WOULD APPROVE" if verdict.approved else "WOULD REJECT") + f": {verdict.reason}")
                st.dataframe(pd.DataFrame([{"Check": k, "Passed": v} for k, v in verdict.checks.items()]), hide_index=True)
            else:
                st.warning("Sizing produced zero quantity under the current risk limits - nothing to check.")
            rec = reconcile_paper(engine_for(underlying).broker)
            (st.success if rec.ok else st.error)("Reconciliation: " + rec.summary)
            st.caption(f"Execution mode: PAPER. Live trading stays locked. Hard risk vetoes are enforced in the order path "
                       f"(auto_trader -> risk engine -> broker), not only shown here.")
