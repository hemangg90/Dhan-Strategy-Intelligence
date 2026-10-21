"""The Risk Control screen for one market. NSE and MCX each have their own paper account, limits, drawdown and kill
switch, so each has its own page; both call `render_risk_control(market)`."""
from __future__ import annotations

import dataclasses

import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.database.db import get_session, is_durable
from quant_intelligence.database.models import RiskEvent
from quant_intelligence.execution.engine import engine_for_market
from quant_intelligence.risk import limit_overrides as LO
from quant_intelligence.risk.risk_engine import drawdown_pct, limits_for, loss_streak, risk_multiplier
from quant_intelligence.utils.market_profile import profile_for


def _event_market(event: RiskEvent) -> str:
    """The market a risk event belongs to: stored on limit changes, else derived from the instrument in its details."""
    details = event.details or {}
    if details.get("market"):
        return str(details["market"]).upper()
    instrument = details.get("instrument") or details.get("underlying")
    return profile_for(instrument).name if instrument else "NSE"


def render_risk_control(market: str) -> None:
    market = market.upper()
    engine = engine_for_market(market)
    k = market.lower()  # widget-key prefix: both pages can be opened in one session
    limits = limits_for(market)
    account = engine.account_state()
    multiplier, why = risk_multiplier(account)
    dd = drawdown_pct(account)

    st.title(f"Risk Control: {market}")
    st.caption(
        f"The {market} paper account, its limits and its kill switch. {'NSE' if market == 'MCX' else 'MCX'} has its own "
        "separate page and account. The risk engine is deterministic, independent of the research layer, and can veto any trade."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{market} equity (marked to market)", f"₹{account.equity:,.0f}",
                help=f"Cash + unrealised P&L of open positions (unrealised: ₹{account.unrealised_pnl:,.0f})")
    col2.metric("Daily P&L", f"₹{account.daily_pnl:,.0f}", help="Realised today + unrealised")
    col3.metric("Open Positions", f"{account.open_positions_count} / {limits.max_open_positions or '∞'}")
    col4.metric("Trades Today", f"{account.trades_today} / {limits.max_trades_per_day}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Drawdown", f"{dd:.2f}%", help=f"From peak equity, across days. Hard stop at {limits.max_drawdown_pct:g}%.")
    col6.metric("Drawdown peak", f"₹{account.peak_equity:,.0f}")
    col7.metric("Position size now", f"{multiplier:.0%}", help=why)
    col8.metric("Losing streak today", loss_streak(account))
    if multiplier < 1:
        st.warning(f"New {market} trades are sized at {multiplier:.0%} of normal: {why}.")

    st.subheader(f"{market} risk limits")
    if SETTINGS.demo_lockdown:
        st.info("Demo lockdown is on: limits are shown but cannot be changed here.")
    else:
        st.caption(
            f"These limits apply only to {market} trades. Tightening a limit applies at once. Loosening needs the "
            "confirmation tick and a reason. Each value has a hard ceiling the app will not exceed. New values apply to new "
            "trades; open positions are not re-judged."
        )
        with st.form(f"{k}_risk_limits_form"):
            entered: dict = {}
            cols = st.columns(2)
            for i, row in enumerate(LO.rows(market)):
                kind_float = row["kind"] == "float"
                low = 0.0 if row["off_allowed"] else float(row["lo"])
                entered[row["field"]] = cols[i % 2].number_input(
                    f"{row['label']} ({row['unit']})" + (" - changed" if row["changed"] else ""),
                    min_value=low if kind_float else int(low), max_value=float(row["hi"]) if kind_float else int(row["hi"]),
                    value=float(row["current"]) if kind_float else int(row["current"]),
                    step=0.05 if kind_float else 1,
                    help=f"{row['help']} Allowed {row['lo']:g} to {row['hi']:g}. Default {row['default']:g}.",
                    key=f"{k}_lim_{row['field']}")
            reason = st.text_input("Reason (needed when loosening)", key=f"{k}_lim_reason")
            confirm = st.checkbox("I understand this loosens a safety limit", key=f"{k}_lim_confirm")
            apply_clicked = st.form_submit_button(f"Apply {market} changes")
        if apply_clicked:
            result = LO.apply(entered, reason=reason, confirm_loosen=confirm, kill_switch=engine.kill_switch, market=market)
            if not result.ok:
                for problem in result.errors:
                    st.error(problem)
            elif not result.applied:
                st.info("Nothing changed.")
            else:
                for field, move in result.applied.items():
                    st.success(f"{LO.BY_FIELD[field].label}: {move['old']:g} -> {move['new']:g} ({move['direction']})")
                st.caption("Applied. The figures above refresh on the next interaction.")
        with st.expander(f"Reset {market} limits to defaults"):
            st.caption(f"Back to the values the app started with (environment or built-in defaults) for {market} only, "
                       "and forget its saved changes.")
            reset_ok = st.checkbox("I understand this may loosen limits that I tightened", key=f"{k}_lim_reset_confirm")
            if st.button(f"Reset all {market} limits to defaults", key=f"{k}_lim_reset"):
                outcome = LO.reset_to_defaults(reason=reason or "reset to defaults", confirm_loosen=reset_ok,
                                               kill_switch=engine.kill_switch, market=market)
                for problem in outcome.errors:
                    st.error(problem)
                if outcome.ok:
                    st.success(f"{market} limits reset to defaults.")
        if not is_durable():
            st.caption(
                "Saved changes live in this app's local database. On hosts with a temporary disk (such as Streamlit Cloud) "
                "they are lost when the app restarts: set DATABASE_URL to a hosted database, or put the values in the app's "
                "Secrets, to keep them. The app is shared: anyone who can open this page can change these limits."
            )
        changes = LO.recent_changes(market=market)
        if changes:
            st.markdown(f"**Recent {market} limit changes**")
            st.dataframe(changes, width="stretch", hide_index=True)

    with st.expander(f"All configured {market} limits (read-only)"):
        st.json(dataclasses.asdict(limits))

    with st.expander(f"Reset {market} drawdown peak"):
        st.caption(
            f"Starts a fresh drawdown peak at the current {market} equity, so the drawdown throttle and the "
            f"{limits.max_drawdown_pct:g}% hard stop measure from here. Use it only on purpose, e.g. after "
            "reviewing a losing stretch or adding capital."
        )
        confirm_reset = st.checkbox("I understand this clears the current drawdown", key=f"{k}_confirm_peak_reset")
        if st.button("Reset peak to current equity", disabled=not confirm_reset, key=f"{k}_peak_reset"):
            new_peak = engine.reset_drawdown_peak()
            st.success(f"{market} drawdown peak reset to ₹{new_peak:,.0f}")
            st.rerun()

    st.divider()
    st.subheader(f"{market} emergency kill switch")
    kill = st.toggle(f"Engage the {market} kill switch (blocks ALL new {market} trades immediately)",
                     value=engine.kill_switch, key=f"{k}_kill")
    engine.kill_switch = kill
    if kill:
        st.error(f"{market} KILL SWITCH ENGAGED. No new {market} trades will be approved by the risk engine.")

    st.divider()
    st.subheader(f"Recent {market} risk events (audit trail)")
    with get_session() as session:
        events = session.query(RiskEvent).order_by(RiskEvent.id.desc()).limit(300).all()
    rows = [
        {"Timestamp": e.timestamp, "Decision ID": e.decision_id, "Type": e.event_type, "Reason": e.reason}
        for e in events if _event_market(e) == market
    ][:100]
    st.dataframe(rows, width="stretch", hide_index=True)
