import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import RiskEvent
from quant_intelligence.execution.engine import ENGINE
from quant_intelligence.risk.risk_engine import drawdown_pct, loss_streak, risk_multiplier
from quant_intelligence.ui.state import get_account_state, init_session_state
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Risk Control")
st.caption("Deterministic risk engine - independent of the research/AI layer, with VETO authority.")

account = get_account_state()
limits = SETTINGS.risk
multiplier, why = risk_multiplier(account)
dd = drawdown_pct(account)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Equity (marked to market)", f"₹{account.equity:,.0f}",
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
    st.warning(f"New trades are sized at {multiplier:.0%} of normal: {why}.")

st.subheader("Risk limits")
from quant_intelligence.database.db import is_durable  # noqa: E402
from quant_intelligence.risk import limit_overrides as LO  # noqa: E402

if SETTINGS.demo_lockdown:
    st.info("Demo lockdown is on: limits are shown but cannot be changed here.")
else:
    st.caption(
        "Tightening a limit applies at once. Loosening needs the confirmation tick and a reason. Each value has a hard "
        "ceiling the app will not exceed. New values apply to new trades; open positions are not re-judged."
    )
    with st.form("risk_limits_form"):
        entered: dict = {}
        cols = st.columns(2)
        for i, row in enumerate(LO.rows()):
            kind_float = row["kind"] == "float"
            low = 0.0 if row["off_allowed"] else float(row["lo"])
            entered[row["field"]] = cols[i % 2].number_input(
                f"{row['label']} ({row['unit']})" + (" - changed" if row["changed"] else ""),
                min_value=low if kind_float else int(low), max_value=float(row["hi"]) if kind_float else int(row["hi"]),
                value=float(row["current"]) if kind_float else int(row["current"]),
                step=0.05 if kind_float else 1, help=f"{row['help']} Allowed {row['lo']:g} to {row['hi']:g}. Default {row['default']:g}.",
                key=f"lim_{row['field']}")
        reason = st.text_input("Reason (needed when loosening)", key="lim_reason")
        confirm = st.checkbox("I understand this loosens a safety limit", key="lim_confirm")
        apply_clicked = st.form_submit_button("Apply changes")
    if apply_clicked:
        result = LO.apply(entered, reason=reason, confirm_loosen=confirm, kill_switch=ENGINE.kill_switch)
        if not result.ok:
            for problem in result.errors:
                st.error(problem)
        elif not result.applied:
            st.info("Nothing changed.")
        else:
            for field, move in result.applied.items():
                st.success(f"{LO.BY_FIELD[field].label}: {move['old']:g} -> {move['new']:g} ({move['direction']})")
            st.caption("Applied. The figures above refresh on the next interaction.")
    with st.expander("Reset to defaults"):
        st.caption("Back to the values the app started with (environment or built-in defaults) and forget the saved changes.")
        reset_ok = st.checkbox("I understand this may loosen limits that I tightened", key="lim_reset_confirm")
        if st.button("Reset all limits to defaults"):
            outcome = LO.reset_to_defaults(reason=reason or "reset to defaults", confirm_loosen=reset_ok, kill_switch=ENGINE.kill_switch)
            for problem in outcome.errors:
                st.error(problem)
            if outcome.ok:
                st.success("Limits reset to defaults.")
    if not is_durable():
        st.caption(
            "Saved changes live in this app's local database. On hosts with a temporary disk (such as Streamlit Cloud) they "
            "are lost when the app restarts: set DATABASE_URL to a hosted database, or put the values in the app's Secrets, "
            "to keep them. The app is shared: anyone who can open this page can change these limits."
        )
    changes = LO.recent_changes()
    if changes:
        st.markdown("**Recent limit changes**")
        st.dataframe(changes, width="stretch", hide_index=True)

with st.expander("All configured limits (read-only)"):
    st.json(dataclasses.asdict(SETTINGS.risk))

with st.expander("Reset drawdown peak"):
    st.caption(
        "Starts a fresh drawdown peak at the current equity, so the drawdown throttle and the "
        f"{limits.max_drawdown_pct:g}% hard stop measure from here. Use it only on purpose, e.g. after "
        "reviewing a losing stretch or adding capital."
    )
    confirm_reset = st.checkbox("I understand this clears the current drawdown", key="confirm_peak_reset")
    if st.button("Reset peak to current equity", disabled=not confirm_reset):
        new_peak = ENGINE.reset_drawdown_peak()
        st.success(f"Drawdown peak reset to ₹{new_peak:,.0f}")
        st.rerun()

st.divider()
st.subheader("Emergency kill switch")
kill = st.toggle("Engage kill switch (blocks ALL new trades immediately)", value=ENGINE.kill_switch)
ENGINE.kill_switch = kill
if kill:
    st.error("KILL SWITCH ENGAGED. No new trades will be approved by the risk engine.")

st.divider()
st.subheader("Recent risk events (audit trail)")
with get_session() as session:
    events = session.query(RiskEvent).order_by(RiskEvent.id.desc()).limit(100).all()
rows = [
    {"Timestamp": e.timestamp, "Decision ID": e.decision_id, "Type": e.event_type, "Reason": e.reason}
    for e in events
]
st.dataframe(rows, width="stretch", hide_index=True)
