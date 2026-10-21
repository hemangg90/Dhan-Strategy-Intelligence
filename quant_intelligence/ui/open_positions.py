"""Shared live open-positions block (cards with live P&L, stop/target bar, manual exit).

Used by the Positions page, the Paper Trading page and both auto-trading pages, so every page shows the
same book the same way. Live prices come from the shared quote cache (ui never polls Dhan on its own).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from quant_intelligence.brokers.dhan_api_client import DhanApiClient
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.engine import ENGINE, MCX_ENGINE
from quant_intelligence.execution.position_monitor import fetch_option_ltp_map
from quant_intelligence.ui import format as F
from quant_intelligence.ui.actions import exit_position, square_off_all
from quant_intelligence.ui.components import empty_state, position_card
from quant_intelligence.ui.position_views import positions_frame, with_live
from quant_intelligence.utils.timeutil import now_ist


def live_open() -> tuple[pd.DataFrame, str | None]:
    """Open paper positions of both markets priced with live premiums (from the shared cache). Returns (frame, error).
    NSE and MCX are separate accounts, so the book is the two brokers' open positions together."""
    rows: list[dict] = []
    ltp_map: dict = {}
    error = None
    client = DhanApiClient()
    for engine in (ENGINE, MCX_ENGINE):
        engine_rows = engine.broker.get_open_positions()
        rows += engine_rows
        if engine_rows and client.is_configured():
            try:
                ltp_map.update(fetch_open_ltp(engine.broker, client))
            except Exception as e:  # a failed quote fetch must not blank the page
                error = str(e)
    return with_live(positions_frame(rows, now=now_ist()), ltp_map), error


def fetch_open_ltp(broker, client) -> dict:
    return fetch_option_ltp_map(broker, client)


def _request_exit(position_id: str) -> None:
    st.session_state["pending_exit"] = position_id


def render_open_positions(market: str | None = None, key: str = "op", toolbar: bool = True,
                          empty_hint: str = "Trades the auto-traders or you place will appear here with live P&L.") -> None:
    """Live cards for open positions. `market` ('NSE'/'MCX') limits the list; `key` namespaces widgets so the
    block can appear on several pages. Refreshes itself every few seconds."""

    @st.fragment(run_every=f"{SETTINGS.ltp_refresh_seconds}s")
    def _view() -> None:
        open_df, error = live_open()
        if market:
            open_df = open_df[open_df["market"] == market]
        if open_df.empty:
            empty_state("No open positions", empty_hint)
            return

        if toolbar:
            c1, c2, c3 = st.columns([2, 2, 1])
            markets = ["All"] + sorted(open_df["market"].unique())
            pick = (c1.segmented_control("Market", markets, default="All", key=f"{key}_market",
                                         label_visibility="collapsed") if not market else "All") or "All"
            order_by = c2.selectbox("Sort by", ["Newest", "Largest P&L", "Worst P&L", "Closest to stop"],
                                    label_visibility="collapsed", key=f"{key}_sort")
            if pick != "All":
                open_df = open_df[open_df["market"] == pick]
            col, asc = {"Largest P&L": ("unrealised", False), "Worst P&L": ("unrealised", True),
                        "Newest": ("opened_at", False), "Closest to stop": ("progress", True)}[order_by]
            open_df = open_df.sort_values(col, ascending=asc, na_position="last")
            if c3.button("Exit all", key=f"{key}_exit_all", help="Close every listed open paper position at its latest price"):
                st.session_state["pending_exit"] = "__ALL__"

        pending = st.session_state.get("pending_exit")
        if pending:
            label = "ALL open positions" if pending == "__ALL__" else next(
                (r["contract"] for _, r in open_df.iterrows() if r["position_id"] == pending), None)
            if label:
                with st.container(border=True):
                    st.warning(f"Exit **{label}** at the live price? This closes the paper position now.")
                    b1, b2, _ = st.columns([1, 1, 4])
                    if b1.button("Confirm exit", type="primary", key=f"{key}_confirm"):
                        if pending == "__ALL__":
                            done = square_off_all()
                            st.toast(f"Closed {len(done)} position(s)" if done else "Nothing closed (no live prices)")
                        else:
                            ok, msg = exit_position(pending)
                            st.toast(msg, icon=None if ok else "⚠️")
                        st.session_state.pop("pending_exit", None)
                        st.rerun()
                    if b2.button("Cancel", key=f"{key}_cancel"):
                        st.session_state.pop("pending_exit", None)
                        st.rerun(scope="fragment")

        invested = float(open_df["invested"].sum())
        unreal = float(open_df["unrealised"].sum(skipna=True))
        charges = float(open_df["unrealised_charges"].sum(skipna=True))
        st.caption(
            f"{len(open_df)} open · invested {F.inr(invested)} · unrealised {F.inr(unreal, signed=True)} "
            f"· est. charges if closed now {F.inr(charges)} · prices updated {now_ist():%H:%M:%S} IST"
        )
        for _, row in open_df.iterrows():
            position_card(row, key=f"{key}_{row['position_id']}", on_exit=_request_exit)
        if error:
            st.warning(f"Live prices unavailable: {error}")
        elif open_df["ltp"].isna().any():
            st.caption("Some contracts have no live quote yet; their P&L shows as empty.")

    _view()
