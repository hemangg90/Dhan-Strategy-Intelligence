"""Commodity Auto Trading (MCX).

Crude oil, crude oil mini and natural gas are pipelined, ranked and automated together,
and every trade is an option on the front-month future. Runs on its own background runner with
its own start/stop and MCX session hours (09:00-23:55 IST), independently of - and at the same
time as - the NSE runner on the Auto Multi-Instrument page. MCX has its OWN paper account, risk
limits, drawdown and kill switch (see Risk Control: MCX); nothing is shared with NSE.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.execution.engine import COMMODITY_RUNNER
from quant_intelligence.ui.components import page_header
from quant_intelligence.ui.multi_instrument_panel import render_multi_instrument_panel
from quant_intelligence.ui.state import init_session_state
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
page_header(
    "Commodity Auto Trading (MCX)",
    "Crude oil, crude oil mini and natural gas - every trade is an option on the front-month future. Runs "
    "independently of the NSE auto-trader, with its own paper account and risk limits.",
)
with st.expander("Lot sizes - please verify", expanded=False):
    st.warning(
        "Lot sizes are MCX contract multipliers from config/watchlist.py (Dhan's scrip master reports 1 for "
        "every MCX contract). Verify them against the exchange before trusting P&L figures."
    )
render_multi_instrument_panel(COMMODITY_RUNNER, commodities=True)
