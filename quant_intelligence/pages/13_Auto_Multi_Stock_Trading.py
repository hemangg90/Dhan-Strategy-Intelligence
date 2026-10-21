"""Auto Multi-Instrument Trading.

The index underlyings (SETTINGS.option_underlyings: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY,
SENSEX by default) and the fixed stock watchlist (config/watchlist.py) are pipelined, ranked and
automated together in one pass, and EVERY trade is an option (index and stock options alike).
All NSE instruments share one result table and the NSE paper account and risk limits (MCX has its own).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.ui.components import page_header
from quant_intelligence.ui.multi_instrument_panel import render_multi_instrument_panel
from quant_intelligence.ui.state import init_session_state
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
page_header(
    "Auto Multi-Instrument Trading",
    f"{', '.join(SETTINGS.option_underlyings)} and the stock watchlist - every trade is an option. Pipeline → ranking → risk engine → paper broker.",
)
render_multi_instrument_panel()
