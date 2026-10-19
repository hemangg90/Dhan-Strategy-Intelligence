"""
Quant Strategy Intelligence & Execution System - entry point and navigation.

Run with:  streamlit run quant_intelligence/app.py
(run from the project root so the `quant_intelligence` package resolves).

This file only routes. Pages are grouped by workflow into five workspaces shown as a top navigation bar; every page
file keeps its own content and its own `apply_theme()` call.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

HERE = Path(__file__).resolve().parent


def _page(path: str, title: str, default: bool = False):
    return st.Page(str(HERE / path), title=title, default=default)


WORKSPACES = {
    "Market Intelligence": [
        _page("home_page.py", "Dashboard", default=True),
        _page("pages/01_Market_State.py", "Market State"),
        _page("pages/05_Regime_Analysis.py", "Regime Analysis"),
        _page("pages/16_NSE_Heatmap.py", "NSE Heatmap"),
    ],
    "Strategy Intelligence": [
        _page("pages/00_Decision_Center.py", "Decision Center"),
        _page("pages/02_Strategy_Intelligence.py", "Strategy Ranking"),
        _page("pages/03_Strategy_Library.py", "Strategy Library"),
    ],
    "Research Lab": [
        _page("pages/04_Backtest_Lab.py", "Backtest Workbench"),
        _page("pages/06_Historical_Analogues.py", "Historical Analogues"),
        _page("pages/10_Research_Reports.py", "Research Reports"),
    ],
    "Risk & Execution": [
        _page("pages/09_Risk_Control.py", "Risk Control"),
        _page("pages/07_Paper_Trading.py", "Paper Trading"),
        _page("pages/13_Auto_Multi_Stock_Trading.py", "Auto Trading: Stocks"),
        _page("pages/14_Commodity_Auto_Trading.py", "Auto Trading: Commodities"),
        _page("pages/12_Live_Options_Trading.py", "Live Orders (locked by default)"),
    ],
    "Performance & Reports": [
        _page("pages/08_Positions_and_Orders.py", "Positions & Orders"),
        _page("pages/15_Daily_Report.py", "Daily Report"),
        _page("pages/11_System_Logs_Health.py", "System Health"),
    ],
}

# Compute the default view once per process, in the background, so the first visitor (or a refreshed tab) does not wait
# for the research pipeline. Later sessions read the shared result.
if not st.session_state.get("_warm_requested"):
    from quant_intelligence.config.settings import SETTINGS
    from quant_intelligence.research.shared_cache import PIPELINE

    PIPELINE.warm(SETTINGS.default_instrument, SETTINGS.default_timeframe, 60)
    from quant_intelligence.regimes.universe import UNIVERSE

    UNIVERSE.warm()
    st.session_state["_warm_requested"] = True

navigation = st.navigation(WORKSPACES, position="top")
navigation.run()
