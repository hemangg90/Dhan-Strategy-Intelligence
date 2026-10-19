"""
Dashboard (the app's home view). Rendered through the navigation router in app.py; it can also run standalone:
    streamlit run quant_intelligence/home_page.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datetime as dt

import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.database.db import init_db
from quant_intelligence.ui.components import chip
from quant_intelligence.ui.credentials_panel import render_credentials_panel
from quant_intelligence.ui.dashboard import decision_banner, hero_header, kpi_cards, probability_bars, ranking_table
from quant_intelligence.ui.decision_view import explanation_rows
from quant_intelligence.utils.timeutil import now_ist
from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme, status_badge

init_db()
apply_theme()
init_session_state()

with st.sidebar:
    st.header("Instrument & Data")
    st.session_state["instrument"] = st.text_input("Instrument", st.session_state["instrument"])
    st.session_state["timeframe"] = st.selectbox("Timeframe", ["5min", "15min", "1min", "3min"], index=0)
    st.session_state["lookback_days"] = st.slider("Lookback window (days)", 10, 180, st.session_state["lookback_days"])

    if st.button("Refresh pipeline", type="primary"):
        with st.spinner("Running data -> features -> regime -> strategy ranking pipeline..."):
            run_pipeline_cached(force=True)

    st.divider()
    st.header("API Keys")
    render_credentials_panel()

    st.divider()
    st.header("Trading Mode")
    if SETTINGS.is_live_mode and SETTINGS.live_mode_fully_authorized:
        st.markdown('<span class="qi-badge qi-badge-live">LIVE MODE ENABLED</span>', unsafe_allow_html=True)
        st.error("LIVE mode is active. Real orders can be sent from the Live Options page after manual confirmation.")
    elif SETTINGS.is_live_mode:
        st.markdown('<span class="qi-badge qi-badge-notrade">LIVE requested, NOT authorized</span>', unsafe_allow_html=True)
        st.warning("TRADING_MODE=LIVE but TRADING_LIVE_CONFIRM is missing/incorrect. Falling back to PAPER behavior.")
    else:
        st.markdown('<span class="qi-badge qi-badge-paper">PAPER MODE</span>', unsafe_allow_html=True)
        st.success("All orders are simulated. No real orders can be sent.")

output = st.session_state.get("pipeline_output")
error = st.session_state.get("pipeline_error")

if output is None and error is None:
    with st.spinner("Running initial pipeline on live Dhan / CSV market data..."):
        output = run_pipeline_cached()
        error = st.session_state.get("pipeline_error")

if error:
    st.title("Quant Strategy Intelligence")
    st.error(f"Pipeline could not run: {error}")
    st.stop()

def _age_text(ts) -> str:
    try:
        minutes = (now_ist().replace(tzinfo=None) - ts.replace(tzinfo=None)).total_seconds() / 60
    except Exception:
        return "age n/a"
    return f"{minutes / 60:.1f} h old" if minutes > 120 else f"{max(minutes, 0):.0f} min old"


quality = output.data_quality_status
ok_quality = quality == "OK"
_prov = output.provenance or {}
mode_chips = [
    chip(f"Data {quality}", "good" if ok_quality else "critical"),
    chip(_prov.get("mode", "n/a"), "good" if _prov.get("mode") == "LIVE" else "warning" if _prov.get("mode") == "DELAYED" else "neutral"),
    chip("PAPER MODE", "info"),
    chip("LIVE ORDERS LOCKED" if SETTINGS.demo_lockdown else "Orders simulated", "neutral"),
]
hero_header(
    "Quant Strategy Intelligence",
    f"<b>{output.instrument}</b> &middot; last bar {output.timestamp:%d %b %Y %H:%M} ({_age_text(output.timestamp)}) "
    "&middot; research-first terminal for Indian index derivatives",
    mode_chips,
)

ranking = output.ranking
eligible = sum(1 for s in output.strategy_intel if s.score.eligible)
kpi_cards([
    {"label": "Market regime", "value": str(output.regime_label).replace("_", " ").title(),
     "sub": f"top score share {output.regime_confidence * 100:.0f}% (heuristic, not a probability)"},
    {"label": "Data quality", "value": quality, "sub": f"{len(output.data_quality_issues or [])} issue(s)",
     "tone": "good" if ok_quality else "critical"},
    {"label": "Strategies evaluated", "value": len(output.strategy_intel), "sub": "17-strategy library"},
    {"label": "Eligible now", "value": eligible, "sub": "clear edge and confidence bars",
     "tone": "good" if eligible else "warning"},
])
for issue in output.data_quality_issues or []:
    st.warning(str(issue))

decision_banner(ranking.is_no_trade, ranking.selected_strategy, ranking.reason, ranking.tie_break)
st.caption("A decision is not an order: it still has to pass the independent risk engine and a real setup must trigger. "
           "Results are historical statistics, not predictions.")

left, right = st.columns([3, 1], gap="large")
with left:
    st.subheader("Strategy ranking")
    ranking_table(explanation_rows(output), show_why=False)
with right:
    st.subheader("Regime scores")
    probability_bars(output.regime_probabilities)
    st.caption("Normalised rule-based scores, not calibrated probabilities: nothing has been fitted to outcomes, and regimes overlap.")

st.divider()
st.caption(
    "Decision Center explains every ranking. Use the sidebar for Market State, Strategy Intelligence, Backtest Lab, "
    "Historical Analogues, Paper Trading, Risk Control, Research Reports, the Daily Report and the NSE Heatmap."
)
