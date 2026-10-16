import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_intelligence.features.feature_engine import FEATURE_DEFINITIONS
from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Market State")

output = run_pipeline_cached()
if output is None:
    st.error(st.session_state.get("pipeline_error", "Pipeline not available."))
    st.stop()

st.caption(f"{output.instrument} feature vector as of {output.timestamp} (data quality: {output.data_quality_status})")

groups = {
    "Trend / Momentum": ["return_1", "return_5", "return_20", "trend_slope", "momentum_20", "market_structure"],
    "Volatility": ["atr_14", "atr_pct_of_price", "atr_percentile_100", "realized_vol_20", "vol_expansion"],
    "Liquidity / Volume": ["relative_volume", "volume_acceleration"],
    "VWAP": ["vwap", "vwap_distance_pct"],
    "Session / Time": ["time_since_open_min", "session_phase", "opening_range_high", "opening_range_low", "gap_pct"],
    "Derivatives (not yet connected)": ["futures_basis", "oi", "oi_change_pct", "pcr", "iv", "iv_percentile", "has_derivatives_data"],
    "Event": ["expiry_proximity_days"],
}

for group_name, cols in groups.items():
    st.subheader(group_name)
    fcols = st.columns(min(len(cols), 4))
    for i, c in enumerate(cols):
        val = output.market_state.features.get(c)
        display_val = "N/A" if val is None or (isinstance(val, float) and pd.isna(val)) else (
            round(val, 4) if isinstance(val, float) else val
        )
        fcols[i % len(fcols)].metric(c, str(display_val))

st.divider()
st.subheader("Feature history (last 300 bars)")
tail = output.features.tail(300)
metric_choice = st.selectbox("Feature to chart", [c for c in output.features.columns if c != "timestamp"], index=list(output.features.columns).index("atr_percentile_100") - 1 if "atr_percentile_100" in output.features.columns else 0)
fig = go.Figure()
fig.add_trace(go.Scatter(x=tail["timestamp"], y=tail[metric_choice], mode="lines", name=metric_choice))
fig.update_layout(template="plotly_white", height=350, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Feature definitions")
with st.expander("Show all feature definitions"):
    for k, v in FEATURE_DEFINITIONS.items():
        st.markdown(f"**{k}**: {v}")
