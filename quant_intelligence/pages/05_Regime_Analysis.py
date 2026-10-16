import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_intelligence.regimes.regime_engine import REGIMES, classify_regime
from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Regime Analysis")
st.caption("Transparent rule/statistical regime classifier. Probabilities, not just a single label.")

output = run_pipeline_cached()
if output is None:
    st.error(st.session_state.get("pipeline_error", "Pipeline not available."))
    st.stop()

st.subheader(f"Current regime: {output.regime_label} ({output.regime_confidence*100:.0f}% confidence)")
fig = go.Figure(go.Bar(x=list(output.regime_probabilities.keys()), y=list(output.regime_probabilities.values())))
fig.update_layout(template="plotly_white", height=350, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Probability")
st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Regime history over the lookback window")
with st.spinner("Classifying regime for each historical bar..."):
    hist = []
    feat_cols = ["trend_slope", "momentum_20", "atr_percentile_100", "vol_expansion", "relative_volume", "market_structure", "expiry_proximity_days"]
    sampled = output.features.iloc[::5]  # sample every 5th bar to keep this fast
    for _, row in sampled.iterrows():
        feats = {c: row.get(c) for c in feat_cols}
        label, probs = classify_regime(feats)
        hist.append({"timestamp": row["timestamp"], "regime": label, "confidence": max(probs.values())})
    hist_df = pd.DataFrame(hist)

fig2 = go.Figure()
for regime in hist_df["regime"].unique():
    sub = hist_df[hist_df["regime"] == regime]
    fig2.add_trace(go.Scatter(x=sub["timestamp"], y=sub["confidence"], mode="markers", name=regime))
fig2.update_layout(template="plotly_white", height=400, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Classifier confidence")
st.plotly_chart(fig2, width="stretch")

st.subheader("Regime frequency")
st.bar_chart(hist_df["regime"].value_counts())

st.caption(f"All regimes tracked: {', '.join(REGIMES)}")
