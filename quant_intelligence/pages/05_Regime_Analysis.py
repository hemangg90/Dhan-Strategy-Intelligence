import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_intelligence.regimes.regime_engine import REGIMES, transition_matrix
from quant_intelligence.regimes.universe import UNIVERSE
from quant_intelligence.regimes.validation import regime_forward_stats
from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Regime Analysis")
st.caption("Transparent rule-based classifier. Scores are normalised heuristics, NOT calibrated probabilities, and the regime "
           "is display-only: it is not an input to the ranking or to any order. Trend strength is measured against each "
           "instrument's own usual range, and the label only changes after a challenger has led for a few bars.")

tab_one, tab_all = st.tabs(["Selected instrument", "All instruments"])

with tab_one:
    output = run_pipeline_cached()
    if output is None:
        st.error(st.session_state.get("pipeline_error", "Pipeline not available."))
    else:
        axes = output.regime_axes or {}
        st.subheader(f"{output.instrument}: {output.regime_label.replace('_', ' ').title()}")
        a, b, c, d = st.columns(4)
        a.metric("Direction", str(axes.get("direction", "n/a")).title())
        b.metric("Volatility", f"{str(axes.get('volatility', 'n/a')).title()} · {str(axes.get('volatility_trend', '')).title()}")
        c.metric("Clarity", "n/a" if output.regime_clarity is None else f"{output.regime_clarity * 100:.0f}%",
                 help="Margin between the top two scores plus agreement of the label with direction and volatility. A heuristic, not a probability.")
        d.metric("In this regime for", "n/a" if output.regime_age_bars is None else f"{output.regime_age_bars} bars")

        fig = go.Figure(go.Bar(x=list(output.regime_probabilities.keys()), y=list(output.regime_probabilities.values())))
        fig.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Normalised score share")
        st.plotly_chart(fig, width="stretch")

        st.divider()
        st.subheader("Regime history over the lookback window")
        hist = output.regime_series
        if hist is None or hist.empty:
            st.info("No regime history available.")
        else:
            order = [r for r in REGIMES if r in set(hist["label"])]
            fig2 = go.Figure(go.Scatter(x=hist["timestamp"], y=[order.index(x) for x in hist["label"]], mode="markers",
                                        marker=dict(size=5), text=hist["label"], hovertemplate="%{x}<br>%{text}<extra></extra>"))
            fig2.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=30, b=20),
                               yaxis=dict(tickmode="array", tickvals=list(range(len(order))), ticktext=order))
            st.plotly_chart(fig2, width="stretch")
            switches = int((hist["label"] != hist["label"].shift()).sum() - 1)
            raw_switches = int((hist["raw_label"] != hist["raw_label"].shift()).sum() - 1)
            st.caption(f"The label changed {switches} times over {len(hist)} bars (bar-by-bar labelling would have changed it "
                       f"{raw_switches} times).")

            left, right = st.columns(2)
            with left:
                st.subheader("Regime frequency")
                st.bar_chart(hist["label"].value_counts())
            with right:
                st.subheader("What followed each label")
                stats = regime_forward_stats(output.ohlcv, hist)
                if stats.empty:
                    st.info("Not enough data.")
                else:
                    shown = stats.assign(
                        fwd_vol_vs_all=stats["fwd_vol_vs_all"].round(2),
                        fwd_ret_bp=(stats["fwd_ret"] * 1e4).round(1),
                    )[["regime", "n", "n_days", "fwd_vol_vs_all", "fwd_ret_bp"]]
                    st.dataframe(shown.rename(columns={"fwd_vol_vs_all": "next-hour volatility vs all", "fwd_ret_bp": "next-hour return (bp)"}),
                                 hide_index=True, width="stretch")
                    st.caption("Descriptive, from this instrument's own history: after the label, how volatile was the next hour "
                               "(1.00 = same as overall) and what was the mean return. Small n means treat with caution; "
                               "trend labels are not claimed to predict direction.")
            with st.expander("How often one label follows another"):
                tm = transition_matrix(hist["label"])
                st.dataframe(tm.round(2), width="stretch")

with tab_all:
    st.caption("Every instrument with cached candles, from the local cache only (no broker calls). Instruments whose data "
               "fails validation show their status and no regime. Refreshes about every 5 minutes.")
    scan = UNIVERSE.peek("5min")
    rows = pd.DataFrame(scan or [])
    if scan is None:
        st.info("Scanning all instruments in the background. Reload this tab in a minute.")
    elif rows.empty:
        st.info("No instruments found.")
    else:
        view = rows.assign(
            regime=rows["regime"].fillna("-").str.replace("_", " "),
            confidence=(rows["confidence"] * 100).round(0),
            clarity=(rows["clarity"] * 100).round(0),
        )[["instrument", "kind", "regime", "direction", "volatility", "volatility_trend", "age_bars", "clarity", "data_status", "category", "last_bar", "note"]]
        st.dataframe(view, hide_index=True, width="stretch", height=min(60 + 35 * len(view), 900))
        counts = rows["regime"].value_counts()
        st.caption("Across the universe: " + ", ".join(f"{k.replace('_', ' ').title()} {v}" for k, v in counts.items()) + ".")
        stale = rows[rows["category"] == "STALE"]
        if len(stale):
            st.warning(f"{len(stale)} instruments have data older than the market clock allows, so these regimes describe the last "
                       "session, not the live market.")

st.caption(f"All regimes tracked: {', '.join(REGIMES)}")
