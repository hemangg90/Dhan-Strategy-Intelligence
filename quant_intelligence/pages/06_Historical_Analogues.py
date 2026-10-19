import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.analogues.analogue_engine import ALL_FEATURES as COMPARISON_FEATURES
from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Historical Analogues")
st.caption(
    "The evidence behind conditional strategy selection: historical observations most similar "
    "to the CURRENT market state, and how each strategy performed in them."
)

output = run_pipeline_cached()
if output is None:
    st.error(st.session_state.get("pipeline_error", "Pipeline not available."))
    st.stop()

st.subheader("Current market state (comparison features)")
current_vals = {f: output.market_state.get(f) for f in COMPARISON_FEATURES}
st.json(current_vals)

st.divider()
choice = st.selectbox("Strategy", [s.strategy_name for s in output.strategy_intel])
sel = next(s for s in output.strategy_intel if s.strategy_name == choice)

if len(sel.analogues) == 0:
    st.warning("No historical analogues found for this strategy in the current lookback window.")
else:
    st.subheader(f"Top {len(sel.analogues)} historical analogues for {choice}")
    display_cols = ["entry_timestamp", "similarity_score", "r_multiple", "outcome"] + [
        c for c in COMPARISON_FEATURES if c in sel.analogues.columns
    ]
    st.dataframe(sel.analogues[display_cols].round(4), width="stretch", hide_index=True, height=400)

    st.subheader("Conditional expectancy computed from these analogues")
    st.json(sel.conditional_metrics)

    if sel.conditional_metrics.get("confidence_label") == "INSUFFICIENT_DATA":
        st.error(
            "INSUFFICIENT DATA: fewer than 10 comparable historical analogues exist. "
            "This strategy's conditional edge cannot be estimated reliably right now."
        )
    elif sel.conditional_metrics.get("confidence_label") == "LOW":
        st.warning("LOW CONFIDENCE: the analogues cover fewer than 6 separate trading days (rows from one session count as one "
                   "piece of evidence). Treat the conditional expectancy estimate with caution.")
    if sel.conditional_metrics.get("analogues_informative") is False:
        st.info("Analogues add no detectable information here: sets built for arbitrary other days differ from the strategy's "
                f"overall mean about as much as this one (p = {sel.conditional_metrics.get('conditioning_p_value'):.2f}), "
                "so the overall estimate is shown instead.")
