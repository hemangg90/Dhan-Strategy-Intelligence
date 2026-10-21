import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.ui.guide_content import GLOSSARY, HOW_TO_READ, LIMITS, STAGES
from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Guide")
st.caption("What each screen means and how to read a result. Everything in this product describes the past or a simulation.")

st.subheader("How to read a candidate")
for line in HOW_TO_READ:
    st.markdown(f"- {line}")

st.subheader("The path every candidate follows")
for i, (name, text) in enumerate(STAGES, start=1):
    st.markdown(f"**{i}. {name}**: {text}")

st.subheader("Walkthrough on the current data")
st.caption("This uses the data the app has right now. Check the label: if the market is closed you are looking at the last "
           "completed session (HISTORICAL), not live prices.")
output = run_pipeline_cached()
if output is None:
    st.info("The research pipeline is not available right now, so there is nothing to walk through. "
            + str(st.session_state.get("pipeline_error") or ""))
else:
    from quant_intelligence.ui.candidate import build_candidate
    from quant_intelligence.ui.state import account_state_for

    prov = output.provenance or {}
    st.markdown(f"**{output.instrument}**, data currency **{prov.get('mode', 'n/a')}**, last bar {output.timestamp:%d %b %H:%M}.")
    walk = build_candidate(output, account_state_for(output.instrument))
    for stage in walk["stages"]:
        st.markdown(f"- **{stage['title']}** ({stage['state']}): {stage['headline']}")
    st.markdown(f"**Result: {walk['verdict']}.** {walk['reason']}")

st.subheader("Glossary")
query = st.text_input("Search the glossary", key="guide_search").strip().lower()
shown = {k: v for k, v in GLOSSARY.items() if not query or query in k.lower() or query in v.lower()}
for term, meaning in shown.items():
    st.markdown(f"**{term}**: {meaning}")
if not shown:
    st.caption("No matching term.")

st.subheader("What this product cannot tell you")
for line in LIMITS:
    st.markdown(f"- {line}")
