import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import html

import pandas as pd
import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data.provenance import strip_text
from quant_intelligence.ui.candidate import CURRENT_CANDIDATE, build_candidate
from quant_intelligence.ui.components import chip
from quant_intelligence.ui.dashboard import decision_banner, hero_header
from quant_intelligence.ui.state import get_account_state, init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()

TONE = {"PASS": "good", "WARN": "warning", "FAIL": "critical", "WAIT": "info", "SKIP": "muted", "INFO": "info"}

output = run_pipeline_cached()
if output is None:
    st.title("Candidate")
    st.error(st.session_state.get("pipeline_error", "Pipeline not available."))
    st.stop()

recent = []
try:
    from quant_intelligence.database.db import get_session
    from quant_intelligence.reports.trace import trade_trace

    with get_session() as session:
        recent = [t for t in trade_trace(session, limit=20, instrument=output.instrument) if t["strategy"] == output.ranking.selected_strategy]
except Exception:
    recent = []  # the outcome stage then reports "no paper trade yet"

candidate = build_candidate(output, get_account_state(), recent)

prov = output.provenance or {}
hero_header(
    "Candidate",
    f"Follow one candidate from its data to the risk engine's verdict. <b>{html.escape(output.instrument)}</b> &middot; "
    f"{html.escape(strip_text(prov))}",
    [chip(prov.get("mode", "n/a"), "good" if prov.get("mode") == "LIVE" else "info"),
     chip("PAPER MODE", "info"), chip("LIVE ORDERS LOCKED" if SETTINGS.demo_lockdown else "Orders simulated", "neutral")],
)

is_ready = candidate["verdict"] == "CANDIDATE READY"
decision_banner(not is_ready, candidate["strategy"], f"{candidate['verdict']}: {candidate['reason']}")
if candidate["blocking_stage"]:
    st.caption(f"Stopped at: **{candidate['blocking_stage']}**")

st.subheader("Stage by stage")
for stage in candidate["stages"]:
    with st.container(border=True):
        left, right = st.columns([1, 6], vertical_alignment="center")
        left.markdown(f"**{stage['title']}**")
        right.markdown(chip(stage["state"], TONE[stage["state"]]) + f" &nbsp; {html.escape(stage['headline'])}", unsafe_allow_html=True)
        for line in stage["detail"]:
            if line:
                st.caption(line)

setup = candidate["setup"]
if setup is not None:
    st.subheader("Setup detail")
    st.markdown(chip(CURRENT_CANDIDATE.replace("_", " "), "good") + " " + chip("not an order", "neutral"), unsafe_allow_html=True)
    rr = "n/a" if setup["risk_reward"] is None else f"1 : {setup['risk_reward']:.2f}"
    a, b, c, d = st.columns(4)
    a.metric("Entry", f"{setup['entry']:,.2f}")
    b.metric("Stop", f"{setup['stop']:,.2f}")
    c.metric("Target", f"{setup['target']:,.2f}")
    d.metric("Risk : reward", rr)
    st.write(setup["invalidation"])
    st.markdown("**Entry assumptions**")
    for line in candidate["assumptions"]:
        st.markdown(f"- {line}")
    if setup["meta"]:
        with st.expander("Strategy notes for this setup"):
            st.json({k: v for k, v in setup["meta"].items() if isinstance(v, (int, float, str, bool))})

if candidate["risk_checks"]:
    st.subheader("Risk checks (preview)")
    rows = "".join(f"<tr><td>{html.escape(r['check'])}</td><td>{chip('PASS' if r['ok'] else 'FAIL', 'good' if r['ok'] else 'critical')}</td></tr>"
                   for r in candidate["risk_checks"])
    st.markdown(f'<table class="qi-table"><thead><tr><th>Check</th><th></th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
    st.caption("A preview records nothing and issues no approval. Checks run in a fixed order and stop at the first failure.")

if candidate["levels"]:
    with st.expander("Relevant levels"):
        st.dataframe(pd.DataFrame({"Level": list(candidate["levels"]), "Price": [round(v, 2) for v in candidate["levels"].values()]}),
                     hide_index=True, width="stretch")

st.subheader("Earlier setups of the selected strategy")
hist = candidate["historical"]
if hist:
    st.caption("HISTORICAL SETUP: past occurrences in this strategy's backtest, not current opportunities.")
    st.dataframe(pd.DataFrame(hist).drop(columns=["status"]).round(3), hide_index=True, width="stretch")
else:
    st.caption("No historical setups to show for the selected strategy.")

def _link(path: str, label: str, icon: str) -> None:
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:  # running outside the navigation (a test or a single-page run): the link cannot resolve
        st.caption(f"{label}: open it from the navigation bar.")


c1, c2 = st.columns(2)
with c1:
    if is_ready:
        _link("pages/07_Paper_Trading.py", "Place the simulated order on Paper Trading", ":material/arrow_forward:")
with c2:
    _link("pages/08_Positions_and_Orders.py", "See paper trades and their trace", ":material/timeline:")
st.caption("Everything here is a historical statistic or a simulation, not a prediction. A candidate is not an order.")
