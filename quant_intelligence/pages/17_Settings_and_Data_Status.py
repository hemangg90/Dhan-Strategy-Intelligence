import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from quant_intelligence.regimes.universe import UNIVERSE
from quant_intelligence.ui.settings_view import housekeeping_rows, mode_rows, risk_rows
from quant_intelligence.ui.state import init_session_state
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Settings & Data Status")
st.caption("Read-only. What the system is allowed to do, the limits the risk engine enforces, and the state of the data behind "
           "every instrument. Nothing here changes a setting; credentials are never shown.")

st.subheader("Mode and switches")
st.dataframe(pd.DataFrame(mode_rows()), hide_index=True, width="stretch")

st.subheader("Risk limits")
st.dataframe(pd.DataFrame(risk_rows()), hide_index=True, width="stretch")

st.subheader("Data status by instrument")
scan = UNIVERSE.peek("5min")
if scan is None:
    st.info("Checking every instrument in the background. Reload this page in a minute.")
else:
    frame = pd.DataFrame(scan)[["instrument", "kind", "mode", "data_status", "category", "last_bar", "rows", "source", "note"]]
    frame = frame.rename(columns={"mode": "currency", "data_status": "quality", "category": "quality category"})
    st.dataframe(frame, hide_index=True, width="stretch", height=min(60 + 35 * len(frame), 900))
    st.caption("Currency: LIVE (session open, latest bar current), DELAYED (open but old), HISTORICAL (session closed; last "
               "completed session). Quality is the separate gate that forces NO TRADE when not OK. Read from the local candle cache.")

with st.expander("Housekeeping"):
    st.dataframe(pd.DataFrame(housekeeping_rows()), hide_index=True, width="stretch")
