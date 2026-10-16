import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.ranking.ranking_engine import tradable
from quant_intelligence.risk.risk_engine import MAX_QUOTE_AGE_SEC, MAX_SPREAD_PCT
from quant_intelligence.ui.decision_view import explanation_rows, global_rows
from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme
from quant_intelligence.utils.timeutil import now_ist

apply_theme()
init_session_state()
st.title("Decision Center")
st.caption(
    "Every number below is a HISTORICAL statistic from past trades, not a prediction. "
    "The risk engine is a separate check that can veto any decision; the ranker cannot override it."
)

output = run_pipeline_cached()
if output is None:
    st.error(st.session_state.get("pipeline_error", "Pipeline not available."))
    st.stop()

# ---- 1. Data status ----------------------------------------------------------------------------
st.subheader("1. Data status")
age_min = None
try:
    age_min = (now_ist().replace(tzinfo=None) - output.timestamp.replace(tzinfo=None)).total_seconds() / 60
except Exception:
    pass
if age_min is None:
    age_text = "n/a"
elif age_min > 120:
    age_text = f"{age_min / 60:.1f} h"
else:
    age_text = f"{age_min:.0f} min"
c1, c2, c3, c4 = st.columns(4)
c1.metric("Instrument", output.instrument)
c2.metric("Last bar", f"{output.timestamp:%d %b %H:%M}")
c3.metric("Data age", age_text)
c4.metric("Data quality", output.data_quality_status)
for issue in output.data_quality_issues or []:
    st.warning(str(issue))

# ---- 2. Decision ---------------------------------------------------------------------------------
st.subheader("2. Decision")
ranking = output.ranking
ok, why_not = tradable(ranking, "PAPER")
if ranking.is_no_trade or not ok:
    st.markdown('<span class="qi-badge qi-badge-notrade" style="font-size:1.1rem;">NO TRADE</span>', unsafe_allow_html=True)
    st.write(f"**Why:** {why_not or ranking.reason}")
else:
    label = "TIE-BREAK" if ranking.tie_break else "SELECTED"
    st.markdown(
        f'<span class="qi-badge qi-badge-trade" style="font-size:1.1rem;">{label}: {ranking.selected_strategy}</span>',
        unsafe_allow_html=True,
    )
    st.write(f"**Why:** {ranking.reason}")
    st.info("A selection is not an order. It must still pass the independent risk engine below, and a real setup must trigger.")
st.caption(f"Regime: **{output.regime_label}** ({output.regime_confidence * 100:.0f}% model confidence, uncalibrated)")

# ---- 3. Ranked strategies with the evidence behind each --------------------------------------------
st.subheader("3. Why each strategy is ranked, eligible or rejected")
st.dataframe(explanation_rows(output), width="stretch", hide_index=True)
st.caption(
    "Conditional expectancy is the mean net R of the most similar past market states (nearest-neighbour analogues), "
    "after costs. Samples are small; check the standard error and distinct days before trusting any row."
)
with st.expander("Unconditional backtest statistics (gross of costs)"):
    st.dataframe(global_rows(output), width="stretch", hide_index=True)

# ---- 4. Independent risk engine ---------------------------------------------------------------------
st.subheader("4. Independent risk engine")
limits = SETTINGS.risk
st.markdown(
    f"""
The risk engine takes only the account state and the proposed order. It never sees the ranking, so a high-ranked or
tie-break strategy is vetoed exactly like any other. **Any failed mandatory check means NO TRADE.**

| Check | Rule |
|---|---|
| Order parameters | Entry, stop, target positive; stop and target on the correct side; whole-lot quantity |
| Data freshness | Price no older than {MAX_QUOTE_AGE_SEC // 60} minutes at order time, and data quality `OK` |
| Spread and slippage | Spread at most {MAX_SPREAD_PCT:g}% and half-spread under half the stop distance; unknown spread fails closed |
| Trade risk | At most {limits.max_risk_per_trade_pct:g}% of equity per trade (scaled down in drawdown) |
| Daily loss | New entries stop at {limits.max_daily_loss_pct:g}% daily loss |
| Drawdown | New entries stop at {limits.max_drawdown_pct:g}% drawdown from peak |
| Exposure | Per-strategy {limits.max_strategy_exposure_pct:g}%, portfolio {limits.max_portfolio_exposure_pct:g}%, correlated-group limits |
| Kill switch, broker link | Manual kill switch and broker connectivity always veto |
"""
)
st.caption(
    "Live order placement is disabled in this session (demo lockdown)."
    if SETTINGS.demo_lockdown
    else "Orders are simulated unless live mode is explicitly authorised."
)
