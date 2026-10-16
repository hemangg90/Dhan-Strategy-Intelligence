import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.ranking.ranking_engine import tradable
from quant_intelligence.risk.risk_engine import MAX_QUOTE_AGE_SEC, MAX_SPREAD_PCT
from quant_intelligence.ui.components import chip
from quant_intelligence.ui.dashboard import decision_banner, hero_header, kpi_cards, ranking_table
from quant_intelligence.ui.decision_view import explanation_rows, global_rows
from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme
from quant_intelligence.utils.timeutil import now_ist

apply_theme()
init_session_state()

output = run_pipeline_cached()
if output is None:
    st.title("Decision Center")
    st.error(st.session_state.get("pipeline_error", "Pipeline not available."))
    st.stop()

age_min = None
try:
    age_min = (now_ist().replace(tzinfo=None) - output.timestamp.replace(tzinfo=None)).total_seconds() / 60
except Exception:
    pass
age_text = "n/a" if age_min is None else (f"{age_min / 60:.1f} h" if age_min > 120 else f"{max(age_min, 0):.0f} min")
quality_ok = output.data_quality_status == "OK"

hero_header(
    "Decision Center",
    "Every number is a <b>historical statistic</b>, not a prediction. The risk engine is a separate check "
    "that can veto any decision; the ranker cannot override it.",
    [chip(f"Data {output.data_quality_status}", "good" if quality_ok else "critical"), chip("PAPER MODE", "info")],
)

kpi_cards([
    {"label": "Instrument", "value": output.instrument, "sub": f"last bar {output.timestamp:%d %b %H:%M}"},
    {"label": "Data age", "value": age_text, "sub": f"orders need data under {MAX_QUOTE_AGE_SEC // 60} min old",
     "tone": "good" if age_min is not None and age_min <= MAX_QUOTE_AGE_SEC / 60 else "critical"},
    {"label": "Data quality", "value": output.data_quality_status, "tone": "good" if quality_ok else "critical",
     "sub": f"{len(output.data_quality_issues or [])} issue(s)"},
    {"label": "Regime", "value": str(output.regime_label).replace("_", " ").title(),
     "sub": f"top score share {output.regime_confidence * 100:.0f}% (heuristic)"},
])
for issue in output.data_quality_issues or []:
    st.warning(str(issue))

ranking = output.ranking
ok, why_not = tradable(ranking, "PAPER")
decision_banner(ranking.is_no_trade or not ok, ranking.selected_strategy, why_not or ranking.reason, ranking.tie_break)
if not (ranking.is_no_trade or not ok):
    st.info("A selection is not an order. It must still pass the independent risk engine below, and a real setup must trigger.")

st.subheader("Why each strategy is ranked, eligible or rejected")
ranking_table(explanation_rows(output), show_why=True)
with st.expander("Unconditional backtest statistics (gross of costs)"):
    st.dataframe(global_rows(output), width="stretch", hide_index=True)

st.subheader("Independent risk engine")
limits = SETTINGS.risk
st.markdown(
    "The risk engine sees only the account state and the proposed order, never the ranking. A top-ranked or "
    "tie-break strategy is vetoed exactly like any other. **Any failed mandatory check means NO TRADE.**"
)
checks = [
    ("Order parameters", "Entry, stop and target positive; stop and target on the correct side; whole-lot quantity"),
    ("Data freshness", f"Price no older than {MAX_QUOTE_AGE_SEC // 60} minutes at order time, and data quality OK"),
    ("Spread and slippage", f"Spread at most {MAX_SPREAD_PCT:g}%; half-spread under half the stop distance; unknown spread fails closed"),
    ("Trade risk", f"At most {limits.max_risk_per_trade_pct:g}% of equity per trade, scaled down in drawdown"),
    ("Daily loss", f"New entries stop at {limits.max_daily_loss_pct:g}% daily loss"),
    ("Drawdown", f"New entries stop at {limits.max_drawdown_pct:g}% drawdown from peak"),
    ("Exposure", f"Strategy {limits.max_strategy_exposure_pct:g}%, portfolio {limits.max_portfolio_exposure_pct:g}%, plus correlated-group limits"),
    ("Kill switch and broker link", "Manual kill switch and broker connectivity always veto"),
]
rows = "".join(f"<tr><td><b>{n}</b></td><td>{r}</td><td>{chip('MANDATORY', 'info')}</td></tr>" for n, r in checks)
st.markdown(
    f'<table class="qi-table"><thead><tr><th>Check</th><th>Rule</th><th></th></tr></thead><tbody>{rows}</tbody></table>',
    unsafe_allow_html=True,
)
st.caption(
    "Live order placement is disabled in this session (demo lockdown)."
    if SETTINGS.demo_lockdown
    else "Orders are simulated unless live mode is explicitly authorised."
)
