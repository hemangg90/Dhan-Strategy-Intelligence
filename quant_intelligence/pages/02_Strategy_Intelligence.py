import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Strategy Intelligence")
st.caption("Global vs. conditional (current-regime/state) performance for every strategy in the library.")

output = run_pipeline_cached()
if output is None:
    st.error(st.session_state.get("pipeline_error", "Pipeline not available."))
    st.stop()

rows = []
for si in output.strategy_intel:
    g = si.global_metrics
    c = si.conditional_metrics
    rows.append(
        {
            "Strategy": si.strategy_name,
            "Global Expected R": round(g.get("expected_r", 0), 3) if g.get("n_trades", 0) else None,
            "Global Win Rate": f"{g.get('win_rate', 0)*100:.1f}%" if g.get("n_trades", 0) else "N/A",
            "Global Sharpe (daily net R)": round(g.get("sharpe", float('nan')), 2) if g.get("n_trades", 0) else None,
            "Global Trades": g.get("n_trades", 0),
            "Conditional Expected R": round(c["expected_r"], 3) if c.get("expected_r") is not None else None,
            "Conditional Win Prob": f"{c['prob_positive_return']*100:.1f}%" if c.get("prob_positive_return") is not None else "N/A",
            "Confidence": c.get("confidence_label"),
            "Analogue Sample Size": c.get("sample_size"),
            "Cost-Adj Expectancy": round(g.get("cost_adjusted_expectancy", 0), 1) if g.get("n_trades", 0) else None,
            "Eligible Now": "YES" if si.score.eligible else "NO",
        }
    )
st.dataframe(rows, width="stretch", hide_index=True)

st.divider()
st.subheader("Drill-down")
choice = st.selectbox("Select strategy", [s.strategy_name for s in output.strategy_intel])
sel = next(s for s in output.strategy_intel if s.strategy_name == choice)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Global (unconditional) metrics**")
    st.json(sel.global_metrics)
with col2:
    st.markdown("**Conditional metrics (current market state analogues)**")
    st.json(sel.conditional_metrics)

st.markdown("**Ranking score components**")
st.json(sel.score.components)

st.markdown("**Historical observations backing this strategy's backtest**")
trade_rows = [
    {
        "Entry": t.setup.timestamp,
        "Direction": t.setup.direction,
        "Entry Price": round(t.setup.entry_price, 2),
        "Exit Price": round(t.exit_price, 2) if t.exit_price else None,
        "R Multiple": round(t.r_multiple, 3) if t.r_multiple is not None else None,
        "Outcome": t.outcome,
        "Holding (bars)": t.holding_period_bars,
    }
    for t in sel.backtest.trades
]
st.dataframe(trade_rows, width="stretch", hide_index=True, height=300)

if sel.backtest.metrics.get("n_trades", 0) < 30:
    st.warning(
        f"Sample size is small ({sel.backtest.metrics.get('n_trades', 0)} trades) over the current lookback window. "
        "Widen the lookback window in the sidebar for more statistically meaningful global metrics."
    )
