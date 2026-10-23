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

# ------------------------------------------------------------------ the evidence behind the ranking
import pandas as pd

from quant_intelligence.ranking.ranking_report import strategy_report_rows, no_trade_explanation

report = strategy_report_rows(output.strategy_intel, output.ranking, output.regime_label)
if output.ranking.is_no_trade:
    st.warning("**NO TRADE.** " + " ".join(no_trade_explanation(output.ranking, report)[:1]))
    with st.expander("Why each strategy is not eligible"):
        for line in no_trade_explanation(output.ranking, report)[1:]:
            st.write("- " + line)
else:
    st.success(f"Selected: **{output.ranking.selected_strategy}**" + (" (tie-break: the top strategies are statistically tied)" if output.ranking.tie_break else ""))
with st.expander("Evidence behind the ranking (net of costs; sample size and uncertainty shown)"):
    st.caption("Strategies are not ranked by win rate. The rank score uses a lower bound of the shrunk conditional edge, then confidence, "
               "probability of a positive result and a drawdown penalty (ranking/ranking_engine.py). Comparable observations are historical "
               "trades in market conditions like today's. 'Options-specific' performance is unavailable: there are no historical option prices.")
    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("score_components", "regime_record", "why_not_eligible", "options_specific", "regime", "selected")}
                          | {"Selected": "yes" if r["selected"] else ""} for r in report])
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Options-specific performance: " + report[0]["options_specific"] if report else "")

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

if len(sel.analogues):
    with st.expander("Which historical observations were chosen as comparable, and why"):
        st.caption("Closest and furthest features are in scaled units (smaller = more alike). 'Outcome known at' is when the trade's result "
                   "became available: an observation whose outcome was not yet known is never used.")
        picked = sel.analogues
        st.dataframe(pd.DataFrame(
            {"Distance": picked.get("distance"), "Similarity": picked.get("similarity_score"), "Net R": picked.get("r_multiple"),
             "Event (entry)": picked.get("entry_timestamp"), "Outcome known at": picked.get("outcome_available_at", picked.get("exit_timestamp"))}).round(3),
            hide_index=True, width="stretch", height=260)
        feats = picked.attrs.get("features_used")
        if feats:
            st.caption("Features compared: " + ", ".join(feats) + (f". Not compared (missing from the history or today): {', '.join(picked.attrs['features_unavailable'])}."
                                                                   if picked.attrs.get("features_unavailable") else "."))

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
