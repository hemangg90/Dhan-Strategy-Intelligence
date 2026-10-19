import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import datetime as dt

import pandas as pd

import plotly.graph_objects as go
import streamlit as st

from quant_intelligence.backtesting.engine import run_backtest, walk_forward
from quant_intelligence.data.data_manager import DataManager, DataUnavailableError
from quant_intelligence.features.feature_engine import compute_features
from quant_intelligence.strategies.registry import STRATEGY_CLASSES, get_strategy
from quant_intelligence.ui import backtest_workbench as wb
from quant_intelligence.ui.state import init_session_state
from quant_intelligence.ui.theme import apply_theme
from quant_intelligence.utils.timeutil import now_ist

apply_theme()
init_session_state()

INK, MUTED, GREEN, RED, LINE = "#111827", "#6b7280", "#059669", "#dc2626", "#e5e7eb"

# Light workbench styling, scoped to this page (the CSS is only emitted when this page renders).
st.markdown(
    f"""
<style>
.stApp {{ background:#f7f8fa; }}
.stApp, .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li {{ color:{INK}; }}
[data-testid="stSidebar"] {{ background:#ffffff; }}
[data-testid="stSidebar"] * {{ color:{INK}; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#ffffff; border-color:{LINE} !important; border-radius:12px; }}
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {{ background:#ffffff; color:{INK}; border-color:{LINE}; }}
.wb-card {{ border:1px solid {LINE}; border-radius:12px; padding:10px 12px; background:#fff; }}
.wb-card.sel {{ background:{INK}; border-color:{INK}; }}
.wb-card.sel, .wb-card.sel * {{ color:#f9fafb !important; }}
.wb-top {{ display:flex; justify-content:space-between; font-weight:600; font-size:0.95rem; }}
.wb-sub {{ color:{MUTED}; font-size:0.78rem; margin:2px 0 6px; }}
.wb-mono {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:0.78rem; color:{MUTED}; display:flex; gap:14px; }}
.wb-kpi-label {{ color:{MUTED}; font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase; }}
.wb-kpi-val {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:1.15rem; font-weight:600; }}
.wb-pos {{ color:{GREEN} !important; }} .wb-neg {{ color:{RED} !important; }}
.wb-big {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:2rem; font-weight:700; text-align:right; }}
.wb-pill {{ background:#eef2f7; color:{MUTED}; border-radius:999px; padding:2px 10px; font-size:0.75rem; margin-left:8px; }}
</style>
""",
    unsafe_allow_html=True,
)


def _sign_class(x):
    return "" if x is None else ("wb-pos" if x > 0 else "wb-neg" if x < 0 else "")


def _fmt_pct(x, nd=1, signed=True):
    return "n/a" if x is None else f"{x:+.{nd}f}%" if signed else f"{x:.{nd}f}%"


def _d(ts):
    return "n/a" if ts is None else f"{ts:%b} {ts.day}, {ts.year}"


def _chart_layout(fig, height, yfmt=None):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12, x=0.02), font=dict(color=MUTED, size=11),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#eef0f3", zeroline=False, tickformat=yfmt)
    return fig


def _new_backtest_form():
    """Existing Backtest Lab functionality: configure and run an in-sample / out-of-sample backtest."""
    c1, c2, c3 = st.columns(3)
    instrument = c1.text_input("Instrument", "NIFTY")
    timeframe = c2.selectbox("Timeframe", ["5min", "15min", "1min", "3min"])
    strategy_name = c3.selectbox("Strategy", list(STRATEGY_CLASSES.keys()))
    c4, c5 = st.columns(2)
    lookback_days = c4.slider("Total lookback (days)", 20, 365, 90)
    oos_fraction = c5.slider("Out-of-sample holdout (most recent fraction)", 0.1, 0.5, 0.25)

    defaults = STRATEGY_CLASSES[strategy_name].default_parameters
    st.caption("Parameters are edited by hand and are NOT optimised on out-of-sample data.")
    pcols = st.columns(len(defaults)) if defaults else []
    params = {}
    for i, (k, v) in enumerate(defaults.items()):
        params[k] = pcols[i].number_input(k, value=float(v)) if isinstance(v, (int, float)) else pcols[i].text_input(k, value=str(v))

    if not st.button("Run backtest", type="primary"):
        return
    end = now_ist()
    start = end - dt.timedelta(days=lookback_days)
    with st.spinner("Fetching data and computing features..."):
        try:
            ohlcv, _ = DataManager().get_ohlcv(instrument, timeframe, start, end)
        except DataUnavailableError as e:
            st.error(str(e))
            return
        if len(ohlcv) < 100:
            st.error("Not enough data for the selected window.")
            return
        features = compute_features(ohlcv)
    split = int(len(ohlcv) * (1 - oos_fraction))
    with st.spinner("Running in-sample and out-of-sample backtests..."):
        is_res = run_backtest(get_strategy(strategy_name, params), ohlcv.iloc[:split].reset_index(drop=True),
                              features.iloc[:split].reset_index(drop=True), instrument, split="IN_SAMPLE")
        oos_res = run_backtest(get_strategy(strategy_name, params), ohlcv.iloc[split:].reset_index(drop=True),
                               features.iloc[split:].reset_index(drop=True), instrument, split="OUT_OF_SAMPLE")
    st.session_state["wb_selected"] = oos_res.run_id if oos_res.metrics.get("n_trades") else is_res.run_id
    is_er, oos_er = is_res.metrics.get("expected_r"), oos_res.metrics.get("expected_r")
    if is_er is not None and oos_er is not None and is_er > 0 and oos_er < is_er * 0.3:
        st.session_state["wb_notice"] = (
            f"In-sample expected R ({is_er:.3f}) does not survive out-of-sample ({oos_er:.3f}). This is evidence "
            "AGAINST a robust edge in this window - do not assume profitability."
        )
    else:
        st.session_state.pop("wb_notice", None)
    st.session_state["wb_last_inputs"] = dict(strategy_name=strategy_name, params=params, instrument=instrument, ohlcv=ohlcv)
    st.rerun()


runs = wb.list_runs()
capital = wb.starting_capital()

left, right = st.columns([1, 2.6], gap="large")

# ---------------------------------------------------------------- left: run list
with left:
    h1, h2 = st.columns([4, 1])
    h1.markdown(f"### Backtest Workbench\n<span style='color:{MUTED};font-size:0.85rem'>{len(runs)} runs total</span>",
                unsafe_allow_html=True)
    query = st.text_input("Search", placeholder="Search runs, strategy, instrument...", label_visibility="collapsed")
    outcome = st.radio("Filter", ["All", "Win", "Loss"], horizontal=True, label_visibility="collapsed")
    shown = wb.filter_runs(runs, query, outcome)
    selected_id = st.session_state.get("wb_selected")
    if shown and selected_id not in {r.run_id for r in shown}:
        selected_id = shown[0].run_id

    with st.container(height=640, border=False):
        if not shown:
            st.info("No backtest runs yet. Open **New backtest** on the right and run one." if not runs else "No runs match.")
        for i, r in enumerate(shown):
            sel = r.run_id == selected_id
            ret_cls = _sign_class(r.total_return_pct)
            wr = f"{r.win_rate * 100:.0f}%" if r.win_rate is not None else "n/a"
            st.markdown(
                f"""<div class="wb-card{' sel' if sel else ''}">
<div class="wb-top"><span>{r.strategy}</span><span class="{ret_cls}">{_fmt_pct(r.total_return_pct)}</span></div>
<div class="wb-sub">{r.instrument} · {r.split.replace('_', ' ').title()}</div>
<div class="wb-mono"><span>DD: {_fmt_pct(r.max_dd_pct)}</span><span>WR: {wr}</span><span>{r.n_trades} trades</span></div>
<div class="wb-sub" style="margin-top:6px">{_d(r.start)} → {_d(r.end)}</div></div>""",
                unsafe_allow_html=True,
            )
            if not sel and st.button("Open", key=f"open_{r.run_id}", use_container_width=True):
                st.session_state["wb_selected"] = r.run_id
                st.rerun()
    st.caption(f"Showing {len(shown)} of {len(runs)} runs")

# ---------------------------------------------------------------- right: detail
with right:
    with st.expander("New backtest", expanded=not runs):
        _new_backtest_form()

    if st.session_state.get("wb_notice"):
        st.error(st.session_state["wb_notice"])

    run = next((r for r in shown if r.run_id == selected_id), None) if shown else None
    if run is None:
        st.stop()

    trades = wb.load_trades(run.run_id)
    equity = wb.equity_curve(trades, capital)
    dd = wb.drawdown_pct(equity, capital)
    daily = wb.daily_returns_pct(trades, capital)
    sharpe = wb.daily_sharpe(daily)
    final = float(equity.iloc[-1]) if len(equity) else None

    t1, t2 = st.columns([3, 1])
    t1.markdown(
        f"## {run.strategy} <span class='wb-pill'>completed</span>\n"
        f"<span style='color:{MUTED}'>{run.instrument} · {run.split.replace('_', ' ').title()} · {_d(run.start)} → {_d(run.end)}</span>",
        unsafe_allow_html=True,
    )
    t2.markdown(f"<div class='wb-big {_sign_class(run.total_return_pct)}'>{_fmt_pct(run.total_return_pct)}</div>",
                unsafe_allow_html=True)

    kpis = [
        ("Max DD", _fmt_pct(float(dd.min()) if len(dd) else None), "wb-neg"),
        ("Sharpe (daily)", "n/a" if sharpe is None else f"{sharpe:.2f}", _sign_class(sharpe)),
        ("Win rate", "n/a" if run.win_rate is None else f"{run.win_rate * 100:.1f}%", ""),
        ("Trades", str(run.n_trades), ""),
        ("Initial", f"Rs {capital:,.0f}", ""),
        ("Final", "n/a" if final is None else f"Rs {final:,.0f}", _sign_class(None if final is None else final - capital)),
    ]
    for col, (label, val, cls) in zip(st.columns(6), kpis):
        col.markdown(f"<div class='wb-kpi-label'>{label}</div><div class='wb-kpi-val {cls}'>{val}</div>", unsafe_allow_html=True)
    notes = [f"Only {run.n_trades} trades: statistics may be unreliable."] if run.n_trades < 30 else []
    notes.append("Net of modelled costs. Win rate is gross R > 0. Historical backtest, not a prediction.")
    st.caption(" ".join(notes))

    if len(equity) == 0:
        st.warning("This run has no stored trades, so no curves can be drawn.")
        st.stop()

    with st.container(border=True):
        st.markdown("**Equity Curve** <span class='wb-pill'>PRIMARY</span>", unsafe_allow_html=True)
        st.caption("Strategy account value after each closed trade vs buy-and-hold of the instrument (when cached prices exist).")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Strategy", line=dict(color=INK, width=2),
                                 fill="tozeroy", fillcolor="rgba(17,24,39,0.05)"))
        bench = wb.benchmark_curve(run.instrument, run.start, run.end, capital)
        if bench is not None:
            fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="Benchmark (buy & hold)",
                                     line=dict(color="#9ca3af", width=1.5, dash="dash")))
        lo = min(float(equity.min()), float(bench.min()) if bench is not None else float(equity.min()))
        hi = max(float(equity.max()), float(bench.max()) if bench is not None else float(equity.max()))
        pad = (hi - lo) * 0.15 or capital * 0.01
        fig.update_yaxes(range=[lo - pad, hi + pad])
        st.plotly_chart(_chart_layout(fig, 300, ",.0f"), width="stretch", key="wb_equity")

    with st.container(border=True):
        st.markdown("**Drawdown**")
        st.caption("Peak-to-trough decline of the account after each closed trade.")
        fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, line=dict(color=RED, width=1.5), fill="tozeroy",
                                   fillcolor="rgba(220,38,38,0.12)", name="Drawdown"))
        st.plotly_chart(_chart_layout(fig, 220, ".1f"), width="stretch", key="wb_dd")

    with st.container(border=True):
        st.markdown("**Daily Returns**")
        st.caption("Day-by-day net P&L as a percent of the account at the start of that day.")
        colors = [GREEN if v >= 0 else RED for v in daily.values]
        fig = go.Figure(go.Bar(x=daily.index, y=daily.values, marker_color=colors, name="Daily return"))
        st.plotly_chart(_chart_layout(fig, 220, ".1f"), width="stretch", key="wb_daily")

    with st.expander("Parameter sensitivity (settings at 0.8x and 1.2x)"):
        last_inputs = st.session_state.get("wb_last_inputs")
        if not last_inputs:
            st.caption("Run a new backtest first; sensitivity reuses its data and parameters.")
        else:
            st.caption("Re-runs the strategy with each numeric setting moved 20% down and up. A result that flips sign or "
                       "vanishes was probably fitted to this sample. It reuses the same data, so it shows fragility, not proof.")
            if st.button("Run sensitivity"):
                from quant_intelligence.research.pipeline import default_quantity
                from quant_intelligence.research.sensitivity import parameter_sensitivity

                with st.spinner("Re-running the strategy with changed settings..."):
                    st.session_state["wb_sensitivity"] = parameter_sensitivity(
                        last_inputs["strategy_name"], last_inputs["ohlcv"], compute_features(last_inputs["ohlcv"]),
                        last_inputs["instrument"], default_quantity(last_inputs["instrument"]), last_inputs["params"])
            sens = st.session_state.get("wb_sensitivity")
            if sens and sens["strategy"] == last_inputs["strategy_name"]:
                base = sens["base"]
                base_r = "n/a" if base["mean_net_r"] is None else f"{base['mean_net_r']:.3f}"
                st.markdown(f"Base: **{base['n_trades']} trades**, net R per trade **{base_r}**")
                st.dataframe(pd.DataFrame(sens["rows"]), hide_index=True, width="stretch")
                if sens["sign_survives"] is None:
                    st.info("Too few trades to judge whether the sign holds.")
                elif sens["sign_survives"]:
                    st.success(f"The sign of the result held in all {sens['variants_compared']} comparable variants.")
                else:
                    st.warning("The sign of the result changed in at least one variant: this result is fragile.")

    with st.expander("Walk-forward (4 folds)"):
        last = st.session_state.get("wb_last_inputs")
        if not last:
            st.caption("Run a new backtest first; walk-forward reuses its data and parameters.")
        elif st.button("Run walk-forward"):
            feats = compute_features(last["ohlcv"])
            res = walk_forward(lambda: get_strategy(last["strategy_name"], last["params"]), last["ohlcv"], feats,
                               last["instrument"], n_folds=4)
            st.session_state["wb_selected"] = res[-1].run_id if res else st.session_state.get("wb_selected")
            st.rerun()
