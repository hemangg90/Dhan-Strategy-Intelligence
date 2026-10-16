"""Positions & Orders - a broker-style view of the paper account.

Live open positions with P&L, stop/target progress and a manual exit; today's trades; full trade
history; order book; fills; and performance analytics. Numbers use Indian grouping; P&L is shown net of
charges at Dhan's brokerage and the statutory option rates. Positions closed by the current paper broker carry the
charges it booked; older rows only stored a gross figure, so their charges are estimated here.
"""
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.database.db import get_session, is_durable
from quant_intelligence.database.models import Fill, Order, Position
from quant_intelligence.execution.capital import capital_summary
from quant_intelligence.execution.engine import ENGINE
from quant_intelligence.reports.history_io import import_positions
from quant_intelligence.reports.performance import unresolved
from quant_intelligence.ui import format as F
from quant_intelligence.ui.charts import equity_curve_chart, pnl_bar_chart
from quant_intelligence.ui.components import empty_state, kpi_row, page_header
from quant_intelligence.ui.open_positions import live_open, render_open_positions
from quant_intelligence.ui.position_views import (
    charges_breakdown, default_statuses, equity_curve, fills_frame, orders_frame, pnl_by, positions_frame, trade_stats, with_live,
)
from quant_intelligence.ui.state import init_session_state
from quant_intelligence.ui.tables import fills_table, history_table, orders_table, trades_table
from quant_intelligence.ui.theme import apply_theme
from quant_intelligence.utils.timeutil import now_ist

apply_theme()
init_session_state()
page_header(
    "Positions & Orders",
    "Paper account - live positions, order book, trade history and performance. P&L is net of estimated Dhan charges.",
)


# ---------------------------------------------------------------------------------------------- data
def _history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Closed/stale positions, orders and fills from the database (open ones come live from the broker)."""
    with get_session() as session:
        # No cap on positions: the full history is the point. Orders/fills are bounded only to keep the page fast.
        pos = session.query(Position).filter(Position.status != "OPEN").order_by(Position.id.desc()).all()
        orders = session.query(Order).order_by(Order.id.desc()).limit(20000).all()
        fills = session.query(Fill).order_by(Fill.id.desc()).limit(20000).all()
    orders_df = orders_frame(orders)
    return positions_frame(pos, now=now_ist()), orders_df, fills_frame(fills, orders_df)


def _today(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    start = dt.datetime.combine(now_ist().date(), dt.time.min)
    return df[(df["status"] == "CLOSED") & (pd.to_datetime(df["closed_at"]) >= start)]


# ---------------------------------------------------------------------------------------------- KPIs (live)
@st.fragment(run_every=f"{SETTINGS.ltp_refresh_seconds}s")
def live_kpis() -> None:
    hist, _, _ = _history()
    open_df, error = live_open()
    today = _today(hist)
    realised = float(today["net_pnl"].sum()) if len(today) else 0.0
    unrealised = float(open_df["unrealised_net"].sum(skipna=True)) if len(open_df) else 0.0
    cap = capital_summary(ENGINE.broker)
    stats = trade_stats(today)
    day = realised + unrealised
    spark = list(equity_curve(today)["equity"]) if len(today) else []
    kpi_row(
        [
            {"label": "Day P&L (net)", "value": F.inr(day, signed=True), "tone": F.tone(day), "spark": spark,
             "help": f"Realised {F.inr(realised, signed=True)} (closed today, after charges) + open positions "
                     f"{F.inr(unrealised, signed=True)} (live, after estimated charges)",
             "delta": f"{F.inr(realised, signed=True)} realised"},
            {"label": "Capital used", "value": F.inr(cap["capital_used"]),
             "delta": f"{cap['capital_used'] / cap['account_value'] * 100:.1f}% of account" if cap["account_value"] else None},
            {"label": "Available", "value": F.inr(cap["available"]), "help": "Account value minus capital in open BUY positions"},
            {"label": "Account value", "value": F.inr(cap["account_value"]),
             "help": f"Starting capital {F.inr(cap['starting_capital'])} + realised P&L booked by the paper broker"},
            {"label": "Open positions", "value": str(len(open_df))},
            {"label": "Win rate today",
             "value": f"{stats['win_rate']:.0f}%" if stats["win_rate"] is not None else "–",
             "delta": f"{stats['wins']}W / {stats['losses']}L of {stats['trades']}" if stats["trades"] else "no closed trades"},
        ]
    )
    if error:
        st.warning(f"Live prices unavailable: {error}")


live_kpis()

tab_open, tab_today, tab_hist, tab_orders, tab_fills, tab_stats = st.tabs(
    ["Open positions", "Today's trades", "Trade history", "Order book", "Fills", "Analytics"]
)


with tab_open:
    render_open_positions(key="p08")

# ---------------------------------------------------------------------------------------------- history tabs
hist_df, orders_df, fills_df = _history()
closed_all = hist_df[hist_df["status"] == "CLOSED"] if len(hist_df) else hist_df

with tab_today:
    today = _today(hist_df)
    if today.empty:
        empty_state("No closed trades today", "Closed trades show gross P&L, itemised charges and net P&L.")
    else:
        s = trade_stats(today)
        kpi_row([
            {"label": "Trades", "value": str(s["trades"])},
            {"label": "Net P&L", "value": F.inr(s["net"], signed=True), "tone": F.tone(s["net"]),
             "delta": f"Gross {F.inr(s['gross'], signed=True)} · charges {F.inr(s['charges'])}"},
            {"label": "Win rate", "value": f"{s['win_rate']:.0f}%", "delta": f"{s['wins']}W / {s['losses']}L"},
            {"label": "Best / worst", "value": f"{F.inr(s['best'], signed=True)}", "delta": f"worst {F.inr(s['worst'], signed=True)}"},
        ])
        st.dataframe(trades_table(today.sort_values("closed_at", ascending=False)), width="stretch", hide_index=True)

with tab_hist:
    if not is_durable():
        st.warning(
            "History is stored in a local SQLite file. On Streamlit Cloud it is erased whenever the app reboots or "
            "sleeps. Set DATABASE_URL (a free Neon/Supabase Postgres) in the app's Secrets to keep every trade - see the README.",
            icon="⚠️",
        )
    _un = unresolved(hist_df) if len(hist_df) else {"stale": 0, "invested": 0.0, "dates": []}
    if _un["stale"]:
        st.info(
            f"{_un['stale']} position(s) from {', '.join(_un['dates'][:4])}{'...' if len(_un['dates']) > 4 else ''} were left "
            f"open when the app stopped, so they have no exit and no result ({F.inr(_un['invested'])} entry value). "
            "They are shown as STALE and excluded from every statistic."
        )
    with st.expander("Restore earlier trades from a CSV download"):
        st.caption("Upload a file saved from this tab's 'Download positions (CSV)'. Trades already stored are skipped, "
                   "so it is safe to import the same file twice.")
        up = st.file_uploader("Positions CSV", type=["csv"], key="import_positions_csv")
        if up is not None and st.button("Import trades", key="import_positions_go"):
            try:
                result = import_positions(pd.read_csv(up))
                st.success(f"Imported {result['inserted']} trade(s); {result['skipped_existing']} already stored.")
                for line, why in result["invalid"][:10]:
                    st.warning(f"Line {line}: {why}")
                if result["inserted"]:
                    st.rerun()
            except Exception as e:
                st.error(f"Could not import that file: {e}")
    if hist_df.empty:
        empty_state("No trade history yet")
    else:
        f1, f2, f3, f4 = st.columns(4)
        status_options = sorted(hist_df["status"].dropna().unique())
        status_pick = f1.multiselect("Status", status_options, default=default_statuses(status_options))
        inst_pick = f2.multiselect("Instrument", sorted(hist_df["instrument"].dropna().unique()))
        strat_pick = f3.multiselect("Strategy", sorted(hist_df["strategy"].dropna().unique()))
        dates = pd.to_datetime(hist_df["opened_at"]).dropna()
        span = f4.date_input("Opened between", value=(dates.min().date(), dates.max().date()) if len(dates) else ())
        view = hist_df
        if status_pick:
            view = view[view["status"].isin(status_pick)]
        if inst_pick:
            view = view[view["instrument"].isin(inst_pick)]
        if strat_pick:
            view = view[view["strategy"].isin(strat_pick)]
        if isinstance(span, tuple) and len(span) == 2:
            opened = pd.to_datetime(view["opened_at"]).dt.date
            view = view[(opened >= span[0]) & (opened <= span[1])]
        st.caption(f"{len(view)} of {len(hist_df)} position(s). STALE rows were left open by an earlier day and are excluded from statistics.")
        st.dataframe(history_table(view.sort_values("opened_at", ascending=False)), width="stretch", hide_index=True)
        st.download_button("Download positions (CSV)", view.drop(columns=["held"]).to_csv(index=False).encode("utf-8"),
                           file_name=f"positions_{now_ist():%Y%m%d}.csv", mime="text/csv", key="dl_positions")

with tab_orders:
    if orders_df.empty:
        empty_state("No orders yet")
    else:
        counts = orders_df["status"].value_counts()
        kpi_row([
            {"label": "Orders", "value": str(len(orders_df))},
            {"label": "Filled", "value": str(int(counts.get("FILLED", 0)))},
            {"label": "Rejected", "value": str(int(counts.get("REJECTED", 0))),
             "help": "Rejected orders are usually risk-engine vetoes or insufficient funds - see the Reason column"},
        ])
        pick = st.segmented_control("Status", ["All", "FILLED", "REJECTED"], default="All", key="order_status",
                                    label_visibility="collapsed") or "All"
        view = orders_df if pick == "All" else orders_df[orders_df["status"] == pick]
        st.dataframe(orders_table(view), width="stretch", hide_index=True)
        st.download_button("Download orders (CSV)", view.to_csv(index=False).encode("utf-8"),
                           file_name=f"orders_{now_ist():%Y%m%d}.csv", mime="text/csv", key="dl_orders")

with tab_fills:
    if fills_df.empty:
        empty_state("No fills yet", "Each filled paper order records its simulated fill and slippage here.")
    else:
        total_slip = float((fills_df["slippage"] * fills_df["quantity"]).sum())
        kpi_row([{"label": "Fills", "value": str(len(fills_df))},
                 {"label": "Total slippage cost", "value": F.inr(total_slip, 0),
                  "help": "Simulated adverse slippage (ticks x tick size x quantity) across all fills"}])
        st.dataframe(fills_table(fills_df), width="stretch", hide_index=True)
        st.download_button("Download fills (CSV)", fills_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"fills_{now_ist():%Y%m%d}.csv", mime="text/csv", key="dl_fills")

# ---------------------------------------------------------------------------------------------- analytics
with tab_stats:
    if closed_all.empty:
        empty_state("No closed trades to analyse yet", "Statistics appear after the first position closes.")
    else:
        s = trade_stats(closed_all)
        kpi_row([
            {"label": "Closed trades", "value": str(s["trades"]), "delta": f"{s['wins']}W / {s['losses']}L"},
            {"label": "Net P&L", "value": F.inr(s["net"], signed=True), "tone": F.tone(s["net"]),
             "delta": f"after {F.inr(s['charges'])} charges"},
            {"label": "Win rate", "value": f"{s['win_rate']:.1f}%"},
            {"label": "Profit factor", "value": "∞" if s["profit_factor"] == float("inf") else (
                f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else "–"),
             "help": "Gross profit divided by gross loss. Above 1 means winners outweigh losers"},
            {"label": "Expectancy / trade", "value": F.inr(s["expectancy"], signed=True), "tone": F.tone(s["expectancy"])},
            {"label": "Max drawdown", "value": F.inr(s["max_drawdown"]), "help": "Largest peak-to-trough fall of cumulative net P&L"},
        ])
        kpi_row([
            {"label": "Average win", "value": F.inr(s["avg_win"], signed=True)},
            {"label": "Average loss", "value": F.inr(s["avg_loss"], signed=True)},
            {"label": "Payoff ratio", "value": f"{s['payoff']:.2f}" if s["payoff"] else "–",
             "help": "Average win divided by average loss"},
            {"label": "Best / worst trade", "value": F.inr(s["best"], signed=True), "delta": f"worst {F.inr(s['worst'], signed=True)}"},
            {"label": "Avg holding time", "value": F.duration(s["avg_held"])},
            {"label": "Max streaks", "value": f"{s['max_win_streak']}W / {s['max_loss_streak']}L"},
        ])
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Cumulative net P&L**")
            st.plotly_chart(equity_curve_chart(equity_curve(closed_all)), width="stretch", config={"displayModeBar": False})
        with right:
            st.markdown("**Charges paid (closed trades)**")
            br = charges_breakdown(closed_all)
            st.dataframe(
                pd.DataFrame({"Charge": list(br), "Amount": [F.inr(v, 2) for v in br.values()]})
                .pipe(lambda d: pd.concat([d, pd.DataFrame({"Charge": ["Total"], "Amount": [F.inr(sum(br.values()), 2)]})])),
                width="stretch", hide_index=True,
            )
        a, b = st.columns(2)
        with a:
            st.markdown("**Net P&L by strategy**")
            st.plotly_chart(pnl_bar_chart(pnl_by(closed_all, "strategy"), "strategy"), width="stretch",
                            config={"displayModeBar": False})
        with b:
            st.markdown("**Net P&L by instrument**")
            st.plotly_chart(pnl_bar_chart(pnl_by(closed_all, "instrument"), "instrument"), width="stretch",
                            config={"displayModeBar": False})
