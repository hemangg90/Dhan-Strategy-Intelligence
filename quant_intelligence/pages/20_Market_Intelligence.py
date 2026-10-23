"""Market Intelligence: the underlying's regime, trend, volatility, VWAP, levels, liquidity zones and market structure.

Everything here is calculated from the instrument's price history (MarketContext). Nothing is a signal and nothing places an
order. Each block says what kind of number it is: a measured value, a calculated feature, a heuristic, or unavailable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from quant_intelligence.intelligence.market_context import context_from_pipeline
from quant_intelligence.ui.charts import price_chart
from quant_intelligence.ui.components import chip, empty_state, page_header
from quant_intelligence.ui.state import init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme
from quant_intelligence.utils.market_profile import profile_for

apply_theme()
init_session_state()
page_header(
    "Market Intelligence",
    "Regime, trend, volatility, VWAP, support and resistance, liquidity zones and market structure for the selected instrument. "
    "Measured and calculated from price history; none of it is a trade signal.",
)

output = run_pipeline_cached()
if output is None:
    st.error(st.session_state.get("pipeline_error", "The analysis pipeline is not available."))
    st.stop()

ctx = context_from_pipeline(output, st.session_state.get("option_chain"))
tone = {"OK": "good", "WARNING": "warning", "CRITICAL": "critical"}[ctx.status.level]
st.markdown(chip(f"Data: {ctx.status.level}", tone, icon="") + " " + chip(f"{ctx.underlying} · {ctx.timeframe}", "info", icon="")
            + " " + chip(f"{ctx.bars_used} bars to {ctx.info_cutoff:%d %b %H:%M}", "muted", icon=""), unsafe_allow_html=True)
if ctx.status.reason:
    st.caption(f"{ctx.status.reason}")
if not ctx.status.ok:
    empty_state("Market context unavailable", ctx.status.reason)
    st.stop()

# ------------------------------------------------------------------ chart
profile = profile_for(ctx.underlying)
levels = []
for lv in ctx.levels:
    if not lv.significant:
        continue
    levels.append({"label": lv.kind.replace("_", " ").title(), "price": lv.price, "tone": "good" if lv.price <= (ctx.last_close or 0) else "critical"})
for z in ctx.liquidity.standing[:6]:
    levels.append({"label": z.source.replace("_", " ").title() + " (liquidity zone)", "price": z.level, "tone": "info"})
st.plotly_chart(price_chart(output.ohlcv.tail(300), profile=profile, levels=levels[:14], height=480), width="stretch",
                config={"displayModeBar": False})
st.caption("Green lines sit below the last close, red above. Liquidity zones are structural hypotheses (an obvious swing, equal "
           "highs or lows, a prior session extreme): price data cannot show where stop orders actually rest.")

tabs = st.tabs(["Overview", "Levels", "Liquidity", "Structure"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime", ctx.regime.get("label") or "-", help="The regime engine's label.")
    c2.metric("Structure trend", ctx.trend["structure_trend"].title(), help="From confirmed swings and breaks of structure.")
    c3.metric("ATR (14)", f"{ctx.atr:,.2f}" if ctx.atr else "-")
    c4.metric("Relative volume", f"{ctx.volume['relative_volume']:.2f}" if ctx.volume.get("relative_volume") is not None else "-")
    st.caption("Regime scores are the engine's heuristic scores, not calibrated probabilities. " + (
        f"Clarity {ctx.regime['clarity']:.2f}, {ctx.regime['age_bars']} bars in this label." if ctx.regime.get("clarity") is not None else ""))
    v = ctx.vwap
    st.markdown("**VWAP**")
    if v["available"]:
        a, b, c, d = st.columns(4)
        a.metric("VWAP", f"{v['vwap']:,.2f}")
        b.metric("Distance", f"{v['distance_pct']:+.2f}%", help=None if v["distance_atr"] is None else f"{v['distance_atr']:+.2f} ATR")
        c.metric("Slope (ATR/bar)", f"{v['slope_atr_per_bar']:+.3f}" if v["slope_atr_per_bar"] is not None else "-")
        d.metric("Crossings today", v["crossings_session"], help=f"{v['rejections_recent']} rejection(s) in the last 12 bars")
    else:
        st.info(f"VWAP unavailable: {v['reason']}.")
    p = ctx.volume_profile
    st.markdown("**Volume profile (approximate)**")
    if p["available"]:
        a, b, c = st.columns(3)
        a.metric("POC", f"{p['poc']:,.2f}")
        b.metric("Value area high", f"{p['vah']:,.2f}")
        c.metric("Value area low", f"{p['val']:,.2f}")
        st.caption(f"Approximation: {p['approximation']}.")
    else:
        st.info(f"Volume profile unavailable: {p['reason']}.")
    vol = ctx.volatility
    st.caption(f"Volatility: ATR is {vol['atr_pct_of_price'] * 100:.2f}% of price"
               + (f", {vol['atr_percentile_100'] * 100:.0f}th percentile of the last 100 bars" if vol.get("atr_percentile_100") is not None else "")
               + (f", volatility expansion {vol['vol_expansion']:.2f}x" if vol.get("vol_expansion") is not None else "") + "." if vol.get("atr_pct_of_price") else "")

with tabs[1]:
    st.caption("Distance is in ATR units. 'Significant' levels (weight 1.5 or more) are the ones the trade check uses; open-interest peaks are "
               "context only and never count as support or resistance.")
    rows = []
    for lv in sorted(ctx.levels, key=lambda x: -x.price):
        dist = (lv.price - ctx.last_close) if ctx.last_close else None
        rows.append({"Level": lv.kind.replace("_", " ").title(), "Price": lv.price, "Side": "above" if dist and dist > 0 else "below",
                     "Distance (ATR)": round(abs(dist) / ctx.atr, 2) if dist is not None and ctx.atr else None,
                     "Weight": lv.weight, "Significant": "yes" if lv.significant else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch") if rows else st.info("No levels could be built (ATR or history missing).")
    n = ctx.nearest
    for key, label in (("support", "Nearest support"), ("resistance", "Nearest resistance")):
        if n.get(key):
            st.markdown(f"**{label}:** {n[key]['kind'].replace('_', ' ').title()} {n[key]['price']:,.2f} "
                        f"({n[key]['distance_atr']:.2f} ATR away)" if n[key]["distance_atr"] is not None else f"**{label}:** {n[key]['price']:,.2f}")

with tabs[2]:
    st.markdown("**Active liquidity events**")
    active = ctx.liquidity.active_events()
    if active:
        st.dataframe(pd.DataFrame([{"Event": e.kind.replace("_", " ").title(), "Side": e.side.replace("_", " ").title(), "Level": e.level,
                                    "Zone": e.source.replace("_", " ").title(), "Breach bar": e.breach_index, "Known at bar": e.known_index,
                                    "Valid until bar": e.expires_index, "Reading": e.note} for e in active]), hide_index=True, width="stretch")
        st.caption("A sweep is a pattern in price, never an instruction to trade; each event expires and is cancelled by a close beyond its extreme.")
    else:
        st.info("No active sweep, accepted breakout or failed breakout at a known zone.")
    st.markdown("**Standing zones (nothing has breached them)**")
    if ctx.liquidity.standing:
        st.dataframe(pd.DataFrame([{"Zone": z.source.replace("_", " ").title(), "Side": z.side.replace("_", " ").title(), "Level": z.level,
                                    "Touches": z.touches, "Known since": z.known_at} for z in ctx.liquidity.standing]),
                     hide_index=True, width="stretch")
    else:
        st.info("No standing zones.")

with tabs[3]:
    s = ctx.structure
    a, b, c = st.columns(3)
    a.metric("Trend", s.trend.title())
    b.metric("Last swing high", f"{s.last_swing_high.price:,.2f}" if s.last_swing_high else "-",
             help=None if not s.last_swing_high else f"confirmed at {s.last_swing_high.known_at:%d %b %H:%M}")
    c.metric("Last swing low", f"{s.last_swing_low.price:,.2f}" if s.last_swing_low else "-",
             help=None if not s.last_swing_low else f"confirmed at {s.last_swing_low.known_at:%d %b %H:%M}")
    st.markdown("**Breaks of structure (BOS) and shifts (MSS)**")
    if s.events:
        st.dataframe(pd.DataFrame([{"Type": e.kind, "Direction": e.direction.title(), "Broken level": e.level, "Swing bar": e.swing_index,
                                    "Break bar": e.break_index, "Known at": e.known_at} for e in s.events[-8:]][::-1]),
                     hide_index=True, width="stretch")
    else:
        st.info("No confirmed swing has been broken yet in this history.")
    st.markdown("**Fair value gaps still open**")
    gaps = s.open_fvgs()
    if gaps:
        st.dataframe(pd.DataFrame([{"Direction": g.direction.title(), "Low": g.low, "High": g.high, "Formed (bar)": g.index,
                                    "Status": g.status.title(), "Filled": f"{g.fill_pct:.0%}"} for g in gaps[-8:]][::-1]),
                     hide_index=True, width="stretch")
    else:
        st.info("No open fair value gap.")
    if s.displacements:
        last = s.displacements[-1]
        st.caption(f"Latest displacement: {last.direction.title()} bar {last.index}, body {last.body_atr:.1f} ATR.")
    st.caption("Swings need 3 bars after them to be confirmed, so the newest turning point is never shown early. Definitions: "
               "features/market_structure.py.")
