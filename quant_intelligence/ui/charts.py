"""Plotly charts for the trading UI.

Follows the data-viz rules: one y-axis per chart, thin marks, recessive solid hairline grid, a legend
for two or more series (single-series charts are named by their title), selective direct labels, series
colours from the validated categorical order (blue, orange, aqua), and status colours only where the
colour MEANS good/bad (P&L sign, up/down candles) - with a symbol alongside so it never stands alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quant_intelligence.ui.format import inr, num
from quant_intelligence.ui.theme import (
    AQUA, BLUE, BORDER, CARD, CRITICAL, GOOD, MUTED, ORANGE, SURFACE, TEXT, TEXT_2, TONE_COLOR,
)
from quant_intelligence.utils.market_profile import NSE, MarketProfile

GRID = "#eef0f3"
_FONT = dict(family="Inter, Segoe UI, system-ui, sans-serif", color=TEXT_2, size=12)


def _base_layout(fig: go.Figure, height: int, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=_FONT,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=TEXT_2)),
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BORDER, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=BORDER, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def _rangebreaks(profile: MarketProfile) -> list[dict]:
    """Hide weekends and the hours outside the session so intraday candles run together."""
    open_h = profile.open.hour + profile.open.minute / 60.0
    close_h = profile.close.hour + profile.close.minute / 60.0
    return [dict(bounds=["sat", "mon"]), dict(bounds=[close_h, open_h], pattern="hour")]


def _session_vwap(df: pd.DataFrame) -> pd.Series | None:
    if "volume" not in df or float(df["volume"].fillna(0).sum()) <= 0:
        return None
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    day = pd.to_datetime(df["timestamp"]).dt.date
    return (typical * df["volume"]).groupby(day).cumsum() / df["volume"].groupby(day).cumsum()


def price_chart(
    df: pd.DataFrame,
    *,
    profile: MarketProfile = NSE,
    levels: list[dict] | None = None,
    markers: list[dict] | None = None,
    emas: tuple[int, ...] = (20,),
    vwap: bool = True,
    height: int = 470,
) -> go.Figure:
    """Candlesticks with optional VWAP / EMA overlays, volume, horizontal trade levels and markers.

    `levels`: [{"label": "Entry", "price": 22610, "tone": "info"|"good"|"critical"}]
    `markers`: [{"time": ts, "price": p, "text": "ORB long", "direction": "LONG"|"SHORT"}]
    """
    has_volume = "volume" in df and float(df["volume"].fillna(0).sum()) > 0
    fig = make_subplots(
        rows=2 if has_volume else 1, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.8, 0.2] if has_volume else [1.0],
    )
    x = pd.to_datetime(df["timestamp"])
    fig.add_trace(
        go.Candlestick(
            x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price",
            increasing=dict(line=dict(color=GOOD, width=1), fillcolor=GOOD),
            decreasing=dict(line=dict(color=CRITICAL, width=1), fillcolor=CRITICAL),
            showlegend=False,
        ),
        row=1, col=1,
    )
    if vwap:
        v = _session_vwap(df)
        if v is not None:
            fig.add_trace(go.Scatter(x=x, y=v, name="VWAP", mode="lines", line=dict(color=ORANGE, width=1.6)), row=1, col=1)
    for color, span in zip((BLUE, AQUA), emas):
        fig.add_trace(
            go.Scatter(x=x, y=df["close"].ewm(span=span, adjust=False).mean(), name=f"EMA {span}",
                       mode="lines", line=dict(color=color, width=1.4)),
            row=1, col=1,
        )
    if has_volume:
        up = (df["close"] >= df["open"]).to_numpy()
        fig.add_trace(
            go.Bar(x=x, y=df["volume"], name="Volume", showlegend=False,
                   marker=dict(color=np.where(up, "rgba(12,163,12,0.45)", "rgba(208,59,59,0.45)")), hoverinfo="skip"),
            row=2, col=1,
        )
    for lv in levels or []:
        color = TONE_COLOR.get(lv.get("tone", "info"), BLUE)
        fig.add_hline(
            y=lv["price"], line=dict(color=color, width=1), row=1, col=1,
            annotation_text=f"{lv['label']} {num(lv['price'], 2)}", annotation_position="right",
            annotation_font=dict(color=color, size=11),
        )
    for mk in markers or []:
        long = mk.get("direction") == "LONG"
        fig.add_trace(
            go.Scatter(
                x=[mk["time"]], y=[mk["price"]], mode="markers", showlegend=False, hovertext=mk.get("text", ""),
                marker=dict(symbol="triangle-up" if long else "triangle-down", size=11,
                            color=GOOD if long else CRITICAL, line=dict(color=SURFACE, width=2)),
            ),
            row=1, col=1,
        )
    _base_layout(fig, height)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=_rangebreaks(profile))
    fig.update_yaxes(side="right", row=1, col=1)
    if has_volume:
        fig.update_yaxes(showgrid=False, showticklabels=False, row=2, col=1)
    fig.update_yaxes(tickformat=",.0f" if float(df["close"].median()) >= 1000 else ",.2f", row=1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=8, r=120, t=28, b=8))
    return fig


def equity_curve_chart(curve: pd.DataFrame, height: int = 300, by_time: bool = False) -> go.Figure:
    """Cumulative net P&L: one line, a zero baseline, the end value labelled. The x-axis is the closed-trade number,
    or (by_time=True) the dates in `curve["time"]` - then each point is one day."""
    fig = go.Figure()
    if len(curve):
        x = list(pd.to_datetime(curve["time"])) if by_time else list(range(1, len(curve) + 1))
        fig.add_trace(
            go.Scatter(x=x, y=curve["equity"], mode="lines+markers", name="Cumulative net P&L",
                       line=dict(color=BLUE, width=2), marker=dict(size=6, color=BLUE, line=dict(color=SURFACE, width=2)),
                       customdata=np.stack([curve["trade"], curve["pnl"]], axis=-1),
                       hovertemplate=("%{customdata[0]}<br>Day P&L %{customdata[1]:,.0f}<br>" if by_time else
                                      "Trade %{x}: %{customdata[0]}<br>Trade P&L %{customdata[1]:,.0f}<br>")
                                     + "Cumulative %{y:,.0f}<extra></extra>")
        )
        last = float(curve["equity"].iloc[-1])
        fig.add_annotation(x=x[-1], y=last, text=inr(last, signed=True), showarrow=False, xanchor="left",
                           xshift=8, font=dict(color=TEXT, size=12))
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    _base_layout(fig, height, legend=False)
    if by_time:
        fig.update_xaxes(title_text="Day", title_font=dict(color=MUTED))
    else:
        fig.update_xaxes(title_text="Closed trade #", title_font=dict(color=MUTED), dtick=1 if len(curve) <= 15 else None)
    fig.update_layout(margin=dict(l=8, r=60, t=12, b=8), hovermode="closest")
    return fig


def pnl_bar_chart(df: pd.DataFrame, label: str, value: str = "net_pnl", height: int | None = None) -> go.Figure:
    """Horizontal bars of net P&L per `label`, coloured by sign with the value written at the bar end."""
    d = df.sort_values(value, ascending=True)
    colors = [GOOD if v > 0 else CRITICAL for v in d[value]]
    fig = go.Figure(
        go.Bar(
            y=d[label], x=d[value], orientation="h", marker=dict(color=colors),
            text=[("▲ " if v > 0 else "▼ ") + inr(v, signed=True) for v in d[value]],
            textposition="outside", cliponaxis=False, textfont=dict(color=TEXT),
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        )
    )
    _base_layout(fig, height or max(180, 44 * len(d) + 40), legend=False)
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    lo, hi = min(0.0, float(d[value].min())), max(0.0, float(d[value].max()))
    pad = max(hi - lo, 1.0) * 0.32
    fig.update_xaxes(range=[lo - pad if lo < 0 else 0, hi + pad if hi > 0 else pad], tickformat=",.0f")
    fig.update_yaxes(automargin=True)
    fig.update_layout(margin=dict(l=8, r=20, t=8, b=8), hovermode="closest")
    return fig


# Diverging pair for "up" and "down": blue and orange around a dark neutral, so the green/red used for profit and loss
# elsewhere keep their meaning. Direction is never colour-only: every tile carries an arrow and a signed percentage.
UP_COLOR, DOWN_COLOR, FLAT_COLOR = "#1d5fbf", "#b85a0b", "#9ca3af"


def heatmap_treemap(df: pd.DataFrame, size_col: str | None = None, color_limit: float = 3.0, height: int = 520) -> go.Figure:
    """Treemap of instruments grouped group -> sector, coloured by `change_pct` (needs columns symbol, name, group,
    sector, change_pct, last). Equal-sized tiles unless `size_col` is given (e.g. 'day_range_pct': busier = bigger).
    Rows without a change are left out - the caller lists them."""
    d = df.dropna(subset=["change_pct"]).copy()
    fig = go.Figure()
    if d.empty:
        return _base_layout(fig, height, legend=False)
    sizes = d[size_col].fillna(0).clip(lower=0.05) if size_col else pd.Series(1.0, index=d.index)
    ids, labels, parents, values, colors, text, hover = ["All"], ["All"], [""], [0.0], [0.0], [""], [""]
    for group in [g for g in ("Index", "Stock", "Commodity") if g in set(d["group"])]:
        ids.append(group); labels.append(group); parents.append("All"); values.append(0.0); colors.append(0.0); text.append(""); hover.append("")
        for sector in sorted(set(d.loc[d["group"] == group, "sector"])):
            sid = f"{group}/{sector}"
            if sector != group:
                ids.append(sid); labels.append(sector); parents.append(group); values.append(0.0); colors.append(0.0); text.append(""); hover.append("")
            for idx, r in d[(d["group"] == group) & (d["sector"] == sector)].iterrows():
                pct = float(r["change_pct"])
                ids.append(f"{sid}/{r['symbol']}"); labels.append(r["symbol"]); parents.append(sid if sector != group else group)
                values.append(float(sizes[idx])); colors.append(pct)
                text.append(f"{'▲' if pct > 0 else '▼' if pct < 0 else '■'} {pct:+.2f}%")
                last = r.get("last")
                hover.append(f"{r['name']}<br>{pct:+.2f}%" + (f"<br>last {last:,.2f}" if pd.notna(last) else ""))
    fig.add_trace(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, text=text, customdata=hover,
        texttemplate="<b>%{label}</b><br>%{text}", hovertemplate="%{customdata}<extra></extra>",
        marker=dict(colors=colors, colorscale=[[0, DOWN_COLOR], [0.5, FLAT_COLOR], [1, UP_COLOR]], cmid=0,
                    cmin=-color_limit, cmax=color_limit, line=dict(width=2, color=SURFACE),
                    colorbar=dict(title=dict(text="% change"), thickness=10, len=0.6, ticksuffix="%")),
        textfont=dict(color="#ffffff", size=14), tiling=dict(pad=3), root_color="rgba(0,0,0,0)",
    ))
    _base_layout(fig, height, legend=False)
    fig.update_layout(hovermode="closest")
    return fig


def r_ci_chart(table: pd.DataFrame, height: int | None = None) -> go.Figure:
    """Mean R per strategy with its 95% interval (mean +/- 1.96 standard errors) against a zero line.

    A hollow marker means too few trades to conclude; a filled one has enough. A whisker that crosses zero says
    the result cannot be told apart from no edge - the honest default for most strategies on a short history."""
    d = table.dropna(subset=["avg_r"]).sort_values("avg_r", ascending=True)
    fig = go.Figure()
    if len(d):
        enough = d["verdict"] != "TOO_FEW"
        err = (1.96 * d["r_se"].fillna(0.0)).to_numpy(float)
        labels = [f"{n} (n={int(t)})" for n, t in zip(d["strategy"], d["trades"])]
        fig.add_trace(go.Scatter(
            x=d["avg_r"], y=labels, mode="markers",
            marker=dict(size=11, color=[BLUE if e else SURFACE for e in enough], symbol="circle",
                        line=dict(color=BLUE, width=2)),
            error_x=dict(type="data", array=err, color=MUTED, thickness=1.5, width=5),
            customdata=np.stack([d["verdict_text"], d["t_stat"].fillna(0.0)], axis=-1),
            hovertemplate="%{y}<br>Mean R %{x:+.2f}<br>%{customdata[0]}<br>t = %{customdata[1]:.2f}<extra></extra>",
        ))
        fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    _base_layout(fig, height or max(200, 46 * len(d) + 70), legend=False)
    fig.update_xaxes(title_text="Mean R per trade (net of charges), 95% interval", title_font=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(automargin=True)
    fig.update_layout(margin=dict(l=8, r=20, t=8, b=8), hovermode="closest")
    return fig


def oi_chart(chain, window: int = 12, height: int = 320) -> go.Figure:
    """Open interest by strike (calls vs puts) around the money, with the spot price marked."""
    rows = sorted(chain.strikes, key=lambda s: s.strike)
    if not rows:
        return _base_layout(go.Figure(), height)
    atm_i = min(range(len(rows)), key=lambda i: abs(rows[i].strike - chain.spot_price))
    view = rows[max(0, atm_i - window): atm_i + window + 1]
    strikes = [f"{r.strike:g}" for r in view]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=strikes, y=[r.ce_oi or 0 for r in view], name="Call OI", marker=dict(color=BLUE)))
    fig.add_trace(go.Bar(x=strikes, y=[r.pe_oi or 0 for r in view], name="Put OI", marker=dict(color=ORANGE)))
    atm_label = f"{rows[atm_i].strike:g}"
    atm_pos = max(0, atm_i - max(0, atm_i - window))  # index of the ATM strike inside the plotted slice
    # Mark ATM by category INDEX: a numeric-looking label ("22700") is parsed as a number on the category axis,
    # which stretches the axis to 22,700 categories and squashes every bar into a sliver.
    fig.add_shape(type="line", x0=atm_pos, x1=atm_pos, y0=0, y1=1, xref="x", yref="paper",
                  line=dict(color=TEXT_2, width=1))
    fig.add_annotation(x=atm_pos, y=1.0, xref="x", yref="paper", yanchor="bottom", showarrow=False,
                       text=f"ATM {atm_label} (spot {num(chain.spot_price, 2)})", font=dict(color=TEXT_2, size=11))
    _base_layout(fig, height)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="x unified")
    fig.update_xaxes(type="category", tickangle=-60)
    fig.update_yaxes(tickformat=".2s")
    return fig


def regime_chart(probabilities: dict, height: int = 210, top: int = 5) -> go.Figure:
    """Emphasis form: the leading regime in blue, the rest in neutral grey."""
    items = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:top][::-1]
    labels = [k.replace("_", " ").title() for k, _ in items]
    values = [v * 100 for _, v in items]
    colors = [BLUE if i == len(items) - 1 else "#9ca3af" for i in range(len(items))]
    fig = go.Figure(
        go.Bar(y=labels, x=values, orientation="h", marker=dict(color=colors),
               text=[f"{v:.0f}%" for v in values], textposition="outside", cliponaxis=False,
               textfont=dict(color=TEXT_2), hoverinfo="skip")
    )
    _base_layout(fig, height, legend=False)
    fig.update_xaxes(visible=False, range=[0, max(values + [1]) * 1.25])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=8, r=40, t=4, b=4), hovermode="closest")
    return fig
