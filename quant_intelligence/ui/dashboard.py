"""Presentation components for the hero dashboard and Decision Center (HTML, light theme).

Pure rendering: every value comes from the pipeline output; a missing value renders as "n/a", never a placeholder.
"""
from __future__ import annotations

import html

import streamlit as st

from quant_intelligence.ui.components import chip

DASH_CSS = """
<style>
html { font-size: 17px; }  /* projector readability */
[data-testid="stMetricValue"] { font-size: 1.7rem; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a { font-size: 0.95rem; }
.qi-hero { display:flex; flex-wrap:wrap; align-items:flex-end; justify-content:space-between; gap:12px; margin:2px 0 14px 0; }
.qi-hero h1 { margin:0 !important; }
.qi-hero .sub { color: var(--qi-text-2); font-size:0.95rem; margin-top:2px; }
.qi-hero .chips { display:flex; gap:8px; flex-wrap:wrap; }

.qi-kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:6px 0 16px 0; }
.qi-kpi { background:var(--qi-card); border:1px solid var(--qi-border); border-radius:14px; padding:14px 16px; }
.qi-kpi .l { color:var(--qi-muted); font-size:0.72rem; letter-spacing:.08em; text-transform:uppercase; }
.qi-kpi .v { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:1.9rem; font-weight:700; line-height:1.2; margin-top:2px; }
.qi-kpi .s { color:var(--qi-text-2); font-size:0.82rem; margin-top:2px; }
.qi-kpi.good .v { color:var(--qi-good); } .qi-kpi.critical .v { color:var(--qi-critical); } .qi-kpi.warning .v { color:var(--qi-warning); }

.qi-banner { border-radius:16px; padding:18px 22px; margin:6px 0 16px 0; border:1px solid var(--qi-border); background:var(--qi-card); }
.qi-banner .k { font-size:0.72rem; letter-spacing:.1em; text-transform:uppercase; color:var(--qi-muted); }
.qi-banner .t { font-size:2.1rem; font-weight:750; letter-spacing:-0.01em; margin:2px 0 4px 0; }
.qi-banner .r { color:var(--qi-text-2); font-size:0.98rem; max-width:1100px; }
.qi-banner.no { border-left:8px solid var(--qi-critical); } .qi-banner.no .t { color:#b91c1c; }
.qi-banner.go { border-left:8px solid var(--qi-good); } .qi-banner.go .t { color:#047857; }
.qi-banner.tie { border-left:8px solid var(--qi-warning); } .qi-banner.tie .t { color:#b45309; }

.qi-prob { display:grid; grid-template-columns:150px 1fr 48px; gap:10px; align-items:center; margin:5px 0; font-size:0.9rem; }
.qi-prob .bar { height:10px; background:var(--qi-card-2); border-radius:999px; overflow:hidden; }
.qi-prob .bar > div { height:100%; background:var(--qi-blue); border-radius:999px; }
.qi-prob.top .bar > div { background:var(--qi-text); } .qi-prob.top { font-weight:650; }
.qi-prob .n { text-align:right; font-family:ui-monospace,Menlo,Consolas,monospace; color:var(--qi-text-2); }

.qi-table { width:100%; border-collapse:separate; border-spacing:0; background:var(--qi-card); border:1px solid var(--qi-border); border-radius:14px; overflow:hidden; font-size:0.9rem; }
.qi-table th { text-align:left; color:var(--qi-muted); font-size:0.7rem; letter-spacing:.08em; text-transform:uppercase; padding:10px 12px; background:var(--qi-card-2); border-bottom:1px solid var(--qi-border); }
.qi-table td { padding:10px 12px; border-bottom:1px solid var(--qi-border); vertical-align:middle; }
.qi-table tr:last-child td { border-bottom:none; }
.qi-table tr.sel td { background:#f1f5f9; }
.qi-table .mono { font-family:ui-monospace,Menlo,Consolas,monospace; }
.qi-table .why { color:var(--qi-text-2); font-size:0.8rem; }
.qi-exp { display:flex; align-items:center; gap:8px; min-width:170px; }
.qi-exp .track { position:relative; flex:1; height:8px; background:var(--qi-card-2); border-radius:999px; }
.qi-exp .mid { position:absolute; left:50%; top:-2px; width:1px; height:12px; background:var(--qi-muted); }
.qi-exp .fill { position:absolute; top:0; height:8px; border-radius:999px; }
.qi-exp .val { width:58px; text-align:right; font-family:ui-monospace,Menlo,Consolas,monospace; }
.qi-note { color:var(--qi-text-2); font-size:0.82rem; margin-top:8px; }
</style>
"""

_STATUS_TONE = {"SELECTED": "good", "TIE-BREAK": "warning", "ELIGIBLE": "info", "REJECTED": "muted"}


def inject_css() -> None:
    st.markdown(DASH_CSS, unsafe_allow_html=True)


def hero_header(title: str, subtitle: str, chips: list[str]) -> None:
    st.markdown(
        f'<div class="qi-hero"><div><h1>{html.escape(title)}</h1><div class="sub">{subtitle}</div></div>'
        f'<div class="chips">{"".join(chips)}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_cards(items: list[dict]) -> None:
    """items: {label, value, sub?, tone?} where tone is good / critical / warning / None."""
    cards = "".join(
        f'<div class="qi-kpi {it.get("tone") or ""}"><div class="l">{html.escape(it["label"])}</div>'
        f'<div class="v">{html.escape(str(it["value"]))}</div>'
        f'<div class="s">{html.escape(str(it.get("sub") or ""))}</div></div>'
        for it in items
    )
    st.markdown(f'<div class="qi-kpis">{cards}</div>', unsafe_allow_html=True)


def decision_banner(is_no_trade: bool, selected: str | None, reason: str, tie_break: bool = False) -> None:
    if is_no_trade:
        cls, kicker, title = "no", "Decision", "NO TRADE"
    elif tie_break:
        cls, kicker, title = "tie", "Decision - tie-break", f"{selected}"
    else:
        cls, kicker, title = "go", "Decision", f"{selected}"
    st.markdown(
        f'<div class="qi-banner {cls}"><div class="k">{kicker}</div><div class="t">{html.escape(str(title))}</div>'
        f'<div class="r">{html.escape(reason)}</div></div>',
        unsafe_allow_html=True,
    )


def probability_bars(probs: dict, top_n: int = 6) -> None:
    items = sorted(probs.items(), key=lambda kv: -kv[1])[:top_n]
    rows = "".join(
        f'<div class="qi-prob{" top" if i == 0 else ""}"><span>{html.escape(str(name).replace("_", " ").title())}</span>'
        f'<div class="bar"><div style="width:{max(0.0, min(p, 1.0)) * 100:.1f}%"></div></div>'
        f'<span class="n">{p * 100:.0f}%</span></div>'
        for i, (name, p) in enumerate(items)
    )
    st.markdown(rows, unsafe_allow_html=True)


def _dd_text(value) -> str:
    return "n/a" if value is None else "%.1fR" % value


def _exp_cell(value) -> str:
    if value is None:
        return '<span class="mono" style="color:var(--qi-muted)">n/a</span>'
    scale = 1.0  # bar spans +/-1R either side of the midline, clipped
    frac = max(-1.0, min(float(value) / scale, 1.0))
    color = "var(--qi-good)" if value > 0 else "var(--qi-critical)"
    left = 50 if frac >= 0 else 50 + frac * 50
    width = abs(frac) * 50
    return (
        f'<div class="qi-exp"><div class="track"><div class="mid"></div>'
        f'<div class="fill" style="left:{left:.1f}%;width:{width:.1f}%;background:{color}"></div></div>'
        f'<span class="val">{value:+.2f}R</span></div>'
    )


def ranking_table(rows: list[dict], show_why: bool = True) -> None:
    """rows are `decision_view.explanation_rows` output: ranked strategies with their evidence."""
    body = []
    for r in rows:
        status = r["Status"]
        cells = [
            f'<td><b>{html.escape(r["Strategy"])}</b></td>',
            f'<td>{chip(status, _STATUS_TONE.get(status, "muted"))}</td>',
            f'<td>{_exp_cell(r["Cond. expectancy (R, net of costs)"])}</td>',
            f'<td class="mono">{r["Win rate (historical)"] or "n/a"}</td>',
            f'<td class="mono">{r["Analogue sample"]}</td>',
            f'<td>{html.escape(str(r["Confidence"]).replace("_", " ").title())}</td>',
            f'<td class="mono">{_dd_text(r["Max drawdown (R)"])}</td>',
        ]
        if show_why:
            cells.append(f'<td class="why">{html.escape(r["Why"])}</td>')
        body.append(f'<tr class="{"sel" if status in ("SELECTED", "TIE-BREAK") else ""}">{"".join(cells)}</tr>')
    head = ["Strategy", "Status", "Net expectancy", "Win rate", "Sample", "Confidence", "Max DD"] + (["Why"] if show_why else [])
    st.markdown(
        f'<table class="qi-table"><thead><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
        '<div class="qi-note">Historical, cost-adjusted conditional statistics from similar past market states. '
        "Not a prediction. Small samples: read the confidence column.</div>",
        unsafe_allow_html=True,
    )
