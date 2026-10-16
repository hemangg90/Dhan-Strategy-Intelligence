"""Styled tables for the trading pages: Indian-grouped numbers, signed P&L (so the sign carries the
meaning, colour only reinforces it), human labels. Underlying values stay numeric so columns sort."""
from __future__ import annotations

import pandas as pd

from quant_intelligence.ui import format as F
from quant_intelligence.ui.position_views import exit_reason_label

_GOOD, _BAD = "#059669", "#dc2626"


def _pnl_color(v) -> str:
    if v is None or pd.isna(v) or v == 0:
        return ""
    return f"color: {_GOOD}; font-weight: 600" if v > 0 else f"color: {_BAD}; font-weight: 600"


def _signed0(v) -> str:
    return F.num(v, 0, signed=True)


def _held(v) -> str:
    return F.duration(v) if v is not None and not pd.isna(v) else "–"


def trades_table(df: pd.DataFrame):
    """Closed trades: gross, charges and net P&L like a contract note."""
    d = pd.DataFrame(
        {
            "Closed": pd.to_datetime(df["closed_at"]),
            "Contract": df["contract"],
            "Side": df["side"],
            "Qty": df["qty"],
            "Entry": df["entry"],
            "Exit": df["exit_price"],
            "Gross P&L": df["gross_pnl"],
            "Charges": df["charges"],
            "Net P&L": df["net_pnl"],
            "Return %": df["return_pct"],
            "Held": df["held"].map(_held),
            "Exit reason": df["exit_reason"].map(exit_reason_label),
            "Strategy": df["strategy"],
            "Tag": df["tag"].fillna(""),
        }
    )
    return (
        d.style.format(
            {"Entry": lambda v: F.num(v, 2), "Exit": lambda v: F.num(v, 2), "Gross P&L": _signed0,
             "Charges": lambda v: F.num(v, 2), "Net P&L": _signed0, "Return %": lambda v: F.pct(v, 1),
             "Closed": lambda v: f"{v:%d %b %H:%M}" if pd.notna(v) else "–"}
        ).map(_pnl_color, subset=["Gross P&L", "Net P&L", "Return %"])
    )


def orders_table(df: pd.DataFrame):
    d = pd.DataFrame(
        {
            "Time": pd.to_datetime(df["timestamp"]),
            "Order ID": df["order_id"],
            "Contract": df["contract"],
            "Direction": df["side"],
            "Qty": df["quantity"],
            "Type": df["order_type"],
            "Price": df["price"],
            "Status": df["status"].map(lambda s: {"FILLED": "✓ FILLED", "REJECTED": "✖ REJECTED"}.get(s, s)),
            "Reason": df["reject_reason"].fillna(""),
            "Strategy": df["strategy_name"],
            "Tag": df["tag"].fillna(""),
            "Mode": df["mode"],
        }
    )
    return d.style.format({"Price": lambda v: F.num(v, 2), "Time": lambda v: f"{v:%d %b %H:%M:%S}" if pd.notna(v) else "–"})


def fills_table(df: pd.DataFrame):
    d = pd.DataFrame(
        {
            "Time": pd.to_datetime(df["timestamp"]),
            "Order ID": df["order_id"],
            "Contract": df["contract"],
            "Qty": df["quantity"],
            "Fill price": df["fill_price"],
            "Slippage /unit": df["slippage"],
            "Slippage cost": df["slippage"] * df["quantity"],
        }
    )
    return d.style.format(
        {"Fill price": lambda v: F.num(v, 2), "Slippage /unit": lambda v: F.num(v, 2),
         "Slippage cost": lambda v: F.inr(v, 2), "Time": lambda v: f"{v:%d %b %H:%M:%S}" if pd.notna(v) else "–"}
    )


def history_table(df: pd.DataFrame):
    """Every position (open, closed or stale) for the audit trail."""
    d = pd.DataFrame(
        {
            "Opened": pd.to_datetime(df["opened_at"]),
            "Contract": df["contract"],
            "Side": df["side"],
            "Qty": df["qty"],
            "Entry": df["entry"],
            "Exit": df["exit_price"],
            "Net P&L": df["net_pnl"],
            "Status": df["status"],
            "Exit reason": df["exit_reason"].map(exit_reason_label),
            "Strategy": df["strategy"],
            "Tag": df["tag"].fillna("") if "tag" in df else "",
            "Expected R": df["expected_r"] if "expected_r" in df else None,
            "Position ID": df["position_id"],
        }
    )
    return d.style.format(
        {"Entry": lambda v: F.num(v, 2), "Exit": lambda v: F.num(v, 2), "Net P&L": _signed0,
         "Expected R": lambda v: f"{v:+.2f}" if pd.notna(v) else "–",
         "Opened": lambda v: f"{v:%d %b %H:%M}" if pd.notna(v) else "–"}
    ).map(_pnl_color, subset=["Net P&L"])


def _oi(v) -> str:
    """Open interest in lakh/crore-style short form: 1,234,500 -> '12.35L'."""
    if v is None or pd.isna(v):
        return "–"
    a = abs(v)
    if a >= 1e7:
        return f"{v / 1e7:.2f}Cr"
    if a >= 1e5:
        return f"{v / 1e5:.2f}L"
    return F.num(v, 0)


def ladder_table(df: pd.DataFrame, spot: float):
    """Broker-style option chain: calls | strike | puts. The ATM row is highlighted and in-the-money
    sides are lightly shaded (calls below spot, puts above)."""
    d = pd.DataFrame(
        {
            "Call OI": df["ce_oi"], "Call ΔOI": df["ce_oi_change"], "Call IV": df["ce_iv"], "Call LTP": df["ce_ltp"],
            "Strike": df["strike"],
            "Put LTP": df["pe_ltp"], "Put IV": df["pe_iv"], "Put ΔOI": df["pe_oi_change"], "Put OI": df["pe_oi"],
        }
    ).reset_index(drop=True)
    atm = df["is_atm"].tolist()
    strikes = df["strike"].tolist()
    n = len(d.columns)

    def style_row(row):
        i = row.name
        css = [""] * n
        if strikes[i] < spot:  # calls in the money
            css[:4] = ["background-color: rgba(57,135,229,0.10)"] * 4
        if strikes[i] > spot:  # puts in the money
            css[5:] = ["background-color: rgba(217,89,38,0.10)"] * 4
        css[4] = "font-weight: 700; background-color: rgba(17,24,39,0.05)"
        if atm[i]:
            css = ["background-color: rgba(57,135,229,0.28); font-weight: 700"] * n
        return css

    return (
        d.style.apply(style_row, axis=1)
        .format({"Call OI": _oi, "Put OI": _oi, "Call ΔOI": lambda v: _oi(v) if pd.notna(v) else "–",
                 "Put ΔOI": lambda v: _oi(v) if pd.notna(v) else "–", "Call IV": lambda v: F.num(v, 1),
                 "Put IV": lambda v: F.num(v, 1), "Call LTP": lambda v: F.num(v, 2), "Put LTP": lambda v: F.num(v, 2),
                 "Strike": lambda v: F.num(v, 0)})
    )


def ranking_frame(strategy_intel, selected: str | None) -> pd.DataFrame:
    """Every strategy's standing for the current market state, best score first."""
    rows = []
    for si in strategy_intel:
        sc = si.score
        rows.append(
            {
                "strategy": si.strategy_name,
                "score": sc.score,
                "expected_r": sc.expected_r,
                "win_prob": None if sc.prob_positive_return is None else sc.prob_positive_return * 100.0,
                "confidence": sc.confidence_label,
                "samples": sc.sample_size,
                "eligible": bool(sc.eligible),
                "note": sc.ineligibility_reason or ("Selected" if si.strategy_name == selected else "Eligible"),
                "selected": si.strategy_name == selected,
            }
        )
    out = pd.DataFrame(rows, columns=["strategy", "score", "expected_r", "win_prob", "confidence", "samples",
                                      "eligible", "note", "selected"])
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def ranking_table(df: pd.DataFrame):
    d = pd.DataFrame(
        {
            "#": range(1, len(df) + 1),
            "Strategy": [("★ " if sel else "") + name for name, sel in zip(df["strategy"], df["selected"])],
            "Score": df["score"],
            "Expected R": df["expected_r"],
            "Win prob": df["win_prob"],
            "Confidence": df["confidence"],
            "Samples": df["samples"],
            "Status": [("✓ " if ok else "✖ ") + note for ok, note in zip(df["eligible"], df["note"])],
        }
    )
    sel = df["selected"].tolist()
    return d.style.format(
        {"Score": lambda v: F.num(v, 3), "Expected R": lambda v: F.num(v, 3), "Win prob": lambda v: F.pct(v, 0, signed=False)}
    ).apply(lambda row: ["background-color: rgba(57,135,229,0.18); font-weight: 600" if sel[row.name] else ""] * len(row), axis=1)
