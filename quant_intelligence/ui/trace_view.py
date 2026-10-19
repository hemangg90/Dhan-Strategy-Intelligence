"""Tables for the trade trace (pure functions over `reports.trace` output)."""
from __future__ import annotations

import pandas as pd


def _r(x, nd: int = 2):
    return None if x is None or x != x else round(float(x), nd)


def trace_summary_frame(traces: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Opened": t["opened_at"], "Instrument": t["instrument"], "Strategy": t["strategy"], "Status": t["status"],
        "Exit": t["exit_reason"], "Expected R": _r(t["expected_r"]), "Realised R": _r(t["realised_r"]),
        "R basis": t["r_basis"], "Net P&L": _r(t["net_pnl"], 1), "Gross P&L": _r(t["gross_pnl"], 1),
        "Missing links": ", ".join(t["missing_links"]) or "none", "Type": t["label"],
    } for t in traces])


def timeline_frame(events: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Time (IST)": e["time"], "Step": e["stage"], "What happened": e["detail"]} for e in events])
