"""Downloadable forms of a daily report: Excel (one sheet per section), Markdown and JSON."""
from __future__ import annotations

import io
import json

import pandas as pd


def _sheet(rows) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame({"note": ["(none)"]})


def sheets(report: dict) -> dict[str, pd.DataFrame]:
    """Name -> table for every section of a report's JSON (names are Excel-safe, 31 characters max)."""
    s, f = report["summary"], report["funnel"]
    out = {
        "Summary": _sheet([
            {"period": "Today", **{k: v for k, v in s["today"].items() if not isinstance(v, (dict, list))}},
            {"period": "All time", **{k: v for k, v in s["cumulative"].items() if not isinstance(v, (dict, list))}},
        ]),
        "Narrative": pd.DataFrame({"how things went": report["narrative"]}),
        "Strategies (all time)": _sheet(report["strategies"]["all_time"]),
        "Strategies (30 days)": _sheet(next((v for k, v in report["strategies"].items() if k.startswith("last_")), [])),
        "Strategies (today)": _sheet(report["strategies"]["today"]),
        "Signals by strategy": _sheet(f["by_strategy"]),
        "No-trade reasons": _sheet(f["no_trade_reasons"]),
        "Veto reasons": _sheet(f["veto_reasons"]),
        "Expected vs realised": _sheet(report["expected_vs_realised"]),
        "Daily": _sheet(report["daily"]),
        "Order rejects": _sheet(report["orders"]["reject_reasons"]),
        "Risk vetoes": _sheet(report["risk"]["vetoes"]),
        "Setups not traded": _sheet(report.get("rejected_setups")),
        "Data quality": _sheet((report.get("data_quality") or {}).get("items")),
        "Methodology": pd.DataFrame({"how to read this report": report.get("methodology") or ["(not recorded in this report)"]}),
    }
    for key, rows in report["breakdowns"].items():
        out[f"By {key}"[:31]] = _sheet(rows)
    return out


def to_excel(report: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets(report).items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


def _inline(text: str) -> str:
    import html
    import re

    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text))


def markdown_to_html(markdown: str, title: str) -> str:
    """Self-contained, print-friendly HTML from the report's own Markdown (headings, bullets, tables, paragraphs). Open it in a
    browser and print to PDF. No external files or scripts."""
    import html

    body, in_list, table = [], False, []

    def close_list():
        nonlocal in_list
        if in_list:
            body.append("</ul>")
            in_list = False

    def flush_table():
        nonlocal table
        if table:
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table
                     if not set(r.replace("|", "").strip()) <= set("-: ")]
            if cells:
                head, rows = cells[0], cells[1:]
                body.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                            + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>")
            table = []

    for line in markdown.splitlines():
        if line.startswith("|"):
            close_list()
            table.append(line)
            continue
        flush_table()
        if line.startswith("## "):
            close_list()
            body.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            close_list()
            body.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("- "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{_inline(line[2:])}</li>")
        elif line.strip():
            close_list()
            body.append(f"<p>{_inline(line)}</p>")
        else:
            close_list()
    close_list()
    flush_table()
    style = ("body{font:14px/1.5 system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#111}"
             "table{border-collapse:collapse;width:100%;margin:8px 0 16px}th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;font-size:13px}"
             "th{background:#f3f4f6}h1{font-size:22px}h2{font-size:17px;margin-top:22px;border-bottom:1px solid #ddd}"
             "@media print{body{margin:0}}")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f"<style>{style}</style></head><body>{''.join(body)}</body></html>")


def to_json(report: dict) -> bytes:
    return json.dumps(report, indent=2, default=str).encode("utf-8")
