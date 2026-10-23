"""End-of-day report: how the system worked today, how every strategy has done so far, and what it all means.

Built ONLY from what the system recorded (positions, orders, fills, risk events, the scan decision log, system
events). Nothing is estimated or invented, and every conclusion is worded no more strongly than the sample allows
(see `performance.py`). The narrative is plain rules over those numbers - no language model. A report can be
re-generated for any past date; the stored copy for that date and scope is replaced.
"""
from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from quant_intelligence.reports import performance as perf
from quant_intelligence.ui import format as F
from quant_intelligence.ui.position_views import positions_frame
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeutil import now_ist

SCOPES = ("ALL", "NSE", "MCX")
_UTC_TO_IST = dt.timedelta(hours=5, minutes=30)  # risk_events / system_events are stamped in UTC
STRATEGY_DAYS = 30


# ------------------------------------------------------------------------------------------ helpers
def jsonable(obj):
    """JSON-safe copy: NaN/inf -> None; numpy/pandas scalars, dates and Timestamps -> python / ISO strings."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.total_seconds()
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, int):
        return obj
    return str(obj)


def records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    return jsonable(df.head(limit).to_dict("records") if limit else df.to_dict("records"))


def _count_table(items, label: str, top: int = 6) -> list[dict]:
    counts: dict[str, int] = {}
    for it in items:
        key = (it or "(none)")
        counts[key] = counts.get(key, 0) + 1
    return [{label: k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:top]]


def _in_scope(instrument: str | None, scope: str) -> bool:
    return scope == "ALL" or profile_for(instrument).name == scope


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.0f}%"


# ------------------------------------------------------------------------------------------ data access
def _load(report_date: dt.date, scope: str) -> dict:
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import MarketDataMetadata, Order, Position, RiskEvent, ScanDecision, SystemEvent

    day_start = dt.datetime.combine(report_date, dt.time.min)
    day_end = day_start + dt.timedelta(days=1)
    with get_session() as session:
        positions = [p for p in session.query(Position).all() if _in_scope(p.underlying or p.instrument, scope)]
        orders = [o for o in session.query(Order).filter(Order.timestamp >= day_start, Order.timestamp < day_end).all()
                  if _in_scope(o.instrument, scope)]
        decisions = session.query(ScanDecision).filter(ScanDecision.decided_at >= day_start, ScanDecision.decided_at < day_end).all()
        decisions = [d for d in decisions if scope == "ALL" or d.market == scope]
        utc_lo, utc_hi = day_start - _UTC_TO_IST, day_end - _UTC_TO_IST
        risk = session.query(RiskEvent).filter(RiskEvent.timestamp >= utc_lo, RiskEvent.timestamp < utc_hi).all()
        events = session.query(SystemEvent).filter(SystemEvent.timestamp >= utc_lo, SystemEvent.timestamp < utc_hi,
                                                   SystemEvent.level.in_(["WARNING", "ERROR"])).all()
        quality = session.query(MarketDataMetadata).filter(MarketDataMetadata.fetched_at >= utc_lo,
                                                           MarketDataMetadata.fetched_at < utc_hi).all()
        quality = [{"instrument": m.instrument, "status": m.quality_status, "source": m.source,
                    "category": (m.quality_report or {}).get("category"), "issues": list((m.quality_report or {}).get("issues") or [])}
                   for m in quality if _in_scope(m.instrument, scope)]
        # detach plain values so the session can close
        return {
            "data_quality": quality,
            "positions": positions_frame(positions, now=now_ist()),
            "orders": [{"status": o.status, "reject_reason": o.reject_reason, "strategy": o.strategy_name} for o in orders],
            "decisions": [{"status": d.status, "reason_class": d.reason_class, "strategy": d.strategy, "tie_break": bool(d.tie_break),
                           "reason": d.reason, "instrument": d.instrument, "bar_ts": d.bar_ts, "expected_r": d.expected_r}
                          for d in decisions],
            "risk": [{"event_type": r.event_type, "reason": r.reason} for r in risk],
            "events": [{"component": e.component, "level": e.level, "message": e.message} for e in events],
        }


# ------------------------------------------------------------------------------------------ sections
def _funnel(decisions: list[dict]) -> dict:
    by_status: dict[str, int] = {}
    for d in decisions:
        by_status[d["status"]] = by_status.get(d["status"], 0) + 1
    signals = [d for d in decisions if d["status"] in ("TRIGGERED", "VETOED", "REJECTED", "FILLED")]
    strategies: dict[str, dict] = {}
    for d in signals:
        s = strategies.setdefault(d["strategy"] or "(none)", {"strategy": d["strategy"] or "(none)", "signals": 0, "filled": 0, "vetoed": 0, "rejected": 0})
        s["signals"] += 1
        s["filled"] += d["status"] == "FILLED"
        s["vetoed"] += d["status"] == "VETOED"
        s["rejected"] += d["status"] == "REJECTED"
    return {
        "bars_evaluated": len(decisions),
        "by_status": by_status,
        "signals": len(signals),
        "filled": by_status.get("FILLED", 0),
        "vetoed": by_status.get("VETOED", 0),
        "rejected": by_status.get("REJECTED", 0),
        "tie_break": sum(1 for d in decisions if d["tie_break"] and d["status"] != "NO_TRADE"),
        "direction_filtered": sum(1 for d in decisions if d["reason_class"] == "DIRECTION_FILTER"),
        "no_room": sum(1 for d in decisions if d["reason_class"] == "NO_ROOM"),
        "no_size": sum(1 for d in decisions if d["reason_class"] == "NO_SIZE"),
        "no_size_instruments": sorted({d.get("instrument") for d in decisions if d["reason_class"] == "NO_SIZE" and d.get("instrument")}),
        "no_trade_reasons": _count_table([d["reason_class"] for d in decisions if d["status"] == "NO_TRADE"], "reason"),
        "veto_reasons": _count_table([d["reason_class"] for d in decisions if d["status"] == "VETOED"], "reason"),
        "by_strategy": sorted(strategies.values(), key=lambda r: -r["signals"]),
    }


REJECTED_SETUP_STATUSES = ("TRIGGERED", "VETOED", "REJECTED", "SKIPPED", "CHAIN_ERROR")


def _rejected_setups(decisions: list[dict], limit: int = 100) -> list[dict]:
    """Setups that fired (or were looked at) and did NOT become a trade, with the reason, newest bar first."""
    rows = [d for d in decisions if d["status"] in REJECTED_SETUP_STATUSES]
    rows.sort(key=lambda d: str(d.get("bar_ts")), reverse=True)
    return [{"bar": d.get("bar_ts"), "instrument": d.get("instrument"), "strategy": d.get("strategy"), "status": d["status"],
             "reason_class": d.get("reason_class"), "reason": (d.get("reason") or "")[:200]} for d in rows[:limit]]


def _data_quality(items: list[dict], top: int = 20) -> dict:
    """What the data-quality gate found while the day's data was loaded: how many loads, how many were not OK, and what."""
    not_ok = [i for i in items if i["status"] != "OK"]
    seen: dict[tuple, dict] = {}
    for i in not_ok:
        key = (i["instrument"], i["status"], (i["issues"] or [""])[0][:100])
        seen.setdefault(key, {"instrument": i["instrument"], "status": i["status"], "category": i["category"],
                              "source": i["source"], "issue": key[2], "times": 0})["times"] += 1
    return {"loads": len(items), "not_ok": len(not_ok),
            "items": sorted(seen.values(), key=lambda r: -r["times"])[:top]}


def _health(data: dict, unresolved: dict) -> dict:
    from quant_intelligence.config.credentials import token_status
    from quant_intelligence.config.settings import SETTINGS

    ts = token_status(SETTINGS.dhan_access_token)
    events = data["events"]
    return {
        "token": {"state": ts["state"], "expires_at": ts["expires_at"]},
        "warnings": sum(1 for e in events if e["level"] == "WARNING"),
        "errors": sum(1 for e in events if e["level"] == "ERROR"),
        "by_component": _count_table([e["component"] for e in events], "component"),
        "top_messages": _count_table([(e["message"] or "")[:120] for e in events], "message", top=5),
        "unresolved": unresolved,
    }


def _narrative(rep: dict) -> list[str]:
    s = rep["summary"]["today"]
    f = rep["funnel"]
    out: list[str] = []
    if s["trades"]:
        out.append(f"{s['trades']} trade(s) closed today: net P&L {F.inr(s['net'], signed=True)} after charges "
                   f"(win rate {_pct(s['win_rate'])}, {s['wins']}W/{s['losses']}L; best {F.inr(s['best'], signed=True)}, worst {F.inr(s['worst'], signed=True)}).")
    else:
        out.append("No trade closed today.")
    if f["bars_evaluated"]:
        stood = f["by_status"].get("NO_TRADE", 0)
        line = f"{f['bars_evaluated']} instrument-bars were evaluated; the system stood aside on {stood}"
        if f["no_trade_reasons"]:
            top = f["no_trade_reasons"][0]
            line += f" (most often: {top['reason']}, {top['count']}x)"
        out.append(line + ".")
        if f["signals"]:
            out.append(f"{f['signals']} setup(s) triggered: {f['filled']} filled, {f['vetoed']} stopped by the risk engine"
                       + (f" (mostly {f['veto_reasons'][0]['reason']})" if f["veto_reasons"] else "")
                       + f", {f['rejected']} rejected by the broker.")
        if f["tie_break"]:
            out.append(f"{f['tie_break']} decision(s) came from the tie-break rule (top strategies statistically tied).")
        if f.get("direction_filtered"):
            out.append(f"{f['direction_filtered']} setup(s) were skipped because they pointed against the heatmap mover's direction "
                       "(an up-mover only takes calls, a down-mover only puts).")
        if f.get("no_room"):
            out.append(f"{f['no_room']} setup(s) were skipped because a support/resistance or liquidity level sat too close "
                       "in the trade's way (less than the required room to run).")
        if f["no_size"]:
            names = ", ".join(f["no_size_instruments"][:6])
            out.append(f"{f['no_size']} setup(s) were skipped because one lot's stop-loss exceeds the per-trade loss budget "
                       f"({names}{'...' if len(f['no_size_instruments']) > 6 else ''}). A tighter stop (STOP_ATR_SCALE) or a larger "
                       "budget would let those instruments trade.")
    else:
        out.append("No scan decisions were recorded today - the auto-traders were probably not running (or the market was closed).")
    out.append(rep["leader"]["text"])
    for r in rep["expected_vs_realised"]:
        if r["gap_t"] is not None and r["gap_t"] <= -2 and r["trades"] >= perf.MIN_TRADES:
            out.append(f"{r['strategy']} is under-delivering its backtest: expected {r['expected_r']:+.2f}R per trade, realised "
                       f"{r['realised_r']:+.2f}R over {r['trades']} trades.")
    tb = {b["tag_label"]: b for b in rep["breakdowns"]["tag_label"]}
    if "TIE-BREAK" in tb and tb["TIE-BREAK"]["trades"]:
        t = tb["TIE-BREAK"]
        out.append(f"Tie-break trades so far: {t['trades']} closed, net {F.inr(t['net_pnl'], signed=True)} "
                   + ("(too few to judge)." if t["trades"] < perf.MIN_TRADES else "."))
    h = rep["health"]
    if h["token"]["state"] in ("expired", "missing"):
        out.append("The Dhan access token is expired or missing: the data feed is paused and no new trades can be priced until a new token is entered.")
    if h["unresolved"]["stale"] or h["unresolved"]["open"]:
        out.append(f"{h['unresolved']['stale']} position(s) were left unresolved because the app was not running at the close "
                   f"(capital tied {F.inr(h['unresolved']['invested'])}); they have no result and are excluded from statistics.")
    if h["errors"] or h["warnings"]:
        top = h["by_component"][0]["component"] if h["by_component"] else "?"
        out.append(f"System log: {h['errors']} error(s) and {h['warnings']} warning(s) today, mostly from {top}.")
    return out


METHODOLOGY = [
    "Trade results are SIMULATED: paper orders filled at the last price plus modelled slippage, with modelled brokerage and "
    "statutory charges. They are not exchange fills.",
    "Expected R is a HISTORICAL estimate: the strategy's net R in past situations most similar to the one at entry, shrunk toward "
    "its own overall mean. It is not a forecast.",
    "R = net P&L after charges divided by the amount risked at the stop. A strategy is judged on mean R and its t-statistic, "
    "never on a single day.",
    "The signals-vs-trades funnel counts one decision per instrument per closed bar; the strongest outcome on a bar is kept.",
    "Results exclude positions left open when the app was not running (they have no outcome). The data-quality gate must pass "
    "for a decision to be made; bars that fail it produce NO TRADE, not a guess.",
    "Nothing here is investment advice or evidence that a strategy will be profitable.",
]
LABELS = {
    "trades": "SIMULATED (paper trading)",
    "expected_r": "HISTORICAL (backtest-based estimate)",
    "decisions": "Recorded by the system while it scanned",
}

CAVEATS = [
    "Paper trading: fills use simple slippage on the last price; real fills depend on liquidity and spread.",
    "Charges are modelled at Dhan's brokerage and the statutory option rates; check them against a contract note.",
    f"A strategy is only called good or bad with at least {perf.MIN_TRADES} closed trades; smaller samples are noise.",
    "R = net P&L after charges / premium risked at the stop. Verdicts use the t-statistic of mean R (|t| >= 2).",
    "History is only as complete as the database: positions left open when the app was asleep have no result.",
]


# ------------------------------------------------------------------------------------------ build
def build_report(report_date: dt.date, scope: str = "ALL", now: dt.datetime | None = None) -> dict:
    """Return {"content_json": dict, "content_markdown": str} for one date and scope."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    now = now or now_ist()
    data = _load(report_date, scope)
    frame = perf.with_r(data["positions"]) if len(data["positions"]) else data["positions"]
    closed = perf.closed_only(frame) if len(frame) else frame

    today = closed[closed["date"] == report_date] if len(closed) else closed
    before = closed[closed["date"] < report_date] if len(closed) else closed
    prev = None
    if len(before):
        last_day = max(before["date"])
        prev = {"date": last_day.isoformat(), "net": float(before[before["date"] == last_day]["net_pnl"].sum()),
                "trades": int((before["date"] == last_day).sum())}
    cumulative = perf.summarise(closed[closed["date"] <= report_date]) if len(closed) else perf.summarise(closed)
    unresolved = perf.unresolved(frame) if len(frame) else {"stale": 0, "open": 0, "invested": 0.0, "dates": []}
    upto = closed[closed["date"] <= report_date] if len(closed) else closed
    table = perf.strategy_table(upto) if len(upto) else perf.strategy_table(closed)
    recent = upto[upto["date"] >= report_date - dt.timedelta(days=STRATEGY_DAYS)] if len(upto) else upto

    rep = {
        "date": report_date.isoformat(), "scope": scope, "generated_at": now.isoformat(),
        "summary": {"today": perf.summarise(today), "previous": prev, "cumulative": cumulative,
                    "open_positions": int((frame["status"] == "OPEN").sum()) if len(frame) else 0},
        "funnel": _funnel(data["decisions"]),
        "strategies": {"all_time": records(table), f"last_{STRATEGY_DAYS}_days": records(perf.strategy_table(recent)) if len(recent) else [],
                       "today": records(perf.strategy_table(today)) if len(today) else []},
        "leader": perf.leader(table),
        "breakdowns": {k: records(perf.breakdown(upto, k)) for k in
                       ("instrument", "market", "option_type", "exit_reason", "hour", "weekday", "tag_label")},
        "expected_vs_realised": records(perf.expected_vs_realised(upto)),
        "daily": records(perf.daily_series(upto).tail(60)),
        "orders": {"total": len(data["orders"]), "filled": sum(1 for o in data["orders"] if o["status"] == "FILLED"),
                   "rejected": sum(1 for o in data["orders"] if o["status"] == "REJECTED"),
                   "reject_reasons": _count_table([(o["reject_reason"] or "")[:100] for o in data["orders"] if o["status"] == "REJECTED"], "reason")},
        "risk": {"vetoes": _count_table([r["reason"] for r in data["risk"] if r["event_type"] == "VETO"], "reason", top=8),
                 "approved": sum(1 for r in data["risk"] if r["event_type"] == "APPROVED")},
        "health": _health(data, unresolved),
        "rejected_setups": _rejected_setups(data["decisions"]),
        "data_quality": _data_quality(data.get("data_quality", [])),
        "methodology": METHODOLOGY,
        "labels": LABELS,
        "caveats": CAVEATS,
    }
    rep["summary"]["today"] = {k: v for k, v in rep["summary"]["today"].items() if k not in ("avg_held",)}
    rep["summary"]["cumulative"] = {k: v for k, v in rep["summary"]["cumulative"].items() if k not in ("avg_held",)}
    rep["narrative"] = _narrative(rep)
    rep = jsonable(rep)
    return {"content_json": rep, "content_markdown": to_markdown(rep)}


# ------------------------------------------------------------------------------------------ markdown
def _md_table(rows: list[dict], columns: list[tuple[str, str]], fmt: dict | None = None) -> list[str]:
    if not rows:
        return ["_none_", ""]
    fmt = fmt or {}
    lines = ["| " + " | ".join(h for _, h in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for r in rows:
        cells = []
        for key, _ in columns:
            v = r.get(key)
            cells.append("" if v is None else (fmt[key](v) if key in fmt else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def to_markdown(rep: dict) -> str:
    s, f = rep["summary"], rep["funnel"]
    inr = lambda v: F.inr(v, signed=True)  # noqa: E731
    out = [f"# Daily report - {rep['date']} ({rep['scope']})", f"Generated {rep['generated_at'][:16].replace('T', ' ')} IST", "",
           "## How things went", *[f"- {line}" for line in rep["narrative"]], "",
           "## Today at a glance"]
    t, c = s["today"], s["cumulative"]
    out += _md_table([
        {"w": "Today", "n": t["trades"], "net": t["net"], "wr": t["win_rate"], "pf": t["profit_factor"], "ex": t["expectancy"], "dd": t["max_drawdown"]},
        {"w": "All time", "n": c["trades"], "net": c["net"], "wr": c["win_rate"], "pf": c["profit_factor"], "ex": c["expectancy"], "dd": c["max_drawdown"]},
    ], [("w", ""), ("n", "Trades"), ("net", "Net P&L"), ("wr", "Win rate"), ("pf", "Profit factor"), ("ex", "Expectancy"), ("dd", "Max drawdown")],
        {"net": inr, "wr": lambda v: f"{v:.0f}%", "pf": lambda v: f"{v:.2f}", "ex": inr, "dd": lambda v: F.inr(v)})
    out += ["## Strategy leaderboard (all time)", rep["leader"]["text"], ""]
    out += _md_table(rep["strategies"]["all_time"], [("strategy", "Strategy"), ("trades", "Trades"), ("win_rate", "Win rate"),
                                                     ("net_pnl", "Net P&L"), ("expectancy", "Per trade"), ("avg_r", "Mean R"),
                                                     ("profit_factor", "Profit factor"), ("verdict_text", "Verdict")],
                     {"win_rate": lambda v: f"{v:.0f}%", "net_pnl": inr, "expectancy": inr, "avg_r": lambda v: f"{v:+.2f}",
                      "profit_factor": lambda v: f"{v:.2f}"})
    out += ["## Signals vs trades", f"Bars evaluated: {f['bars_evaluated']}; signals: {f['signals']}; filled: {f['filled']}; "
            f"vetoed: {f['vetoed']}; rejected: {f['rejected']}.", ""]
    out += _md_table(f["no_trade_reasons"], [("reason", "Why the system stood aside"), ("count", "Times")])
    out += _md_table(f["by_strategy"], [("strategy", "Strategy"), ("signals", "Signals"), ("filled", "Filled"), ("vetoed", "Vetoed"), ("rejected", "Rejected")])
    out += ["## Realised vs backtest-expected R"]
    out += _md_table(rep["expected_vs_realised"], [("strategy", "Strategy"), ("trades", "Trades"), ("expected_r", "Expected R"),
                                                   ("realised_r", "Realised R"), ("gap", "Gap")],
                     {"expected_r": lambda v: f"{v:+.2f}", "realised_r": lambda v: f"{v:+.2f}", "gap": lambda v: f"{v:+.2f}"})
    for key, title in (("instrument", "By instrument"), ("exit_reason", "By exit reason"), ("tag_label", "TIE-BREAK vs normal"),
                       ("hour", "By entry hour"), ("option_type", "Calls vs puts")):
        out += [f"## {title}"]
        out += _md_table(rep["breakdowns"][key], [(key, key.replace("_", " ").title()), ("trades", "Trades"), ("win_rate", "Win rate"),
                                                  ("net_pnl", "Net P&L"), ("expectancy", "Per trade")],
                         {"win_rate": lambda v: f"{v:.0f}%", "net_pnl": inr, "expectancy": inr})
    h = rep["health"]
    out += ["## System and data health",
            f"- Token: {h['token']['state']}" + (f" (expires {h['token']['expires_at'][:16].replace('T', ' ')})" if h["token"]["expires_at"] else ""),
            f"- Log: {h['errors']} errors, {h['warnings']} warnings",
            f"- Orders today: {rep['orders']['total']} ({rep['orders']['filled']} filled, {rep['orders']['rejected']} rejected)",
            f"- Unresolved positions: {h['unresolved']['stale']} stale, {h['unresolved']['open']} open", ""]
    out += _md_table(rep["risk"]["vetoes"], [("reason", "Risk-engine vetoes"), ("count", "Times")])
    out += ["## Setups that did not become trades"]
    out += _md_table(rep.get("rejected_setups", [])[:40], [("bar", "Bar"), ("instrument", "Instrument"), ("strategy", "Strategy"),
                                                       ("status", "Outcome"), ("reason_class", "Reason"), ("reason", "Detail")])
    dq = rep.get("data_quality", {"loads": 0, "not_ok": 0, "items": []})
    out += ["## Data quality", f"{dq['loads']} data loads recorded; {dq['not_ok']} were not OK (those bars produce NO TRADE).", ""]
    out += _md_table(dq["items"], [("instrument", "Instrument"), ("status", "Status"), ("category", "Category"), ("issue", "Finding"), ("times", "Times")])
    out += ["## Methodology", *[f"- {m}" for m in rep.get("methodology", [])], "",
            "## What each number is", *[f"- {k.replace('_', ' ').capitalize()}: {v}" for k, v in rep.get("labels", {}).items()], ""]
    out += ["## Caveats", *[f"- {c}" for c in rep["caveats"]], ""]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ storage
def save_report(report_date: dt.date, scope: str, report: dict) -> None:
    """Insert or replace the stored report for (date, scope)."""
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import DailyReport, utcnow

    with get_session() as session:
        row = session.query(DailyReport).filter_by(report_date=report_date, scope=scope).first()
        if row is None:
            session.add(DailyReport(report_date=report_date, scope=scope, content_json=report["content_json"],
                                    content_markdown=report["content_markdown"]))
        else:
            row.content_json, row.content_markdown, row.generated_at = report["content_json"], report["content_markdown"], utcnow()


def is_complete(content_json) -> bool:
    """Whether a stored report has every summary field the current page reads. Stored reports outlive code versions, so
    one saved by an older version (or a minimal fixture) may lack fields added since; displaying it would crash the page."""
    try:
        import pandas as pd

        summary = content_json["summary"]
        needed = set(perf.summarise(pd.DataFrame())) - {"avg_held"}
        return needed <= set(summary["today"]) and needed <= set(summary["cumulative"])
    except Exception:
        return False


def load_report(report_date: dt.date, scope: str = "ALL") -> dict | None:
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import DailyReport

    with get_session() as session:
        row = session.query(DailyReport).filter_by(report_date=report_date, scope=scope).first()
        return None if row is None else {"content_json": row.content_json, "content_markdown": row.content_markdown,
                                         "generated_at": row.generated_at}


def list_reports(scope: str = "ALL") -> list[dict]:
    """Every stored day, newest first: date, trades, net P&L, win rate, cumulative net."""
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import DailyReport

    with get_session() as session:
        rows = session.query(DailyReport).filter_by(scope=scope).order_by(DailyReport.report_date.desc()).all()
        out = []
        for r in rows:
            j = r.content_json or {}
            t = (j.get("summary") or {}).get("today") or {}
            c = (j.get("summary") or {}).get("cumulative") or {}
            out.append({"date": r.report_date, "trades": t.get("trades", 0), "net_pnl": t.get("net", 0.0),
                        "win_rate": t.get("win_rate"), "cumulative": c.get("net", 0.0), "generated_at": r.generated_at})
        return out


def generate_and_save(report_date: dt.date, scope: str = "ALL") -> dict:
    report = build_report(report_date, scope)
    save_report(report_date, scope, report)
    return report
