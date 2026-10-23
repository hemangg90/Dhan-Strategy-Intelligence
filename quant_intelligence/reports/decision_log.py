"""Record what the system decided on every closed bar, so a report can show signals vs trades.

One row per (instrument, bar): the strongest thing that happened on that bar wins - a fill outranks a risk veto,
which outranks a triggered setup, which outranks "waiting", which outranks NO TRADE. Scans repeat every minute on
the same bar, so an in-memory memo skips the database unless a decision is new or has become more significant.
Logging never raises into a trading cycle.
"""
from __future__ import annotations

import datetime as dt

from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeutil import now_ist

PRIORITY = {
    "NO_TRADE": 1, "SKIPPED": 2, "WAITING": 3, "EXPIRED": 3, "NO_SETUP": 3, "CHAIN_ERROR": 3,
    "TRIGGERED": 4, "VETOED": 5, "REJECTED": 5, "FILLED": 6,
}
_seen: dict[tuple[str, str], int] = {}
_SEEN_LIMIT = 5000


def classify_no_trade(reason: str | None) -> str:
    """Short code for why the ranker stood aside."""
    r = (reason or "").lower()
    if "quality" in r:
        return "DATA_QUALITY"
    if "distinguishable" in r or "tied" in r or "gap" in r:
        return "TIED"
    if "threshold" in r or "eligible" in r or "edge" in r or "confidence" in r:
        return "NO_EDGE"
    return "OTHER"


def _veto_class(reason: str) -> str:
    r = reason.lower()
    for needle, code in (("daily loss", "DAILY_LOSS"), ("drawdown", "DRAWDOWN"), ("trades per day", "MAX_TRADES"),
                         ("risk per trade", "TRADE_RISK"), ("strategy exposure", "STRATEGY_EXPOSURE"),
                         ("portfolio exposure", "PORTFOLIO_EXPOSURE"), ("liquidity", "LIQUIDITY"),
                         ("kill switch", "KILL_SWITCH"), ("data quality", "DATA_QUALITY"), ("volatility", "VOL_SPIKE")):
        if needle in r:
            return code
    return "RISK_OTHER"


def classify_cycle(output, result) -> dict:
    """Turn a pipeline output + an auto-cycle result into one decision-log record."""
    ranking = output.ranking
    selected = next((s for s in getattr(ranking, "ranked", []) if s.strategy_name == ranking.selected_strategy), None)
    reason = (result.reason or "").strip()
    order = result.order
    if ranking.is_no_trade:
        status, code = "NO_TRADE", classify_no_trade(ranking.reason)
        reason = ranking.reason
    elif order is not None:
        status = "FILLED" if getattr(order, "status", "") == "FILLED" else "REJECTED"
        code = "FILLED" if status == "FILLED" else "ORDER_REJECTED"
    elif "RISK ENGINE VETO" in reason:
        status, code = "VETOED", _veto_class(reason)
    elif reason.startswith("Direction filtered"):
        status, code = "SKIPPED", "DIRECTION_FILTER"  # a setup fired against the heatmap mover's direction
    elif reason.startswith("LEVEL CHECK"):
        status, code = "SKIPPED", "NO_ROOM"  # a setup fired but the next support/resistance level leaves no room to run
    elif reason.startswith("SETUP VALIDATION"):
        # "SETUP VALIDATION [CODE]: ..." - the validation stage stopped the trade; the bracketed code is the reason class
        code = reason.split("[", 1)[1].split("]", 1)[0] if "[" in reason else "VALIDATION"
        status = "WAITING" if code in ("SETUP_NOT_ACTIVE", "SETUP_EXPIRED") else "SKIPPED"
    elif reason.startswith("RECONCILIATION"):
        status, code = "SKIPPED", "RECONCILIATION"
    elif reason.startswith("NO SIZE"):
        status, code = "TRIGGERED", "NO_SIZE"  # a setup fired but one lot is above the per-trade loss budget
    elif result.setup_status == "CHAIN_ERROR":
        status, code = "CHAIN_ERROR", "CHAIN_ERROR"
    elif result.setup_status and "TRIGGERED" in str(result.setup_status):
        status, code = "TRIGGERED", "NOT_FILLED"
    elif reason.startswith("Skipped"):
        status, code = "SKIPPED", "ALREADY_OPEN"
    elif result.setup_status and "WAIT" in str(result.setup_status):
        status, code = "WAITING", "WAITING_FOR_SETUP"
    else:
        status, code = "NO_SETUP", "NO_SETUP"
    return {
        "instrument": result.instrument, "market": profile_for(result.instrument).name,
        "bar_ts": output.timestamp.to_pydatetime() if hasattr(output.timestamp, "to_pydatetime") else output.timestamp,
        "strategy": ranking.selected_strategy, "status": status, "reason_class": code, "reason": reason[:500],
        "tie_break": bool(getattr(ranking, "tie_break", False)),
        "expected_r": getattr(selected, "expected_r", None), "confidence": getattr(selected, "confidence_label", None),
        "decision_id": result.risk_decision_id, "order_id": getattr(order, "order_id", None) if order is not None else None,
        "tag": result.tag,
    }


def record(decision: dict) -> bool:
    """Insert, or upgrade, the row for (instrument, bar). True if the database was written."""
    key = (decision["instrument"], str(decision["bar_ts"]))
    priority = PRIORITY.get(decision["status"], 1)
    if _seen.get(key, 0) >= priority:
        return False
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import ScanDecision

    with get_session() as session:
        row = session.query(ScanDecision).filter_by(instrument=decision["instrument"], bar_ts=decision["bar_ts"]).first()
        if row is None:
            session.add(ScanDecision(decided_at=now_ist(), **decision))
        elif PRIORITY.get(row.status, 1) < priority:
            for k, v in decision.items():
                setattr(row, k, v)
            row.decided_at = now_ist()
    if len(_seen) > _SEEN_LIMIT:
        _seen.clear()
    _seen[key] = priority
    return True


def log_cycle(output, result) -> None:
    """Record one cycle's decision. Never raises."""
    try:
        record(classify_cycle(output, result))
    except Exception as e:
        log_event("decision_log", f"Could not record the scan decision for {getattr(result, 'instrument', '?')}: {e}", level="WARNING")


def log_chain_error(instrument: str, output, detail: str) -> None:
    """The chain fetch failed before the cycle could run (multi_cycle)."""
    try:
        ranking = output.ranking
        record({
            "instrument": instrument, "market": profile_for(instrument).name,
            "bar_ts": output.timestamp.to_pydatetime() if hasattr(output.timestamp, "to_pydatetime") else output.timestamp,
            "strategy": ranking.selected_strategy, "status": "CHAIN_ERROR", "reason_class": "CHAIN_ERROR",
            "reason": str(detail)[:500], "tie_break": bool(getattr(ranking, "tie_break", False)),
            "expected_r": None, "confidence": None, "decision_id": None, "order_id": None, "tag": None,
        })
    except Exception:
        pass


def prune(days: int = 90) -> int:
    """Delete decisions older than `days`. Returns rows removed."""
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import ScanDecision

    cutoff = now_ist() - dt.timedelta(days=days)
    with get_session() as session:
        return session.query(ScanDecision).filter(ScanDecision.decided_at < cutoff).delete()
