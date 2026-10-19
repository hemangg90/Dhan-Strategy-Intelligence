"""Follow a paper trade from the decision that produced it to its outcome, and list what was decided but not traded.

The tables are linked by `decision_id` (and `order_id`), not by foreign keys, so the chain is joined here. Missing links are
reported, never hidden: a trade older than the decision log has no scan-decision row, and an approval that never became an
order is an orphan the user should see.

Times: scan decisions, orders, fills and positions are stored in naive IST; risk events are stored in UTC and are converted
to IST here so the timeline is in one clock.
"""
from __future__ import annotations

import datetime as dt

IST_OFFSET = dt.timedelta(hours=5, minutes=30)
REJECTED_STATUSES = ("NO_TRADE", "SKIPPED", "WAITING", "NO_SETUP", "CHAIN_ERROR", "TRIGGERED", "VETOED", "REJECTED", "EXPIRED")


def _risk_amount(p) -> float | None:
    try:
        r = abs(float(p.entry_price) - float(p.stop_price)) * float(p.quantity)
        return r if r > 0 else None
    except (TypeError, ValueError):
        return None


def realised_r(position) -> tuple[float | None, str | None]:
    """(R multiple, basis). Rows with booked `costs` carry a net P&L; legacy rows hold gross only and say so."""
    risk = _risk_amount(position)
    if risk is None or position.status != "CLOSED":
        return None, None
    if position.costs is not None and position.net_pnl is not None:
        return float(position.net_pnl) / risk, "net"
    if position.net_pnl is not None:
        return float(position.net_pnl) / risk, "gross (legacy row, charges not booked)"
    return None, None


def _event(time, stage: str, detail: str) -> dict:
    return {"time": time, "stage": stage, "detail": detail}


def _timeline(position, decision, risk, order, fills) -> list[dict]:
    events = []
    if decision is not None:
        events.append(_event(decision.bar_ts, "Signal bar", f"{decision.strategy or 'strategy'} on the {decision.bar_ts:%d %b %H:%M} bar: "
                                                         f"{decision.status} ({decision.reason_class})"))
        events.append(_event(decision.decided_at, "Decision recorded", decision.reason or decision.status))
    if risk is not None:
        passed = [k for k, v in (risk.details or {}).items() if v is True]
        events.append(_event(risk.timestamp + IST_OFFSET, f"Risk engine: {risk.event_type}",
                             f"{risk.reason}. Checks passed: {len(passed)}"))
    if order is not None:
        events.append(_event(order.timestamp, "Order (simulated)",
                             f"{order.direction} {order.quantity} {order.instrument}, {order.order_type} at {order.price}, status {order.status}"
                             + (f" ({order.reject_reason})" if order.reject_reason else "")))
    for f in fills:
        events.append(_event(f.timestamp, "Fill assumption", f"filled at {f.fill_price} for {f.quantity}; modelled slippage {f.slippage}"))
    if position.closed_at is not None:
        events.append(_event(position.closed_at, f"Exit: {position.exit_reason or 'closed'}", f"exit price {position.exit_price}"))
    return sorted(events, key=lambda e: (e["time"] is None, e["time"]))


def trade_trace(session, limit: int = 100, instrument: str | None = None, since: dt.datetime | None = None) -> list[dict]:
    """One dict per position (newest first) with the joined records, a timeline, expected vs realised R and the missing links."""
    from quant_intelligence.database.models import Fill, Order, Position, RiskEvent, ScanDecision

    q = session.query(Position)
    if instrument:
        q = q.filter(Position.instrument == instrument)
    if since is not None:
        q = q.filter(Position.opened_at >= since)
    positions = q.order_by(Position.opened_at.desc()).limit(limit).all()
    traces = []
    for p in positions:
        order = session.query(Order).filter_by(order_id=p.order_id).first() if p.order_id else None
        fills = session.query(Fill).filter_by(order_id=p.order_id).order_by(Fill.timestamp).all() if p.order_id else []
        risk = (session.query(RiskEvent).filter_by(decision_id=p.decision_id, event_type="APPROVED").order_by(RiskEvent.id).first()
                if p.decision_id else None)
        decision = session.query(ScanDecision).filter_by(decision_id=p.decision_id).first() if p.decision_id else None
        r, basis = realised_r(p)
        missing = [name for name, found in (("scan decision", decision), ("risk approval", risk), ("order", order), ("fill", fills)) if not found]
        gross = p.gross_pnl if p.costs is not None else p.net_pnl
        traces.append({
            "position_id": p.position_id, "decision_id": p.decision_id, "order_id": p.order_id,
            "instrument": p.instrument, "strategy": p.strategy_name, "direction": p.direction, "status": p.status,
            "opened_at": p.opened_at, "closed_at": p.closed_at, "exit_reason": p.exit_reason,
            "entry": p.entry_price, "stop": p.stop_price, "target": p.target_price, "exit": p.exit_price,
            "quantity": p.quantity, "regime_at_entry": p.regime, "confidence_at_entry": p.confidence,
            "expected_r": p.expected_r, "realised_r": r, "r_basis": basis,
            "r_difference": None if r is None or p.expected_r is None else r - p.expected_r,
            "gross_pnl": gross, "costs": p.costs, "net_pnl": p.net_pnl if p.costs is not None else None,
            "mode": p.mode, "label": "SIMULATED" if (p.mode or "PAPER") == "PAPER" else (p.mode or ""),
            "timeline": _timeline(p, decision, risk, order, fills), "missing_links": missing,
        })
    return traces


def rejected_decisions(session, limit: int = 300, reason_class: str | None = None, since: dt.datetime | None = None) -> list[dict]:
    """Bars where the system looked and did NOT trade (newest first): stood aside, waited, vetoed, rejected."""
    from quant_intelligence.database.models import ScanDecision

    q = session.query(ScanDecision).filter(ScanDecision.status.in_(REJECTED_STATUSES))
    if reason_class:
        q = q.filter(ScanDecision.reason_class == reason_class)
    if since is not None:
        q = q.filter(ScanDecision.decided_at >= since)
    return [{"bar": d.bar_ts, "decided_at": d.decided_at, "instrument": d.instrument, "strategy": d.strategy, "status": d.status,
             "reason_class": d.reason_class, "reason": d.reason, "expected_r": d.expected_r, "confidence": d.confidence,
             "decision_id": d.decision_id, "tie_break": bool(d.tie_break)}
            for d in q.order_by(ScanDecision.decided_at.desc()).limit(limit).all()]


def orphans(session, limit: int = 200) -> dict[str, list[dict]]:
    """Records whose chain stops early: an approval or a filled order with no position, a position with no approval."""
    from quant_intelligence.database.models import Order, Position, RiskEvent

    with_position = {x for (x,) in session.query(Position.decision_id).filter(Position.decision_id.isnot(None)).all()}
    order_with_position = {x for (x,) in session.query(Position.order_id).filter(Position.order_id.isnot(None)).all()}
    approvals = session.query(RiskEvent).filter_by(event_type="APPROVED").order_by(RiskEvent.id.desc()).limit(limit).all()
    filled = session.query(Order).filter(Order.status == "FILLED").order_by(Order.id.desc()).limit(limit).all()
    approved_without_position = [
        {"decision_id": a.decision_id, "time": a.timestamp + IST_OFFSET, "reason": a.reason}
        for a in approvals if a.decision_id not in with_position]
    filled_without_position = [
        {"order_id": o.order_id, "decision_id": o.decision_id, "time": o.timestamp, "instrument": o.instrument}
        for o in filled if o.order_id not in order_with_position]
    recent_positions = session.query(Position).order_by(Position.id.desc()).limit(limit).all()
    ids = [p.decision_id for p in recent_positions if p.decision_id]
    approved_ids = {x for (x,) in session.query(RiskEvent.decision_id).filter(
        RiskEvent.event_type == "APPROVED", RiskEvent.decision_id.in_(ids)).all()} if ids else set()
    positions_without_approval = [
        {"position_id": p.position_id, "decision_id": p.decision_id, "instrument": p.instrument, "opened_at": p.opened_at}
        for p in recent_positions if p.decision_id not in approved_ids]
    return {"approved_without_position": approved_without_position, "filled_order_without_position": filled_without_position,
            "position_without_approval": positions_without_approval}
