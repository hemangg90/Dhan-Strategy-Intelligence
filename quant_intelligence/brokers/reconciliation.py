"""
Position reconciliation: does the book the application believes in match the book it can be checked against?

Two uses, one comparison (`reconcile_books`, quantities by instrument identifier, signed: long +, short -):
  * PAPER   the engine's in-memory open positions against the open rows in the database. The paper broker saves every
            position, but a failed save is logged and swallowed, so memory and database can drift apart unnoticed (and a restart
            would then rebuild a different book). `reconcile_paper` makes that visible.
  * LIVE    an expected book (from the orders this session sent and saw filled) against the broker's reported positions
            (`/positions`: `securityId`, `netQty`). A broker that cannot be reached is CRITICAL - never read as "flat".
            The broker's order book is not available through the application's client, so order-level reconciliation
            (acknowledged vs filled vs rejected for each order) is NOT done here; the live broker's lock on an order with an
            unknown outcome (`unreconciled_order`) remains the guard for that.
A report is OK only when both sides were read and every quantity agrees; any difference lists exactly what disagrees.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from quant_intelligence.intelligence.status import DataStatus, critical, good
from quant_intelligence.utils.timeutil import now_ist


@dataclass
class ReconciliationReport:
    ok: bool
    mode: str
    checked_at: dt.datetime
    local_count: int = 0
    remote_count: int = 0
    missing_remotely: list[dict] = field(default_factory=list)  # we think it is open; the other side does not have it
    unexpected_remotely: list[dict] = field(default_factory=list)  # the other side has it; we do not
    quantity_mismatches: list[dict] = field(default_factory=list)
    reason: str = ""
    status: DataStatus = field(default_factory=lambda: good("reconciliation"))

    @property
    def summary(self) -> str:
        if self.ok:
            return f"{self.mode}: books agree ({self.local_count} open position(s))"
        return f"{self.mode}: {self.reason}"


def reconcile_books(local: dict[str, float], remote: dict[str, float], mode: str, tolerance: float = 0.0,
                    now: dt.datetime | None = None) -> ReconciliationReport:
    now = now or now_ist()
    local = {k: v for k, v in local.items() if v}
    remote = {k: v for k, v in remote.items() if v}
    missing = [{"id": k, "local_qty": v} for k, v in sorted(local.items()) if k not in remote]
    unexpected = [{"id": k, "remote_qty": v} for k, v in sorted(remote.items()) if k not in local]
    mismatched = [{"id": k, "local_qty": local[k], "remote_qty": remote[k]} for k in sorted(local)
                  if k in remote and abs(local[k] - remote[k]) > tolerance]
    problems = []
    if missing:
        problems.append(f"{len(missing)} position(s) we hold are not on the other side ({', '.join(m['id'] for m in missing[:4])})")
    if unexpected:
        problems.append(f"{len(unexpected)} position(s) exist on the other side that we do not hold ({', '.join(u['id'] for u in unexpected[:4])})")
    if mismatched:
        problems.append(f"{len(mismatched)} quantity mismatch(es) ({', '.join(m['id'] for m in mismatched[:4])})")
    ok = not problems
    reason = "; ".join(problems)
    return ReconciliationReport(ok, mode, now, len(local), len(remote), missing, unexpected, mismatched, reason,
                                good("reconciliation") if ok else critical("reconciliation", reason))


def local_book(positions: list[dict]) -> dict[str, float]:
    """Signed open quantity by security id (BUY/LONG +, SELL/SHORT -) of position dicts; positions with no security id
    (direct equity paper positions) are keyed by instrument."""
    book: dict[str, float] = {}
    for p in positions:
        if p.get("status", "OPEN") != "OPEN":
            continue
        key = str(p.get("security_id") or p.get("instrument"))
        side = p.get("transaction") or p.get("direction")
        sign = 1.0 if side in ("BUY", "LONG") else -1.0
        book[key] = book.get(key, 0.0) + sign * float(p.get("quantity") or 0)
    return book


def dhan_book(raw_positions: list[dict]) -> dict[str, float]:
    """Net quantity by `securityId` from the broker's `/positions` rows (`netQty`, signed)."""
    book: dict[str, float] = {}
    for p in raw_positions or []:
        key = str(p.get("securityId"))
        book[key] = book.get(key, 0.0) + float(p.get("netQty") or 0)
    return book


def reconcile_paper(broker, now: dt.datetime | None = None) -> ReconciliationReport:
    """The paper broker's open positions against the open rows in the database."""
    now = now or now_ist()
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import Position

        with get_session() as session:
            rows = [{"position_id": r.position_id, "security_id": r.security_id, "instrument": r.instrument, "status": r.status,
                     "transaction": r.transaction, "direction": r.direction, "quantity": r.quantity}
                    for r in session.query(Position).filter(Position.status == "OPEN").all()]
        # Compare by position id so two positions on the same security cannot hide a missing one.
        mine = {p["position_id"]: p for p in broker.get_open_positions()}
        theirs = {r["position_id"]: r for r in rows}
        local = {pid: local_book([p]).get(str(p.get("security_id") or p.get("instrument")), 0.0) for pid, p in mine.items()}
        remote = {pid: local_book([r]).get(str(r.get("security_id") or r.get("instrument")), 0.0) for pid, r in theirs.items()}
        return reconcile_books(local, remote, "PAPER (memory vs database)", now=now)
    except Exception as e:
        return ReconciliationReport(False, "PAPER (memory vs database)", now, reason=f"the database could not be read: {e}",
                                    status=critical("reconciliation", f"the database could not be read: {e}"))


def reconcile_live(client, expected_book: dict[str, float], now: dt.datetime | None = None) -> ReconciliationReport:
    """The expected book against the broker's reported positions. An unreachable broker is a failed reconciliation."""
    now = now or now_ist()
    try:
        raw = client.get_positions()
    except Exception as e:
        reason = f"the broker's positions could not be read ({e}); reconciliation failed, not 'flat'"
        return ReconciliationReport(False, "LIVE (expected vs broker)", now, reason=reason, status=critical("reconciliation", reason))
    return reconcile_books(expected_book, dhan_book(raw), "LIVE (expected vs broker)", now=now)
