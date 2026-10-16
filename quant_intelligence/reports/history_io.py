"""Restore trade history from an earlier CSV download (Positions & Orders -> Trade history -> Download).

Use it after a reboot wiped a local database, or to load trades into a new durable (Postgres) database. Rows are
matched on `position_id`, so importing the same file twice - or a file that overlaps what is already stored -
never duplicates a trade. A position that was still OPEN in the file is imported as STALE: it cannot be resumed,
so it is kept for the record and excluded from statistics, like any position left open by a restart.
"""
from __future__ import annotations

import math

import pandas as pd

REQUIRED = ("position_id", "instrument", "status")
VALID_STATUS = {"OPEN", "CLOSED", "STALE"}


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    if hasattr(value, "item") and not isinstance(value, (pd.Timestamp, str)):
        return value.item()
    return value


def _ts(value):
    v = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(v) else v.to_pydatetime()


def _booked_pnl(r: dict):
    """The GROSS booked premium P&L of an exported row (the `gross_pnl` column, or net + charges)."""
    for key in ("gross_pnl",):
        if _clean(r.get(key)) is not None:
            return float(r[key])
    net = _clean(r.get("net_pnl"))
    if net is None:
        return None
    return float(net) + float(_clean(r.get("charges")) or 0.0)


def _pnl_fields(r: dict) -> dict:
    """Position P&L columns for an imported row, restored exactly as they were stored. Only when the file says the charges
    were BOOKED (`charges_booked`) is the row stored in the current convention (gross, costs, net = gross - costs);
    otherwise it is a legacy row: `net_pnl` holds the gross and `costs` stays NULL (the charges in the file were only
    estimates). Files written before this column existed therefore restore as legacy rows, which is what they were."""
    gross = _booked_pnl(r)
    charges = _clean(r.get("charges"))
    booked = str(_clean(r.get("charges_booked"))).strip().lower() in ("true", "1", "1.0")
    if gross is not None and charges is not None and booked:
        return {"gross_pnl": gross, "costs": float(charges), "net_pnl": gross - float(charges), "costs_model": "IMPORTED"}
    return {"gross_pnl": None, "costs": None, "net_pnl": gross, "costs_model": None}


def rows_from_frame(df: pd.DataFrame) -> tuple[list[dict], list[tuple[int, str]]]:
    """Validate and convert a downloaded positions CSV into Position column dicts. (rows, [(line, problem)])."""
    problems: list[tuple[int, str]] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [], [(0, f"missing column(s): {', '.join(missing)} - is this a Trade history download?")]
    out = []
    for i, raw in enumerate(df.to_dict("records"), start=2):  # line 1 is the header
        r = {k: _clean(v) for k, v in raw.items()}
        pid, instrument = r.get("position_id"), r.get("instrument")
        status = str(r.get("status") or "").upper()
        if not pid or not instrument:
            problems.append((i, "position_id and instrument are required"))
            continue
        if status not in VALID_STATUS:
            problems.append((i, f"unknown status {status!r}"))
            continue
        option_type = r.get("option_type")
        entry = r.get("entry", r.get("entry_price"))
        qty = r.get("qty", r.get("quantity"))
        out.append({
            "position_id": str(pid), "instrument": str(instrument), "underlying": str(instrument) if option_type else None,
            "strategy_name": r.get("strategy", r.get("strategy_name")),
            "direction": "LONG" if option_type == "CE" else ("SHORT" if option_type == "PE" else r.get("direction") or "LONG"),
            "quantity": int(qty) if qty is not None else None, "entry_price": float(entry) if entry is not None else None,
            "stop_price": r.get("stop", r.get("stop_price")), "target_price": r.get("target", r.get("target_price")),
            "status": "STALE" if status == "OPEN" else status,
            "opened_at": _ts(r.get("opened_at")), "closed_at": _ts(r.get("closed_at")),
            "exit_price": r.get("exit_price"), **_pnl_fields(r),
            "exit_reason": "STALE_ON_RESTART" if status == "OPEN" else r.get("exit_reason"),
            "mode": "PAPER", "security_id": None if r.get("security_id") is None else str(r.get("security_id")),
            "option_type": option_type, "strike": r.get("strike"), "expiry": None if r.get("expiry") is None else str(r.get("expiry")),
            "transaction": r.get("side", r.get("transaction")) if option_type else None, "tag": r.get("tag"),
            "order_id": r.get("order_id"), "decision_id": r.get("decision_id"), "expected_r": r.get("expected_r"),
            "confidence": r.get("confidence"), "regime": r.get("regime"),
        })
    return out, problems


def import_positions(df: pd.DataFrame) -> dict:
    """Insert the positions of a downloaded CSV that are not stored yet. Returns
    {"inserted": n, "skipped_existing": n, "invalid": [(line, problem), ...]}."""
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import Position

    rows, problems = rows_from_frame(df)
    inserted = skipped = 0
    with get_session() as session:
        ids = [r["position_id"] for r in rows]
        existing: set[str] = set()
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            existing.update(p for (p,) in session.query(Position.position_id).filter(Position.position_id.in_(chunk)).all())
        seen: set[str] = set()
        for r in rows:
            if r["position_id"] in existing or r["position_id"] in seen:
                skipped += 1
                continue
            seen.add(r["position_id"])
            session.add(Position(**r))
            inserted += 1
    return {"inserted": inserted, "skipped_existing": skipped, "invalid": problems}
