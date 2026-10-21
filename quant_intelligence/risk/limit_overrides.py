"""Editable risk limits: validated, bounded, persisted and audited.

The limits live in the frozen `SETTINGS.risk` (read from the environment at import). Every read site looks at
`SETTINGS.risk` on each call, so changing a field in place takes effect on the next evaluation - the same way
`config/credentials.py` applies a new token. Rules:

* Only the fields in `EDITABLE` can change, and only within their hard range (a ceiling the app cannot exceed).
* Tightening applies at once. Loosening needs an explicit confirmation and a reason, and is refused while the kill
  switch is engaged. Nothing can be changed under DEMO_LOCKDOWN.
* A change is stored (table `risk_limit_overrides`) and audited (a `LIMIT_CHANGE` risk event) in ONE transaction. If
  that fails, nothing is applied (fail closed).
* Saved values are reloaded at start by `load_persisted()` (never at import, so tests keep their own environment).
* Already-open positions and already-issued approvals are not re-judged; new values apply to new evaluations.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.logging_utils import log_event

LOOSER, TIGHTER = "LOOSER", "TIGHTER"
MIN_REASON_CHARS = 5


@dataclass(frozen=True)
class Spec:
    field: str
    label: str
    kind: str  # "float" | "int"
    lo: float  # smallest accepted non-zero value
    hi: float  # hard ceiling
    off_allowed: bool  # 0 switches the check off (the code already treats 0 that way)
    looser: str  # "higher": a larger number is a looser limit; "lower": a smaller number is
    unit: str = ""
    help: str = ""


EDITABLE: tuple[Spec, ...] = (
    Spec("max_risk_per_trade_pct", "Max risk per trade", "float", 0.05, 2.0, False, "higher", "% of equity",
         "What one stopped trade may cost. Sizing divides this by the stop distance."),
    Spec("max_capital_per_trade_pct", "Max capital per trade", "float", 0.5, 25.0, True, "higher", "% of equity",
         "Cap on premium x quantity in one trade. 0 switches the cap off."),
    Spec("max_daily_loss_pct", "Max daily loss", "float", 0.25, 5.0, False, "higher", "% of equity",
         "New entries stop once the day has lost this much."),
    Spec("max_drawdown_pct", "Max drawdown", "float", 1.0, 20.0, False, "higher", "% from peak",
         "New entries stop at this fall from peak equity."),
    Spec("max_trades_per_day", "Max trades per day", "int", 1, 30, False, "higher", "trades", ""),
    Spec("max_open_positions", "Max open positions", "int", 1, 15, True, "higher", "positions", "0 switches the cap off."),
    Spec("max_strategy_exposure_pct", "Strategy exposure", "float", 1.0, 50.0, False, "higher", "% of equity",
         "Open risk in one strategy."),
    Spec("max_portfolio_exposure_pct", "Portfolio exposure", "float", 1.0, 100.0, False, "higher", "% of equity",
         "Open risk in total. Must be at least the strategy exposure."),
    Spec("loss_streak_pause", "Loss-streak pause after", "int", 1, 10, True, "higher", "losses",
         "No new entries after this many consecutive losers. 0 switches the pause off."),
    Spec("loss_streak_pause_min", "Loss-streak pause length", "int", 5, 480, False, "lower", "minutes",
         "How long entries stay paused."),
)
BY_FIELD = {s.field: s for s in EDITABLE}

# What the environment/defaults gave at start: the target of "reset to defaults".
BASELINE: dict[str, float] = {s.field: getattr(SETTINGS.risk, s.field) for s in EDITABLE}


def current() -> dict[str, float]:
    return {s.field: getattr(SETTINGS.risk, s.field) for s in EDITABLE}


def _number(spec: Spec, raw) -> tuple[float | int | None, str | None]:
    if isinstance(raw, bool) or raw is None:
        return None, f"{spec.label}: not a number"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"{spec.label}: not a number"
    if not math.isfinite(value):
        return None, f"{spec.label}: not a finite number"
    if spec.kind == "int":
        if value != int(value):
            return None, f"{spec.label}: must be a whole number"
        value = int(value)
    if value == 0 and spec.off_allowed:
        return value, None
    if not spec.lo <= value <= spec.hi:
        off = " (or 0 to switch it off)" if spec.off_allowed else ""
        return None, f"{spec.label}: must be between {spec.lo:g} and {spec.hi:g}{off}"
    return value, None


def validate(changes: dict) -> tuple[dict, list[str]]:
    """Clean values for the fields in `changes`, plus the problems found. Nothing is applied here."""
    clean: dict = {}
    errors: list[str] = []
    for field, raw in changes.items():
        spec = BY_FIELD.get(field)
        if spec is None:
            errors.append(f"{field}: this limit cannot be changed from the app")
            continue
        value, problem = _number(spec, raw)
        if problem:
            errors.append(problem)
        else:
            clean[field] = value
    merged = {**current(), **clean}
    if merged["max_portfolio_exposure_pct"] < merged["max_strategy_exposure_pct"]:
        errors.append("Portfolio exposure must be at least the strategy exposure")
    return clean, errors


def _effective(spec: Spec, value: float) -> float:
    """A comparable 'looseness' score: higher is looser. 0 on a switch-off limit is the loosest of all."""
    if value == 0 and spec.off_allowed:
        return math.inf
    return value if spec.looser == "higher" else -value


def direction(field: str, old: float, new: float) -> str | None:
    spec = BY_FIELD[field]
    a, b = _effective(spec, old), _effective(spec, new)
    if a == b:
        return None
    return LOOSER if b > a else TIGHTER


@dataclass
class Result:
    ok: bool
    applied: dict  # field -> {"old", "new", "direction"}
    errors: list[str]


def _store(rows: dict, decision_id: str, reason: str, details: dict, clear: bool) -> None:
    """Upsert the override rows and write the audit event in one transaction."""
    from quant_intelligence.database.db import get_session, init_db
    from quant_intelligence.database.models import RiskEvent, RiskLimitOverride, utcnow

    init_db()
    with get_session() as session:
        if clear:
            session.query(RiskLimitOverride).delete()
        else:
            for field, value in rows.items():
                row = session.get(RiskLimitOverride, field)
                if row is None:
                    session.add(RiskLimitOverride(field=field, value=float(value), reason=reason))
                else:
                    row.value, row.reason = float(value), reason
                    row.updated_at = utcnow()
        session.add(RiskEvent(decision_id=decision_id, event_type="LIMIT_CHANGE", reason=reason, details=details))


def apply(changes: dict, *, reason: str = "", confirm_loosen: bool = False, kill_switch: bool = False,
          _clear: bool = False) -> Result:
    """Validate, persist, audit, then apply `changes`. Returns what was applied, or why nothing was."""
    if SETTINGS.demo_lockdown:
        return Result(False, {}, ["Risk limits cannot be changed under demo lockdown"])
    clean, errors = validate(changes)
    if errors:
        return Result(False, {}, errors)
    now = current()
    diff = {f: v for f, v in clean.items() if v != now[f]}
    moves = {f: {"old": now[f], "new": v, "direction": direction(f, now[f], v)} for f, v in diff.items()}
    reason = (reason or "").strip()
    if any(m["direction"] == LOOSER for m in moves.values()):
        if kill_switch:
            return Result(False, {}, ["The kill switch is engaged: limits cannot be loosened until it is released"])
        if not confirm_loosen:
            return Result(False, {}, ["Loosening a limit needs the confirmation tick"])
        if len(reason) < MIN_REASON_CHARS:
            return Result(False, {}, ["Loosening a limit needs a reason (a few words)"])
    if not moves and not _clear:
        return Result(True, {}, [])
    decision_id = f"LIMIT-{uuid.uuid4().hex[:12]}"
    summary = ", ".join(f"{f} {m['old']:g} -> {m['new']:g} ({m['direction']})" for f, m in moves.items()) or "reset to defaults"
    details = {"changes": moves, "reason": reason, "reset": _clear}
    try:
        _store(diff if not _clear else {}, decision_id, reason or summary, details, _clear)
    except Exception as exc:  # fail closed: an unrecorded change is not made
        log_event("risk_engine", f"Risk limit change NOT applied (could not be recorded): {type(exc).__name__}", level="ERROR")
        return Result(False, {}, [f"The change could not be recorded, so it was not applied ({type(exc).__name__})"])
    for field, value in diff.items():
        object.__setattr__(SETTINGS.risk, field, value)
    log_event("risk_engine", f"Risk limits changed: {summary}", level="WARNING" if any(
        m["direction"] == LOOSER for m in moves.values()) else "INFO", decision_id=decision_id, reason=reason)
    return Result(True, moves, [])


def reset_to_defaults(*, reason: str = "", confirm_loosen: bool = False, kill_switch: bool = False) -> Result:
    """Back to the environment/default values and forget the saved overrides. Loosening rules still apply."""
    return apply(dict(BASELINE), reason=reason, confirm_loosen=confirm_loosen, kill_switch=kill_switch, _clear=True)


def load_persisted() -> dict:
    """Apply the saved overrides at start. Invalid or out-of-range rows are skipped and logged. Returns what was applied."""
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import RiskLimitOverride

        with get_session() as session:
            saved = {r.field: r.value for r in session.query(RiskLimitOverride).all()}
    except Exception as exc:
        log_event("risk_engine", f"Saved risk limits could not be read: {type(exc).__name__}", level="WARNING")
        return {}
    applied: dict = {}
    for field, raw in saved.items():
        clean, errors = validate({field: raw})
        if errors or field not in clean:
            log_event("risk_engine", f"Saved risk limit ignored: {'; '.join(errors) or field}", level="WARNING")
            continue
        merged = {**current(), **clean}
        if merged["max_portfolio_exposure_pct"] < merged["max_strategy_exposure_pct"]:
            log_event("risk_engine", f"Saved risk limit {field} ignored: portfolio exposure below strategy exposure", level="WARNING")
            continue
        object.__setattr__(SETTINGS.risk, field, clean[field])
        applied[field] = clean[field]
    if applied:
        log_event("risk_engine", f"Loaded saved risk limits: {applied}", level="INFO")
    return applied


def rows() -> list[dict]:
    """One row per editable limit, for the UI."""
    now = current()
    return [
        {"field": s.field, "label": s.label, "unit": s.unit, "help": s.help, "kind": s.kind, "lo": s.lo, "hi": s.hi,
         "off_allowed": s.off_allowed, "current": now[s.field], "default": BASELINE[s.field],
         "changed": now[s.field] != BASELINE[s.field]}
        for s in EDITABLE
    ]


def recent_changes(limit: int = 30) -> list[dict]:
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import RiskEvent

    with get_session() as session:
        events = (session.query(RiskEvent).filter(RiskEvent.event_type == "LIMIT_CHANGE")
                  .order_by(RiskEvent.id.desc()).limit(limit).all())
    out = []
    for e in events:
        moves = (e.details or {}).get("changes", {})
        out.append({
            "When (UTC)": e.timestamp,
            "Change": ", ".join(f"{BY_FIELD[f].label if f in BY_FIELD else f}: {m['old']:g} -> {m['new']:g} ({m['direction']})"
                                for f, m in moves.items()) or "reset to defaults",
            "Reason": (e.details or {}).get("reason") or e.reason,
        })
    return out
