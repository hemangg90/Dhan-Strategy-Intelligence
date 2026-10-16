"""System diagnostics: what is actually working, what is not, and what has not been tested.

Each check reports OK, WARN, FAIL or NOT_TESTED. NOT_TESTED means no check was made: it is never shown as healthy. Nothing
here places an order, and nothing returns a credential (token state and expiry only).
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from quant_intelligence.config.settings import DATA_CACHE_DIR, LOGS_DIR, SETTINGS
from quant_intelligence.utils.logging_utils import redact

OK, WARN, FAIL, NOT_TESTED = "OK", "WARN", "FAIL", "NOT_TESTED"

# (module, required?, why it matters)
DEPENDENCIES = [
    ("pandas", True, "all calculations"), ("numpy", True, "all calculations"), ("scipy", True, "statistics"),
    ("statsmodels", True, "statistics"), ("sklearn", True, "analytics"), ("sqlalchemy", True, "persistence"),
    ("pyarrow", True, "candle cache (Parquet)"), ("requests", True, "Dhan API"), ("streamlit", True, "the application"),
    ("plotly", True, "charts"), ("arch", False, "optional volatility models"), ("openpyxl", False, "Excel report export"),
    ("fastapi", False, "optional read-only website"), ("hypothesis", False, "property-based tests (development)"),
]


@dataclass
class Check:
    name: str
    status: str
    detail: str


def dependency_checks() -> list[Check]:
    checks = []
    for module, required, why in DEPENDENCIES:
        try:
            version = getattr(importlib.import_module(module), "__version__", "installed")
            checks.append(Check(f"Dependency: {module}", OK, f"{version} ({why})"))
        except Exception as exc:
            checks.append(Check(f"Dependency: {module}", FAIL if required else WARN,
                                f"missing ({type(exc).__name__}); needed for {why}" + ("" if required else " (optional)")))
    return checks


def execution_mode_check() -> Check:
    if SETTINGS.demo_lockdown:
        return Check("Execution mode", OK, "PAPER only: demo lockdown is on, so the live-order gate is closed")
    if SETTINGS.live_mode_fully_authorized:
        return Check("Execution mode", WARN, "LIVE ORDERS ARE AUTHORISED (TRADING_MODE=LIVE and the confirmation phrase are set)")
    if SETTINGS.is_live_mode:
        return Check("Execution mode", OK, "LIVE requested but not confirmed: live orders are blocked")
    return Check("Execution mode", OK, "PAPER: no real orders can be sent")


def credentials_check() -> Check:
    from quant_intelligence.config.credentials import token_status

    ts = token_status(SETTINGS.dhan_access_token)
    state = ts["state"]
    if state == "missing":
        return Check("Dhan credentials", FAIL, "no access token configured")
    if state == "expired":
        return Check("Dhan credentials", FAIL, f"access token expired {ts['expires_at']:%d %b %H:%M} IST")
    if state == "expiring":
        return Check("Dhan credentials", WARN, f"access token expires {ts['expires_at']:%d %b %H:%M} IST")
    if state == "unknown":
        return Check("Dhan credentials", WARN, "a token is set but its expiry cannot be read")
    return Check("Dhan credentials", OK, f"access token valid until {ts['expires_at']:%d %b %H:%M} IST")


def database_check() -> Check:
    try:
        from sqlalchemy import text

        from quant_intelligence.database.db import backend_name, get_session, is_durable

        with get_session() as session:
            session.execute(text("SELECT 1"))
        if not is_durable():
            return Check("Database", WARN, f"reachable ({backend_name()}), but local SQLite is not durable on ephemeral hosts")
        return Check("Database", OK, f"reachable ({backend_name()})")
    except Exception as exc:
        return Check("Database", FAIL, f"not reachable ({type(exc).__name__}): trade history cannot be saved")


def cache_check(cache_dir: Path | None = None) -> Check:
    import pandas as pd

    directory = Path(cache_dir) if cache_dir else DATA_CACHE_DIR / "parquet_cache"
    files = sorted(directory.glob("*.parquet")) if directory.exists() else []
    if not files:
        return Check("Candle cache", NOT_TESTED, "no cached candle files yet")
    unreadable = []
    for f in files:
        try:
            pd.read_parquet(f, columns=["timestamp"])
        except Exception:
            unreadable.append(f.name)
    if unreadable:
        shown = ", ".join(unreadable[:3]) + ("..." if len(unreadable) > 3 else "")
        return Check("Candle cache", FAIL, f"{len(unreadable)} of {len(files)} file(s) unreadable: {shown}")
    return Check("Candle cache", OK, f"{len(files)} file(s), all readable")


def data_freshness_check(output=None, max_age_minutes: float = 15.0) -> Check:
    """Freshness of the last computed pipeline result. NOT_TESTED if none has been computed in this process."""
    if output is None:
        return Check("Market data", NOT_TESTED, "the research pipeline has not run yet in this session")
    from quant_intelligence.utils.timeutil import now_ist

    age = (now_ist().replace(tzinfo=None) - output.timestamp.replace(tzinfo=None)).total_seconds() / 60
    detail = f"{output.instrument}: last bar {output.timestamp:%d %b %H:%M} ({age:.0f} min old), quality {output.data_quality_status}"
    if output.data_quality_status != "OK":
        issues = "; ".join(str(i) for i in (output.data_quality_issues or [])[:2])
        return Check("Market data", FAIL, detail + (f" - {issues}" if issues else ""))
    return Check("Market data", OK if age <= max_age_minutes else WARN,
                 detail + ("" if age <= max_age_minutes else " (older than the order freshness limit; normal outside market hours)"))


def unresolved_order_check(live_broker=None) -> Check:
    """Whether a live order is left in an unknown state. NOT_TESTED if no live broker exists in this session."""
    if live_broker is None:
        return Check("Live order state", NOT_TESTED, "the live broker has not been opened in this session")
    pending = getattr(live_broker, "unreconciled_order", None)
    if pending:
        return Check("Live order state", FAIL, f"an order ({pending}) has an UNKNOWN outcome: check Dhan's order book; new live orders are blocked")
    return Check("Live order state", OK, "no unresolved live order")


def api_check(run: bool = False) -> Check:
    """A real round trip to Dhan, only when asked: otherwise NOT_TESTED (never assumed healthy)."""
    if not run:
        return Check("Dhan API", NOT_TESTED, "no connection test has been run: press the test button")
    from quant_intelligence.ui.credentials_panel import test_connection

    ok, message = test_connection()
    return Check("Dhan API", OK if ok else FAIL, redact(message))


def recent_errors(limit: int = 10, log_file: Path | None = None, max_bytes: int = 400_000) -> list[dict]:
    """The latest WARNING / ERROR lines of the application log, with credential-looking text removed."""
    path = Path(log_file) if log_file else Path(LOGS_DIR) / "system.log"
    if not path.exists():
        return []
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(fh.tell() - max_bytes, 0))
        lines = fh.read().decode("utf-8", errors="replace").splitlines()
    found = []
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("level") in ("ERROR", "WARNING", "CRITICAL"):
            msg = record.get("msg")
            message = msg.get("message") if isinstance(msg, dict) else str(msg)
            found.append({"time": record.get("ts"), "level": record["level"], "component": record.get("component", ""),
                          "message": redact(str(message))[:300]})
            if len(found) >= limit:
                break
    return found


def collect(output=None, live_broker=None, test_api: bool = False) -> list[Check]:
    return [
        execution_mode_check(), credentials_check(), api_check(run=test_api), data_freshness_check(output),
        cache_check(), database_check(), unresolved_order_check(live_broker), *dependency_checks(),
    ]
