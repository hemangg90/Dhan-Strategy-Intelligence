"""Pre-demo checklist. Run from the project root:  python scripts/demo_preflight.py

Prints PASS / FAIL per check and exits non-zero if any mandatory check fails. It makes no orders and never
prints credentials.
"""
import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEMO_LOCKDOWN", "1")  # the demo is always run locked down

import pandas as pd  # noqa: E402

from quant_intelligence.config.credentials import token_status  # noqa: E402
from quant_intelligence.config.settings import DATA_CACHE_DIR, SETTINGS  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "", mandatory: bool = True) -> None:
    print(f"[{'PASS' if ok else ('FAIL' if mandatory else 'WARN')}] {name}" + (f" - {detail}" if detail else ""))
    if not ok and mandatory:
        failures.append(name)


check("Demo lockdown is on", SETTINGS.demo_lockdown, "set DEMO_LOCKDOWN=1" if not SETTINGS.demo_lockdown else "")
check("Live order gate is closed", not SETTINGS.live_mode_fully_authorized)

ts = token_status(SETTINGS.dhan_access_token)
if ts["state"] == "missing":
    check("Dhan access token present", False, "no token configured")
else:
    hours = ts["delta"].total_seconds() / 3600 if ts["delta"] is not None else None
    ok = ts["state"] in ("valid", "unknown") or (ts["state"] == "expiring" and hours is not None and hours > 0)
    detail = ts["state"] if hours is None else f"{ts['state']}, {hours:+.1f} h to expiry"
    check("Dhan access token usable", ok, detail + ("" if ok else " - generate a new token before the demo"))
    if ts["state"] == "expiring":
        print("       token expires within 2 hours: renew it shortly before going on stage")

try:
    from quant_intelligence.database.db import get_session  # noqa: E402
    from sqlalchemy import text  # noqa: E402

    with get_session() as session:
        session.execute(text("SELECT 1"))
    check("Database reachable", True)
except Exception as exc:  # pragma: no cover
    check("Database reachable", False, type(exc).__name__)

now = dt.datetime.now()
for name in ("NIFTY", "BANKNIFTY"):
    path = DATA_CACHE_DIR / "parquet_cache" / f"{name}_5min.parquet"
    if not path.exists():
        check(f"{name} 5min candles cached", False, "missing")
        continue
    last = pd.to_datetime(pd.read_parquet(path, columns=["timestamp"])["timestamp"]).max()
    age_h = (now - last.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600
    # Stale candles are not fatal (the app will honestly show NO TRADE), but they make a poor demo.
    check(f"{name} candles fresh (last bar {last:%d %b %H:%M})", age_h < 96, f"{age_h:.0f} h old", mandatory=False)

print()
if failures:
    print("PREFLIGHT FAILED:", "; ".join(failures))
    sys.exit(1)
print("PREFLIGHT OK (warnings, if any, are above)")
