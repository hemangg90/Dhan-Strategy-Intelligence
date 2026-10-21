"""The end-of-day job run by the data keeper: after each market closes, settle any position still open, build and
store that day's report, and trim old log rows so a free-tier database does not fill up.

Once per market session: NSE ~10 minutes after 15:30, MCX ~10 minutes after 23:55 (so its report lands just after
midnight, dated for the session that ended). The combined "ALL" report for a date is regenerated after each of
its two markets has closed, so the later one is complete. Failure-isolated, retried after a pause, and it works
without a Dhan token (only the forced square-off needs live prices; without it the open positions are reported).
"""
from __future__ import annotations

import datetime as dt
import time

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.market_profile import MCX, NSE
from quant_intelligence.utils.timeutil import now_ist
from quant_intelligence.volatility.job import SETTLE_MINUTES, VolJob  # same "latest finished session" rule

FORCED_REASON = "EOD_FORCED"


def session_key(now: dt.datetime, market: str) -> dt.date | None:
    """Date of the latest finished session of NSE/MCX, once settled; else None."""
    return VolJob.session_key(now, "NIFTY" if market == "NSE" else "CRUDEOIL")


class EodJob:
    def __init__(self, retry_after_sec: float = 900.0) -> None:
        self.retry_after_sec = retry_after_sec
        self.done: dict[str, str] = {}  # scope -> key last completed
        self.next_try: dict[str, float] = {}
        self.last_error: str | None = None
        self.last_report_at: dt.datetime | None = None
        self.last_report: tuple[dt.date, str] | None = None
        self._pruned_on: dt.date | None = None

    def due(self, now: dt.datetime | None = None) -> list[tuple[str, dt.date, str]]:
        """[(scope, report_date, key)] still to do. ALL is keyed by both markets so it re-runs after the later close."""
        now = now or now_ist()
        nse, mcx = session_key(now, "NSE"), session_key(now, "MCX")
        jobs = []
        if nse is not None:
            jobs.append(("NSE", nse, str(nse)))
        if mcx is not None:
            jobs.append(("MCX", mcx, str(mcx)))
        if nse is not None:
            jobs.append(("ALL", nse, f"{nse}|{mcx}"))
        mono = time.monotonic()
        return [j for j in jobs if self.done.get(j[0]) != j[2] and mono >= self.next_try.get(j[0], 0.0)]

    @staticmethod
    def force_square_off(market: str) -> int:
        """Close paper positions still open after the close at the latest traded price. Returns how many closed."""
        from quant_intelligence.brokers.dhan_api_client import DhanApiClient
        from quant_intelligence.execution.engine import engine_for_market
        from quant_intelligence.execution.square_off import square_off_positions

        owner = engine_for_market(market)  # each market is its own account
        return len(square_off_positions(owner.broker, DhanApiClient(), NSE if market == "NSE" else MCX,
                                        owner.lock, owner.add_pnl, reason=FORCED_REASON))

    def run_one(self, scope: str, report_date: dt.date, key: str) -> bool:
        from quant_intelligence.reports import decision_log, eod

        try:
            if SETTINGS.eod_force_square_off and scope in ("NSE", "MCX"):
                try:
                    n = self.force_square_off(scope)
                    if n:
                        log_event("eod_job", f"{scope}: closed {n} position(s) still open after the close", level="INFO")
                except Exception as e:  # prices unavailable etc.: the report will list the open positions
                    log_event("eod_job", f"{scope}: forced square-off skipped: {e}", level="WARNING")
            eod.generate_and_save(report_date, scope)
            self.done[scope] = key
            self.last_report_at, self.last_report = now_ist(), (report_date, scope)
            log_event("eod_job", f"Daily report stored for {report_date} ({scope})", level="INFO")
            if self._pruned_on != report_date:
                self._prune(decision_log)
                self._pruned_on = report_date
            return True
        except Exception as e:
            self.last_error = f"{scope} {report_date}: {type(e).__name__}: {e}"
            self.next_try[scope] = time.monotonic() + self.retry_after_sec
            log_event("eod_job", f"Daily report failed for {report_date} ({scope}): {e}", level="WARNING")
            return False

    @staticmethod
    def _prune(decision_log) -> None:
        """Trim old rows: keeps a free-tier database small. Trades, orders, fills and reports are never touched."""
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import MarketDataMetadata, SystemEvent, utcnow

        cutoff = utcnow() - dt.timedelta(days=SETTINGS.log_retention_days)
        with get_session() as session:
            session.query(SystemEvent).filter(SystemEvent.timestamp < cutoff).delete()
            session.query(MarketDataMetadata).filter(MarketDataMetadata.fetched_at < cutoff).delete()
        decision_log.prune(SETTINGS.decision_retention_days)

    def run_round(self, now: dt.datetime | None = None) -> int:
        """Do every due report. Returns how many were attempted."""
        if not SETTINGS.eod_report_enabled:
            return 0
        done = 0
        for scope, report_date, key in self.due(now):
            self.run_one(scope, report_date, key)
            done += 1
        return done

    def status(self) -> dict:
        return {"enabled": SETTINGS.eod_report_enabled, "done": dict(self.done), "last_report": self.last_report,
                "last_report_at": self.last_report_at, "error": self.last_error}
