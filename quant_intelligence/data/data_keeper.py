"""Background data keeper: keeps every watchlist instrument's candles current, always.

Candles used to be downloaded only when a runner scanned or a page ran the pipeline. With the
auto-traders stopped and nobody on the page, nothing refreshed them: after the last visit the data aged
past the 30-minute tolerance and every decision became "data quality DEGRADED -> NO TRADE" until someone
pressed a button. The keeper removes that dependency - it is a daemon thread that, every minute, asks the
DataManager for each NSE and MCX watchlist instrument. When an instrument's cache already holds the most
recently closed bar this is a no-op (no network, no database write); otherwise it costs one small tail
request, spaced by the Dhan rate limiter (about 25 instruments x one request per closed bar).

It runs whatever the markets are doing: after a close it captures the final bars, on weekends/holidays the
cache is already current so it does nothing, and when Dhan rejects the token it simply reports why and keeps
trying quietly until a valid token appears (and `wake()` makes it retry immediately, e.g. right after you
save a new token).
"""
from __future__ import annotations

import datetime as dt
import threading

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.config.watchlist import WATCHLIST_COMMODITIES, WATCHLIST_STOCKS
from quant_intelligence.data.data_manager import DataManager
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.timeutil import now_ist

LOOKBACK_DAYS = 60


def watchlist_symbols() -> list[str]:
    """NSE indices, the stock watchlist and the MCX commodities."""
    return list(SETTINGS.option_underlyings) + [s["symbol"] for s in WATCHLIST_STOCKS] + [c["symbol"] for c in WATCHLIST_COMMODITIES]


class DataKeeper:
    def __init__(self, interval_seconds: int | None = None, timeframe: str | None = None) -> None:
        self.interval = max(15, int(interval_seconds or SETTINGS.data_keeper_interval_sec))
        self.timeframe = timeframe or SETTINGS.default_timeframe
        self._manager = DataManager()
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.rounds = 0
        self.last_round_at: dt.datetime | None = None
        self.last_error: str | None = None
        from quant_intelligence.volatility.job import VolJob

        from quant_intelligence.reports.eod_job import EodJob

        self.eod_job = EodJob()  # daily report after each close (EOD_REPORT); works without a token
        self.vol_job = VolJob()  # once-a-day volatility models (VOL_MODELS_ENABLED); never runs in a scan cycle
        self.paused = False  # True while the token is unusable: no requests are sent until a valid one appears
        self.instruments: dict[str, dict] = {}  # symbol -> {"last_bar", "quality", "issues", "checked_at"}

    # ---- lifecycle -------------------------------------------------------------
    def start(self) -> bool:
        """Start the background thread (idempotent). Returns True if this call started it."""
        if not SETTINGS.data_keeper_enabled:
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="data-keeper", daemon=True)
            self._thread.start()
        log_event("data_keeper", f"Data keeper started (every {self.interval}s, {self.timeframe})", level="INFO")
        return True

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        """Run a refresh round now instead of waiting for the next tick (e.g. after a new token is saved)."""
        self._wake.set()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _loop(self) -> None:
        from quant_intelligence.brokers.dhan_auth import TOKEN_KEEPER

        while not self._stop.is_set():
            try:
                TOKEN_KEEPER.ensure_fresh()  # renew before expiry so the refresh below never meets a dead token
                self.run_once()
            except Exception as e:  # a bad round must never kill the keeper
                self.last_error = f"{type(e).__name__}: {e}"
                log_event("data_keeper", f"Refresh round failed: {e}", level="ERROR")
            try:
                self.eod_job.run_round()
            except Exception as e:  # a failed report must never disturb the candle refresh
                log_event("data_keeper", f"EOD job failed: {e}", level="ERROR")
            if not self.paused:
                try:
                    self.vol_job.run_round(watchlist_symbols(), self._stop)
                except Exception as e:  # failure-isolated: models must never disturb the candle refresh
                    log_event("data_keeper", f"Volatility job failed: {e}", level="ERROR")
            self._wake.wait(self.interval)
            self._wake.clear()

    # ---- one round --------------------------------------------------------------
    @staticmethod
    def _token_problem() -> str | None:
        """Why no Dhan request can succeed right now (expired / rejected token), or None.

        A missing token is not a problem here: CSV files can still supply candles."""
        from quant_intelligence.brokers.dhan_rate_limit import LIMITER
        from quant_intelligence.config.credentials import token_status

        token = SETTINGS.dhan_access_token
        if not token:
            return None
        status = token_status(token)
        if status["state"] == "expired":
            return (f"Dhan access token expired {status['expires_at']:%d %b %H:%M} IST - "
                    "enter a new token")
        if LIMITER.auth_block_remaining(token) > 0:
            return ("Dhan rejected the access token (401) - enter a new token")
        return None

    def run_once(self, symbols: list[str] | None = None) -> dict:
        """Refresh every instrument once. Returns {"refreshed": n, "stale": [...], "error": str | None}."""
        end = now_ist()
        problem = self._token_problem()
        if problem:
            # Every call would fail the same way and log two warnings per instrument per minute. Say it once,
            # send nothing, and carry on automatically when a valid token arrives (wake() fires on save).
            if problem != self.last_error:
                log_event("data_keeper", f"Data feed paused: {problem}", level="WARNING")
            self.paused, self.last_error = True, problem
            self.rounds += 1
            self.last_round_at = end
            return {"refreshed": 0, "stale": sorted(self.instruments), "error": problem, "paused": True}
        if self.paused:
            log_event("data_keeper", "Data feed resuming: a usable token is available", level="INFO")
            self.paused = False
        start = end - dt.timedelta(days=LOOKBACK_DAYS)
        stale: list[str] = []
        first_error: str | None = None
        refreshed = 0
        for sym in symbols or watchlist_symbols():
            if self._stop.is_set():
                break
            try:
                _, meta = self._manager.get_ohlcv(sym, self.timeframe, start, end, quiet=True)
            except Exception as e:  # no data at all (token rejected, nothing cached ...)
                first_error = first_error or str(e)
                self.instruments[sym] = {"last_bar": None, "quality": "NO DATA", "issues": [str(e)], "checked_at": end}
                stale.append(sym)
                continue
            issues = meta["quality_report"].get("issues", [])
            self.instruments[sym] = {"last_bar": meta["end_ts"], "quality": meta["quality_status"],
                                     "issues": issues, "checked_at": end}
            if meta["source"] == "cache+tail":
                refreshed += 1
            if meta["quality_status"] != "OK":
                stale.append(sym)
                first_error = first_error or (next((i for i in issues if "refresh failed" in i), None))
        self.rounds += 1
        self.last_round_at = end
        if first_error != self.last_error:
            log_event("data_keeper", f"Data feed problem: {first_error}" if first_error else "Data feed healthy again",
                      level="WARNING" if first_error else "INFO")
        self.last_error = first_error
        return {"refreshed": refreshed, "stale": stale, "error": first_error}

    # ---- status ----------------------------------------------------------------
    def status(self) -> dict:
        """Summary for the UI: how many instruments are current, which are not, and the newest/oldest bar."""
        bad = sorted(s for s, v in self.instruments.items() if v["quality"] != "OK")
        bars = [v["last_bar"] for v in self.instruments.values() if v["last_bar"] is not None]
        return {
            "running": self.running,
            "rounds": self.rounds,
            "last_round_at": self.last_round_at,
            "tracked": len(self.instruments),
            "ok": len(self.instruments) - len(bad),
            "not_ok": bad,
            "paused": self.paused,
            "newest_bar": max(bars) if bars else None,
            "oldest_bar": min(bars) if bars else None,
            "error": self.last_error,
        }


DATA_KEEPER = DataKeeper()
