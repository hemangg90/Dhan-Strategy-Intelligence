"""Process-wide cache of research pipeline results, shared by the Streamlit app and the website.

Stale-while-revalidate: once a result exists it is returned immediately, and if it is older than the TTL a single
background thread recomputes it. Only the very first request (or `force`) waits for a run, so a ~30-40 s pipeline never
blocks a page after warm-up, and a second browser tab does not pay for it again. Every result carries its own bar
timestamp, so the UI always shows how old the data really is.
"""
from __future__ import annotations

import datetime as dt
import threading
import time

from quant_intelligence.utils.timeutil import now_ist

PIPELINE_TTL_SEC = 60.0
LOOKBACK_DAYS = 60


class PipelineCache:
    def __init__(self, runner=None, ttl: float = PIPELINE_TTL_SEC):
        self._runner = runner
        self._ttl = ttl
        self._lock = threading.Lock()  # guards the dict and the "refreshing" set
        self._run_lock = threading.Lock()  # one pipeline run at a time
        self._items: dict[tuple, tuple[float, object]] = {}
        self._refreshing: set[tuple] = set()
        self.last_error: str | None = None

    def _compute(self, key, requested_at: float | None = None):
        runner = self._runner
        if runner is None:
            from quant_intelligence.research.pipeline import run_pipeline as runner
        with self._run_lock:
            if requested_at is not None:
                # Someone else finished a run while we waited for the lock: that result is newer than our request,
                # so reuse it instead of paying for the same ~40 s computation twice.
                with self._lock:
                    hit = self._items.get(key)
                if hit and hit[0] >= requested_at:
                    return hit[1]
            end = now_ist()
            output = runner(key[0], key[1], end - dt.timedelta(days=key[2]), end)
        with self._lock:
            self._items[key] = (time.monotonic(), output)
            self.last_error = None
        return output

    def _refresh_in_background(self, key):
        with self._lock:
            if key in self._refreshing:
                return
            self._refreshing.add(key)

        requested_at = time.monotonic()

        def work():
            try:
                self._compute(key, requested_at)
            except Exception as exc:  # keep serving the previous result
                self.last_error = str(exc)
            finally:
                with self._lock:
                    self._refreshing.discard(key)

        threading.Thread(target=work, daemon=True).start()

    def get(self, instrument: str, timeframe: str, force: bool = False, lookback_days: int = LOOKBACK_DAYS):
        key = (instrument.upper(), timeframe, int(lookback_days))
        requested_at = time.monotonic()
        with self._lock:
            hit = self._items.get(key)
        if hit and not force:
            if time.monotonic() - hit[0] >= self._ttl:
                self._refresh_in_background(key)
            return hit[1]
        return self._compute(key, requested_at)

    def warm(self, instrument: str = "NIFTY", timeframe: str = "5min", lookback_days: int = LOOKBACK_DAYS) -> None:
        """Compute a view in the background so the first visitor does not wait."""
        self._refresh_in_background((instrument.upper(), timeframe, int(lookback_days)))


PIPELINE = PipelineCache()
