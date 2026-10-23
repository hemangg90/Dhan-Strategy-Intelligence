"""Builds the implied-volatility history the application does not otherwise have.

The broker serves live option chains only, so IV rank and percentile (`options/iv_history.py`) can exist only after the chain has
been read regularly for a while. The data keeper calls `IvRecorder.run_round` each tick: for each option underlying whose market is
open and which has not been read for `RECORD_INTERVAL_MIN` minutes, it fetches the chain through the same cache and rate limiter the
traders use (`fetch_chain`), and that fetch records the ATM IV. At most `max_per_round` underlyings per round, a failure is isolated
to its underlying and retried later, and nothing here places an order. Switch off with IV_RECORDER=0.
"""
from __future__ import annotations

import datetime as dt
import time

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.options.iv_history import RECORD_INTERVAL_MIN
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeutil import is_market_open, now_ist


class IvRecorder:
    def __init__(self, interval_min: float = RECORD_INTERVAL_MIN, max_per_round: int = 2, retry_after_sec: float = 600.0) -> None:
        self.interval_sec = interval_min * 60.0
        self.max_per_round = max_per_round
        self.retry_after_sec = retry_after_sec
        self.next_due: dict[str, float] = {}  # underlying -> monotonic time of the next attempt
        self.last_ok: dict[str, dt.datetime] = {}
        self.last_error: dict[str, str] = {}
        self.rounds = 0

    def due(self, underlyings: list[str], now: dt.datetime | None = None) -> list[str]:
        now = now or now_ist()
        mono = time.monotonic()
        return [u for u in underlyings if is_market_open(now, profile_for(u)) and mono >= self.next_due.get(u, 0.0)]

    def run_round(self, underlyings: list[str], stop_event=None, now: dt.datetime | None = None, client=None, fetch=None) -> dict:
        """Read the due chains. Returns {"recorded": [...], "failed": {...}}. Never raises."""
        out: dict = {"recorded": [], "failed": {}}
        if not SETTINGS.iv_recorder_enabled:
            return {**out, "disabled": True}
        self.rounds += 1
        try:
            from quant_intelligence.brokers.dhan_api_client import DhanApiClient
            from quant_intelligence.options.option_selector import fetch_chain

            fetch = fetch or fetch_chain
            client = client or DhanApiClient()
            for underlying in self.due(underlyings, now)[: self.max_per_round]:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    fetch(client, underlying, SETTINGS.option_expiry_preference, max_age=0)
                    self.next_due[underlying] = time.monotonic() + self.interval_sec
                    self.last_ok[underlying] = now or now_ist()
                    self.last_error.pop(underlying, None)
                    out["recorded"].append(underlying)
                except Exception as e:  # one underlying's failure (no expiry, rate limit, token) must not stop the others
                    self.next_due[underlying] = time.monotonic() + self.retry_after_sec
                    self.last_error[underlying] = f"{type(e).__name__}: {e}"
                    out["failed"][underlying] = self.last_error[underlying]
        except Exception as e:
            log_event("iv_recorder", f"IV recording round failed: {e}", level="WARNING")
            out["error"] = str(e)
        return out
