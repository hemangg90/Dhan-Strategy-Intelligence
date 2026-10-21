"""Process-wide rate limiter for Dhan API calls.

Dhan's limits apply to the whole ACCOUNT (orders 10/s, data/candles 5/s, quote/LTP 1/s,
non-trading 20/s, option chain 1 per 3 s) and exceeding them risks the account being blocked. Every
request the app makes goes through `LIMITER`, so the NSE runner, the commodity runner, every browser
tab and every page share one budget:

  * `acquire(category)` spaces calls in each category by a minimum interval (set ~30% under Dhan's
    limit - see config.settings.DhanLimits);
  * `record_429` / `record_success` feed a short circuit breaker: after several consecutive 429s,
    READ calls fast-fail for a cooldown (30 s, doubling to 2 min) instead of hammering Dhan.
    Orders are never blocked by the breaker, so exits can always go out;
  * `snapshot()` exposes per-category usage for the health page.

It cannot see requests made by OTHER processes using the same account (a second copy of the app, a
script) - run one instance.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from quant_intelligence.config.settings import SETTINGS, DhanLimits
from quant_intelligence.utils.logging_utils import log_event

CATEGORIES = ("quote", "data", "non_trading", "orders", "option_chain")
READ_CATEGORIES = tuple(c for c in CATEGORIES if c != "orders")


class RateLimiter:
    def __init__(self, limits: DhanLimits | None = None, clock=time.monotonic, sleep=time.sleep):
        self._limits = limits
        self.clock = clock
        self.sleep = sleep
        self._locks = {c: threading.Lock() for c in CATEGORIES}
        self._state_lock = threading.Lock()
        self.reset()

    @property
    def limits(self) -> DhanLimits:
        return self._limits or SETTINGS.dhan

    def reset(self) -> None:
        with self._state_lock:
            self._next_allowed = {c: 0.0 for c in CATEGORIES}
            self._calls = {c: deque(maxlen=5000) for c in CATEGORIES}
            self._total_429 = 0
            self._total_429_by_category = {c: 0 for c in CATEGORIES}
            self._consecutive_429 = 0
            self._breaker_opens = 0
            self._cooldown_until = 0.0
            self._auth_failed_token = None
            self._auth_failed_until = 0.0
            self._last_ok_wall = None

    def min_interval(self, category: str) -> float:
        return {
            "quote": self.limits.min_interval_quote,
            "data": self.limits.min_interval_data,
            "non_trading": self.limits.min_interval_non_trading,
            "orders": self.limits.min_interval_orders,
            "option_chain": self.limits.min_interval_option_chain,
        }[category]

    # ---- spacing ---------------------------------------------------------------
    def acquire(self, category: str) -> None:
        """Block until a request in `category` is allowed, then reserve the slot. Concurrent
        callers queue up and are spaced `min_interval` apart."""
        interval = self.min_interval(category)
        with self._locks[category]:
            wait = self._next_allowed[category] - self.clock()
            if wait > 0:
                self.sleep(wait)
            now = self.clock()
            self._next_allowed[category] = now + interval
            with self._state_lock:
                self._calls[category].append(now)

    # ---- circuit breaker ---------------------------------------------------------
    def cooldown_remaining(self, category: str) -> float:
        """Seconds left in the breaker cooldown for this category (always 0 for orders)."""
        if category == "orders":
            return 0.0
        return max(0.0, self._cooldown_until - self.clock())

    def record_429(self, category: str, retry_after: float | None = None) -> bool:
        """Note a 429. Returns True if this opened (or re-opened) the breaker."""
        opened = False
        with self._state_lock:
            self._total_429 += 1
            self._total_429_by_category[category] += 1
            if category != "orders":
                self._consecutive_429 += 1
                if self._consecutive_429 >= self.limits.breaker_threshold:
                    self._breaker_opens += 1
                    cooldown = min(
                        self.limits.breaker_cooldown_sec * 2 ** (self._breaker_opens - 1),
                        self.limits.breaker_max_cooldown_sec,
                    )
                    if retry_after:
                        cooldown = max(cooldown, min(retry_after, self.limits.breaker_max_cooldown_sec))
                    self._cooldown_until = self.clock() + cooldown
                    # Half-open afterwards: ONE more 429 re-opens it straight away.
                    self._consecutive_429 = self.limits.breaker_threshold - 1
                    opened = True
        log_event(
            "dhan_rate_limit",
            f"Dhan 429 on '{category}' calls" + (f"; reads paused for {self.cooldown_remaining(category):.0f}s" if opened else ""),
            level="WARNING",
        )
        return opened

    def record_success(self, category: str) -> None:
        if category == "orders":
            return
        with self._state_lock:
            self._last_ok_wall = time.time()
            self._consecutive_429 = 0
            if self.clock() >= self._cooldown_until:
                self._breaker_opens = 0

    # ---- rejected credentials ------------------------------------------------------
    def record_auth_failure(self, token: str) -> None:
        """Dhan answered 401 for this access token: stop using it until it changes (or the cooldown ends)."""
        with self._state_lock:
            first = self._auth_failed_token != token or self.clock() >= self._auth_failed_until
            self._auth_failed_token = token
            self._auth_failed_until = self.clock() + self.limits.auth_failure_cooldown_sec
        if first:
            log_event(
                "dhan_rate_limit",
                f"Dhan rejected the access token (401); no Dhan requests will be sent with it for "
                f"{self.limits.auth_failure_cooldown_sec:.0f}s or until the token is replaced",
                level="ERROR",
            )

    def auth_block_remaining(self, token: str) -> float:
        """Seconds for which requests with THIS token are suppressed (0 for a different/new token)."""
        if token != self._auth_failed_token:
            return 0.0
        return max(0.0, self._auth_failed_until - self.clock())

    # ---- observability -----------------------------------------------------------
    def snapshot(self, window: float = 60.0, token: str | None = None) -> dict:
        """`token`: the access token in use now. When given, a 401 recorded for a different (replaced) token is not reported."""
        now = self.clock()
        with self._state_lock:
            per_category = {
                c: {
                    "calls_last_window": sum(1 for t in self._calls[c] if now - t <= window),
                    "min_interval_sec": self.min_interval(c),
                    "total_429": self._total_429_by_category[c],
                }
                for c in CATEGORIES
            }
            return {
                "categories": per_category,
                "total_429": self._total_429,
                "cooldown_remaining": max(0.0, self._cooldown_until - now),
                "auth_block_remaining": (
                    0.0 if token is not None and token != self._auth_failed_token
                    else max(0.0, self._auth_failed_until - now)
                ),
                "last_ok_age_sec": None if self._last_ok_wall is None else time.time() - self._last_ok_wall,
                "window_sec": window,
            }


LIMITER = RateLimiter()
