"""Token keeper: keeps the Dhan access token valid without anyone pasting a new one.

Dhan access tokens last 24 hours. `RenewToken` swaps a still-active token for a fresh 24h one, so the keeper
does this when less than TOKEN_RENEW_BEFORE_HOURS (default 8) remain and one working token keeps itself alive.
An expired token, or one Dhan has rejected (401 / DH-901) before its expiry, cannot be renewed: a new one has to
be entered by hand under API Keys.

Safety: one attempt at a time (single-flight); after a failure, 1 minute before retrying a network/server error and
10 minutes otherwise; tokens never reach logs or error messages. The new token is applied through
`config.credentials.update_dhan_credentials` (saved to .env where the filesystem allows it, always applied in
memory).
"""
from __future__ import annotations

import datetime as dt
import os
import threading
import time

import requests

from quant_intelligence.brokers.dhan_rate_limit import LIMITER
from quant_intelligence.config.credentials import token_status, update_dhan_credentials
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.logging_utils import log_event

RENEW_URL = "https://api.dhan.co/v2/RenewToken"
RETRY_AFTER_FAILURE_SEC = 600.0  # never hammer auth
RETRY_AFTER_TRANSIENT_SEC = 60.0  # Dhan unreachable or a 5xx: nothing was wrong with the credentials
_TIMEOUT = 15


class AuthCallFailed(RuntimeError):
    """The renew call failed. `transient`: worth retrying soon."""

    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


def _http_failure(action: str, resp, hint: str = "") -> AuthCallFailed:
    return AuthCallFailed(f"{action}: Dhan answered HTTP {resp.status_code} without a token{hint}",
                          transient=resp.status_code >= 500)


def _json(resp):
    try:
        return resp.json()
    except ValueError:
        return None


def _extract_token(payload) -> str | None:
    if isinstance(payload, dict):
        for key in ("accessToken", "access_token", "token"):
            value = payload.get(key)
            if isinstance(value, str) and value.count(".") >= 2:
                return value
        data = payload.get("data")
        if isinstance(data, dict):
            return _extract_token(data)
    return None


def _apply(client_id: str, token: str) -> None:
    """Make `token` the live credential. Saving to .env is best effort (read-only on some hosts)."""
    try:
        update_dhan_credentials(client_id, token)
    except OSError:
        os.environ["DHAN_ACCESS_TOKEN"] = token
        object.__setattr__(SETTINGS, "dhan_access_token", token)
        object.__setattr__(SETTINGS, "dhan_client_id", client_id)


class TokenKeeper:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.last_action: str | None = None  # "renewed"
        self.last_success_at: dt.datetime | None = None
        self.last_error: str | None = None
        self._next_attempt = 0.0  # monotonic

    # ---- the Dhan call -------------------------------------------------------
    def _renew(self) -> str:
        try:
            resp = requests.get(RENEW_URL, headers={"access-token": SETTINGS.dhan_access_token,
                                                    "dhanClientId": SETTINGS.dhan_client_id}, timeout=_TIMEOUT)
        except requests.RequestException as e:
            raise AuthCallFailed(f"renew: could not reach Dhan ({type(e).__name__})", transient=True) from None
        token = _extract_token(_json(resp)) if resp.ok else None
        if not token:
            raise _http_failure("renew", resp)
        return token

    # ---- policy -------------------------------------------------------------------
    def ensure_fresh(self, force: bool = False) -> str:
        """Renew the token if it is due. Returns what happened:
        "ok" (nothing needed) | "renewed" | "failed" | "manual" (expired or rejected: enter a new token) |
        "waiting" (a recent attempt failed) | "disabled" | "busy" (another thread is already doing it).

        A token Dhan rejected with 401 counts as dead even before its expiry: renewing it cannot work."""
        if not SETTINGS.token_keeper_enabled:
            return "disabled"
        if not SETTINGS.dhan_client_id:
            return "ok"
        status = token_status(SETTINGS.dhan_access_token)
        state = status["state"]
        rejected = state != "missing" and LIMITER.auth_block_remaining(SETTINGS.dhan_access_token) > 0
        if state == "unknown" and not rejected:
            return "ok"  # not a JWT with an expiry: cannot judge, leave it alone
        dead = rejected or state in ("missing", "expired")
        due = dead or status["delta"] < dt.timedelta(hours=SETTINGS.token_renew_before_hours)
        if not due and not force:
            return "ok"
        if dead:
            what = "Dhan rejected the access token" if rejected else "Token expired"
            self.last_error = f"{what}: enter a new token under API Keys."
            return "manual"
        if not self._lock.acquire(blocking=False):
            return "busy"
        try:
            if time.monotonic() < self._next_attempt and not force:
                return "waiting"
            try:
                token = self._renew()
            except AuthCallFailed as e:
                self.last_error = str(e)
                self._next_attempt = time.monotonic() + (RETRY_AFTER_TRANSIENT_SEC if e.transient
                                                         else RETRY_AFTER_FAILURE_SEC)
                log_event("token_keeper", f"Token refresh failed: {self.last_error}", level="WARNING")
                return "failed"
            _apply(SETTINGS.dhan_client_id, token)
            self.last_action, self.last_success_at, self.last_error = "renewed", dt.datetime.now(), None
            self._next_attempt = 0.0
            expires = token_status(token)["expires_at"]
            log_event("token_keeper", "Dhan access token renewed"
                                      + (f" (valid until {expires:%d %b %H:%M} IST)" if expires else ""), level="INFO")
            return "renewed"
        finally:
            self._lock.release()

    def status(self) -> dict:
        wait = max(0.0, self._next_attempt - time.monotonic())
        return {"enabled": SETTINGS.token_keeper_enabled, "last_action": self.last_action, "last_success_at": self.last_success_at,
                "last_error": self.last_error, "retry_in": wait}


TOKEN_KEEPER = TokenKeeper()
