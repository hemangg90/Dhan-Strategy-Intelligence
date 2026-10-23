"""Runtime updates to Dhan API credentials.

Persists to .env, updates os.environ, and mutates the (frozen) SETTINGS
instance in place so modules that already imported SETTINGS see the change.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import os

from dotenv import set_key

from quant_intelligence.config.settings import PROJECT_ROOT, SETTINGS

ENV_PATH = PROJECT_ROOT / ".env"


def mask_secret(value: str, visible: int = 4) -> str:
    if not value:
        return "(not set)"
    if len(value) <= visible:
        return "*" * len(value)
    return "*" * 8 + value[-visible:]


def update_dhan_credentials(client_id: str, access_token: str) -> None:
    client_id, access_token = client_id.strip(), access_token.strip()
    if not client_id or not access_token:
        raise ValueError("Both Client ID and Access Token are required.")

    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), "DHAN_CLIENT_ID", client_id, quote_mode="never")
    set_key(str(ENV_PATH), "DHAN_ACCESS_TOKEN", access_token, quote_mode="never")

    os.environ["DHAN_CLIENT_ID"] = client_id
    os.environ["DHAN_ACCESS_TOKEN"] = access_token
    object.__setattr__(SETTINGS, "dhan_client_id", client_id)
    object.__setattr__(SETTINGS, "dhan_access_token", access_token)


_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def token_expiry(token: str | None) -> dt.datetime | None:
    """Expiry time (naive IST) of a Dhan access token, read from the token itself - no network call.

    Dhan access tokens are signed JWTs that carry an `exp` claim and last about 24 hours. Returns None if the
    token is empty or not in that format (then validity can only be learned by calling Dhan)."""
    try:
        payload = (token or "").strip().split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp")
        return None if exp is None else dt.datetime.fromtimestamp(int(exp), _IST).replace(tzinfo=None)
    except Exception:
        return None


def token_status(token: str | None, now: dt.datetime | None = None) -> dict:
    """{"state": "missing"|"unknown"|"expired"|"expiring"|"valid", "expires_at": dt|None, "delta": timedelta|None}
    `delta` is time until expiry (negative once expired). 'expiring' means under two hours left."""
    if not (token or "").strip():
        return {"state": "missing", "expires_at": None, "delta": None}
    expires = token_expiry(token)
    if expires is None:
        return {"state": "unknown", "expires_at": None, "delta": None}
    now = now or dt.datetime.now(_IST).replace(tzinfo=None)
    delta = expires - now
    state = "expired" if delta.total_seconds() <= 0 else ("expiring" if delta < dt.timedelta(hours=2) else "valid")
    return {"state": state, "expires_at": expires, "delta": delta}
