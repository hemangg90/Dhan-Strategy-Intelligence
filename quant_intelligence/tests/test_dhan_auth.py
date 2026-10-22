"""Token keeper: RFC 6238 codes, renew before expiry, generate when expired, backoff, no secrets in messages.
All network calls are mocked."""
import base64
import dataclasses
import datetime as dt
import json
import time

import pytest
import requests

from quant_intelligence.brokers import dhan_auth
from quant_intelligence.brokers.dhan_auth import TokenKeeper, totp_now
from quant_intelligence.config import credentials as cred_module
from quant_intelligence.config import settings as settings_module

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def make_token(hours_left: float) -> str:
    def b64(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    exp = (dt.datetime.now(IST) + dt.timedelta(hours=hours_left)).timestamp()
    return f"{b64({'alg': 'HS256'})}.{b64({'exp': int(exp)})}.sig"


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body
        self.ok = 200 <= status < 300

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


@pytest.fixture
def env(monkeypatch):
    """Keeper enabled, a client id set, credentials persisted nowhere (in-memory only)."""
    state = {"settings": None, "saved": []}

    def configure(token_hours, pin="", secret=""):
        state["settings"] = dataclasses.replace(
            settings_module.SETTINGS, token_keeper_enabled=True, dhan_client_id="1000000001",
            dhan_access_token=make_token(token_hours) if token_hours is not None else "",
            dhan_pin=pin, dhan_totp_secret=secret, token_renew_before_hours=8.0)
        monkeypatch.setattr(dhan_auth, "SETTINGS", state["settings"])

    def fake_update(client_id, token):
        state["saved"].append(token)
        object.__setattr__(state["settings"], "dhan_access_token", token)

    monkeypatch.setattr(dhan_auth, "update_dhan_credentials", fake_update)
    monkeypatch.setattr(dhan_auth, "log_event", lambda *a, **k: None)
    state["configure"] = configure
    return state


# ---------------------------------------------------------------- TOTP (RFC 6238 appendix B, SHA-1)
def test_totp_matches_rfc_6238_vectors():
    secret = base64.b32encode(b"12345678901234567890").decode()
    assert totp_now(secret, 59, digits=8) == "94287082"
    assert totp_now(secret, 1111111109, digits=8) == "07081804"
    assert totp_now(secret, 2000000000, digits=8) == "69279037"
    assert len(totp_now(secret)) == 6


def test_totp_accepts_spaced_lowercase_secrets():
    secret = base64.b32encode(b"12345678901234567890").decode().lower()
    spaced = " ".join(secret[i:i + 4] for i in range(0, len(secret), 4))
    assert totp_now(spaced, 59, digits=8) == "94287082"


# ---------------------------------------------------------------- renew
def test_a_healthy_token_is_left_alone(env, monkeypatch):
    env["configure"](20)
    monkeypatch.setattr(dhan_auth.requests, "get", lambda *a, **k: pytest.fail("must not call Dhan"))
    assert TokenKeeper().ensure_fresh() == "ok"


def test_a_token_close_to_expiry_is_renewed(env, monkeypatch):
    env["configure"](5)
    new = make_token(24)
    seen = {}

    def fake_get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers)
        return Resp(200, {"accessToken": new, "expiryTime": "2026-10-03T10:00:00"})

    monkeypatch.setattr(dhan_auth.requests, "get", fake_get)
    keeper = TokenKeeper()
    assert keeper.ensure_fresh() == "renewed"
    assert seen["url"].endswith("/RenewToken")
    assert seen["headers"]["dhanClientId"] == "1000000001" and "access-token" in seen["headers"]
    assert env["saved"] == [new] and keeper.status()["last_action"] == "renewed" and keeper.last_error is None


def test_renew_failure_backs_off_and_does_not_retry_for_ten_minutes(env, monkeypatch):
    env["configure"](5)
    calls = []
    monkeypatch.setattr(dhan_auth.requests, "get", lambda *a, **k: calls.append(1) or Resp(401, {}))
    keeper = TokenKeeper()
    assert keeper.ensure_fresh() == "failed" and "401" in keeper.last_error
    assert keeper.ensure_fresh() == "waiting" and len(calls) == 1
    assert 0 < keeper.status()["retry_in"] <= dhan_auth.RETRY_AFTER_FAILURE_SEC


def test_network_errors_are_reported_without_leaking_urls(env, monkeypatch):
    env["configure"](5)

    def boom(*a, **k):
        raise requests.ConnectionError("HTTPSConnectionPool host=api.dhan.co ... access-token=SECRET")

    monkeypatch.setattr(dhan_auth.requests, "get", boom)
    keeper = TokenKeeper()
    assert keeper.ensure_fresh() == "failed"
    assert "SECRET" not in keeper.last_error and "ConnectionError" in keeper.last_error


# ---------------------------------------------------------------- generate
def test_an_expired_token_without_totp_asks_for_manual_entry(env, monkeypatch):
    env["configure"](-3)
    monkeypatch.setattr(dhan_auth.requests, "get", lambda *a, **k: pytest.fail("an expired token cannot be renewed"))
    monkeypatch.setattr(dhan_auth.requests, "post", lambda *a, **k: pytest.fail("no TOTP configured"))
    keeper = TokenKeeper()
    assert keeper.ensure_fresh() == "manual" and "enter a new token" in keeper.last_error


def test_an_expired_token_is_regenerated_with_pin_and_totp(env, monkeypatch):
    secret = base64.b32encode(b"12345678901234567890").decode()
    env["configure"](-3, pin="123456", secret=secret)
    new = make_token(24)
    seen = {}

    def fake_post(url, params=None, timeout=None):
        seen.update(url=url, params=params)
        return Resp(200, {"accessToken": new})

    monkeypatch.setattr(dhan_auth.requests, "get", lambda *a, **k: pytest.fail("expired: do not try renew"))
    monkeypatch.setattr(dhan_auth.requests, "post", fake_post)
    keeper = TokenKeeper()
    assert keeper.ensure_fresh() == "generated"
    assert seen["url"] == dhan_auth.GENERATE_URL
    assert seen["params"]["dhanClientId"] == "1000000001" and seen["params"]["pin"] == "123456"
    assert len(seen["params"]["totp"]) == 6 and env["saved"] == [new]


def test_renew_failure_falls_back_to_generate(env, monkeypatch):
    secret = base64.b32encode(b"12345678901234567890").decode()
    env["configure"](2, pin="123456", secret=secret)
    new = make_token(24)
    monkeypatch.setattr(dhan_auth.requests, "get", lambda *a, **k: Resp(500, None))
    monkeypatch.setattr(dhan_auth.requests, "post", lambda *a, **k: Resp(200, {"data": {"accessToken": new}}))
    assert TokenKeeper().ensure_fresh() == "generated" and env["saved"] == [new]


def test_wrong_pin_never_leaks_into_the_error_and_backs_off(env, monkeypatch):
    secret = base64.b32encode(b"12345678901234567890").decode()
    env["configure"](-1, pin="987654", secret=secret)

    def boom(url, params=None, timeout=None):
        raise requests.ConnectionError(f"failed for {url}?pin={params['pin']}&totp={params['totp']}")

    monkeypatch.setattr(dhan_auth.requests, "post", boom)
    keeper = TokenKeeper()
    assert keeper.ensure_fresh() == "failed"
    assert "987654" not in keeper.last_error and "totp" not in keeper.last_error
    assert keeper.ensure_fresh() == "waiting"


def test_a_bad_totp_secret_is_reported_not_raised(env):
    env["configure"](-1, pin="123456", secret="not base32 !!")
    keeper = TokenKeeper()
    assert keeper.ensure_fresh() == "failed" and "not a valid base32" in keeper.last_error


# ---------------------------------------------------------------- guards
def test_disabled_keeper_and_missing_client_id_do_nothing(env, monkeypatch):
    env["configure"](1)
    monkeypatch.setattr(dhan_auth, "SETTINGS", dataclasses.replace(env["settings"], token_keeper_enabled=False))
    assert TokenKeeper().ensure_fresh() == "disabled"
    monkeypatch.setattr(dhan_auth, "SETTINGS", dataclasses.replace(env["settings"], dhan_client_id=""))
    assert TokenKeeper().ensure_fresh() == "ok"


def test_only_one_attempt_runs_at_a_time(env, monkeypatch):
    env["configure"](5)
    keeper = TokenKeeper()
    keeper._lock.acquire()
    monkeypatch.setattr(dhan_auth.requests, "get", lambda *a, **k: pytest.fail("single-flight"))
    assert keeper.ensure_fresh() == "busy"
    keeper._lock.release()


def test_force_ignores_the_backoff(env, monkeypatch):
    env["configure"](5)
    new = make_token(24)
    monkeypatch.setattr(dhan_auth.requests, "get", lambda *a, **k: Resp(500, None))
    keeper = TokenKeeper()
    assert keeper.ensure_fresh() == "failed"
    monkeypatch.setattr(dhan_auth.requests, "get", lambda *a, **k: Resp(200, {"accessToken": new}))
    assert keeper.ensure_fresh(force=True) == "renewed"


# ---------------------------------------------------------------- the API client uses the keeper
def test_client_recovers_from_an_expired_token_once_and_picks_up_the_new_one(monkeypatch):
    from quant_intelligence.brokers import dhan_api_client as client_module

    expired, fresh = make_token(-2), make_token(24)
    live = dataclasses.replace(settings_module.SETTINGS, dhan_client_id="1000000001", dhan_access_token=expired)
    monkeypatch.setattr(client_module, "SETTINGS", live)

    class FakeKeeper:
        calls = 0

        def ensure_fresh(self):
            FakeKeeper.calls += 1
            object.__setattr__(live, "dhan_access_token", fresh)
            return "renewed"

    monkeypatch.setattr(dhan_auth, "TOKEN_KEEPER", FakeKeeper())
    sent = {}

    class R:
        status_code, text, ok = 200, "{}", True

        def json(self):
            return {"availabelBalance": 1}

    monkeypatch.setattr(client_module.requests, "get", lambda url, headers=None, timeout=None: sent.update(h=headers) or R())
    monkeypatch.setattr(client_module.LIMITER, "acquire", lambda *a, **k: None, raising=False)
    client = client_module.DhanApiClient()
    client.get_fund_limit()
    assert FakeKeeper.calls == 1 and sent["h"]["access-token"] == fresh


def test_client_with_explicit_credentials_never_triggers_the_keeper(monkeypatch):
    from quant_intelligence.brokers import dhan_api_client as client_module

    monkeypatch.setattr(dhan_auth, "TOKEN_KEEPER", type("K", (), {"ensure_fresh": lambda s: pytest.fail("explicit creds")})())
    client = client_module.DhanApiClient("1000000001", make_token(-2))
    with pytest.raises(client_module.DhanApiError, match="expired"):
        client.get_fund_limit()


# ---------------------------------------------------------------- rejected tokens and refused PIN/TOTP
SECRET = base64.b32encode(b"12345678901234567890").decode()


@pytest.fixture
def reject(monkeypatch):
    """Mark a token as rejected by Dhan (401) the way the API client does."""
    from quant_intelligence.brokers import dhan_rate_limit

    monkeypatch.setattr(dhan_rate_limit, "log_event", lambda *a, **k: None)
    return dhan_rate_limit.LIMITER.record_auth_failure


def test_a_rejected_but_unexpired_token_is_regenerated_not_renewed(env, monkeypatch, reject):
    env["configure"](20, pin="123456", secret=SECRET)
    reject(env["settings"].dhan_access_token)
    new = make_token(24)
    monkeypatch.setattr(dhan_auth.requests, "get", lambda *a, **k: pytest.fail("a rejected token cannot be renewed"))
    monkeypatch.setattr(dhan_auth.requests, "post", lambda *a, **k: Resp(200, {"accessToken": new}))
    assert TokenKeeper().ensure_fresh() == "generated" and env["saved"] == [new]


def test_a_rejected_token_without_totp_asks_for_manual_entry(env, monkeypatch, reject):
    env["configure"](20)
    reject(env["settings"].dhan_access_token)
    monkeypatch.setattr(dhan_auth.requests, "get", lambda *a, **k: pytest.fail("no network"))
    keeper = TokenKeeper()
    assert keeper.ensure_fresh() == "manual" and "rejected" in keeper.last_error


def test_a_refused_pin_is_not_retried_until_it_changes(env, monkeypatch):
    env["configure"](-1, pin="111111", secret=SECRET)
    calls = []
    monkeypatch.setattr(dhan_auth.requests, "post", lambda *a, **k: calls.append(1) or Resp(401, {"errorCode": "DH-901"}))
    keeper = TokenKeeper()
    assert keeper.ensure_fresh() == "failed" and "paused until the PIN" in keeper.last_error
    keeper._next_attempt = 0.0  # even once the backoff is over...
    assert keeper.ensure_fresh() == "manual" and len(calls) == 1 and keeper.status()["totp_refused"]

    object.__setattr__(env["settings"], "dhan_pin", "222222")  # ...until the PIN is corrected
    new = make_token(24)
    monkeypatch.setattr(dhan_auth.requests, "post", lambda *a, **k: Resp(200, {"accessToken": new}))
    assert keeper.ensure_fresh() == "generated" and env["saved"] == [new]


def test_server_errors_are_retried_after_a_minute_not_ten(env, monkeypatch):
    env["configure"](-1, pin="123456", secret=SECRET)
    monkeypatch.setattr(dhan_auth.requests, "post", lambda *a, **k: Resp(503, None))
    keeper = TokenKeeper()
    assert keeper.ensure_fresh() == "failed"
    assert 0 < keeper.status()["retry_in"] <= dhan_auth.RETRY_AFTER_TRANSIENT_SEC and not keeper.status()["totp_refused"]


def test_a_code_about_to_roll_over_waits_for_the_next_one(env, monkeypatch):
    env["configure"](-1, pin="123456", secret=SECRET)
    slept = []
    monkeypatch.setattr(dhan_auth.time, "time", lambda: 30 * 56_666_667 - 1.0)  # 1 s left in the 30 s step
    monkeypatch.setattr(dhan_auth.time, "sleep", slept.append)
    monkeypatch.setattr(dhan_auth.requests, "post", lambda *a, **k: Resp(200, {"accessToken": make_token(24)}))
    assert TokenKeeper().ensure_fresh() == "generated" and slept and slept[0] > 1.0


# ---------------------------------------------------------------- the API client recovers from 401 / missing token
class _ClientResp:
    def __init__(self, status, body):
        self.status_code, self._body, self.ok, self.headers = status, body, 200 <= status < 300, {}
        self.text = json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def client_env(monkeypatch):
    from quant_intelligence.brokers import dhan_api_client as client_module
    from quant_intelligence.brokers import dhan_rate_limit

    live = dataclasses.replace(settings_module.SETTINGS, dhan_client_id="1000000001", dhan_access_token=make_token(20))
    monkeypatch.setattr(client_module, "SETTINGS", live)
    monkeypatch.setattr(client_module.LIMITER, "acquire", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(dhan_rate_limit, "log_event", lambda *a, **k: None)
    fresh = make_token(24)

    class FakeKeeper:
        calls = 0

        def ensure_fresh(self):
            FakeKeeper.calls += 1
            object.__setattr__(live, "dhan_access_token", fresh)
            return "generated"

    monkeypatch.setattr(dhan_auth, "TOKEN_KEEPER", FakeKeeper())
    return client_module, live, fresh, FakeKeeper


def test_client_replaces_a_rejected_token_and_retries_a_read_once(client_env, monkeypatch):
    client_module, live, fresh, keeper = client_env
    sent = []

    def fake_get(url, headers=None, timeout=None):
        sent.append(headers["access-token"])
        return _ClientResp(200, {"availabelBalance": 1}) if headers["access-token"] == fresh else \
            _ClientResp(401, {"errorCode": "DH-901", "errorMessage": "Invalid_Authentication"})

    monkeypatch.setattr(client_module.requests, "get", fake_get)
    assert client_module.DhanApiClient().get_fund_limit() == {"availabelBalance": 1}
    assert keeper.calls == 1 and sent[-1] == fresh and len(sent) == 2


def test_client_gives_up_after_one_recovery(client_env, monkeypatch):
    client_module, live, fresh, keeper = client_env
    sent = []
    monkeypatch.setattr(client_module.requests, "get",
                        lambda url, headers=None, timeout=None: sent.append(1) or _ClientResp(401, {"errorCode": "DH-901"}))
    with pytest.raises(client_module.DhanApiError):
        client_module.DhanApiClient().get_fund_limit()
    assert len(sent) == 2


def test_client_never_resends_an_order_after_a_401(client_env, monkeypatch):
    client_module, live, fresh, keeper = client_env
    sent = []
    monkeypatch.setattr(client_module.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: sent.append(1) or _ClientResp(401, {}))
    with pytest.raises(client_module.DhanApiError):
        client_module.DhanApiClient().place_order({"x": 1})
    assert len(sent) == 1 and keeper.calls == 1  # the token is still replaced, so the NEXT order works


def test_client_generates_a_token_when_none_is_set(client_env, monkeypatch):
    client_module, live, fresh, keeper = client_env
    object.__setattr__(live, "dhan_access_token", "")
    sent = {}
    monkeypatch.setattr(client_module.requests, "get",
                        lambda url, headers=None, timeout=None: sent.update(h=headers) or _ClientResp(200, {}))
    client_module.DhanApiClient().get_fund_limit()
    assert keeper.calls == 1 and sent["h"]["access-token"] == fresh
