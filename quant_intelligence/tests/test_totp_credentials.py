"""Saving the PIN + TOTP secret from the sidebar: validated, saved to .env, applied live, never echoed back."""
import dataclasses

import pytest
from streamlit.testing.v1 import AppTest

from quant_intelligence.config import credentials
from quant_intelligence.config.settings import SETTINGS

SECRET = "JBSWY3DPEHPK3PXP"


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = dataclasses.replace(SETTINGS, dhan_client_id="", dhan_pin="", dhan_totp_secret="")
    monkeypatch.setattr(credentials, "SETTINGS", settings)
    monkeypatch.setattr(credentials, "ENV_PATH", tmp_path / ".env")
    for key in ("DHAN_CLIENT_ID", "DHAN_PIN", "DHAN_TOTP_SECRET"):
        monkeypatch.delenv(key, raising=False)
    return settings, tmp_path / ".env"


def test_saves_to_env_file_and_applies_in_memory(env):
    settings, path = env
    assert credentials.update_dhan_totp(" 1000000001 ", "123456", "jbsw y3dp-ehpk 3pxp") is True
    text = path.read_text()
    assert "DHAN_PIN=123456" in text and f"DHAN_TOTP_SECRET={SECRET}" in text and "DHAN_CLIENT_ID=1000000001" in text
    assert (settings.dhan_client_id, settings.dhan_pin, settings.dhan_totp_secret) == ("1000000001", "123456", SECRET)


@pytest.mark.parametrize("args", [("", "1", SECRET), ("id", "", SECRET), ("id", "1", ""), ("id", "1", "not base32!")])
def test_rejects_missing_or_invalid_values_without_saving(env, args):
    settings, path = env
    with pytest.raises(ValueError):
        credentials.update_dhan_totp(*args)
    assert not path.exists() and not settings.dhan_pin


def test_unwritable_env_still_applies_in_memory(env, monkeypatch):
    settings, _ = env

    def boom(*a, **k):
        raise OSError("read-only")

    monkeypatch.setattr(credentials, "set_key", boom)
    assert credentials.update_dhan_totp("1000000001", "123456", SECRET) is False
    assert settings.dhan_totp_secret == SECRET


def _panel_app():
    from quant_intelligence.ui.credentials_panel import render_credentials_panel

    render_credentials_panel()


def test_sidebar_form_generates_a_token_when_none_is_set(monkeypatch):
    from quant_intelligence.brokers import dhan_auth
    from quant_intelligence.ui import credentials_panel as panel

    saved, calls = [], []
    monkeypatch.setattr(panel, "update_dhan_totp", lambda *a: saved.append(a) or True)
    monkeypatch.setattr(panel, "test_connection", lambda: (True, "Dhan accepted the credentials."))
    monkeypatch.setattr(dhan_auth.TOKEN_KEEPER, "ensure_fresh", lambda: calls.append(1) or "generated")
    monkeypatch.setattr(panel, "SETTINGS", dataclasses.replace(SETTINGS, demo_lockdown=False, dhan_access_token=""))

    at = AppTest.from_function(_panel_app).run()
    at.text_input(key="dhan_totp_client_id").input("1000000001")
    pins = [t for t in at.text_input if t.label in ("Dhan PIN", "TOTP secret (setup key)")]
    pins[0].input("123456")
    pins[1].input(SECRET)
    next(b for b in at.button if b.label == "Save PIN and TOTP").click()
    at.run()

    assert not at.exception
    assert saved == [("1000000001", "123456", SECRET)] and calls == [1]
    assert any("New access token generated" in s.value for s in at.success)
    page = " ".join(str(e.value) for e in at.markdown) + " ".join(str(e.value) for e in at.caption)
    assert "123456" not in page and SECRET not in page
