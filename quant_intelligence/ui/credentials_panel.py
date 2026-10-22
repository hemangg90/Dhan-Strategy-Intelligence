"""Streamlit widget for viewing, changing and TESTING the Dhan API credentials from the dashboard.

Shows when the saved access token expires (read from the token itself), validates a newly saved token by
making one cheap Dhan call, and offers a Test connection button - so "saved" never silently means "saved a
token that does not work".
"""
from __future__ import annotations

import datetime as dt

import streamlit as st

from quant_intelligence.brokers.dhan_api_client import DhanApiClient, DhanApiError
from quant_intelligence.config.credentials import (mask_secret, token_status, update_dhan_credentials,
                                                 update_dhan_totp)
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.ui import format as F
from quant_intelligence.ui.state import invalidate_cached_analysis


def test_connection() -> tuple[bool, str]:
    """One non-trading call (fund limit) to prove the saved Client ID + token are accepted by Dhan."""
    client = DhanApiClient()
    if not client.is_configured():
        return False, "No credentials are set."
    try:
        data = client.get_fund_limit()
    except DhanApiError as e:
        return False, str(e)
    except Exception as e:  # network
        return False, f"Could not reach Dhan: {e}"
    balance = data.get("availabelBalance") if isinstance(data, dict) else None
    return True, "Dhan accepted the credentials" + (f" (available balance {F.inr(balance)})." if balance is not None else ".")


def _lockdown_status() -> str:
    """Token state and expiry only, with nothing derived from the secret itself."""
    ts = token_status(SETTINGS.dhan_access_token)
    state = ts["state"]
    if state == "missing":
        return "token not set"
    if state == "unknown":
        return "token present (expiry unknown)"
    when = f"{ts['expires_at']:%d %b %H:%M} IST"
    return {"valid": f"token valid until {when}", "expiring": f"token expires {when}", "expired": f"token EXPIRED {when}"}[state]


def _token_line() -> str:
    ts = token_status(SETTINGS.dhan_access_token)
    state = ts["state"]
    if state == "missing":
        return "Access token: **not set**"
    if state == "expired":
        return (f"Access token: `{mask_secret(SETTINGS.dhan_access_token)}` ✖ **EXPIRED** {ts['expires_at']:%d %b %H:%M} IST "
                f"({F.duration(-ts['delta'])} ago) - generate a new one in your Dhan account")
    if state == "expiring":
        return (f"Access token: `{mask_secret(SETTINGS.dhan_access_token)}` ▲ expires {ts['expires_at']:%d %b %H:%M} IST "
                f"(in {F.duration(ts['delta'])})")
    if state == "valid":
        return (f"Access token: `{mask_secret(SETTINGS.dhan_access_token)}` ● valid until {ts['expires_at']:%d %b %H:%M} IST "
                f"(in {F.duration(ts['delta'])})")
    return f"Access token: `{mask_secret(SETTINGS.dhan_access_token)}` (expiry unknown)"


def _keeper_line() -> str:
    """One line on automatic renewal: what it will do, and what it last did or why it could not."""
    from quant_intelligence.brokers.dhan_auth import TOKEN_KEEPER

    ks = TOKEN_KEEPER.status()
    if not ks["enabled"]:
        return "Auto-renew: **off** (TOKEN_KEEPER=false)"
    mode = ("renews before expiry, but Dhan refused the saved PIN / TOTP secret: correct them below to re-enable "
            "automatic recovery" if ks["totp_refused"] else
            "renews before expiry and generates a new token with your PIN + TOTP if it ever expires or is rejected"
            if ks["totp"] else
            f"renews the token when under {SETTINGS.token_renew_before_hours:g}h remain; "
            "add DHAN_PIN and DHAN_TOTP_SECRET in Secrets to also recover from an expired token")
    line = f"Auto-renew: **on** - {mode}."
    if ks["last_success_at"]:
        line += f" Last {ks['last_action']} {ks['last_success_at']:%d %b %H:%M}."
    if ks["last_error"]:
        line += f" Last problem: {ks['last_error']}"
    return line


def render_credentials_panel() -> None:
    if SETTINGS.demo_lockdown:
        # Presentation mode: status only. No client ID, no token (not even its last characters), no form.
        st.caption(f"Dhan: {_lockdown_status()} (credentials hidden: demo lockdown)")
        return
    expired = token_status(SETTINGS.dhan_access_token)["state"] in ("missing", "expired")
    with st.expander("Dhan API credentials", expanded=expired):
        st.caption(f"Client ID: `{SETTINGS.dhan_client_id or '(not set)'}`")
        st.markdown(_token_line())
        st.caption(_keeper_line())
        if st.button("Test connection", key="dhan_test_conn"):
            ok, msg = test_connection()
            (st.success if ok else st.error)(msg)
        with st.form("dhan_credentials_form", clear_on_submit=True):
            client_id = st.text_input("Client ID", value=SETTINGS.dhan_client_id)
            token = st.text_input("Access token", type="password", placeholder="Paste new token")
            submitted = st.form_submit_button("Save credentials")
        if submitted:
            try:
                update_dhan_credentials(client_id, token)
            except ValueError as e:
                st.error(str(e))
                return
            # Drop clients/brokers built with the old credentials, and every cached analysis computed with them:
            # otherwise the page keeps showing the old "data quality degraded" answer after the problem is fixed.
            st.session_state["dhan_api_client"] = DhanApiClient()
            st.session_state.pop("dhan_broker", None)
            invalidate_cached_analysis()
            ok, msg = test_connection()
            if ok:
                st.success(f"Saved to .env and applied. {msg}")
            else:
                st.error(f"Saved, but Dhan did not accept it: {msg}")
            if ok:
                st.info("Cached analyses were cleared and candles are being refreshed in the background - the page "
                        "recomputes on its next run.")
            st.caption("On Streamlit Cloud this panel only changes the running app (it resets on reboot and is shared by "
                       "everyone who opens it). Put the token in the app's Secrets for a lasting change.")
        _render_totp_form()


def _render_totp_form() -> None:
    """PIN + TOTP secret for the token keeper, so an expired (or never entered) token is replaced automatically."""
    from quant_intelligence.brokers.dhan_auth import TOKEN_KEEPER

    st.markdown("**Automatic token (PIN + TOTP)**")
    st.caption("Enable TOTP in Dhan (Profile > Access DhanHQ APIs) and paste the setup key shown under the QR code. "
               "These let the app create new tokens for your account: treat them like a password."
               + (" PIN and TOTP secret are set; saving replaces them." if TOKEN_KEEPER.totp_configured() else ""))
    with st.form("dhan_totp_form", clear_on_submit=True):
        client_id = st.text_input("Client ID", value=SETTINGS.dhan_client_id, key="dhan_totp_client_id")
        pin = st.text_input("Dhan PIN", type="password")
        secret = st.text_input("TOTP secret (setup key)", type="password")
        submitted = st.form_submit_button("Save PIN and TOTP")
    if not submitted:
        return
    try:
        saved = update_dhan_totp(client_id, pin, secret)
    except ValueError as e:
        st.error(str(e))
        return
    st.success("Saved to .env and applied." if saved else
               "Applied to the running app, but .env is not writable: put DHAN_PIN and DHAN_TOTP_SECRET in Secrets to keep them.")
    if token_status(SETTINGS.dhan_access_token)["state"] not in ("missing", "expired"):
        st.info("The current token is still valid; a new one will be generated with these if it ever expires.")
        return
    result = TOKEN_KEEPER.ensure_fresh()
    if result == "generated":
        st.session_state["dhan_api_client"] = DhanApiClient()
        st.session_state.pop("dhan_broker", None)
        invalidate_cached_analysis()
        ok, msg = test_connection()
        (st.success if ok else st.error)(f"New access token generated. {msg}")
    elif result == "waiting":
        st.warning(f"A recent attempt failed; the next try is in {F.duration(dt.timedelta(seconds=TOKEN_KEEPER.status()['retry_in']))} "
                   "(spaced out so a wrong PIN cannot lock the account).")
    elif result == "disabled":
        st.warning("Token keeper is off (TOKEN_KEEPER=false): no token was generated.")
    else:
        st.error(f"Could not generate a token: {TOKEN_KEEPER.status()['last_error'] or result}")
