"""Failure injection for the Dhan request layer: bounded, jittered retries for reads; never for orders."""
from unittest.mock import MagicMock, patch

import pytest
import requests
from hypothesis import given, settings, strategies as st

from quant_intelligence.brokers.dhan_api_client import DhanApiClient, DhanApiError, backoff_delay
from quant_intelligence.brokers.dhan_rate_limit import LIMITER

MODULE = "quant_intelligence.brokers.dhan_api_client"


def _client():
    return DhanApiClient(client_id="1000000001", access_token="a.b.c", base_url="https://example.test/v2")


def _resp(status=200, body=None, text="{}"):
    r = MagicMock()
    r.ok = 200 <= status < 300
    r.status_code = status
    r.headers = {}
    r.text = text
    if body is None:
        r.json.side_effect = ValueError("no json")
    else:
        r.json.return_value = body
    return r


def _run(side_effect, method="get", call=lambda c: c._get("/fundlimit")):
    with patch(f"{MODULE}.requests.{method}", side_effect=side_effect) as http, patch(f"{MODULE}.time.sleep") as sleep, \
            patch.object(LIMITER, "sleep"):
        try:
            result, error = call(_client()), None
        except Exception as exc:  # the test inspects the exact exception type
            result, error = None, exc
    return result, error, http, sleep


def test_a_read_that_times_out_is_retried_with_growing_delays_then_succeeds():
    result, error, http, sleep = _run([requests.Timeout("t"), requests.ConnectionError("c"), _resp(200, {"ok": 1})])
    assert error is None and result == {"ok": 1} and http.call_count == 3
    delays = [c.args[0] for c in sleep.call_args_list]
    assert len(delays) == 2 and delays[0] < delays[1] and all(0.8 <= d <= 10.0 for d in delays)


def test_a_read_that_never_answers_ends_in_a_dhan_error_after_bounded_attempts():
    result, error, http, sleep = _run([requests.Timeout("t")] * 10)
    assert isinstance(error, DhanApiError) and "did not respond" in str(error)
    assert http.call_count == 3 and sleep.call_count == 2  # bounded: DHAN_429_MAX_ATTEMPTS attempts, no sleep after the last


def test_transient_server_errors_are_retried_for_reads():
    result, error, http, _ = _run([_resp(503, {"remarks": "busy"}), _resp(200, {"ok": 1})])
    assert error is None and result == {"ok": 1} and http.call_count == 2
    _, error, http, _ = _run([_resp(502, {"remarks": "bad gateway"})] * 5)
    assert isinstance(error, DhanApiError) and error.status_code == 502 and http.call_count == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_errors_are_not_retried(status):
    _, error, http, sleep = _run([_resp(status, {"remarks": "no"})] * 5)
    assert isinstance(error, DhanApiError) and error.status_code == status
    assert http.call_count == 1 and sleep.call_count == 0


def test_an_order_is_never_retried_and_a_transport_error_is_not_disguised_as_a_rejection():
    _, error, http, sleep = _run([requests.Timeout("read timed out")] * 5, method="post", call=lambda c: c.place_order({"x": 1}))
    assert isinstance(error, requests.Timeout)  # NOT DhanApiError: the broker must treat the outcome as unknown
    assert http.call_count == 1 and sleep.call_count == 0


def test_a_server_error_on_an_order_is_not_retried_either():
    _, error, http, sleep = _run([_resp(503, {"remarks": "busy"})] * 5, method="post", call=lambda c: c.place_order({"x": 1}))
    assert isinstance(error, DhanApiError) and http.call_count == 1 and sleep.call_count == 0


def test_a_success_status_with_a_non_json_body_is_an_error_not_empty_data():
    _, error, _, _ = _run([_resp(200, None, text="<html>Service Unavailable</html>")])
    assert isinstance(error, DhanApiError) and "non-JSON" in str(error)
    result, error, _, _ = _run([_resp(200, None, text="")])
    assert error is None and result == {}  # a genuinely empty body is still just "nothing"


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=1, max_value=12), st.floats(min_value=0.1, max_value=30.0), st.floats(min_value=1.0, max_value=120.0))
def test_backoff_is_exponential_jittered_and_capped(attempt, base, cap):
    delay = backoff_delay(attempt, base, cap)
    nominal = min(cap, base * 2 ** (attempt - 1))
    assert 0.8 * nominal - 1e-9 <= delay <= 1.2 * nominal + 1e-9
