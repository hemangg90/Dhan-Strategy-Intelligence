"""The IV recorder: reads due chains in market hours, throttles, isolates failures, and can be switched off."""
import datetime as dt
import threading

import pytest

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.options.iv_recorder import IvRecorder

OPEN = dt.datetime(2030, 1, 7, 11, 0)  # Monday, NSE open
CLOSED = dt.datetime(2030, 1, 7, 20, 0)


@pytest.fixture
def enabled():
    original = SETTINGS.iv_recorder_enabled
    object.__setattr__(SETTINGS, "iv_recorder_enabled", True)
    yield
    object.__setattr__(SETTINGS, "iv_recorder_enabled", original)


def _fetcher(calls, fail=()):
    def fetch(client, underlying, preference, max_age=None):
        calls.append((underlying, max_age))
        if underlying in fail:
            raise RuntimeError("rate limited")
    return fetch


def test_due_underlyings_are_read_during_the_session_only(enabled):
    calls = []
    rec = IvRecorder(max_per_round=5)
    out = rec.run_round(["NIFTY", "BANKNIFTY"], now=OPEN, client=object(), fetch=_fetcher(calls))
    assert out["recorded"] == ["NIFTY", "BANKNIFTY"] and calls == [("NIFTY", 0), ("BANKNIFTY", 0)]  # a fresh read, not a cached one
    assert IvRecorder().run_round(["NIFTY"], now=CLOSED, client=object(), fetch=_fetcher(calls))["recorded"] == []


def test_an_underlying_is_not_read_again_inside_the_interval(enabled):
    calls = []
    rec = IvRecorder(interval_min=30)
    rec.run_round(["NIFTY"], now=OPEN, client=object(), fetch=_fetcher(calls))
    again = rec.run_round(["NIFTY"], now=OPEN, client=object(), fetch=_fetcher(calls))
    assert again["recorded"] == [] and len(calls) == 1 and "NIFTY" in rec.last_ok


def test_only_a_few_underlyings_are_read_per_round(enabled):
    calls = []
    out = IvRecorder(max_per_round=2).run_round(["A", "B", "C", "D"], now=OPEN, client=object(), fetch=_fetcher(calls))
    assert len(out["recorded"]) == 2


def test_one_failure_does_not_stop_the_others_and_is_retried_later(enabled):
    calls = []
    rec = IvRecorder(max_per_round=5, retry_after_sec=600)
    out = rec.run_round(["NIFTY", "BANKNIFTY"], now=OPEN, client=object(), fetch=_fetcher(calls, fail={"NIFTY"}))
    assert out["recorded"] == ["BANKNIFTY"] and "rate limited" in out["failed"]["NIFTY"] and "NIFTY" in rec.last_error
    assert rec.due(["NIFTY"], OPEN) == []  # backing off


def test_it_can_be_switched_off_and_stops_on_request(enabled):
    object.__setattr__(SETTINGS, "iv_recorder_enabled", False)
    calls = []
    assert IvRecorder().run_round(["NIFTY"], now=OPEN, client=object(), fetch=_fetcher(calls))["disabled"] and calls == []
    object.__setattr__(SETTINGS, "iv_recorder_enabled", True)
    stop = threading.Event()
    stop.set()
    assert IvRecorder().run_round(["NIFTY"], stop, now=OPEN, client=object(), fetch=_fetcher(calls))["recorded"] == []
