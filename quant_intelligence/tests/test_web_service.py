import threading
import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from quant_intelligence.web import service
from quant_intelligence.web.app import app


def test_clean_makes_values_json_safe():
    out = service.clean({"a": np.float64("nan"), "b": np.int64(3), "c": pd.Timestamp("2026-01-02 09:15"), "d": [np.inf, 1.5]})
    assert out == {"a": None, "b": 3, "c": "2026-01-02T09:15:00", "d": [None, 1.5]}


def test_clean_keeps_a_date_index_as_a_list_of_iso_dates():
    idx = pd.DatetimeIndex(["2026-01-02 09:15", "2026-01-03 09:15"])
    assert service.clean({"x": idx, "y": np.array([1.0, np.nan])}) == {
        "x": ["2026-01-02T09:15:00", "2026-01-03T09:15:00"], "y": [1.0, None]}


def test_cache_serves_stale_result_while_refreshing_in_background():
    calls, gate = [], threading.Event()

    def runner(inst, tf, start, end):
        calls.append(1)
        if len(calls) > 1:
            gate.wait(2)  # the refresh is slow
        return f"result-{len(calls)}"

    cache = service.PipelineCache(runner=runner, ttl=0.0)
    assert cache.get("NIFTY", "5min") == "result-1"  # first call waits for a run
    t0 = time.monotonic()
    assert cache.get("NIFTY", "5min") == "result-1"  # stale result returned at once, refresh started
    assert time.monotonic() - t0 < 0.5
    gate.set()
    got = "result-1"
    for _ in range(60):
        got = cache.get("NIFTY", "5min")
        if got != "result-1":
            break
        time.sleep(0.05)
    assert got != "result-1" and len(calls) >= 2  # the background refresh replaced the stale result


def test_website_is_read_only_and_serves_the_app():
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json()["ok"] is True
    for path in ("/api/overview", "/api/risk", "/api/backtests"):
        assert client.post(path).status_code == 405
        assert client.put(path).status_code == 405
        assert client.delete(path).status_code == 405
    routes = {r.path for r in app.routes}
    assert not any("order" in p.lower() for p in routes)


def test_unknown_run_and_strategy_return_404():
    client = TestClient(app)
    assert client.get("/api/backtests/nope").status_code == 404


def test_a_visitor_arriving_during_warmup_reuses_the_warmup_result():
    """Without this the visitor waited for the warm-up and then ran the whole pipeline a second time."""
    calls = []

    def slow_runner(inst, tf, start, end):
        calls.append(1)
        time.sleep(0.4)
        return f"run-{len(calls)}"

    cache = service.PipelineCache(runner=slow_runner, ttl=60.0)
    cache.warm("NIFTY", "5min", 60)
    time.sleep(0.05)  # the warm-up is running
    t0 = time.monotonic()
    assert cache.get("NIFTY", "5min") == "run-1"
    assert len(calls) == 1 and time.monotonic() - t0 < 1.5


def test_the_cache_keeps_separate_results_per_lookback_window():
    cache = service.PipelineCache(runner=lambda i, t, s, e: (e - s).days, ttl=60.0)
    assert cache.get("NIFTY", "5min", lookback_days=30) == 30
    assert cache.get("NIFTY", "5min", lookback_days=90) == 90
