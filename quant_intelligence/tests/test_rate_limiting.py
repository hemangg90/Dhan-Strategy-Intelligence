"""Staying under Dhan's account-wide limits: spacing, 429 handling, circuit breaker, shared caches,
lazy chain fetching and bar-close candle refresh. No network: requests/clocks are faked."""
import datetime as dt
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant_intelligence.brokers import dhan_api_client as client_module
from quant_intelligence.brokers.dhan_api_client import DhanApiClient, DhanApiError, DhanRateLimited
from quant_intelligence.brokers.dhan_cache import QuoteCache, TtlCache
from quant_intelligence.brokers.dhan_rate_limit import LIMITER, RateLimiter
from quant_intelligence.config.settings import DhanLimits


class FakeTime:
    def __init__(self):
        self.t = 1000.0

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _resp(status, payload=None, headers=None):
    r = MagicMock(ok=status < 400, status_code=status, text=str(payload))
    r.json.return_value = payload if payload is not None else {}
    r.headers = headers or {}
    return r


# ---------------------------------------------------------------- limiter: spacing
def test_calls_in_one_category_are_spaced_by_the_minimum_interval():
    ft = FakeTime()
    lim = RateLimiter(DhanLimits(), clock=ft.clock, sleep=ft.sleep)
    start = ft.t
    for _ in range(5):
        lim.acquire("quote")
    assert ft.t - start == pytest.approx(4 * 1.25)  # Dhan allows 1 quote request/s; we use 1.25 s


def test_categories_do_not_slow_each_other_down():
    ft = FakeTime()
    lim = RateLimiter(DhanLimits(), clock=ft.clock, sleep=ft.sleep)
    lim.acquire("option_chain")
    before = ft.t
    lim.acquire("data")
    lim.acquire("non_trading")
    assert ft.t == before  # no waiting: separate budgets


def test_default_spacing_is_inside_every_published_dhan_limit():
    d = DhanLimits()
    assert d.min_interval_quote > 1.0  # limit 1/s
    assert d.min_interval_data > 1 / 5  # limit 5/s
    assert d.min_interval_non_trading > 1 / 20  # limit 20/s
    assert d.min_interval_orders > 1 / 10  # limit 10/s
    assert d.min_interval_option_chain > 3.0  # limit 1 per 3 s


def test_snapshot_reports_usage_per_category():
    ft = FakeTime()
    lim = RateLimiter(DhanLimits(), clock=ft.clock, sleep=ft.sleep)
    for _ in range(3):
        lim.acquire("data")
    snap = lim.snapshot()
    assert snap["categories"]["data"]["calls_last_window"] == 3
    assert snap["categories"]["quote"]["calls_last_window"] == 0


# ---------------------------------------------------------------- circuit breaker
def test_breaker_opens_after_consecutive_429s_blocks_reads_but_never_orders():
    ft = FakeTime()
    lim = RateLimiter(DhanLimits(), clock=ft.clock, sleep=ft.sleep)
    assert not lim.record_429("quote") and not lim.record_429("quote")
    assert lim.record_429("quote")  # third in a row opens it
    assert lim.cooldown_remaining("quote") == pytest.approx(30.0)
    assert lim.cooldown_remaining("option_chain") > 0  # reads share the pause
    assert lim.cooldown_remaining("orders") == 0  # exits must always be able to go out
    ft.t += 31
    assert lim.cooldown_remaining("quote") == 0  # auto-resumes


def test_breaker_is_half_open_and_cooldown_doubles_then_caps():
    ft = FakeTime()
    lim = RateLimiter(DhanLimits(), clock=ft.clock, sleep=ft.sleep)
    for _ in range(3):
        lim.record_429("data")
    ft.t += 31
    assert lim.record_429("data")  # ONE more 429 after the pause re-opens it immediately
    assert lim.cooldown_remaining("data") == pytest.approx(60.0)
    ft.t += 61
    lim.record_429("data")
    ft.t += 1
    assert lim.cooldown_remaining("data") <= 120.0  # capped


def test_a_success_resets_the_consecutive_count():
    ft = FakeTime()
    lim = RateLimiter(DhanLimits(), clock=ft.clock, sleep=ft.sleep)
    lim.record_429("quote")
    lim.record_429("quote")
    lim.record_success("quote")
    assert not lim.record_429("quote")  # back to 1, breaker stays closed


# ---------------------------------------------------------------- client behaviour
def test_during_cooldown_reads_fail_fast_with_no_network_call_but_orders_still_go():
    client = DhanApiClient("id", "tok")
    for _ in range(3):
        LIMITER.record_429("quote")
    with patch("quant_intelligence.brokers.dhan_api_client.requests.post") as post:
        with pytest.raises(DhanRateLimited) as exc:
            client.get_ltp({"NSE_FNO": [1]})
        assert post.call_count == 0  # nothing sent to Dhan while paused
        assert exc.value.status_code == 429 and exc.value.retry_in > 0

        post.return_value = _resp(200, {"status": "success"})
        with patch.object(LIMITER, "sleep"):
            client.place_order({"x": 1})
        assert post.call_count == 1  # orders are exempt


def test_orders_are_throttled_but_never_retried_on_429():
    client = DhanApiClient("id", "tok")
    with patch("quant_intelligence.brokers.dhan_api_client.requests.post", return_value=_resp(429, {"remarks": "slow"})) as post, \
            patch("quant_intelligence.brokers.dhan_api_client.time.sleep") as sleep, \
            patch.object(LIMITER, "sleep"):
        with pytest.raises(DhanApiError) as exc:
            client.place_order({"x": 1})
    assert exc.value.status_code == 429
    assert post.call_count == 1 and sleep.call_count == 0  # a retried order could be placed twice


def test_retry_after_header_is_honoured_over_the_default_backoff():
    client = DhanApiClient("id", "tok")
    responses = [_resp(429, {}, {"Retry-After": "11"}), _resp(200, {"data": {}})]
    with patch("quant_intelligence.brokers.dhan_api_client.requests.post", side_effect=responses), \
            patch("quant_intelligence.brokers.dhan_api_client.time.sleep") as sleep, \
            patch.object(LIMITER, "sleep"):
        client.get_ltp({"NSE_FNO": [1]})
    assert sleep.call_args.args[0] == 11.0  # > the 6 s default backoff


def test_every_endpoint_is_routed_to_its_dhan_category():
    seen = []

    def fake_acquire(category):
        seen.append(category)

    client = DhanApiClient("id", "tok")
    ok = _resp(200, {"data": []})
    with patch.object(LIMITER, "acquire", side_effect=fake_acquire), \
            patch("quant_intelligence.brokers.dhan_api_client.requests.post", return_value=ok), \
            patch("quant_intelligence.brokers.dhan_api_client.requests.get", return_value=ok):
        client.get_ltp({"NSE_FNO": [1]})
        client.get_fund_limit()
        client.get_positions()
        client.get_intraday_minute("1", "NSE_EQ", "EQUITY", "5", "a", "b")
        client.get_historical_daily("1", "NSE_EQ", "EQUITY", "a", "b")
        client.get_option_chain(13, "IDX_I", "2026-10-06")
        client.place_order({})
    assert seen == ["quote", "non_trading", "non_trading", "data", "data", "option_chain", "orders"]


# ---------------------------------------------------------------- shared caches
class CountingClient:
    def __init__(self, delay=0.0, fail=False):
        self.calls = []
        self.delay, self.fail = delay, fail

    def get_ltp(self, by_segment):
        self.calls.append({k: list(v) for k, v in by_segment.items()})
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise DhanRateLimited("paused", retry_in=30)
        return {"data": {seg: {str(i): {"last_price": 100.0 + i} for i in ids} for seg, ids in by_segment.items()}}


def test_quote_cache_serves_repeat_requests_from_one_call():
    cache, client = QuoteCache(), CountingClient()
    first = cache.get_quotes(client, {"NSE_FNO": [11]}, max_age=60)
    second = cache.get_quotes(client, {"NSE_FNO": [11]}, max_age=60)
    assert len(client.calls) == 1 and first == second
    assert first["NSE_FNO"]["11"]["last_price"] == 111.0


def test_many_threads_asking_at_once_cause_a_single_dhan_request():
    """Ten open browser tabs refreshing together must cost one quote request, not ten."""
    cache, client = QuoteCache(), CountingClient(delay=0.2)
    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.get_quotes(client, {"NSE_FNO": [5]}, max_age=60)))
               for _ in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(client.calls) == 1
    assert len(results) == 10 and all(r["NSE_FNO"]["5"]["last_price"] == 105.0 for r in results)


def test_quote_cache_batches_what_other_callers_recently_asked_for():
    cache, client = QuoteCache(), CountingClient()
    cache.get_quotes(client, {"NSE_FNO": [1]}, max_age=60)
    cache.get_quotes(client, {"NSE_FNO": [2]}, max_age=60)  # miss: also refreshes id 1 in the same request
    assert sorted(client.calls[1]["NSE_FNO"]) == [1, 2]
    cache.get_quotes(client, {"NSE_FNO": [1, 2]}, max_age=60)
    assert len(client.calls) == 2  # both now cached


def test_quote_cache_max_age_zero_always_refetches():
    cache, client = QuoteCache(), CountingClient()
    cache.get_quotes(client, {"NSE_FNO": [1]}, max_age=0)
    cache.get_quotes(client, {"NSE_FNO": [1]}, max_age=0)
    assert len(client.calls) == 2


def test_quote_cache_falls_back_to_recent_prices_only_when_allowed():
    ft = FakeTime()
    cache = QuoteCache(clock=ft.clock)
    client = CountingClient()
    cache.get_quotes(client, {"NSE_FNO": [1]}, max_age=60)
    ft.t += 10
    client.fail = True
    shown = cache.get_quotes(client, {"NSE_FNO": [1]}, max_age=2, stale_ok_for=30)  # display: stale is fine
    assert shown["NSE_FNO"]["1"]["last_price"] == 101.0
    with pytest.raises(DhanRateLimited):
        cache.get_quotes(client, {"NSE_FNO": [1]}, max_age=2)  # exits: never act on stale prices


def test_ttl_cache_single_flight_and_errors_are_not_cached():
    cache, calls = TtlCache(), []

    def slow_loader():
        calls.append(1)
        time.sleep(0.15)
        return "chain"

    out = []
    threads = [threading.Thread(target=lambda: out.append(cache.get("k", slow_loader, 60))) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1 and out == ["chain"] * 8

    def boom():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        cache.get("other", boom, 60)
    assert cache.get("other", lambda: "ok", 60) == "ok"  # the failure was not remembered


def test_fetch_chain_is_cached_per_underlying_and_can_be_forced_fresh(monkeypatch):
    from quant_intelligence.options import option_selector

    calls = []
    monkeypatch.setattr(option_selector, "_fetch_chain_uncached", lambda c, u, p: calls.append(u) or SimpleNamespace(u=u))
    client = SimpleNamespace(base_url="x")
    option_selector.fetch_chain(client, "NIFTY")
    option_selector.fetch_chain(client, "nifty")  # same underlying, any case
    option_selector.fetch_chain(client, "BANKNIFTY")
    assert calls == ["NIFTY", "BANKNIFTY"]
    option_selector.fetch_chain(client, "NIFTY", max_age=0)  # e.g. just before a live order
    assert calls == ["NIFTY", "BANKNIFTY", "NIFTY"]


# ---------------------------------------------------------------- lazy chain fetching
def _output(no_trade=True, strategy="Opening Range Breakout"):
    return SimpleNamespace(
        ranking=SimpleNamespace(tie_break=False, is_no_trade=no_trade, selected_strategy=None if no_trade else strategy,
                                reason="no edge"),
        ohlcv=SimpleNamespace(tail=lambda n: None),
        market_state=SimpleNamespace(get=lambda k, d=None: d),
        data_quality_status="OK",
    )


def test_a_scan_of_25_instruments_fetches_chains_only_for_triggered_setups(monkeypatch):
    from quant_intelligence.execution import multi_cycle

    symbols = [f"S{i}" for i in range(25)]
    triggered = {"S3", "S17"}
    fetched = []
    no_trade = SimpleNamespace(order=None, strategy_name=None, setup_status=None, reason="x",
                               capital_required=None, capital_used=0.0, tag=None)
    monkeypatch.setattr(multi_cycle, "run_pipeline", lambda sym, *a: SimpleNamespace(sym=sym, ranking=SimpleNamespace(
        tie_break=False, is_no_trade=sym not in triggered, selected_strategy="ORB")))
    monkeypatch.setattr(multi_cycle, "chain_needed", lambda out, sym, broker: sym in triggered)
    monkeypatch.setattr(multi_cycle, "fetch_chain", lambda c, sym, pref: fetched.append(sym) or f"chain-{sym}")
    monkeypatch.setattr(multi_cycle, "run_auto_option_cycle", lambda *a, **k: no_trade)
    monkeypatch.setattr(multi_cycle, "monitor_option_positions", lambda *a, **k: [])
    client = SimpleNamespace(is_configured=lambda: True)

    rows, _ = multi_cycle.run_multi_instrument_cycle(
        symbols[:17], symbols[17:], "5min", 30, broker=None, client=client, get_account=lambda: None
    )
    assert len(rows) == 25
    assert sorted(fetched) == ["S17", "S3"]  # 2 chain requests instead of 25


def test_chain_error_for_a_triggered_instrument_is_reported_and_the_scan_continues(monkeypatch):
    from quant_intelligence.execution import multi_cycle

    no_trade = SimpleNamespace(order=None, strategy_name=None, setup_status=None, reason="x",
                               capital_required=None, capital_used=0.0, tag=None)
    monkeypatch.setattr(multi_cycle, "run_pipeline", lambda sym, *a: SimpleNamespace(ranking=SimpleNamespace(
        tie_break=False, is_no_trade=False, selected_strategy="ORB")))
    monkeypatch.setattr(multi_cycle, "chain_needed", lambda *a: True)

    def fetch(c, sym, pref):
        if sym == "A":
            raise DhanRateLimited("paused", retry_in=30)
        return "chain"

    monkeypatch.setattr(multi_cycle, "fetch_chain", fetch)
    monkeypatch.setattr(multi_cycle, "run_auto_option_cycle", lambda *a, **k: no_trade)
    monkeypatch.setattr(multi_cycle, "monitor_option_positions", lambda *a, **k: [])
    rows, _ = multi_cycle.run_multi_instrument_cycle(
        ["A", "B"], [], "5min", 30, broker=None, client=SimpleNamespace(is_configured=lambda: True), get_account=lambda: None
    )
    assert [r["status"] for r in rows][0] == "CHAIN_ERROR" and len(rows) == 2


def test_exits_are_checked_before_the_scan_and_survive_a_failing_instrument(monkeypatch):
    from quant_intelligence.execution import multi_cycle

    calls = []
    monkeypatch.setattr(multi_cycle, "run_pipeline", lambda sym, *a: SimpleNamespace(ranking=SimpleNamespace(
        tie_break=False, is_no_trade=False, selected_strategy="ORB")))
    monkeypatch.setattr(multi_cycle, "chain_needed", lambda *a: False)

    def boom(*a, **k):
        calls.append("scan")
        raise RuntimeError("order path failed")

    monkeypatch.setattr(multi_cycle, "run_auto_option_cycle", boom)
    monkeypatch.setattr(multi_cycle, "monitor_option_positions", lambda *a, **k: calls.append("exits") or [])
    rows, _ = multi_cycle.run_multi_instrument_cycle(
        ["A", "B"], [], "5min", 30, broker=None, client=SimpleNamespace(is_configured=lambda: True), get_account=lambda: None
    )
    assert calls == ["exits", "scan", "scan", "exits"]  # before the scan, and again after it despite both failing
    assert [r["status"] for r in rows] == ["CYCLE_ERROR", "CYCLE_ERROR"]


def test_chain_needed_is_false_for_no_trade_and_open_positions_true_for_chain_strategies(monkeypatch):
    from quant_intelligence.execution import auto_trader
    from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES

    broker = SimpleNamespace(get_open_positions=lambda: [])
    assert not auto_trader.chain_needed(_output(no_trade=True), "NIFTY", broker)

    busy = SimpleNamespace(get_open_positions=lambda: [{"instrument": "NIFTY"}])
    assert not auto_trader.chain_needed(_output(no_trade=False), "NIFTY", busy)

    chain_name = next(iter(CHAIN_AWARE_STRATEGY_NAMES))
    assert auto_trader.chain_needed(_output(no_trade=False, strategy=chain_name), "NIFTY", broker)

    monkeypatch.setattr(auto_trader, "get_strategy", lambda n: object())
    for status, expected in ((auto_trader.SETUP_TRIGGERED, True), (auto_trader.WAITING_FOR_SETUP, False)):
        monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, _s=status, **k: SimpleNamespace(status=_s))
        assert auto_trader.chain_needed(_output(no_trade=False), "NIFTY", broker) is expected


def test_the_chain_provider_is_never_called_when_there_is_nothing_to_trade():
    from quant_intelligence.execution.auto_trader import run_auto_option_cycle

    called = []
    result = run_auto_option_cycle(_output(no_trade=True), "NIFTY", SimpleNamespace(get_open_positions=lambda: []),
                                   client=None, chain=lambda: called.append(1), account=None)
    assert called == [] and "NO TRADE" in result.reason


# ---------------------------------------------------------------- candles refresh on bar close
@pytest.fixture
def dm(monkeypatch, tmp_path):
    from quant_intelligence.data import data_manager as dm_module

    monkeypatch.setattr(dm_module, "CACHE_DIR", tmp_path)
    dm_module._last_attempt.clear()
    dm_module._last_refresh_error.clear()
    manager = dm_module.DataManager()
    manager.fetches = []
    return manager, dm_module, tmp_path


def _bars(first, last):
    ts = pd.date_range(first, last, freq="5min")
    return pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0})


def _seed(tmp_path, first, last):
    _bars(first, last).to_parquet(tmp_path / "TCS_5min.parquet", index=False)


END = dt.datetime(2026, 9, 29, 10, 12)  # bars through 10:05 have closed; 10:10 is still forming
START = END - dt.timedelta(days=2)


def test_up_to_date_cache_makes_no_request(dm):
    manager, _, tmp = dm
    _seed(tmp, "2026-09-29 09:15", "2026-09-29 10:05")
    manager._fetch = lambda *a: manager.fetches.append(a) or ("dhan", pd.DataFrame())
    df, meta = manager.get_ohlcv("TCS", "5min", START, END)
    assert manager.fetches == [] and meta["source"] == "cache"
    assert df["timestamp"].max() == pd.Timestamp("2026-09-29 10:05")


def test_a_closed_bar_triggers_one_small_tail_request_that_is_merged(dm):
    manager, _, tmp = dm
    _seed(tmp, "2026-09-29 09:15", "2026-09-29 09:55")  # 10:00 and 10:05 have since closed

    def fake_fetch(instrument, timeframe, start, end, prefer):
        manager.fetches.append((start, end))
        return "dhan", _bars("2026-09-29 09:50", "2026-09-29 10:05")  # overlaps cache on purpose

    manager._fetch = fake_fetch
    df, meta = manager.get_ohlcv("TCS", "5min", START, END)
    assert len(manager.fetches) == 1
    assert manager.fetches[0][0] == pd.Timestamp("2026-09-29 09:45")  # last cached bar minus two bars: a tail, not 60 days
    assert df["timestamp"].max() == pd.Timestamp("2026-09-29 10:05")
    assert df["timestamp"].is_unique and meta["source"] == "cache+tail"
    assert pd.read_parquet(tmp / "TCS_5min.parquet")["timestamp"].max() == pd.Timestamp("2026-09-29 10:05")  # persisted


def test_repeat_callers_do_not_re_request_while_the_new_bar_is_not_published(dm):
    manager, _, tmp = dm
    _seed(tmp, "2026-09-29 09:15", "2026-09-29 09:55")
    manager._fetch = lambda *a: manager.fetches.append(a) or ("dhan", pd.DataFrame())  # bar not out yet
    for _ in range(5):  # five cycles / tabs in a row
        manager.get_ohlcv("TCS", "5min", START, END)
    assert len(manager.fetches) == 1


def test_a_failed_refresh_serves_cached_bars_instead_of_raising(dm):
    manager, _, tmp = dm
    _seed(tmp, "2026-09-29 09:15", "2026-09-29 09:55")
    manager._fetch = lambda *a: manager.fetches.append(a) or ("none", pd.DataFrame())  # e.g. rate limited
    df, meta = manager.get_ohlcv("TCS", "5min", START, END)
    assert meta["source"] == "cache" and df["timestamp"].max() == pd.Timestamp("2026-09-29 09:55")


def test_a_very_old_cache_is_rebuilt_in_full_not_patched(dm):
    manager, _, tmp = dm
    _seed(tmp, "2026-07-01 09:15", "2026-07-01 15:25")

    def fake_fetch(instrument, timeframe, start, end, prefer):
        manager.fetches.append((start, end))
        return "dhan", _bars("2026-09-27 09:15", "2026-09-29 10:05")

    manager._fetch = fake_fetch
    manager.get_ohlcv("TCS", "5min", dt.datetime(2026, 6, 25), END)
    assert manager.fetches[0][0] == dt.datetime(2026, 6, 25)  # the original, full window


# ---------------------------------------------------------------- rejected credentials (HTTP 401)
def test_a_401_stops_all_further_requests_with_that_token_until_it_changes():
    client = DhanApiClient("id", "old-token")
    rejected = _resp(401, {"data": {"808": "Authentication Failed"}, "status": "failed"})
    with patch("quant_intelligence.brokers.dhan_api_client.requests.post", return_value=rejected) as post, \
            patch.object(LIMITER, "sleep"):
        with pytest.raises(DhanApiError) as first:
            client.get_ltp({"NSE_FNO": [1]})
        assert first.value.status_code == 401 and post.call_count == 1

        for call in (lambda: client.get_ltp({"NSE_FNO": [1]}),
                     lambda: client.get_intraday_minute("1", "NSE_EQ", "EQUITY", "5", "a", "b"),
                     lambda: client.get_option_chain(13, "IDX_I", "2026-10-06")):
            with pytest.raises(DhanApiError) as e:
                call()
            assert e.value.status_code == 401 and "not sending requests" in str(e.value)
        assert post.call_count == 1  # 25 instruments x every cycle no longer hammer Dhan with a dead token

        fresh = DhanApiClient("id", "new-token")  # user pasted a new token: works straight away
        post.return_value = _resp(200, {"data": {}})
        fresh.get_ltp({"NSE_FNO": [1]})
        assert post.call_count == 2


def test_the_auth_block_expires_on_its_own():
    ft = FakeTime()
    lim = RateLimiter(DhanLimits(), clock=ft.clock, sleep=ft.sleep)
    lim.record_auth_failure("t")
    assert lim.auth_block_remaining("t") == pytest.approx(300.0)
    assert lim.auth_block_remaining("other") == 0.0
    ft.t += 301
    assert lim.auth_block_remaining("t") == 0.0


def test_the_snapshot_stops_reporting_a_block_once_the_token_is_replaced():
    ft = FakeTime()
    lim = RateLimiter(DhanLimits(), clock=ft.clock, sleep=ft.sleep)
    lim.record_auth_failure("old")
    assert lim.snapshot(token="old")["auth_block_remaining"] == pytest.approx(300.0)
    assert lim.snapshot(token="new")["auth_block_remaining"] == 0.0  # the red banner must clear when a new token is saved
    assert lim.snapshot()["auth_block_remaining"] == pytest.approx(300.0)  # no token given: unchanged behaviour


# ---------------------------------------------------------------- "why is the data degraded?" must be answerable
def test_a_failed_refresh_is_reported_as_the_reason_the_data_is_old(dm):
    manager, _, tmp = dm
    _seed(tmp, "2026-09-29 09:15", "2026-09-29 09:55")

    def failing_fetch(*a):
        manager._last_errors = ["Dhan rejected your credentials (401)"]
        return "none", pd.DataFrame()

    manager._fetch = failing_fetch
    df, meta = manager.get_ohlcv("TCS", "5min", START, END)
    issues = meta["quality_report"]["issues"]
    assert any("latest refresh failed: Dhan rejected your credentials (401)" in i for i in issues)
    assert meta["quality_status"] != "OK"

    # a second caller inside the retry-spacing window still sees the same explanation
    _, meta2 = manager.get_ohlcv("TCS", "5min", START, END)
    assert any("latest refresh failed" in i for i in meta2["quality_report"]["issues"])


def test_stale_message_is_human_readable():
    from quant_intelligence.data.quality import validate_ohlcv

    ts = pd.date_range("2026-09-29 09:15", periods=60, freq="5min")
    df = pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0})
    report = validate_ohlcv(df, 5, now=dt.datetime(2026, 10, 1, 16, 42))
    stale = next(i for i in report.issues if "stale" in i)
    assert "29 Sep 14:10" in stale and "01 Oct 15:30" in stale and "2d" in stale  # NSE closed at 15:30
    assert "+05:30" not in stale and "000" not in stale  # no raw timestamps / microseconds


def test_no_trade_reason_names_the_data_problem(monkeypatch):
    from quant_intelligence.data.data_manager import DataManager
    from quant_intelligence.data_adapters.synthetic import SyntheticAdapter
    from quant_intelligence.research.pipeline import run_pipeline

    end = dt.datetime(2024, 6, 28, 15, 30)
    start = end - dt.timedelta(days=40)
    frame = SyntheticAdapter().get_ohlcv("NIFTY", "5min", start, end)
    meta = {"quality_status": "DEGRADED", "quality_report": {
        "issues": ["data is stale: last bar 29 Sep 15:00, 2d 1h behind the 01 Oct 16:42 IST market clock",
                   "latest refresh failed: Dhan rejected your credentials (401)"]}}
    monkeypatch.setattr(DataManager, "get_ohlcv", lambda self, *a, **k: (frame, meta))
    out = run_pipeline("NIFTY", "5min", start, end)
    assert out.ranking.is_no_trade
    assert "DEGRADED" in out.ranking.reason and "Dhan rejected your credentials" in out.ranking.reason
    assert out.data_quality_issues == meta["quality_report"]["issues"]
