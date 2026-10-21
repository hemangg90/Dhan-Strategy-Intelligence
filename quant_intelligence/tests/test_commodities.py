"""MCX commodities: instrument resolution, session profile, segment routing and the second runner."""
import datetime as dt
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from quant_intelligence.data_adapters import dhan_instrument_master as mod
from quant_intelligence.data_adapters.dhan_adapter import _align_to_ist_session
from quant_intelligence.data.quality import out_of_session_mask
from quant_intelligence.execution import engine as engine_module
from quant_intelligence.execution import multi_cycle, position_monitor
from quant_intelligence.utils.market_calendar import most_recent_expected_bar_time
from quant_intelligence.utils.market_profile import MCX, NSE, profile_for
from quant_intelligence.utils.timeutil import is_market_open

_HEADER = (
    "SEM_EXM_EXCH_ID,SEM_SEGMENT,SEM_SMST_SECURITY_ID,SEM_INSTRUMENT_NAME,SEM_TRADING_SYMBOL,"
    "SEM_LOT_UNITS,SEM_EXPIRY_DATE,SEM_STRIKE_PRICE,SM_SYMBOL_NAME\n"
)


def _write_master(tmp_path):
    path = tmp_path / "scrip_master.csv"
    path.write_text(
        _HEADER
        # CRUDEOIL: an expired future, then two live ones (the nearer live one is the front month)
        + "MCX,M,1,FUTCOM,CRUDEOIL-01Jan2000-FUT,1.0,2000-01-01 23:30:00,0.0,CRUDEOIL\n"
        + "MCX,M,569900,FUTCOM,CRUDEOIL-19Oct2099-FUT,1.0,2099-10-19 23:30:00,0.0,CRUDEOIL\n"
        + "MCX,M,573422,FUTCOM,CRUDEOIL-19Nov2099-FUT,1.0,2099-11-19 23:30:00,0.0,CRUDEOIL\n"
        + "MCX,M,580499,OPTFUT,CRUDEOIL-15Oct2099-8000-CE,1.0,2099-10-15 23:30:00,8000,CRUDEOIL\n"
        + "MCX,M,580500,OPTFUT,CRUDEOIL-15Oct2099-8050-CE,1.0,2099-10-15 23:30:00,8050,CRUDEOIL\n"
        + "MCX,M,580501,OPTFUT,CRUDEOIL-17Nov2099-8000-CE,1.0,2099-11-17 23:30:00,8000,CRUDEOIL\n"
        # GOLD has a future but no options -> not tradeable as an option underlying
        + "MCX,M,483079,FUTCOM,GOLD-05Oct2099-FUT,1.0,2099-10-05 23:30:00,0.0,GOLD\n"
        # a same-named NSE row must be ignored
        + "NSE,D,999,OPTSTK,CRUDEOIL-Oct2099-1-CE,10,2099-10-27 14:30:00,1,CRUDEOIL\n",
        encoding="utf-8",
    )
    return path


def _patched_master(monkeypatch, path):
    monkeypatch.setattr(mod, "_CACHE_FILE", path)
    monkeypatch.setattr(mod, "_cache", None)
    monkeypatch.setattr(mod, "_fno_cache", None)
    monkeypatch.setattr(mod, "_mcx_cache", None)
    return patch.object(mod, "_cache_is_fresh", return_value=True)


def test_resolve_mcx_picks_front_month_future_and_uses_contract_multiplier(tmp_path, monkeypatch):
    with _patched_master(monkeypatch, _write_master(tmp_path)):
        info = mod.resolve_mcx("crudeoil")
    assert info["security_id"] == "569900"  # expired 2000 future skipped; Oct before Nov
    assert info["security_ids"] == ["569900", "573422"]
    assert info["seg"] == "MCX_COMM"
    assert info["strike_step"] == 50.0  # nearest option expiry (Oct), min gap 8050-8000
    assert info["lot_size"] == 100  # multiplier from config, NOT the scrip master's 1.0


def test_resolve_mcx_rejects_commodities_without_options_or_off_watchlist(tmp_path, monkeypatch):
    with _patched_master(monkeypatch, _write_master(tmp_path)):
        assert mod.resolve_mcx("GOLD") is None  # future exists but no listed options
        assert mod.resolve_mcx("ZINC") is None  # not on the watchlist


def test_get_underlying_info_routes_commodities_to_mcx(tmp_path, monkeypatch):
    from quant_intelligence.options.option_selector import get_underlying_info

    with _patched_master(monkeypatch, _write_master(tmp_path)):
        assert get_underlying_info("CRUDEOIL")["seg"] == "MCX_COMM"


def test_fetch_chain_falls_back_to_next_month_future_when_front_has_no_expiries(monkeypatch):
    from quant_intelligence.options import option_selector

    info = {"security_id": "1", "security_ids": ["1", "2"], "seg": "MCX_COMM", "strike_step": 50.0, "lot_size": 100}
    monkeypatch.setattr(option_selector, "get_underlying_info", lambda u: info)
    calls = []

    class Client:
        def get_expiry_list(self, scrip, seg):
            calls.append(("expiry", scrip, seg))
            return [] if scrip == "1" else ["2099-11-17"]

        def get_option_chain(self, scrip, seg, expiry):
            calls.append(("chain", scrip, seg, expiry))
            return {"data": {}}

    monkeypatch.setattr(option_selector, "parse_option_chain", lambda u, e, raw: (u, e))
    assert option_selector.fetch_chain(Client(), "CRUDEOIL") == ("CRUDEOIL", "2099-11-17")
    assert calls == [("expiry", "1", "MCX_COMM"), ("expiry", "2", "MCX_COMM"), ("chain", "2", "MCX_COMM", "2099-11-17")]


def test_profile_for_and_market_hours():
    assert profile_for("CRUDEOIL") is MCX and profile_for("naturalgas") is MCX
    assert profile_for("NIFTY") is NSE and profile_for("TCS") is NSE and profile_for(None) is NSE

    evening = dt.datetime(2026, 9, 29, 21, 0)  # a Tuesday
    assert is_market_open(evening, profile=MCX) and not is_market_open(evening)  # NSE default unchanged
    assert not is_market_open(dt.datetime(2026, 9, 29, 8, 59), profile=MCX)
    assert not is_market_open(dt.datetime(2026, 9, 27, 21, 0), profile=MCX)  # Sunday


def test_session_mask_and_staleness_use_the_mcx_session():
    ts = pd.Series(pd.to_datetime(["2026-09-29 09:00", "2026-09-29 20:00", "2026-09-29 23:50", "2026-09-29 08:55"]))
    assert list(out_of_session_mask(ts, MCX)) == [False, False, False, True]
    assert list(out_of_session_mask(ts))[1:3] == [True, True]  # the same bars are stray under NSE hours

    evening = dt.datetime(2026, 9, 29, 21, 0)
    assert most_recent_expected_bar_time(evening, MCX).replace(tzinfo=None) == evening  # MCX still open
    assert most_recent_expected_bar_time(evening).time() == NSE.close  # NSE already closed


def test_dhan_timestamps_are_aligned_against_the_mcx_session():
    # Dhan epochs are UTC-based: 03:30 UTC == 09:00 IST, 18:00 UTC == 23:30 IST.
    utc = pd.date_range("2026-09-29 03:30", "2026-09-29 18:00", freq="5min")
    df = pd.DataFrame({"timestamp": utc, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    out = _align_to_ist_session(df, MCX)
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-09-29 09:00")
    assert out["timestamp"].iloc[-1] == pd.Timestamp("2026-09-29 23:30")


def test_evening_tail_refresh_is_shifted_to_ist_not_kept_as_utc():
    # A 15:40-19:45 IST tail arrives as 10:10-14:15 UTC, which also fits inside 09:00-23:55 unshifted.
    # Keeping it unshifted filed evening bars 5h30m early and left MCX data stale all evening.
    utc = pd.date_range("2026-10-01 10:10", "2026-10-01 14:15", freq="5min")
    df = pd.DataFrame({"timestamp": utc, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    out = _align_to_ist_session(df, MCX, end=dt.datetime(2026, 10, 1, 19, 47))
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-10-01 15:40")
    assert out["timestamp"].iloc[-1] == pd.Timestamp("2026-10-01 19:45")


def test_ist_labelled_evening_bars_are_not_pushed_into_the_future():
    ist = pd.date_range("2026-10-01 15:40", "2026-10-01 19:40", freq="5min")
    df = pd.DataFrame({"timestamp": ist, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    out = _align_to_ist_session(df, MCX, end=dt.datetime(2026, 10, 1, 19, 47))
    assert out["timestamp"].iloc[-1] == pd.Timestamp("2026-10-01 19:40")


def test_tail_rows_older_than_the_request_never_overwrite_the_cache():
    from quant_intelligence.data.data_manager import DataManager

    cached = pd.DataFrame({"timestamp": pd.date_range("2026-10-01 09:00", "2026-10-01 15:50", freq="5min"),
                           "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    stale_tail = pd.DataFrame({"timestamp": pd.date_range("2026-10-01 10:10", periods=5, freq="5min"),
                               "open": 9.0, "high": 9.0, "low": 9.0, "close": 9.0, "volume": 9.0})
    dm = DataManager()
    dm._fetch = lambda *a, **k: ("dhan", stale_tail)
    merged = dm._refresh_tail("CRUDEOIL", "5min", cached, dt.datetime(2026, 10, 1, 15, 50), 5,
                              dt.datetime(2026, 10, 1, 19, 47), None)
    assert merged["close"].eq(1.0).all() and len(merged) == len(cached)


def test_option_orders_and_ltp_use_the_commodity_segment():
    positions = [
        {"security_id": "580499", "underlying": "CRUDEOIL", "option_type": "CE"},
        {"security_id": "42", "underlying": "NIFTY", "option_type": "PE"},
    ]
    seen = {}

    class Client:
        def get_ltp(self, payload):
            seen.update(payload)
            return {"data": {"MCX_COMM": {"580499": {"last_price": 9.5}}, "NSE_FNO": {"42": {"last_price": 3.0}}}}

    quotes = position_monitor._fetch_quotes(Client(), positions)
    assert seen == {"MCX_COMM": [580499], "NSE_FNO": [42]}
    assert quotes["580499"]["last_price"] == 9.5 and quotes["42"]["last_price"] == 3.0


def test_multi_cycle_tags_commodities_and_reports_fills_under_the_lock(monkeypatch):
    class Lock:
        held = False

        def __enter__(self):
            Lock.held = True

        def __exit__(self, *exc):
            Lock.held = False

    fills = []
    filled = SimpleNamespace(
        order=SimpleNamespace(status="FILLED"), strategy_name="S", setup_status="SETUP_TRIGGERED",
        reason="ok", capital_required=10.0, capital_used=10.0, tag=None,
    )
    monkeypatch.setattr(
        multi_cycle, "run_pipeline",
        lambda *a: SimpleNamespace(ranking=SimpleNamespace(tie_break=False, is_no_trade=False, selected_strategy="S"), ohlcv=None),
    )

    def fake_cycle(*a, **k):
        assert Lock.held  # order placement must happen inside the shared broker lock
        return filled

    monkeypatch.setattr(multi_cycle, "run_auto_option_cycle", fake_cycle)

    class Client:
        def is_configured(self):
            return False

    rows, _ = multi_cycle.run_multi_instrument_cycle(
        [], ["CRUDEOIL", "TCS"], "5min", 30, broker=None, client=Client(), get_account=lambda: None,
        broker_lock=Lock(), on_fill=lambda: fills.append(1),
    )
    assert [r["type"] for r in rows] == ["commodity", "stock"]
    assert len(fills) == 2


def _fake_cycle(monkeypatch, calls, delay=0.0):
    def fake(index, stocks, tf, lb, broker, client, get_account, **kwargs):
        calls.append((threading.current_thread().name, list(stocks)))
        time.sleep(delay)
        return [{"symbol": "X", "status": "NO_TRADE"}], 0.0

    monkeypatch.setattr(engine_module, "run_multi_instrument_cycle", fake)
    monkeypatch.setattr(engine_module, "DhanApiClient", lambda: object())


def test_nse_and_commodity_runners_run_concurrently_and_independently(monkeypatch):
    calls = []
    _fake_cycle(monkeypatch, calls, delay=0.3)  # a slow scan must not block the other runner
    monkeypatch.setattr(engine_module, "is_market_open", lambda **kw: True)
    nse = engine_module.TradingEngine()
    mcx = engine_module.TradingEngine("commodities", MCX)
    assert mcx.owner is mcx and mcx.broker is not nse.broker  # separate paper accounts
    assert mcx.lock is not nse.lock and mcx.profile is MCX

    nse.start([], ["TCS"], "5min", 30, interval_seconds=0.05, market_hours_only=False)
    mcx.start([], ["CRUDEOIL"], "5min", 30, interval_seconds=0.05, market_hours_only=False)
    deadline = time.time() + 5
    while (nse.cycles < 1 or mcx.cycles < 1) and time.time() < deadline:
        time.sleep(0.02)
    try:
        assert nse.status()["running"] and mcx.status()["running"]
        assert nse.cycles >= 1 and mcx.cycles >= 1
        names = {c[0] for c in calls}
        assert {"auto-paper-trader-nse", "auto-paper-trader-commodities"} <= names

        mcx.stop()  # stopping one leaves the other running
        assert not mcx.running and nse.running
    finally:
        nse.stop()
        mcx.stop()


def test_kill_switch_is_per_market_and_gate_uses_each_runners_own_session(monkeypatch):
    calls = []
    _fake_cycle(monkeypatch, calls)
    seen_profiles = []
    monkeypatch.setattr(engine_module, "is_market_open", lambda profile=NSE, **kw: seen_profiles.append(profile) or profile is MCX)
    monkeypatch.setattr(engine_module, "in_close_window", lambda *a: False)
    nse = engine_module.TradingEngine()
    mcx = engine_module.TradingEngine("commodities", MCX)

    nse.start([], ["TCS"], "5min", 30, interval_seconds=0.05)
    mcx.start([], ["CRUDEOIL"], "5min", 30, interval_seconds=0.05)
    deadline = time.time() + 5
    while mcx.cycles < 1 and time.time() < deadline:
        time.sleep(0.02)
    try:
        assert mcx.cycles >= 1 and nse.cycles == 0  # NSE closed, MCX open
        assert "waiting for 09:15" in nse.last_status

        nse.kill_switch = True  # the NSE kill switch is NSE's own: the MCX runner keeps scanning
        time.sleep(0.4)
        done = mcx.cycles
        time.sleep(0.4)
        assert mcx.cycles > done

        mcx.kill_switch = True  # the MCX switch halts MCX
        time.sleep(0.4)
        done = mcx.cycles
        time.sleep(0.4)
        assert mcx.cycles == done and "Kill switch" in mcx.last_status
    finally:
        nse.stop()
        mcx.stop()
    assert NSE in seen_profiles and MCX in seen_profiles


def test_option_chain_throttle_is_shared_across_clients(monkeypatch):
    from quant_intelligence.brokers import dhan_api_client as client_module
    from quant_intelligence.brokers.dhan_rate_limit import LIMITER

    sleeps = []
    monkeypatch.setattr(LIMITER, "sleep", lambda s: sleeps.append(s))
    a, b = client_module.DhanApiClient("id", "tok"), client_module.DhanApiClient("id", "tok")
    a._throttle_option_chain()
    b._throttle_option_chain()  # a different client, immediately after
    assert sleeps and sleeps[-1] > 2.0


def test_inconsistent_session_opening_bar_is_dropped_but_mid_session_violations_still_fail():
    from quant_intelligence.data.quality import QUALITY_FAIL, clean_ohlcv, validate_ohlcv

    def frame(bad_at):
        ts = list(pd.date_range("2026-09-07 09:00", periods=12, freq="5min")) + \
            list(pd.date_range("2026-09-08 09:00", periods=12, freq="5min"))
        df = pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0})
        df.loc[bad_at, "open"] = 95.0  # open below the bar's own low
        return df

    opening = frame(12)  # 09:00 on the second day: Dhan's opening-call quirk
    cleaned = clean_ohlcv(opening, 5, MCX)
    assert len(cleaned) == 23 and validate_ohlcv(cleaned, 5, now=dt.datetime(2026, 9, 8, 10, 0), profile=MCX).status != QUALITY_FAIL

    mid = clean_ohlcv(frame(15), 5, MCX)  # same violation mid-session is left in
    assert len(mid) == 24
    assert validate_ohlcv(mid, 5, now=dt.datetime(2026, 9, 8, 10, 0), profile=MCX).status == QUALITY_FAIL
