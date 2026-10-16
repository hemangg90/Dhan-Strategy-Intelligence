"""The market-data gate: each kind of bad data gets an explicit, observable verdict. Synthetic frames are test fixtures."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from quant_intelligence.data.quality import (
    QUALITY_DEGRADED, QUALITY_FAIL, QUALITY_OK, clean_ohlcv, validate_ohlcv,
)

NOW = dt.datetime(2026, 10, 16, 12, 0)  # a Friday, mid-session


def _frame(n=30, start="2026-10-16 09:15", freq="5min"):
    ts = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0})


def _check(df, **kw):
    return validate_ohlcv(df, 5, now=kw.pop("now", NOW), **kw)


def test_clean_data_is_valid_and_ok():
    r = _check(_frame())
    assert (r.status, r.category, r.issues) == (QUALITY_OK, "VALID", [])


def test_empty_data_is_unavailable():
    r = _check(_frame().iloc[0:0])
    assert r.status == QUALITY_FAIL and r.category == "UNAVAILABLE"


@pytest.mark.parametrize("column", ["timestamp", "open", "high", "low", "close", "volume"])
def test_a_missing_required_column_is_a_verdict_not_a_crash(column):
    r = _check(_frame().drop(columns=[column]))
    assert r.status == QUALITY_FAIL and r.category == "INVALID" and column in r.issues[0]


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_missing_or_infinite_price_is_invalid(column, bad):
    df = _frame()
    df.loc[10, column] = bad
    r = _check(df)
    assert r.status == QUALITY_FAIL and r.category == "INVALID" and r.n_nonfinite == 1


def test_a_non_numeric_price_is_invalid():
    df = _frame().astype({"close": object})
    df.loc[5, "close"] = "n/a"
    assert _check(df).status == QUALITY_FAIL


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_zero_and_negative_prices_are_invalid(value):
    df = _frame()
    df.loc[3, "low"] = value
    assert _check(df).status == QUALITY_FAIL


def test_high_below_low_and_close_outside_the_range_are_invalid():
    df = _frame()
    df.loc[4, "high"] = 98.0  # below low
    assert _check(df).n_ohlc_violations >= 1
    df = _frame()
    df.loc[4, "close"] = 105.0  # above high
    assert _check(df).status == QUALITY_FAIL


def test_unparseable_timestamps_are_invalid():
    df = _frame().astype({"timestamp": object})
    df.loc[2, "timestamp"] = "not a time"
    r = _check(df)
    assert r.status == QUALITY_FAIL and r.n_bad_timestamps == 1


def test_duplicate_timestamps_make_the_data_incomplete_not_invalid():
    df = pd.concat([_frame(), _frame().iloc[[5]]], ignore_index=True)
    r = _check(df)
    assert (r.status, r.category, r.n_duplicates) == (QUALITY_DEGRADED, "INCOMPLETE", 1)
    assert clean_ohlcv(df, 5)["timestamp"].is_unique  # the deterministic clean-up removes them


def test_a_missing_intra_session_candle_is_incomplete():
    df = _frame().drop(index=[10, 11]).reset_index(drop=True)
    r = _check(df)
    assert (r.status, r.category, r.n_missing_candles) == (QUALITY_DEGRADED, "INCOMPLETE", 1)


def test_overnight_and_weekend_gaps_are_not_missing_candles():
    two_days = pd.concat([_frame(20, "2026-10-15 09:15"), _frame(20, "2026-10-16 09:15")], ignore_index=True)
    assert _check(two_days).n_missing_candles == 0
    weekend = pd.concat([_frame(20, "2026-10-09 09:15"), _frame(20, "2026-10-12 09:15")], ignore_index=True)
    assert _check(weekend, now=dt.datetime(2026, 10, 12, 11, 0)).n_missing_candles == 0


def test_stale_data_is_labelled_stale_and_not_ok():
    r = validate_ohlcv(_frame(), 5, now=dt.datetime(2026, 10, 16, 15, 0))  # last bar 11:40, market open until 15:30
    assert (r.status, r.category, r.is_stale) == (QUALITY_DEGRADED, "STALE", True)
    assert any("stale" in i for i in r.issues)


def test_data_is_not_stale_just_after_the_close_when_the_last_bar_is_the_final_one():
    df = _frame(75, "2026-10-16 09:15")  # a full session: last bar starts 15:25
    assert not validate_ohlcv(df, 5, now=dt.datetime(2026, 10, 16, 16, 30)).is_stale


def test_a_timezone_shifted_feed_is_invalid():
    r = _check(_frame(30, "2026-10-16 03:45"))  # the same session expressed in UTC
    assert r.status == QUALITY_FAIL and any("outside" in i for i in r.issues)


def test_out_of_order_rows_are_reported_and_sorted_not_hidden():
    df = _frame().iloc[::-1].reset_index(drop=True)
    r = _check(df)
    assert r.n_out_of_order > 0 and r.status == QUALITY_OK and r.category == "VALID_WITH_WARNINGS"


def test_bars_at_the_wrong_timeframe_are_flagged():
    r = _check(_frame(60, freq="1min"), now=dt.datetime(2026, 10, 16, 10, 16))  # 1-minute data presented as 5-minute
    assert r.n_irregular_intervals > 0 and r.category == "VALID_WITH_WARNINGS"


def test_negative_or_missing_volume_makes_data_incomplete():
    df = _frame()
    df.loc[2, "volume"] = -5.0
    df.loc[3, "volume"] = np.nan
    r = _check(df)
    assert r.n_missing_volume == 2 and r.category == "INCOMPLETE"


def test_too_few_rows_cannot_be_trusted():
    assert _check(_frame(3)).status == QUALITY_FAIL


def test_the_report_serialises_with_its_category():
    d = _check(_frame()).as_dict()
    assert d["category"] == "VALID" and d["status"] == QUALITY_OK


@settings(max_examples=150, deadline=None)
@given(st.integers(min_value=0, max_value=29), st.sampled_from(["open", "high", "low", "close"]),
       st.sampled_from([np.nan, np.inf, -np.inf, 0.0, -3.0]))
def test_property_invalid_prices_can_never_be_rated_ok(row, column, bad):
    df = _frame()
    df.loc[row, column] = bad
    assert _check(df).status != QUALITY_OK
