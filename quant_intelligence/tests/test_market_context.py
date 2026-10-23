"""VWAP and volume-profile features, the shared data status, and the typed MarketContext."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.features.vwap_volume import APPROXIMATION, volume_profile, vwap_features
from quant_intelligence.intelligence.market_context import (
    MIN_BARS,
    build_market_context,
    context_from_pipeline,
    validate_ohlcv_for_context,
)
from quant_intelligence.intelligence.status import CRITICAL, OK, WARNING, DataStatus, critical, good, warning, worst


def _walk(n=240, seed=11, volume=True, start="2030-01-02 09:15"):
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 0.4, n))
    rows, prev = [], closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + abs(rng.normal(0, 0.2)), min(prev, c) - abs(rng.normal(0, 0.2)), c))
        prev = c
    stamps = []
    day, per_day = pd.Timestamp(start), 75
    for i in range(n):
        stamps.append(day + pd.Timedelta(days=i // per_day) + pd.Timedelta(minutes=5 * (i % per_day)))
    df = pd.DataFrame({"timestamp": stamps, "open": [r[0] for r in rows], "high": [r[1] for r in rows],
                       "low": [r[2] for r in rows], "close": [r[3] for r in rows],
                       "volume": rng.integers(500, 1500, n) if volume else 0})
    return df


# ------------------------------------------------------------------ status
def test_status_levels_and_combination():
    assert good("a").ok and good("a").clean
    assert warning("a", "slow").ok and not warning("a", "slow").clean
    assert not critical("a", "no data").ok
    combined = worst("ctx", [good("a"), warning("b", "stale", ("x",)), critical("c", "gone", ("y", "x"))])
    assert combined.level == CRITICAL and combined.unavailable == ("x", "y") and "b: stale" in combined.reason
    assert worst("ctx", []).level == CRITICAL
    with pytest.raises(ValueError):
        DataStatus("a", "MAYBE")
    assert "unavailable: x" in str(warning("a", "r", ("x",)))


# ------------------------------------------------------------------ VWAP
def test_vwap_matches_a_hand_calculation_and_resets_each_session():
    stamps = list(pd.date_range("2030-01-02 09:15", periods=3, freq="5min")) + list(pd.date_range("2030-01-03 09:15", periods=2, freq="5min"))
    df = pd.DataFrame({"timestamp": stamps, "high": [11, 12, 13, 21, 22], "low": [9, 10, 11, 19, 20],
                       "close": [10, 11, 12, 20, 21], "volume": [100, 200, 300, 10, 30]})
    out = vwap_features(df)
    # day 2: typical prices 20, 21 weighted by 10, 30 = (200 + 630) / 40
    assert out["available"] and out["vwap"] == pytest.approx((20 * 10 + 21 * 30) / 40)
    assert out["distance_pct"] == pytest.approx((21 - 20.75) / 20.75 * 100)


def test_vwap_slope_crossings_and_rejections():
    df = _walk()
    out = vwap_features(df, atr=0.5)
    assert out["available"] and out["distance_atr"] is not None
    assert isinstance(out["crossings_session"], int) and out["crossings_session"] >= 0
    assert isinstance(out["rejections_recent"], int) and 0 <= out["rejections_recent"] <= 12
    assert out["slope_atr_per_bar"] is not None


def test_a_series_without_volume_has_no_vwap_and_says_why():
    out = vwap_features(_walk(volume=False))
    assert not out["available"] and out["vwap"] is None and "no volume" in out["reason"]


def test_vwap_needs_the_columns():
    assert not vwap_features(pd.DataFrame({"close": [1, 2]}))["available"]
    assert not vwap_features(None)["available"]


# ------------------------------------------------------------------ volume profile
def test_the_profile_finds_the_busiest_price_and_a_value_area_holding_70_percent():
    stamps = pd.date_range("2030-01-02 09:15", periods=4, freq="5min")
    df = pd.DataFrame({"timestamp": stamps, "high": [101, 105, 101, 101], "low": [99, 95, 99, 99], "volume": [1000, 10, 1000, 1000]})
    prof = volume_profile(df, bins=20)
    assert prof["available"] and 99 <= prof["poc"] <= 101 and prof["val"] <= prof["poc"] <= prof["vah"]
    assert prof["approximation"] == APPROXIMATION and prof["bins"] == 20


def test_the_profile_is_unavailable_without_volume():
    prof = volume_profile(_walk(volume=False))
    assert not prof["available"] and prof["poc"] is None and "no volume" in prof["reason"]


def test_the_profile_uses_only_the_requested_sessions():
    df = _walk()
    one, two = volume_profile(df, sessions=1), volume_profile(df, sessions=2)
    assert one["available"] and two["available"] and (one["poc"], one["vah"]) != (two["poc"], two["vah"])


# ------------------------------------------------------------------ input checks
def test_good_data_passes_the_checks():
    assert validate_ohlcv_for_context(_walk()).level == OK


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d.head(MIN_BARS - 1), "bars"),
    (lambda d: d.iloc[::-1].reset_index(drop=True), "order"),
    (lambda d: pd.concat([d, d.tail(1)], ignore_index=True), "duplicate"),
    (lambda d: d.assign(close=d["close"].where(d.index != len(d) - 3)), "invalid prices"),
    (lambda d: d.assign(low=d["low"].where(d.index != len(d) - 2, d["high"] + 5)), "high below its low"),
    (lambda d: d.drop(columns=["high"]), "missing columns"),
])
def test_bad_data_is_critical_with_a_reason(mutate, expect):
    status = validate_ohlcv_for_context(mutate(_walk()))
    assert status.level == CRITICAL and expect in status.reason


def test_a_stale_last_bar_is_a_warning():
    df = _walk()
    now = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=45)
    assert validate_ohlcv_for_context(df, 20, now).level == WARNING
    assert validate_ohlcv_for_context(df, 60, now).level == OK


# ------------------------------------------------------------------ the context
def test_the_context_carries_every_block_and_a_clean_status():
    df = _walk()
    ctx = build_market_context(df, underlying="NIFTY", timeframe="5min",
                               regime={"label": "TRENDING", "axes": {"direction": "UP"}, "clarity": 0.6, "age_bars": 4,
                                       "scores": {"TRENDING": 0.6, "RANGING": 0.4}})
    assert ctx.underlying == "NIFTY" and ctx.timestamp == ctx.info_cutoff == df["timestamp"].iloc[-1]
    assert ctx.bars_used == len(df) and ctx.first_bar == df["timestamp"].iloc[0]
    assert ctx.status.ok and ctx.atr and ctx.vwap["available"] and ctx.volume_profile["available"]
    assert ctx.regime["label"] == "TRENDING" and "heuristic" in ctx.regime["note"] and ctx.regime["scores_heuristic"]
    assert ctx.structure.as_of_index == len(df) - 1 and ctx.levels
    assert set(ctx.nearest) == {"support", "resistance"}


def test_the_context_is_reproducible_and_json_safe():
    df = _walk()
    a = build_market_context(df, underlying="NIFTY").as_record()
    b = build_market_context(df.copy(), underlying="NIFTY").as_record()
    assert a == b
    assert json.loads(json.dumps(a))["underlying"] == "NIFTY"
    assert a["parameters"]["structure"]["swing_bars"] == 3 and a["info_cutoff"] == str(df["timestamp"].iloc[-1])


def test_no_volume_makes_a_warning_and_names_the_unavailable_fields():
    ctx = build_market_context(_walk(volume=False), underlying="NIFTY")
    assert ctx.status.level == WARNING and set(ctx.status.unavailable) == {"vwap", "volume_profile"}
    assert not ctx.vwap["available"] and ctx.structure.swings  # price-only analytics still work


def test_bad_data_gives_an_empty_critical_context_not_an_exception():
    ctx = build_market_context(_walk().head(10), underlying="NIFTY")
    assert not ctx.status.ok and ctx.bars_used == 0 and ctx.levels == () and ctx.nearest == {}
    assert ctx.as_record()["status"]["level"] == CRITICAL


def test_a_context_needs_an_underlying():
    with pytest.raises(ValueError):
        build_market_context(_walk(), underlying="")


def test_features_from_the_pipeline_are_used_when_given():
    df = _walk()
    feats = pd.DataFrame({"timestamp": df["timestamp"], "atr_14": 0.7, "trend_z": 1.2, "momentum_z": -0.3, "relative_volume": 2.0,
                          "atr_percentile_100": 0.8, "vol_expansion": 1.1, "realized_vol_20": 0.2, "market_structure": 1.0,
                          "trend_slope": 0.001})
    ctx = build_market_context(df, underlying="NIFTY", features=feats)
    assert ctx.atr == 0.7 and ctx.trend["trend_z"] == 1.2 and ctx.volume["expanding"] is True


def test_pipeline_output_adapter_maps_quality_to_status():
    df = _walk()
    out = SimpleNamespace(instrument="NIFTY", ohlcv=df, features=None, regime_label="RANGING", regime_axes={"direction": "FLAT"},
                          regime_clarity=0.4, regime_age_bars=2, regime_probabilities={"RANGING": 1.0}, data_quality_status="DEGRADED")
    ctx = context_from_pipeline(out)
    assert ctx.status.level == WARNING and "DEGRADED" in ctx.status.reason and ctx.regime["label"] == "RANGING"
    out.data_quality_status = "FAIL"
    assert context_from_pipeline(out).status.level == CRITICAL
