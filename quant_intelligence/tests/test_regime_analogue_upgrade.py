"""Regime (instrument-relative, persistent, per-instrument overlays, universe scan) and robust analogues
(scaling, cutoff, per-day cap, kernel weights, informativeness test)."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from quant_intelligence.analogues import analogue_engine as ae
from quant_intelligence.features.feature_engine import compute_features
from quant_intelligence.regimes import regime_engine as re_
from quant_intelligence.regimes.validation import regime_forward_stats


def _candles(n_days=12, vol=0.0008, scale=100.0, seed=3, start="2026-08-03"):
    rng = np.random.default_rng(seed)
    ts = []
    for d in pd.bdate_range(start, periods=n_days):
        ts += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min"))
    close = scale * np.exp(np.cumsum(rng.normal(0, vol, len(ts))))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, vol, len(ts))) * close
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close, "volume": rng.integers(1000, 5000, len(ts))})


# ------------------------------------------------------------------------------------ regime
UP = {"trend_slope": 0.004, "momentum_20": 0.03, "market_structure": 1, "atr_percentile_100": 0.5, "vol_expansion": 1.0}


def test_z_scores_are_used_when_present_and_make_strength_instrument_relative():
    big = {**UP, "trend_slope": 0.0, "momentum_20": 0.0, "trend_z": 3.0, "momentum_z": 3.0}
    flat = {**big, "trend_z": 0.0, "momentum_z": 0.0}
    assert re_.classify_regime(big)[0] == "TREND_UP"
    assert re_.classify_regime(flat)[0] != "TREND_UP"


def test_regime_is_unchanged_when_prices_are_rescaled():
    a, b = _candles(scale=100.0), _candles(scale=100.0 * 250)  # identical shape, 250x the price level
    la = re_.classify_series(compute_features(a))["label"].tolist()
    lb = re_.classify_series(compute_features(b))["label"].tolist()
    assert la == lb


def test_event_overlay_only_for_instruments_with_an_expiry_rule():
    near_expiry = {**UP, "trend_slope": 0.0, "momentum_20": 0.0, "market_structure": 0, "expiry_proximity_days": 0}
    assert re_.classify_regime(near_expiry, "NIFTY")[1]["EVENT_DRIVEN"] > re_.classify_regime(near_expiry, "GOLD")[1]["EVENT_DRIVEN"]
    # no expiry rule: the score is exactly the neutral softmax share of a zero score
    p = re_.classify_regime(near_expiry, "GOLD")[1]
    assert p["EVENT_DRIVEN"] == pytest.approx(p["LIQUIDITY_STRESS"])


def test_liquidity_stress_needs_traded_volume():
    thin = {"atr_percentile_100": 0.9, "relative_volume": 0.2}
    with_volume = re_.classify_regime(thin)[1]["LIQUIDITY_STRESS"]
    no_volume = re_.classify_regime({"atr_percentile_100": 0.9})[1]["LIQUIDITY_STRESS"]
    assert with_volume > no_volume


def test_series_is_causal_and_label_changes_less_often_than_bar_by_bar():
    feats = compute_features(_candles(n_days=20))
    full = re_.classify_series(feats)
    cut = 600
    part = re_.classify_series(feats.iloc[:cut].reset_index(drop=True))
    assert full["label"].iloc[:cut].tolist() == part["label"].tolist()  # future bars never change the past
    assert (full["label"] != full["label"].shift()).sum() < (full["raw_label"] != full["raw_label"].shift()).sum()


def test_label_never_switches_before_the_dwell_requirement():
    feats = compute_features(_candles(n_days=20, seed=9))
    s = re_.classify_series(feats)
    change = np.flatnonzero((s["label"] != s["label"].shift()).to_numpy())[1:]
    assert all(s["age_bars"].iloc[i] == 0 for i in change)
    # after a change at i, the next change cannot come sooner than SWITCH_BARS bars later
    assert np.all(np.diff(change) >= re_.SWITCH_BARS) if len(change) > 1 else True


def test_axes_clarity_and_age_are_reported():
    s = re_.classify_series(compute_features(_candles()))
    assert set(s["direction"]) <= {"UP", "DOWN", "NEUTRAL"} and set(s["volatility"]) <= {"LOW", "NORMAL", "HIGH"}
    assert s["clarity"].between(0, 1).all() and (s["age_bars"] >= 0).all()
    assert abs(sum(s.attrs["probabilities"].values()) - 1.0) < 1e-9


def test_transition_matrix_rows_sum_to_one():
    tm = re_.transition_matrix(pd.Series(list("AABABBBA")))
    assert np.allclose(tm.sum(axis=1), 1.0)


@settings(max_examples=60, deadline=None)
@given(st.dictionaries(st.sampled_from(["trend_slope", "momentum_20", "trend_z", "momentum_z", "atr_percentile_100",
                                        "vol_expansion", "relative_volume", "market_structure", "expiry_proximity_days"]),
                       st.one_of(st.floats(-50, 50), st.none()), max_size=9))
def test_any_feature_dict_gives_a_valid_distribution(features):
    label, probs = re_.classify_regime(features)
    assert label in re_.REGIMES and abs(sum(probs.values()) - 1.0) < 1e-9 and all(0 <= p <= 1 for p in probs.values())


def test_forward_stats_do_not_cross_sessions_and_group_by_label():
    ohlcv = _candles(n_days=10)
    s = re_.classify_series(compute_features(ohlcv))
    stats = regime_forward_stats(ohlcv, s, horizon=12)
    assert stats.iloc[0]["regime"] == "ALL" and stats.iloc[0]["n"] == stats["n"].iloc[1:].sum()
    assert stats.iloc[0]["n"] <= len(ohlcv) - 12 * 10  # the last `horizon` bars of every session have no full window


def test_features_trend_z_is_causal():
    base = _candles(n_days=30)
    full = compute_features(base)
    part = compute_features(base.iloc[:1200])
    for col in ("trend_z", "momentum_z", "or_position", "prev_day_range_pos"):
        a, b = full[col].iloc[:1200].to_numpy(), part[col].to_numpy()
        assert np.allclose(a, b, equal_nan=True), col


# ---------------------------------------------------------------------------------- analogues
def _observations(n_days=30, per_day=4, seed=1, edge=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for k in range(per_day):
            x = rng.normal(size=len(ae.ROBUST_FEATURES))
            r = rng.normal(edge, 1.0)
            row = dict(zip(ae.ROBUST_FEATURES, x))
            row.update({"trend_slope": x[0], "momentum_20": x[1], "r_multiple": r, "outcome": "WIN" if r > 0 else "LOSS",
                        "entry_timestamp": pd.Timestamp("2026-08-03 09:30") + pd.Timedelta(days=d, minutes=15 * k),
                        "time_since_open_min": 15.0 * k})
            rows.append(row)
    return pd.DataFrame(rows)


def _query(obs, row=0, **over):
    q = {c: float(obs.loc[row, c]) for c in ae.ROBUST_FEATURES + ["trend_slope", "momentum_20", "time_since_open_min"]}
    q.update(over)
    return q


def test_legacy_config_matches_the_previous_behaviour_exactly():
    obs = _observations()
    q = _query(obs)
    got = ae.find_analogues(q, obs, top_k=10, config=ae.LEGACY)
    cols = [c for c in ae.COMPARISON_FEATURES]
    z = (obs[cols] - obs[cols].mean()) / obs[cols].std()
    zq = (pd.Series({c: q[c] for c in cols}) - obs[cols].mean()) / obs[cols].std()
    d = np.sqrt(((z - zq) ** 2).sum(axis=1)).to_numpy()
    assert list(got.index) == list(np.argsort(d, kind="stable")[:10])
    assert got["similarity_score"].iloc[0] == pytest.approx(1 - d.min() / d.max())


def test_robust_scaling_resists_an_extreme_outlier():
    obs = _observations()
    obs.loc[5, "trend_z"] = 1e6
    q = _query(obs, row=0)
    legacy_like = ae.AnalogueConfig(name="z", features=("trend_z", "momentum_z"), scaling="zscore")
    robust = ae.AnalogueConfig(name="r", features=("trend_z", "momentum_z"), scaling="robust")
    # with z-scoring the outlier inflates the spread, so every other row looks closer together on trend_z; robust keeps it
    spread_z = ae.AnalogueIndex(obs, q, legacy_like).Z[:, 0]
    spread_r = ae.AnalogueIndex(obs, q, robust).Z[:, 0]
    others = np.delete(np.arange(len(obs)), 5)
    assert np.std(spread_r[others]) > 10 * np.std(spread_z[others])
    assert np.abs(spread_r).max() <= ae.CLIP_Z + 1e-9


def test_per_day_cap_limits_rows_from_one_session_and_k_counts_days():
    obs = _observations(n_days=20, per_day=6)
    out = ae.find_analogues(_query(obs), obs, top_k=15, config=ae.AnalogueConfig(name="c", features=tuple(ae.ROBUST_FEATURES), max_per_day=2))
    per_day = out["entry_timestamp"].dt.date.value_counts()
    assert len(out) == 15 and per_day.max() <= 2


def test_distance_cutoff_returns_fewer_analogues_when_nothing_is_close():
    obs = _observations(n_days=30, per_day=4)
    far = _query(obs, trend_z=40.0, momentum_z=-40.0, atr_percentile_100=40.0)
    robust = ae.AnalogueConfig(name="c", features=tuple(ae.ROBUST_FEATURES), scaling="robust", cutoff_quantile=0.2)
    near = ae.find_analogues(_query(obs), obs, top_k=30, config=robust)
    none = ae.find_analogues(far, obs, top_k=30, config=robust)
    assert len(near) > len(none)
    assert len(none) <= 2  # nothing is close to a point 40 robust standard deviations away


def test_similarity_is_absolute_with_a_cutoff():
    obs = _observations()
    out = ae.find_analogues(_query(obs), obs, top_k=10, config=ae.ROBUST)
    assert out["similarity_score"].between(0, 1).all() and out["similarity_score"].is_monotonic_decreasing


def test_kernel_weights_known_answer_and_effective_n():
    an = pd.DataFrame({"r_multiple": [1.0, -1.0, 0.0] * 5, "weight": [3.0, 1.0, 1.0] * 5,
                       "entry_timestamp": pd.date_range("2026-08-03", periods=15, freq="D")})
    m = ae.conditional_metrics_from_analogues(an)
    assert m["expected_r"] == pytest.approx(np.average([1, -1, 0] * 5, weights=[3, 1, 1] * 5))
    w = np.array([3.0, 1.0, 1.0] * 5)
    assert m["n_effective"] == pytest.approx(w.sum() ** 2 / (w**2).sum())
    assert m["expected_r_ci95"] is not None and m["expected_r_ci95"][0] <= m["expected_r"] <= m["expected_r_ci95"][1]


def test_unweighted_metrics_are_unchanged():
    an = pd.DataFrame({"r_multiple": np.linspace(-1, 2, 12), "entry_timestamp": pd.date_range("2026-08-03", periods=12, freq="D")})
    m = ae.conditional_metrics_from_analogues(an)
    assert m["expected_r"] == pytest.approx(an["r_multiple"].mean()) and "n_effective" not in m


def test_conditioning_test_finds_no_information_in_noise_and_some_in_signal():
    noise = _observations(n_days=40, per_day=4, seed=2)
    cfg = ae.ROBUST
    idx = ae.AnalogueIndex(noise, _query(noise), cfg)
    r = noise["r_multiple"].to_numpy()
    pos = idx.select(idx.distances(), 20, minutes_now=0.0)
    p_noise = ae.conditioning_test(idx, r, float(r[pos].mean()), top_k=20)["p_value"]
    assert p_noise is not None and p_noise > ae.INFORMATIVE_P_MAX  # a typical noisy set is not "informative"

    sig = _observations(n_days=40, per_day=4, seed=2)
    sig["r_multiple"] = 2.0 * sig["trend_z"] + np.random.default_rng(0).normal(0, 0.3, len(sig))  # outcome depends on a feature
    idx2 = ae.AnalogueIndex(sig, _query(sig, trend_z=1.8), cfg)
    r2 = sig["r_multiple"].to_numpy()
    pos2 = idx2.select(idx2.distances(), 20, minutes_now=0.0)
    p_sig = ae.conditioning_test(idx2, r2, float(r2[pos2].mean()), top_k=20)["p_value"]
    assert p_sig is not None and p_sig < p_noise


def test_assessment_falls_back_to_the_overall_estimate_when_analogues_add_nothing():
    from quant_intelligence.research.ranking_core import assess_observations

    obs = _observations(n_days=40, per_day=4, seed=4)
    a = assess_observations("S", obs, _query(obs), {"max_drawdown_r": -2.0}, config=ae.ROBUST)
    c = a.conditional
    assert c["analogue_method"] == "robust" and "conditioning_p_value" in c
    if c.get("analogues_informative") is False:
        assert c["estimate_basis"] == "overall" and c["expected_r"] == pytest.approx(obs["r_multiple"].mean())


def test_analogues_ignore_trades_that_had_not_closed_yet():
    obs = _observations()
    obs["exit_timestamp"] = obs["entry_timestamp"] + pd.Timedelta(hours=1)
    cutoff = obs["entry_timestamp"].iloc[40]
    out = ae.find_analogues(_query(obs), obs, top_k=30, config=ae.ROBUST, as_of=cutoff)
    assert (pd.to_datetime(out["exit_timestamp"]) <= cutoff).all()


def test_changing_data_after_the_query_time_does_not_change_the_result():
    obs = _observations(n_days=40, per_day=4)
    obs["exit_timestamp"] = obs["entry_timestamp"] + pd.Timedelta(hours=1)
    t = obs["entry_timestamp"].iloc[80]
    q = _query(obs, row=0)
    before = ae.find_analogues(q, obs[obs["exit_timestamp"] <= t], 30, ae.ROBUST)
    tampered = obs.copy()
    tampered.loc[tampered["exit_timestamp"] > t, ["trend_z", "r_multiple"]] = [0.0, 99.0]
    after = ae.find_analogues(q, tampered, 30, ae.ROBUST, as_of=t)
    assert before["r_multiple"].tolist() == after["r_multiple"].tolist()


@settings(max_examples=25, deadline=None)
@given(st.integers(1, 3), st.integers(5, 25))
def test_never_more_analogues_than_days_times_cap(cap, k):
    obs = _observations(n_days=12, per_day=5)
    cfg = ae.AnalogueConfig(name="c", features=tuple(ae.ROBUST_FEATURES), max_per_day=cap)
    out = ae.find_analogues(_query(obs), obs, top_k=k, config=cfg)
    assert len(out) <= min(k, 12 * cap)


# ------------------------------------------------------------------------------------ universe
def test_universe_scan_marks_failing_data_and_never_invents_a_regime(tmp_path, monkeypatch):
    from quant_intelligence.regimes import universe

    good = _candles(n_days=15)
    bad = good.copy()
    bad.loc[10, "close"] = np.nan
    monkeypatch.setattr(universe, "CACHE_DIR", tmp_path)
    good.to_parquet(tmp_path / "NIFTY_5min.parquet", index=False)
    bad.to_parquet(tmp_path / "BANKNIFTY_5min.parquet", index=False)
    rows = {r["instrument"]: r for r in universe.scan_universe("5min", [("NIFTY", "index"), ("BANKNIFTY", "index"), ("TCS", "stock")])}
    assert rows["NIFTY"]["regime"] in re_.REGIMES
    assert rows["BANKNIFTY"]["data_status"] == "FAIL" and rows["BANKNIFTY"]["regime"] is None
    assert rows["TCS"]["regime"] is None and rows["TCS"]["note"] == "no cached candles"


def test_universe_cache_serves_the_previous_scan_when_a_refresh_fails():
    from quant_intelligence.regimes.universe import UniverseCache

    calls = {"n": 0}

    def scanner(tf):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("boom")
        return [{"instrument": "X"}]

    cache = UniverseCache(scanner=scanner, ttl=0.0)
    assert cache.get("5min") == [{"instrument": "X"}]
    assert cache.get("5min") == [{"instrument": "X"}]  # stale result is served while the failing refresh runs


def test_regimes_endpoint_is_read_only_and_json():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from quant_intelligence.web import app as web

    client = TestClient(web.app)
    assert client.post("/api/regimes").status_code == 405
