"""Analogue engine: outcome availability, no look-ahead, structure/options features, redundancy control, explanations."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.analogues import analogue_engine as ae
from quant_intelligence.features.market_structure import StructureConfig, analyze_structure, structure_feature_frame
from quant_intelligence.intelligence.context_features import current_feature_vector, market_context_features
from quant_intelligence.intelligence.market_context import build_market_context
from quant_intelligence.research.ranking_core import observations_from_trades

T0 = pd.Timestamp("2030-01-02 09:15")


def _obs(n=80, seed=1, hours=2):
    """n observations, one every `hours` hours; each trade resolves 3 hours after it fired."""
    rng = np.random.default_rng(seed)
    entry = [T0 + pd.Timedelta(hours=hours * i) for i in range(n)]
    df = pd.DataFrame({c: rng.normal(size=n) for c in ae.ROBUST_FEATURES})
    df["time_since_open_min"] = 60.0
    df["entry_timestamp"] = entry
    df["feature_timestamp"] = entry
    df["event_timestamp"] = entry
    df["exit_timestamp"] = [e + pd.Timedelta(hours=3) for e in entry]
    df["outcome_available_at"] = df["exit_timestamp"]
    df["r_multiple"] = rng.normal(0.1, 1.0, n)
    df["outcome"] = np.where(df["r_multiple"] > 0, "WIN", "LOSS")
    return df


def _query(df):
    return {c: float(df[c].iloc[0]) for c in ae.ROBUST_FEATURES} | {"time_since_open_min": 60.0}


CFG = ae.AnalogueConfig(name="plain", features=tuple(ae.ROBUST_FEATURES), scaling="robust")


# ------------------------------------------------------------------ outcome availability
def test_a_trade_that_fired_before_the_decision_but_resolved_after_it_is_excluded():
    obs = _obs()
    asof = obs["entry_timestamp"].iloc[40] + pd.Timedelta(hours=1)  # trade 40 fired earlier; its result lands 2 hours later
    picked, _ = ae.analogue_search(_query(obs), obs, top_k=80, config=CFG, as_of=asof)
    assert (picked["outcome_available_at"] <= asof).all() and (picked["entry_timestamp"] <= asof).all()
    in_flight = obs[(obs["entry_timestamp"] <= asof) & (obs["outcome_available_at"] > asof)]
    assert len(in_flight) >= 1 and not set(in_flight.index) & set(picked.index)
    assert picked.attrs["excluded_outcome_not_yet_known"] == (obs["outcome_available_at"] > asof).sum()
    assert picked.attrs["information_cutoff"] == asof


def test_changing_an_unresolved_outcome_cannot_change_the_decision():
    obs = _obs()
    asof = obs["entry_timestamp"].iloc[40] + pd.Timedelta(hours=1)
    base, _ = ae.analogue_search(_query(obs), obs, top_k=20, config=CFG, as_of=asof)
    tampered = obs.copy()
    later = tampered["outcome_available_at"] > asof
    tampered.loc[later, "r_multiple"] = 99.0
    tampered.loc[later, ae.ROBUST_FEATURES[0]] = 0.0
    again, _ = ae.analogue_search(_query(obs), tampered, top_k=20, config=CFG, as_of=asof)
    assert list(base.index) == list(again.index) and list(base["r_multiple"]) == list(again["r_multiple"])


def test_an_observation_with_no_outcome_time_is_treated_as_unresolved():
    obs = _obs()
    obs.loc[5, "outcome_available_at"] = pd.NaT
    picked, _ = ae.analogue_search(_query(obs), obs, top_k=80, config=CFG, as_of=obs["entry_timestamp"].max() + pd.Timedelta(days=5))
    assert 5 not in picked.index


def test_as_of_without_any_outcome_time_column_is_refused_not_ignored():
    obs = _obs().drop(columns=["outcome_available_at", "exit_timestamp"])
    with pytest.raises(ValueError, match="outcome_available_at"):
        ae.analogue_search(_query(obs), obs, config=CFG, as_of=T0 + pd.Timedelta(days=3))


def test_the_older_exit_timestamp_column_still_works_for_as_of():
    obs = _obs().drop(columns=["outcome_available_at"])
    asof = obs["entry_timestamp"].iloc[30]
    picked, _ = ae.analogue_search(_query(obs), obs, top_k=50, config=CFG, as_of=asof)
    assert (picked["exit_timestamp"] <= asof).all()


def test_nothing_known_yet_returns_an_empty_result_that_says_so():
    obs = _obs()
    picked, index = ae.analogue_search(_query(obs), obs, config=CFG, as_of=T0)
    assert len(picked) == 0 and index is None and picked.attrs["excluded_outcome_not_yet_known"] == len(obs)


# ------------------------------------------------------------------ the three timestamps
def test_observations_keep_feature_event_and_outcome_times_apart():
    setup = SimpleNamespace(timestamp=T0)
    trade = SimpleNamespace(setup=setup, exit_timestamp=T0 + pd.Timedelta(hours=3), r_multiple=1.0, outcome="WIN", gross_pnl=1.0,
                            mfe=1.0, mae=0.0, exit_price=1.0, holding_period_bars=3)
    feats = pd.DataFrame({"timestamp": [T0], "atr_14": [1.0]}).set_index("timestamp")
    extra = pd.DataFrame({"timestamp": [T0], "structure_trend_code": [1.0]}).set_index("timestamp")
    import quant_intelligence.research.ranking_core as rc

    original = rc.net_r_multiple
    rc.net_r_multiple = lambda t, q, m: 0.8
    try:
        obs = observations_from_trades([trade], feats, "Momentum", 1, "NSE", extra_by_ts=extra)
    finally:
        rc.net_r_multiple = original
    row = obs.iloc[0]
    assert row["feature_timestamp"] == row["event_timestamp"] == T0 and row["outcome_available_at"] == T0 + pd.Timedelta(hours=3)
    assert row["structure_trend_code"] == 1.0 and row["r_multiple"] == 0.8


# ------------------------------------------------------------------ structure / options features and redundancy
def test_features_missing_from_the_history_are_reported_not_invented():
    obs = _obs()
    query = _query(obs) | {"structure_trend_code": 1.0, "iv_rank": 55.0, "pcr_oi": 1.1}
    picked, index = ae.analogue_search(query, obs, top_k=20, config=ae.CONTEXT)
    unavailable = picked.attrs["features_unavailable"]
    assert {"iv_rank", "pcr_oi", "structure_trend_code"} <= set(unavailable)
    assert not {"iv_rank", "pcr_oi", "structure_trend_code"} & set(index.cols)


def test_structure_features_in_the_history_are_used_when_both_sides_have_them():
    obs = _obs()
    rng = np.random.default_rng(4)
    obs["structure_trend_code"] = rng.choice([-1.0, 0.0, 1.0], len(obs))
    obs["dist_swing_high_atr"] = rng.normal(size=len(obs))
    query = _query(obs) | {"structure_trend_code": 1.0, "dist_swing_high_atr": 0.2}
    picked, index = ae.analogue_search(query, obs, top_k=20, config=ae.CONTEXT)
    assert {"structure_trend_code", "dist_swing_high_atr"} <= set(index.cols)
    assert "iv_rank" in picked.attrs["features_unavailable"]


def test_duplicate_features_count_about_once_when_redundancy_control_is_on():
    obs = _obs()
    obs["vwap_copy"] = obs["vwap_distance_pct"] * 3.0 + 1.0  # a perfectly correlated duplicate
    query = _query(obs) | {"vwap_copy": float(obs["vwap_distance_pct"].iloc[0]) * 3.0 + 1.0}
    base_features = tuple(ae.ROBUST_FEATURES)
    plain = ae.AnalogueConfig(name="p", features=base_features, scaling="robust")
    dup = ae.AnalogueConfig(name="d", features=base_features + ("vwap_copy",), scaling="robust")
    dup_controlled = ae.AnalogueConfig(name="c", features=base_features + ("vwap_copy",), scaling="robust", correlation_threshold=0.95)
    d_plain = ae.AnalogueIndex(obs, query, plain).distances()
    d_dup = ae.AnalogueIndex(obs, query, dup).distances()
    idx = ae.AnalogueIndex(obs, query, dup_controlled)
    d_ctrl = idx.distances()
    assert not np.allclose(d_plain, d_dup)  # an uncontrolled duplicate changes the geometry
    assert np.allclose(d_plain, d_ctrl, atol=1e-9) and idx.redundancy == {"vwap_distance_pct": 1, "vwap_copy": 1}


def test_match_columns_restrict_to_like_for_like_and_back_off_when_too_few():
    obs = _obs()
    obs["regime_label"] = ["A"] * 60 + ["B"] * 20
    cfg = ae.AnalogueConfig(name="m", features=tuple(ae.ROBUST_FEATURES), scaling="robust", match_columns=("regime_label",), min_after_match=10)
    picked, _ = ae.analogue_search(_query(obs) | {"regime_label": "A"}, obs, top_k=40, config=cfg)
    assert set(picked["regime_label"]) == {"A"} and picked.attrs["match_filters_dropped"] == []
    rare = obs.assign(regime_label=["A"] * 78 + ["B"] * 2)
    picked, _ = ae.analogue_search(_query(rare) | {"regime_label": "B"}, rare, top_k=40, config=cfg)
    assert picked.attrs["match_filters_dropped"] == ["regime_label"] and len(picked) > 2


def test_explanations_name_the_closest_features_and_the_three_times():
    obs = _obs()
    picked, index = ae.analogue_search(_query(obs), obs, top_k=5, config=CFG)
    why = ae.explain_analogues(picked, index)
    assert list(why["rank"]) == [1, 2, 3, 4, 5][: len(why)] and (why["distance"].diff().dropna() >= -1e-12).all()
    assert why["closest_features"].str.contains(r"\(").all() and why["outcome_available_at"].notna().all()
    assert (why["outcome_available_at"] >= why["event_timestamp"]).all()
    assert len(ae.explain_analogues(picked.head(0), index)) == 0 and len(ae.explain_analogues(picked, None)) == 0


# ------------------------------------------------------------------ the structure frame is causal
def _walk(n=260, seed=2):
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 0.6, n))
    rows, prev = [], closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + abs(rng.normal(0, 0.3)), min(prev, c) - abs(rng.normal(0, 0.3)), c))
        prev = c
    stamps = pd.date_range("2030-01-02 09:15", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": stamps, "open": [r[0] for r in rows], "high": [r[1] for r in rows],
                         "low": [r[2] for r in rows], "close": [r[3] for r in rows], "volume": 1000})


@pytest.mark.parametrize("t", [60, 130, 200, 259])
def test_each_row_of_the_frame_equals_the_analysis_made_at_that_bar(t):
    df = _walk()
    frame = structure_feature_frame(df).iloc[t]
    report = analyze_structure(df, as_of_index=t)
    assert frame["structure_trend_code"] == {"UP": 1.0, "DOWN": -1.0}.get(report.trend, 0.0)
    ev = report.last_event
    assert frame["bars_since_structure_event"] == (min(t - ev.break_index, 50) if ev else 50)
    assert frame["last_event_is_mss"] == (1.0 if ev is not None and ev.kind == "MSS" else 0.0)
    if report.last_swing_high is not None and np.isfinite(frame["dist_swing_high_atr"]):
        assert frame["dist_swing_high_atr"] > -50  # sanity: finite and sensible


def test_changing_future_bars_never_changes_an_earlier_row_of_the_frame():
    df = _walk()
    base = structure_feature_frame(df)
    mutated = df.copy()
    rng = np.random.default_rng(8)
    for col in ("open", "high", "low", "close"):
        mutated.loc[150:, col] = mutated.loc[150:, col] + rng.normal(0, 30, len(df) - 150)
    after = structure_feature_frame(mutated)
    pd.testing.assert_frame_equal(base.iloc[:150].reset_index(drop=True), after.iloc[:150].reset_index(drop=True))


def test_the_live_feature_vector_matches_the_historical_frame_for_the_same_bar():
    df = _walk()
    ctx = build_market_context(df, underlying="NIFTY")
    live = market_context_features(ctx)
    last = structure_feature_frame(df, StructureConfig()).iloc[-1]
    for key in ("structure_trend_code", "bars_since_structure_event", "last_event_is_mss"):
        assert live[key] == last[key]
    for key in ("dist_swing_high_atr", "dist_swing_low_atr"):
        if key in live:
            assert live[key] == pytest.approx(last[key], rel=1e-6)


def test_the_query_vector_adds_context_without_overwriting_the_pipeline_features():
    df = _walk()
    ctx = build_market_context(df, underlying="NIFTY")
    vec = current_feature_vector({"trend_z": 1.0, "relative_volume": 1.2}, ctx, None)
    assert vec["trend_z"] == 1.0 and "structure_trend_code" in vec
    bad = build_market_context(df.head(10), underlying="NIFTY")
    assert market_context_features(bad) == {}
