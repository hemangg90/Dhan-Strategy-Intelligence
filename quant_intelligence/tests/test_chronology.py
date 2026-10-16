"""Chronology (look-ahead) tests.

Property: a decision made at time T may only depend on data available at T. We check it by computing on the
FULL history and on a history TRUNCATED at T, then requiring identical values on every row up to T.
Synthetic data (seeded SyntheticAdapter) is used purely as a test fixture.

These tests catch features and strategy signals that peek at later bars (whole-session aggregates, centered
windows, back-fills). They do not prove the absence of every possible form of leakage.
"""
import numpy as np
import pandas as pd
import pytest

from quant_intelligence.features.feature_engine import compute_features
from quant_intelligence.strategies.registry import STRATEGY_CLASSES, get_strategy


def _cut_points(ohlcv: pd.DataFrame):
    """Truncation points: early in a session, mid-session and the last bar of a session."""
    ts = ohlcv["timestamp"]
    days = ts.dt.date.unique()
    day = days[len(days) // 2]
    idx = ohlcv.index[ts.dt.date == day]
    return [int(idx[2]), int(idx[len(idx) // 2]), int(idx[-1])]


def _assert_same_up_to(full: pd.DataFrame, prefix: pd.DataFrame, cols, label):
    n = len(prefix)
    for c in cols:
        a = full[c].iloc[:n].reset_index(drop=True)
        b = prefix[c].iloc[:n].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a):
            both_nan = a.isna() & b.isna()
            same = np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True) | both_nan
        else:
            same = (a == b) | (a.isna() & b.isna())
        assert bool(same.all()), f"{label}: feature '{c}' changes when future bars are removed (look-ahead)"


def test_every_feature_is_unchanged_when_future_bars_are_removed(synthetic_ohlcv):
    full = compute_features(synthetic_ohlcv)
    cols = [c for c in full.columns if c != "timestamp"]
    for cut in _cut_points(synthetic_ohlcv):
        prefix = compute_features(synthetic_ohlcv.iloc[: cut + 1].reset_index(drop=True))
        _assert_same_up_to(full, prefix, cols, f"cut at row {cut}")


@pytest.mark.parametrize("name", [n for n in STRATEGY_CLASSES])
def test_strategy_setups_do_not_depend_on_future_bars(name, synthetic_ohlcv):
    """Setups generated on the full history, restricted to rows <= T, equal setups generated from data ending at T."""
    strategy = get_strategy(name)
    full_feat = compute_features(synthetic_ohlcv)
    full = {(s.timestamp, s.direction): (round(s.entry_price, 6), round(s.stop_price, 6), round(s.target_price, 6))
            for s in strategy.generate_historical_setups(synthetic_ohlcv, full_feat)}
    for cut in _cut_points(synthetic_ohlcv):
        sub = synthetic_ohlcv.iloc[: cut + 1].reset_index(drop=True)
        sub_feat = compute_features(sub)
        limit = sub["timestamp"].iloc[-1]
        got = {(s.timestamp, s.direction): (round(s.entry_price, 6), round(s.stop_price, 6), round(s.target_price, 6))
               for s in strategy.generate_historical_setups(sub, sub_feat)}
        expected = {k: v for k, v in full.items() if k[0] <= limit}
        assert got == expected, f"{name}: setups differ when future bars are removed (cut at row {cut})"
