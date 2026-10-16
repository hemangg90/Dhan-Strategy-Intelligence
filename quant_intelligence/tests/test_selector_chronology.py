"""The selector must not see the future: decisions made before a cut-off are identical however the later data look.

Synthetic data (seeded) is a test fixture only. Passing shows that THIS tampering does not change earlier decisions;
it does not prove the absence of every conceivable leak.
"""
import numpy as np
import pandas as pd

from quant_intelligence.ranking import evaluate_ranker as ev
from quant_intelligence.strategies.registry import get_strategy

# A deliberately small, fast setup (about 13 s per evaluation): 1,500 bars and five representative strategies.
STRATEGIES = ["Momentum", "Donchian Channel Breakout", "VWAP Mean Reversion", "Supertrend Trend Following", "Opening Range Breakout"]
START = 800


def _small(ohlcv: pd.DataFrame) -> pd.DataFrame:
    return ohlcv.sort_values("timestamp").reset_index(drop=True).iloc[:1500].reset_index(drop=True)


def _evaluate(df: pd.DataFrame):
    return ev.evaluate_instrument(df, "NIFTY", warmup_frac=0.0, min_history_bars=START,
                                  strategies=[get_strategy(n) for n in STRATEGIES])


def _tampered_after(ohlcv: pd.DataFrame, cut: int) -> pd.DataFrame:
    out = ohlcv.copy()
    rng = np.random.default_rng(99)
    factor = np.cumprod(1 + rng.normal(0.004, 0.02, len(out) - cut - 1))  # a violently different future
    for c in ("open", "high", "low", "close"):
        out.loc[cut + 1:, c] = out.loc[cut + 1:, c].to_numpy() * factor
    return out


def test_decisions_before_a_cutoff_do_not_change_when_the_future_is_rewritten(synthetic_ohlcv):
    df = _small(synthetic_ohlcv)
    cut = int(len(df) * 0.8)
    base = _evaluate(df)
    changed = _evaluate(_tampered_after(df, cut))

    limit = df["timestamp"].iloc[cut]
    a = {d.timestamp: (d.selected, d.regime_selected, d.fixed_strategy) for d in base if d.timestamp <= limit}
    b = {d.timestamp: (d.selected, d.regime_selected, d.fixed_strategy) for d in changed if d.timestamp <= limit}
    common = set(a) & set(b)
    assert len(common) >= 15, "test needs a meaningful number of decision points before the cut-off"
    assert [a[t] for t in sorted(common)] == [b[t] for t in sorted(common)]


def test_the_fixed_baseline_is_chosen_from_the_warmup_period_only(synthetic_ohlcv):
    df = _small(synthetic_ohlcv)
    base = _evaluate(df)
    # rewrite everything AFTER the warm-up ends: the fixed pick must not move
    other = _evaluate(_tampered_after(df, START))
    pick_a = {d.fixed_strategy for d in base}
    pick_b = {d.fixed_strategy for d in other}
    assert len(pick_a) == 1 and pick_a == pick_b


def test_summary_reports_samples_baselines_and_breakdowns(synthetic_ohlcv):
    decisions = _evaluate(_small(synthetic_ohlcv))
    s = ev.summarize(decisions)
    assert s["decision_points"] == len(decisions)
    assert set(s["baselines"]) == {"fixed_best_on_warmup", "regime_based", "random_pick_every_signal"}
    for stats in s["baselines"].values():
        assert "n_trades" in stats  # every figure carries its sample size
    if s.get("trades_taken"):
        assert "by_strategy" in s and "by_regime" in s and s["selector_trade_stats"]["n_trades"] == s["trades_taken"]
