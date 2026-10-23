"""
Does adding market-structure or options information improve the strategy selector? A paired, chronological comparison.

Four selectors are named in the design:
  A  the existing selector (the configured analogue method on the pipeline's own features)
  B  A plus underlying market-structure features (swing/BOS/MSS state, distances to the last confirmed swings)
  C  A plus options-market features (IV rank, PCR, spread, days to expiry)
  D  A plus both
A and B can be evaluated: both come from the underlying's price history and `ranking.evaluate_ranker.evaluate_instrument`
replays them bar by bar, each seeing only trades that had already closed by the decision bar. C and D need HISTORICAL
option-chain features, and the application has none (it records IV and chains only from now on), so they are reported as
UNAVAILABLE with that reason - they are never approximated from the underlying.

The comparison is paired: on every bar both selectors saw, each one's outcome is the net R of the trade it would have taken
(0 when it stands aside). The mean paired difference carries a 95% interval that resamples whole trading days, because trades
within a day are correlated. The verdict states only what the interval supports:
  "better" / "worse"  the interval excludes 0
  "not distinguishable" otherwise - which, with the small samples a few months of 5-minute data give, is the usual result.
Nothing here is a claim of profitability; it is one backtest period, in-sample for any choice made after seeing it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNAVAILABLE_OPTIONS = ("historical option-chain features do not exist: the application records IV, open interest and quotes only "
                       "from live snapshots going forward, so selectors C and D cannot be evaluated on past bars")


def paired_outcomes(a, b) -> pd.DataFrame:
    """Per-bar net R of two selectors' decisions on the bars both evaluated (0 where a selector stood aside or its pick had no setup)."""
    fa = pd.DataFrame([{"timestamp": d.timestamp, "day": d.day, "a": d.selected_net_r or 0.0, "a_traded": d.selected_net_r is not None}
                       for d in a])
    fb = pd.DataFrame([{"timestamp": d.timestamp, "b": d.selected_net_r or 0.0, "b_traded": d.selected_net_r is not None} for d in b])
    if fa.empty or fb.empty:
        return pd.DataFrame(columns=["timestamp", "day", "a", "b", "a_traded", "b_traded"])
    return fa.merge(fb, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)


def _cluster_bootstrap(values: np.ndarray, days: np.ndarray, n_boot: int = 2000, seed: int = 3):
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    by_day = {d: values[days == d] for d in uniq}
    means = [np.concatenate([by_day[d] for d in rng.choice(uniq, size=len(uniq), replace=True)]).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _stats(r: np.ndarray, traded: np.ndarray) -> dict:
    taken = r[traded]
    losses = -taken[taken < 0].sum()
    wins = taken[taken > 0].sum()
    cum = np.cumsum(r)
    return {
        "bars": int(len(r)), "trades": int(traded.sum()),
        "mean_net_r_per_bar": float(r.mean()) if len(r) else None,
        "mean_net_r_per_trade": float(taken.mean()) if len(taken) else None,
        "win_rate": float((taken > 0).mean()) if len(taken) else None,
        "profit_factor": (float(wins / losses) if losses > 0 else None),
        "total_net_r": float(r.sum()),
        "tail_loss_cvar5_r": float(np.sort(taken)[: max(1, int(np.ceil(0.05 * len(taken))))].mean()) if len(taken) else None,
        "max_drawdown_r": float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else None,
    }


def compare(a, b, name_a: str = "A", name_b: str = "B") -> dict:
    """Paired comparison of two decision lists from `evaluate_instrument` on the same instrument and bars."""
    paired = paired_outcomes(a, b)
    if paired.empty:
        return {"status": "NO_COMMON_BARS", "reason": "the two selectors share no decision bars"}
    ra, rb = paired["a"].to_numpy(float), paired["b"].to_numpy(float)
    days = paired["day"].to_numpy()
    diff = rb - ra
    ci = _cluster_bootstrap(diff, days)
    mean_diff = float(diff.mean())
    if ci is None or ci[0] <= 0 <= ci[1]:
        verdict = "not distinguishable"
    else:
        verdict = f"{name_b} better" if mean_diff > 0 else f"{name_b} worse"
    return {
        "status": "OK", "common_bars": int(len(paired)), "trading_days": int(len(np.unique(days))),
        name_a: _stats(ra, paired["a_traded"].to_numpy(bool)), name_b: _stats(rb, paired["b_traded"].to_numpy(bool)),
        "paired_difference_mean_net_r_per_bar": mean_diff, "paired_difference_ci95": ci, "verdict": verdict,
        "bars_where_they_differ": int((np.abs(diff) > 1e-12).sum()),
        "caveat": ("a single historical period; the interval resamples days and assumes they are independent; it is not a forecast "
                   "and a 'better' verdict is in-sample evidence only"),
    }


def compare_selectors(ohlcv: pd.DataFrame, instrument: str, **kwargs) -> dict:
    """Run A and B on one instrument's history and compare them; C and D are reported unavailable (see the module docstring)."""
    from quant_intelligence.analogues.analogue_engine import default_config
    from quant_intelligence.features.market_structure import structure_feature_frame
    from quant_intelligence.ranking.evaluate_ranker import evaluate_instrument
    from quant_intelligence.analogues import analogue_engine as ae

    sorted_bars = ohlcv.sort_values("timestamp").reset_index(drop=True)
    structure = structure_feature_frame(sorted_bars)
    base_cfg = default_config()
    decisions_a = evaluate_instrument(sorted_bars, instrument, analogue_config=base_cfg, **kwargs)
    context_cfg = ae.AnalogueConfig(
        name="context-structure", features=tuple(dict.fromkeys(base_cfg.features + tuple(ae.STRUCTURE_FEATURES))), weights=base_cfg.weights,
        scaling="robust", max_per_day=base_cfg.max_per_day or 2, cutoff_quantile=base_cfg.cutoff_quantile or 0.30, kernel=True,
        phase_window_min=base_cfg.phase_window_min or 90.0, test_information=True, correlation_threshold=0.8)
    decisions_b = evaluate_instrument(sorted_bars, instrument, analogue_config=context_cfg, extra_frame=structure, **kwargs)
    return {
        "instrument": instrument,
        "A_vs_B": compare(decisions_a, decisions_b, "A", "B"),
        "C": {"status": "UNAVAILABLE", "reason": UNAVAILABLE_OPTIONS},
        "D": {"status": "UNAVAILABLE", "reason": UNAVAILABLE_OPTIONS},
        "selectors": {"A": f"existing ({base_cfg.name})", "B": "existing + market-structure features", "C": "needs option history",
                      "D": "needs option history"},
    }


def main() -> None:
    """python -m quant_intelligence.ranking.selector_comparison NIFTY BANKNIFTY ... [--max-points N] [--json out.json]

    Uses the candles already cached in data_cache/parquet_cache (no network, no credentials, no orders)."""
    import argparse
    import json

    from quant_intelligence.ranking.evaluate_ranker import load_cached_ohlcv

    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", default=["NIFTY"])
    ap.add_argument("--max-points", type=int, default=400)
    ap.add_argument("--json")
    args = ap.parse_args()
    results = {}
    for sym in args.symbols:
        df = load_cached_ohlcv(sym)
        if df is None or len(df) < 800:
            results[sym] = {"status": "SKIPPED", "reason": "no cached history long enough"}
            continue
        results[sym] = compare_selectors(df, sym, max_points=args.max_points)
        v = results[sym]["A_vs_B"]
        print(f"{sym}: {v.get('verdict')} (bars {v.get('common_bars')}, diff {v.get('paired_difference_mean_net_r_per_bar')}, "
              f"ci {v.get('paired_difference_ci95')})", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
