"""Walk-forward evaluation of the strategy ranker.

Question answered: when the ranker picks strategy S at bar t and S has a live setup at t
(exactly what the auto trader acts on), how good is the NET-of-cost result compared with
just taking a strategy at random among those with a setup at t - and did NO TRADE avoid
bad trades?

No look-ahead: each strategy is back-tested once over the whole series, but the ranker at bar t
only sees trades that had already CLOSED by t (exit <= t) and were entered before t. It goes
through the same code as the live pipeline (research/ranking_core.py).

Baselines (each sees only information available at its decision time):
  random pick   - the mean of every strategy that has a setup at t (what an uninformed choice earns);
  fixed best    - the single strategy with the best net expectancy over the WARM-UP period only, traded whenever
                  it has a setup (chosen before the evaluation period starts, never on the test data);
  regime-based  - at t, among strategies with a setup, the one whose already-closed trades in the CURRENT regime
                  label have the best positive mean net R (needs MIN_REGIME_TRADES), else it abstains.
Trades from consecutive decision points can overlap in time; confidence intervals therefore resample whole trading
days, but they remain approximate and the samples are small. Treat every figure as descriptive, not proof of an edge.

Run:  python -m quant_intelligence.ranking.evaluate_ranker [SYMBOL ...] [--json out.json]
Uses the real candles already cached in data_cache/parquet_cache (no network).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quant_intelligence.analytics.metrics import max_drawdown, profit_factor
from quant_intelligence.backtesting.engine import market_for, max_drawdown_in_r, net_r_multiple, run_backtest
from quant_intelligence.config.settings import DATA_CACHE_DIR, SETTINGS
from quant_intelligence.features.feature_engine import compute_features
from quant_intelligence.ranking.ranking_engine import rank_and_select
from quant_intelligence.regimes.regime_engine import classify_regime
from quant_intelligence.research.pipeline import _uses_volume, default_quantity
from quant_intelligence.research.ranking_core import assess_observations, observations_from_trades
from quant_intelligence.strategies.registry import get_all_strategies
from quant_intelligence.utils.market_profile import profile_for

ALLOW_TIE_BREAK = False  # set by --tie-break: also count the trades taken when the top strategies are tied
MIN_REGIME_TRADES = 5  # regime baseline: closed trades needed in the current regime before it will pick a strategy
MIN_TRAINING_TRADES = 10  # fixed baseline: warm-up trades a strategy needs to be eligible for selection


@dataclass
class Decision:
    instrument: str
    timestamp: pd.Timestamp
    day: str
    selected: str | None  # None = NO TRADE
    n_signals: int  # strategies with a setup at this bar
    selected_net_r: float | None  # net R of the selected strategy's trade at this bar (if it had one)
    pick_net_r: float  # mean net R over all strategies' trades at this bar (= picking one at random)
    regime: str | None = None  # regime label at this bar (heuristic; display/analysis only)
    fixed_strategy: str | None = None  # the strategy the fixed baseline chose on the warm-up period
    fixed_net_r: float | None = None  # its net R at this bar, if it had a setup
    regime_selected: str | None = None  # the regime baseline's pick (None = abstained)
    regime_net_r: float | None = None  # its net R at this bar


def evaluate_instrument(
    ohlcv: pd.DataFrame,
    instrument: str,
    quantity: int | None = None,
    warmup_frac: float = 0.4,
    min_history_bars: int = 400,
    strategies=None,
    max_points: int | None = None,
    analogue_config=None,
) -> list[Decision]:
    quantity = quantity or default_quantity(instrument)
    market = market_for(instrument)
    ohlcv = ohlcv.sort_values("timestamp").reset_index(drop=True)
    features = compute_features(ohlcv, profile_for(instrument))
    feat_by_ts = features.set_index("timestamp")
    ts_to_idx = {ts: i for i, ts in enumerate(ohlcv["timestamp"])}

    strategies = strategies if strategies is not None else get_all_strategies()
    if float(ohlcv["volume"].fillna(0).sum()) <= 0:  # same rule as the live pipeline
        strategies = [s for s in strategies if not _uses_volume(s)]

    per_strategy = {}
    for strat in strategies:
        bt = run_backtest(strat, ohlcv, features, instrument, split="EVAL", quantity=quantity, persist=False)
        # observations_from_trades skips trades whose entry ts has no feature row; keep aligned.
        trades = [t for t in bt.trades if t.setup.timestamp in feat_by_ts.index]
        if not trades:
            continue  # never produces a setup -> permanently INSUFFICIENT_DATA, cannot affect selection
        obs = observations_from_trades(trades, feat_by_ts, strat.name, quantity, market)
        entry_idx = np.array([ts_to_idx[t.setup.timestamp] for t in trades])
        exit_idx = np.array([ts_to_idx.get(t.exit_timestamp, len(ohlcv)) for t in trades])
        obs["entry_idx"], obs["exit_idx"] = entry_idx, exit_idx
        entry_regimes = np.array([
            classify_regime(features.iloc[int(i)].drop(labels=["timestamp"]).to_dict(), instrument)[0] for i in entry_idx
        ])
        per_strategy[strat.name] = {
            "regimes": entry_regimes,
            "obs": obs,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "net_r_arr": np.array([net_r_multiple(t, quantity, market) for t in trades]),
            "net_r": {int(ts_to_idx[t.setup.timestamp]): net_r_multiple(t, quantity, market) for t in trades},
        }

    start_idx = max(int(len(ohlcv) * warmup_frac), min_history_bars)
    signal_bars = sorted({i for d in per_strategy.values() for i in d["net_r"] if i >= start_idx})
    if max_points and len(signal_bars) > max_points:
        signal_bars = signal_bars[:: int(np.ceil(len(signal_bars) / max_points))]

    # Fixed baseline: choose ONCE, from trades that had closed before the evaluation period began.
    fixed_name, fixed_best = None, -np.inf
    for name, d in per_strategy.items():
        train = d["net_r_arr"][(d["exit_idx"] < start_idx) & (d["entry_idx"] < start_idx)]
        if len(train) >= MIN_TRAINING_TRADES and train.mean() > fixed_best:
            fixed_name, fixed_best = name, float(train.mean())

    decisions: list[Decision] = []
    for t in signal_bars:
        current = features.iloc[t].drop(labels=["timestamp"]).to_dict()
        scores = []
        for name, d in per_strategy.items():
            known_mask = (d["exit_idx"] <= t) & (d["entry_idx"] < t)
            metrics = {"max_drawdown_r": max_drawdown_in_r(d["net_r_arr"][known_mask])}
            scores.append(assess_observations(name, d["obs"][known_mask], current, metrics, config=analogue_config).score)
        ranking = rank_and_select(scores, "OK", allow_tie_break=ALLOW_TIE_BREAK)

        at_t = {n: d["net_r"][t] for n, d in per_strategy.items() if t in d["net_r"]}
        selected = None if ranking.is_no_trade else ranking.selected_strategy

        regime_now = classify_regime(current, instrument)[0]
        regime_pick, regime_best = None, 0.0  # must beat zero expectancy to trade
        for name in at_t:
            d = per_strategy[name]
            known = (d["exit_idx"] <= t) & (d["entry_idx"] < t) & (d["regimes"] == regime_now)
            if known.sum() >= MIN_REGIME_TRADES and d["net_r_arr"][known].mean() > regime_best:
                regime_pick, regime_best = name, float(d["net_r_arr"][known].mean())
        decisions.append(
            Decision(
                instrument=instrument,
                timestamp=ohlcv["timestamp"].iloc[t],
                day=str(ohlcv["timestamp"].iloc[t].date()),
                selected=selected,
                n_signals=len(at_t),
                selected_net_r=at_t.get(selected) if selected else None,
                pick_net_r=float(np.mean(list(at_t.values()))),
                regime=regime_now,
                fixed_strategy=fixed_name,
                fixed_net_r=at_t.get(fixed_name) if fixed_name else None,
                regime_selected=regime_pick,
                regime_net_r=at_t.get(regime_pick) if regime_pick else None,
            )
        )
    return decisions


def _bootstrap_ci(values: np.ndarray, days: np.ndarray, n_boot: int = 2000, seed: int = 7):
    """95% CI of the mean, resampling whole trading days (trades within a day are correlated)."""
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    by_day = {d: values[days == d] for d in uniq}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        means.append(np.concatenate([by_day[d] for d in pick]).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(decisions: list[Decision]) -> dict:
    df = pd.DataFrame([asdict(d) for d in decisions])
    if df.empty:
        return {"decision_points": 0}
    traded = df[df["selected_net_r"].notna()]
    no_trade = df[df["selected"].isna()]
    waiting = df[df["selected"].notna() & df["selected_net_r"].isna()]  # picked S, but S had no setup at t

    out = {
        "decision_points": int(len(df)),
        "no_trade_points": int(len(no_trade)),
        "selected_but_no_setup": int(len(waiting)),
        "trades_taken": int(len(traded)),
        "avg_signals_per_point": float(df["n_signals"].mean()),
    }
    if len(traded):
        sel, pick = traded["selected_net_r"].to_numpy(), traded["pick_net_r"].to_numpy()
        days = traded["day"].to_numpy()
        out.update(
            {
                "selected_mean_net_r": float(sel.mean()),
                "selected_win_rate": float((sel > 0).mean()),
                "random_pick_mean_net_r": float(pick.mean()),
                "lift_vs_random_pick": float((sel - pick).mean()),
                "selected_mean_net_r_ci95": _bootstrap_ci(sel, days),
                "lift_ci95": _bootstrap_ci(sel - pick, days),
            }
        )
    if len(no_trade):
        out["avoided_by_no_trade_mean_net_r"] = float(no_trade["pick_net_r"].mean())
    out["take_every_signal_mean_net_r"] = float(df["pick_net_r"].mean())

    out["baselines"] = {
        "fixed_best_on_warmup": _trade_stats(df["fixed_net_r"], df["day"]),
        "regime_based": _trade_stats(df["regime_net_r"], df["day"]),
        "random_pick_every_signal": _trade_stats(df["pick_net_r"], df["day"]),
    }
    if len(traded):
        ordered = traded.sort_values("timestamp")
        out["selector_trade_stats"] = _trade_stats(ordered["selected_net_r"], ordered["day"])
        out["by_strategy"] = {k: _trade_stats(g["selected_net_r"], g["day"], ci=False) for k, g in traded.groupby("selected")}
        out["by_regime"] = {str(k): _trade_stats(g["selected_net_r"], g["day"], ci=False) for k, g in traded.groupby("regime")}
        out["trading_days"] = int(df["day"].nunique())
        out["trades_per_trading_day"] = float(len(traded) / max(df["day"].nunique(), 1))
        out["exposure_fraction_of_decision_points"] = float(len(traded) / len(df))
    return out


def _trade_stats(net_r: pd.Series, days: pd.Series, ci: bool = True) -> dict:
    """Descriptive statistics of a series of per-trade net R (NaN = no trade). Includes the sample size."""
    mask = net_r.notna()
    r = net_r[mask].to_numpy(dtype=float)
    if len(r) == 0:
        return {"n_trades": 0}
    stats = {
        "n_trades": int(len(r)),
        "mean_net_r": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": profit_factor(r),
        "total_net_r": float(r.sum()),
        "max_drawdown_r": max_drawdown(np.cumsum(r)),
    }
    if ci:
        stats["mean_net_r_ci95"] = _bootstrap_ci(r, days[mask].to_numpy())
    return stats


def load_cached_ohlcv(symbol: str, timeframe: str = "5min") -> pd.DataFrame | None:
    path = DATA_CACHE_DIR / "parquet_cache" / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def evaluate_symbols(symbols: list[str], timeframe: str = "5min", **kwargs) -> tuple[dict, list[Decision]]:
    all_decisions: list[Decision] = []
    per_symbol = {}
    for sym in symbols:
        df = load_cached_ohlcv(sym, timeframe)
        if df is None or len(df) < 600:
            per_symbol[sym] = {"skipped": "no cached data"}
            continue
        d = evaluate_instrument(df, sym, **kwargs)
        per_symbol[sym] = summarize(d)
        all_decisions += d
    return {"per_symbol": per_symbol, "overall": summarize(all_decisions)}, all_decisions


def main() -> None:
    from quant_intelligence.config.watchlist import WATCHLIST_STOCKS

    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--json")
    ap.add_argument("--max-points", type=int, default=None, help="cap decision points per instrument (speed)")
    ap.add_argument("--tie-break", action="store_true", help="include tie-break trades (production default)")
    ap.add_argument("--analogue", choices=["legacy", "robust"], default=None, help="analogue method (default: the configured one)")
    args = ap.parse_args()
    symbols = args.symbols or list(SETTINGS.option_underlyings) + [s["symbol"] for s in WATCHLIST_STOCKS]
    global ALLOW_TIE_BREAK
    ALLOW_TIE_BREAK = args.tie_break

    from quant_intelligence.config.settings import SETTINGS as _S

    print("flags: tie_break=%s vol_premium_model=%s" % (ALLOW_TIE_BREAK, _S.vol_premium_model_in_backtest))
    from quant_intelligence.analogues.analogue_engine import CONFIGS

    report, _ = evaluate_symbols(symbols, max_points=args.max_points,
                                 analogue_config=CONFIGS[args.analogue] if args.analogue else None)
    for sym, r in report["per_symbol"].items():
        print(f"{sym:11}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not k.endswith("ci95")})
    print("\nOVERALL")
    for k, v in report["overall"].items():
        print(f"  {k:32}", tuple(round(x, 3) for x in v) if isinstance(v, tuple) else (round(v, 3) if isinstance(v, float) else v))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
