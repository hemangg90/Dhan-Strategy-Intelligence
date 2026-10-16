"""Trade-performance analytics for the daily report. Pure pandas: takes the trade frame from
`ui.position_views.positions_frame` and returns tables / dicts. Nothing here touches the database or the network.

Honesty rules baked in:
* Every figure that is a mean carries its sample size, and a strategy is only called good or bad when the sample
  can support it. Under MIN_TRADES closed trades the verdict is "too few trades to conclude" - however good the
  average looks. A statistically significant edge needs |t| >= 2 on the mean R multiple.
* R multiple = net P&L (after charges) / premium risked (|entry - stop| x quantity). It puts trades of different
  size and price on one scale.
* Stale (never-closed) positions have no result; they are reported separately and never counted as wins or losses.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quant_intelligence.ui.position_views import trade_stats

MIN_TRADES = 30  # below this no strategy is called good or bad
SIGNIFICANT_T = 2.0


def with_r(df: pd.DataFrame) -> pd.DataFrame:
    """Add r_multiple, date, hour, weekday and tag_label columns to a positions_frame."""
    out = df.copy()
    risk = (out["entry"].astype(float) - out["stop"].astype(float)).abs() * out["qty"].astype(float)
    out["r_multiple"] = np.where((risk > 0) & out["net_pnl"].notna(), out["net_pnl"].astype(float) / risk.replace(0, np.nan), np.nan)
    closed_at = pd.to_datetime(out["closed_at"])
    opened_at = pd.to_datetime(out["opened_at"])
    out["date"] = closed_at.dt.date
    out["hour"] = opened_at.dt.hour
    out["weekday"] = opened_at.dt.day_name()
    out["tag_label"] = np.where(out["tag"].fillna("") == "", "Normal", out["tag"].fillna(""))
    return out


def closed_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["status"] == "CLOSED") & df["net_pnl"].notna()]


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate (percent). (nan, nan) with no trades."""
    if n <= 0:
        return float("nan"), float("nan")
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half) * 100.0, min(1.0, centre + half) * 100.0


def verdict(n: int, avg_r: float | None, t_stat: float | None) -> tuple[str, str]:
    """(code, plain-English text) - what the evidence supports, no more."""
    if n < MIN_TRADES or avg_r is None or t_stat is None or not math.isfinite(t_stat):
        return "TOO_FEW", f"Too few trades to conclude ({n} of {MIN_TRADES} needed)"
    if t_stat >= SIGNIFICANT_T and avg_r > 0:
        return "EDGE", "Positive and statistically significant"
    if t_stat <= -SIGNIFICANT_T and avg_r < 0:
        return "LOSING", "Losing, and statistically significant"
    if avg_r > 0:
        return "POSITIVE_NS", "Positive on average but not significant"
    return "NO_EDGE", "No evidence of an edge"


def summarise(df: pd.DataFrame) -> dict:
    """Headline numbers for a set of trades (closed ones only are counted)."""
    c = closed_only(df)
    stats = trade_stats(c) if len(c) else trade_stats(c)
    n = stats["trades"]
    r = c["r_multiple"].dropna().to_numpy(float) if n and "r_multiple" in c else np.array([])
    avg_r = float(r.mean()) if len(r) else None
    se = float(r.std(ddof=1) / math.sqrt(len(r))) if len(r) > 1 else None
    t = (avg_r / se) if (avg_r is not None and se and se > 0) else None
    lo, hi = wilson_interval(stats["wins"], n)
    code, text = verdict(len(r), avg_r, t)
    return {**stats, "win_ci_low": lo, "win_ci_high": hi, "avg_r": avg_r, "r_se": se, "t_stat": t,
            "n_r": int(len(r)), "verdict": code, "verdict_text": text}


def strategy_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy, best expectancy first among those with enough trades, then the rest by net P&L."""
    c = closed_only(df)
    rows = []
    for name, g in c.groupby("strategy", dropna=False):
        s = summarise(g)
        rows.append({
            "strategy": name if isinstance(name, str) else "(unknown)", "trades": s["trades"], "wins": s["wins"],
            "win_rate": s["win_rate"], "win_ci_low": s["win_ci_low"], "win_ci_high": s["win_ci_high"],
            "net_pnl": s["net"], "charges": s["charges"], "expectancy": s["expectancy"],
            "profit_factor": s["profit_factor"], "avg_r": s["avg_r"], "r_se": s["r_se"], "t_stat": s["t_stat"],
            "payoff": s["payoff"], "max_drawdown": s["max_drawdown"], "verdict": s["verdict"],
            "verdict_text": s["verdict_text"],
        })
    cols = ["strategy", "trades", "wins", "win_rate", "win_ci_low", "win_ci_high", "net_pnl", "charges", "expectancy",
            "profit_factor", "avg_r", "r_se", "t_stat", "payoff", "max_drawdown", "verdict", "verdict_text"]
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    enough = out["trades"] >= MIN_TRADES
    out["_k1"] = (~enough).astype(int)
    out["_k2"] = np.where(enough, -out["avg_r"].fillna(-1e9), -out["net_pnl"])
    return out.sort_values(["_k1", "_k2"]).drop(columns=["_k1", "_k2"]).reset_index(drop=True)


def breakdown(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Trades, win rate, net P&L, expectancy and mean R by `key` (instrument, market, option_type, exit_reason,
    hour, weekday, tag_label ...), biggest net P&L first."""
    c = closed_only(df)
    cols = [key, "trades", "win_rate", "net_pnl", "expectancy", "avg_r"]
    if c.empty:
        return pd.DataFrame(columns=cols)
    g = c.groupby(key, dropna=False)
    out = pd.DataFrame({
        "trades": g.size(),
        "win_rate": g["net_pnl"].apply(lambda s: (s > 0).mean() * 100.0),
        "net_pnl": g["net_pnl"].sum(),
        "expectancy": g["net_pnl"].mean(),
        "avg_r": g["r_multiple"].mean(),
    }).reset_index()
    out[key] = out[key].fillna("(none)")
    return out.sort_values("net_pnl", ascending=False).reset_index(drop=True)[cols]


def daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Per closing date: trades, wins, net P&L, cumulative net P&L and drawdown from the running peak."""
    c = closed_only(df)
    cols = ["date", "trades", "wins", "net_pnl", "cumulative", "drawdown"]
    if c.empty:
        return pd.DataFrame(columns=cols)
    g = c.groupby("date")
    out = pd.DataFrame({"trades": g.size(), "wins": g["net_pnl"].apply(lambda s: int((s > 0).sum())), "net_pnl": g["net_pnl"].sum()})
    out = out.sort_index().reset_index()
    out["cumulative"] = out["net_pnl"].cumsum()
    out["drawdown"] = out["cumulative"] - np.maximum(out["cumulative"].cummax(), 0.0)  # the start (0) is a peak
    return out[cols]


def expected_vs_realised(df: pd.DataFrame) -> pd.DataFrame:
    """Per strategy: what the ranker expected (mean expected R at entry) against what trading delivered (mean R).

    Only trades that recorded an expectation are compared (older trades do not have one). A negative gap means
    the strategy under-delivers its backtest - the usual overfitting / cost symptom the ranker must not hide."""
    c = closed_only(df)
    cols = ["strategy", "trades", "expected_r", "realised_r", "gap", "gap_se", "gap_t"]
    if c.empty or "expected_r" not in c:
        return pd.DataFrame(columns=cols)
    c = c.dropna(subset=["expected_r", "r_multiple"])
    rows = []
    for name, g in c.groupby("strategy"):
        diff = (g["r_multiple"] - g["expected_r"]).to_numpy(float)
        se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else None
        rows.append({"strategy": name, "trades": len(g), "expected_r": float(g["expected_r"].mean()),
                     "realised_r": float(g["r_multiple"].mean()), "gap": float(diff.mean()), "gap_se": se,
                     "gap_t": (float(diff.mean()) / se) if se else None})
    return pd.DataFrame(rows, columns=cols).sort_values("gap").reset_index(drop=True) if rows else pd.DataFrame(columns=cols)


def unresolved(df: pd.DataFrame) -> dict:
    """Positions with no result: STALE (left open when the app slept) and still OPEN."""
    if df.empty:
        return {"stale": 0, "open": 0, "invested": 0.0, "dates": []}
    stale = df[df["status"] == "STALE"]
    open_ = df[df["status"] == "OPEN"]
    dates = sorted({pd.Timestamp(x).date().isoformat() for x in stale["opened_at"].dropna()})
    return {"stale": int(len(stale)), "open": int(len(open_)), "invested": float(stale["invested"].fillna(0).sum()), "dates": dates}


def leader(table: pd.DataFrame) -> dict:
    """Which strategy is working best - stated only as strongly as the evidence allows."""
    if table.empty:
        return {"name": None, "text": "No closed trades yet, so no strategy can be judged."}
    edge = table[table["verdict"] == "EDGE"]
    if len(edge):
        top = edge.sort_values("avg_r", ascending=False).iloc[0]
        return {"name": top["strategy"], "text": f"{top['strategy']} shows a statistically significant edge "
                f"(mean R {top['avg_r']:+.2f} over {int(top['trades'])} trades)."}
    enough = table[table["trades"] >= MIN_TRADES]
    if len(enough):
        top = enough.sort_values("avg_r", ascending=False).iloc[0]
        return {"name": None, "text": f"No strategy has a statistically significant edge yet. Best of those with "
                f">= {MIN_TRADES} trades: {top['strategy']} (mean R {top['avg_r']:+.2f}, {int(top['trades'])} trades, "
                f"{top['verdict_text'].lower()})."}
    top = table.sort_values("net_pnl", ascending=False).iloc[0]
    return {"name": None, "text": f"No strategy has {MIN_TRADES} closed trades yet, so none can be called best. "
            f"Highest net P&L so far: {top['strategy']} ({top['net_pnl']:+,.0f} over {int(top['trades'])} trades) - "
            "treat that as noise until the sample grows."}
