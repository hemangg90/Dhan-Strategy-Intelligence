"""Data helpers for the Backtest Workbench page: list persisted Backtest Lab runs and derive their curves.

Everything is computed from stored trades (net of costs). Nothing is estimated or filled in: a run with no
trades has no curves, and the benchmark is omitted when no cached price history covers the run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_intelligence.analytics.metrics import annualised_sharpe
from quant_intelligence.config.settings import DATA_CACHE_DIR, SETTINGS

LAB_EXCLUDED_SPLITS = ("RESEARCH",)  # RESEARCH rows are the live pipeline's internal evaluations, not lab runs


@dataclass
class RunSummary:
    run_id: str
    strategy: str
    instrument: str
    split: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    n_trades: int
    win_rate: float | None
    total_return_pct: float | None  # net of costs, on starting capital
    max_dd_pct: float | None  # peak-to-trough on starting capital, from the run's rupee drawdown
    created: pd.Timestamp | None

    @property
    def title(self) -> str:
        return f"{self.strategy}"


def starting_capital() -> float:
    return float(SETTINGS.paper_starting_capital)


def list_runs(limit: int = 200) -> list[RunSummary]:
    """Most recent lab runs that produced at least one trade."""
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import BacktestRun

    cap = starting_capital()
    out: list[RunSummary] = []
    with get_session() as s:
        rows = (
            s.query(BacktestRun)
            .filter(BacktestRun.split.notin_(LAB_EXCLUDED_SPLITS))
            .order_by(BacktestRun.id.desc())
            .limit(limit * 3)
            .all()
        )
        for r in rows:
            m = r.metrics or {}
            n = int(m.get("n_trades") or 0)
            if n == 0:
                continue
            pnl = m.get("net_pnl_total")
            dd = m.get("max_drawdown")
            out.append(
                RunSummary(
                    run_id=r.run_id,
                    strategy=r.strategy_name,
                    instrument=r.instrument,
                    split=r.split or "",
                    start=pd.Timestamp(r.start_date) if r.start_date else None,
                    end=pd.Timestamp(r.end_date) if r.end_date else None,
                    n_trades=n,
                    win_rate=m.get("win_rate"),
                    total_return_pct=(pnl / cap * 100) if pnl is not None else None,
                    max_dd_pct=(dd / cap * 100) if dd is not None else None,
                    created=pd.Timestamp(r.created_at) if r.created_at else None,
                )
            )
            if len(out) >= limit:
                break
    return out


def filter_runs(runs: list[RunSummary], query: str = "", outcome: str = "All") -> list[RunSummary]:
    q = query.strip().lower()
    res = []
    for r in runs:
        if q and q not in f"{r.strategy} {r.instrument} {r.split} {r.run_id}".lower():
            continue
        if outcome == "Win" and not (r.total_return_pct or 0) > 0:
            continue
        if outcome == "Loss" and not (r.total_return_pct or 0) < 0:
            continue
        res.append(r)
    return res


def load_trades(run_id: str) -> pd.DataFrame:
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import BacktestTrade

    with get_session() as s:
        rows = (
            s.query(BacktestTrade.exit_timestamp, BacktestTrade.entry_timestamp, BacktestTrade.net_pnl, BacktestTrade.r_multiple)
            .filter(BacktestTrade.run_id == run_id)
            .all()
        )
    df = pd.DataFrame(rows, columns=["exit", "entry", "net_pnl", "r"])
    df["exit"] = pd.to_datetime(df["exit"]).fillna(pd.to_datetime(df["entry"]))
    return df.dropna(subset=["exit", "net_pnl"]).sort_values("exit").reset_index(drop=True)


def equity_curve(trades: pd.DataFrame, capital: float) -> pd.Series:
    """Account value after each closed trade, starting from `capital`."""
    if trades.empty:
        return pd.Series(dtype=float)
    return pd.Series(capital + trades["net_pnl"].cumsum().to_numpy(), index=pd.DatetimeIndex(trades["exit"]))


def drawdown_pct(equity: pd.Series, capital: float) -> pd.Series:
    if equity.empty:
        return equity
    peak = np.maximum.accumulate(np.concatenate([[capital], equity.to_numpy()]))[1:]
    return (equity / peak - 1.0) * 100


def daily_returns_pct(trades: pd.DataFrame, capital: float) -> pd.Series:
    """Each day's net P&L as a percent of the account value at the start of that day."""
    if trades.empty:
        return pd.Series(dtype=float)
    daily = trades.groupby(pd.to_datetime(trades["exit"]).dt.normalize())["net_pnl"].sum()
    start_equity = capital + daily.cumsum().shift(1, fill_value=0.0)
    return daily / start_equity * 100


def daily_sharpe(daily_pct: pd.Series, min_days: int = 10) -> float | None:
    """Annualised Sharpe of daily returns (risk-free 0). None with too few days or no variance."""
    return annualised_sharpe(daily_pct.to_numpy(), 252, min_days)  # one shared, tested definition


def benchmark_curve(instrument: str, start, end, capital: float, timeframe: str = "5min") -> pd.Series | None:
    """Buy-and-hold of the instrument over the run window from cached candles, scaled to `capital`. None if unavailable."""
    path = DATA_CACHE_DIR / "parquet_cache" / f"{instrument}_{timeframe}.parquet"
    if start is None or end is None or not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        window = df[(df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))]
        if len(window) < 2:
            return None
        daily = window.set_index("timestamp")["close"].resample("1D").last().dropna()
        if len(daily) < 2:
            return None
        return daily / float(daily.iloc[0]) * capital
    except Exception:
        return None
