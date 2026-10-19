"""Read-only data layer for the website. Assembles JSON from the existing research pipeline and stored backtests.

No function here places, modifies or cancels an order, and no credential is ever returned.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.ranking.ranking_engine import tradable
from quant_intelligence.risk.risk_engine import MAX_QUOTE_AGE_SEC, MAX_SLIPPAGE_STOP_FRACTION, MAX_SPREAD_PCT
from quant_intelligence.ui import backtest_workbench as wb
from quant_intelligence.ui.decision_view import (
    evidence_rows, explanation_rows, global_rows, multiple_testing_note, regime_breakdown_rows,
)
from quant_intelligence.utils.timeutil import now_ist


def clean(obj):
    """Recursively make a value JSON-safe: NaN/inf -> None, numpy/pandas scalars -> Python, timestamps -> ISO text."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.ndarray, pd.Series, pd.Index)):
        return [clean(v) for v in obj.tolist()]
    if hasattr(obj, "item"):
        try:
            return clean(obj.item())
        except Exception:
            pass
    return str(obj)


# ---------------------------------------------------------------------------- pipeline cache
# Shared with the Streamlit app: see research/shared_cache.py.
from quant_intelligence.research.shared_cache import LOOKBACK_DAYS, PIPELINE, PIPELINE_TTL_SEC, PipelineCache  # noqa: E402,F401


# ---------------------------------------------------------------------------- views
def _age_minutes(ts) -> float | None:
    try:
        return (now_ist().replace(tzinfo=None) - pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)).total_seconds() / 60
    except Exception:
        return None


def mode_info() -> dict:
    return {
        "paper_only": not SETTINGS.live_mode_fully_authorized,
        "demo_lockdown": SETTINGS.demo_lockdown,
        "label": "LIVE ORDERS LOCKED" if SETTINGS.demo_lockdown else "PAPER MODE",
    }


def overview(output) -> dict:
    ranking = output.ranking
    ok, why_not = tradable(ranking, "PAPER")
    no_trade = bool(ranking.is_no_trade or not ok)
    return clean({
        "instrument": output.instrument,
        "last_bar": output.timestamp,
        "age_minutes": _age_minutes(output.timestamp),
        "max_order_age_minutes": MAX_QUOTE_AGE_SEC / 60,
        "data_quality": output.data_quality_status,
        "data_quality_issues": [str(i) for i in (output.data_quality_issues or [])],
        "regime": {"label": output.regime_label, "confidence": output.regime_confidence,
                   "probabilities": output.regime_probabilities, "axes": output.regime_axes,
                   "clarity": output.regime_clarity, "age_bars": output.regime_age_bars},
        "decision": {"no_trade": no_trade, "strategy": None if no_trade else ranking.selected_strategy,
                     "tie_break": bool(ranking.tie_break), "reason": why_not or ranking.reason},
        "counts": {"evaluated": len(output.strategy_intel),
                   "eligible": sum(1 for s in output.strategy_intel if s.score.eligible)},
        "strategies": explanation_rows(output),
        "provenance": output.provenance,
        "evidence": evidence_rows(output),
        "multiple_testing_note": multiple_testing_note(output),
        "mode": mode_info(),
    })


def strategy_detail(output, name: str) -> dict | None:
    si = next((s for s in output.strategy_intel if s.strategy_name == name), None)
    if si is None:
        return None
    row = next((r for r in explanation_rows(output) if r["Strategy"] == name), None)
    glob = next((r for r in global_rows(output) if r["Strategy"] == name), None)
    return clean({
        "name": name,
        "ranking": row,
        "global_backtest": glob,
        "global_metrics": si.global_metrics,
        "conditional_metrics": si.conditional_metrics,
        "evidence": si.evidence,
        "by_regime": regime_breakdown_rows(output, name),
        "score_components": si.score.components,
        "note": "Historical statistics from past trades, net of modelled costs where stated. Not a prediction.",
    })


def regimes_overview(timeframe: str = "5min") -> dict:
    from quant_intelligence.regimes.universe import UNIVERSE

    rows = UNIVERSE.peek(timeframe)  # never waits: the first call starts the scan and reports it as still running
    return clean({"timeframe": timeframe, "ready": rows is not None, "instruments": rows or [],
                  "note": "Heuristic regime scores from the local candle cache. Display only; not a prediction."})


def backtest_runs(limit: int = 200) -> dict:
    runs = wb.list_runs(limit)
    return clean({"count": len(runs), "runs": [
        {"run_id": r.run_id, "strategy": r.strategy, "instrument": r.instrument, "split": r.split,
         "start": r.start, "end": r.end, "n_trades": r.n_trades, "win_rate": r.win_rate,
         "return_pct": r.total_return_pct, "max_dd_pct": r.max_dd_pct} for r in runs]})


def backtest_detail(run_id: str) -> dict | None:
    run = next((r for r in wb.list_runs(1000) if r.run_id == run_id), None)
    if run is None:
        return None
    cap = wb.starting_capital()
    trades = wb.load_trades(run_id)
    equity = wb.equity_curve(trades, cap)
    dd = wb.drawdown_pct(equity, cap)
    daily = wb.daily_returns_pct(trades, cap)
    bench = wb.benchmark_curve(run.instrument, run.start, run.end, cap)
    return clean({
        "run_id": run_id, "strategy": run.strategy, "instrument": run.instrument, "split": run.split,
        "start": run.start, "end": run.end, "n_trades": run.n_trades, "win_rate": run.win_rate,
        "initial": cap, "final": float(equity.iloc[-1]) if len(equity) else None,
        "return_pct": run.total_return_pct,
        "max_dd_pct": float(dd.min()) if len(dd) else None,
        "sharpe_daily": wb.daily_sharpe(daily),
        "equity": {"x": equity.index, "y": equity.values},
        "drawdown": {"x": dd.index, "y": dd.values},
        "daily": {"x": daily.index, "y": daily.values},
        "benchmark": None if bench is None else {"x": bench.index, "y": bench.values},
        "notes": ["Net of modelled costs. Win rate is gross R > 0.", "Historical backtest, not a prediction."]
        + ([f"Only {run.n_trades} trades: statistics may be unreliable."] if run.n_trades < 30 else []),
    })


def risk_overview(limit: int = 15) -> dict:
    limits = SETTINGS.risk
    checks = [
        ("Order parameters", "Entry, stop and target positive; stop and target on the correct side; whole-lot quantity"),
        ("Data freshness", f"Price no older than {MAX_QUOTE_AGE_SEC // 60} minutes at order time, and data quality OK"),
        ("Spread and slippage", f"Spread at most {MAX_SPREAD_PCT:g}%; half-spread under {MAX_SLIPPAGE_STOP_FRACTION:.0%} of the stop distance; unknown spread fails closed"),
        ("Trade risk", f"At most {limits.max_risk_per_trade_pct:g}% of equity per trade, scaled down in drawdown"),
        ("Daily loss", f"New entries stop at {limits.max_daily_loss_pct:g}% daily loss"),
        ("Drawdown", f"New entries stop at {limits.max_drawdown_pct:g}% drawdown from peak"),
        ("Exposure", f"Strategy {limits.max_strategy_exposure_pct:g}%, portfolio {limits.max_portfolio_exposure_pct:g}%, plus correlated-group limits"),
        ("Kill switch and broker link", "Manual kill switch and broker connectivity always veto"),
    ]
    recent = []
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import RiskEvent

        with get_session() as s:
            for e in s.query(RiskEvent).order_by(RiskEvent.id.desc()).limit(limit).all():
                recent.append({"time": e.timestamp, "type": e.event_type, "reason": e.reason, "decision_id": e.decision_id})
    except Exception:
        pass
    return clean({
        "principle": "The risk engine sees only the account state and the proposed order, never the ranking. "
                     "Any failed mandatory check means NO TRADE.",
        "checks": [{"name": n, "rule": r} for n, r in checks],
        "recent_events": recent,
        "mode": mode_info(),
    })
