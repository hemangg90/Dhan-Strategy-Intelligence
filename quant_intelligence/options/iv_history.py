"""
Implied-volatility history, IV rank and IV percentile.

The application has no source of historical implied volatility (the broker serves live chains only), so the history is
built by RECORDING the ATM IV of every live chain snapshot (`record_iv`, throttled). Until enough history exists, rank and
percentile are UNAVAILABLE with the reason - nothing is estimated from realized volatility or made up.

Definitions (percent IV, all computed from observations strictly BEFORE the decision day, so no future value is used):
  daily series      the last recorded observation of each earlier trading date, newest `window_days` of them
  IV rank           (IV_now - min) / (max - min) x 100 over the window; undefined when max == min
  IV percentile     the share (%) of the window's observations that are strictly below IV_now
  change_Nd         IV_now minus the daily observation N sessions back, in volatility points
  tenor-matched     the same two numbers using only observations whose trading days to expiry are within `tenor_tolerance`
                    of today's. The nearest expiry's ATM IV follows a weekly cycle, so the plain series mixes maturities;
                    the tenor-matched figures are the fairer comparison when enough matching days exist.
A high IV rank is context for option pricing, not an instruction to sell, and a low one is not an instruction to buy.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.timeutil import now_ist

MIN_OBSERVATIONS = 30
WINDOW_DAYS = 252
RECORD_INTERVAL_MIN = 30
_last_recorded: dict[tuple[str, str], dt.datetime] = {}


@dataclass(frozen=True)
class IvStats:
    available: bool
    reason: str = ""
    current_iv: float | None = None
    n_obs: int = 0
    window_days: int = WINDOW_DAYS
    first_observation: dt.date | None = None
    last_observation: dt.date | None = None
    iv_rank: float | None = None
    iv_percentile: float | None = None
    change_1d: float | None = None
    change_5d: float | None = None
    n_tenor: int = 0
    iv_rank_tenor: float | None = None
    iv_percentile_tenor: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def rank_percentile(history: list[float], current: float) -> tuple[float | None, float | None]:
    """(IV rank, IV percentile) of `current` against `history` (see module docstring)."""
    values = np.asarray([h for h in history if h is not None and np.isfinite(h)], float)
    if len(values) == 0 or not np.isfinite(current):
        return None, None
    lo, hi = float(values.min()), float(values.max())
    rank = None if hi <= lo else (current - lo) / (hi - lo) * 100.0
    if rank is not None:
        rank = max(0.0, min(100.0, rank))  # a value outside the historical range is rank 0 or 100
    percentile = float(np.mean(values < current) * 100.0)
    return rank, percentile


def compute_iv_stats(
    daily: list[tuple[dt.date, float, float | None]],
    current_iv: float,
    dte: float | None = None,
    *,
    window_days: int = WINDOW_DAYS,
    min_obs: int = MIN_OBSERVATIONS,
    tenor_tolerance: float = 3.0,
) -> IvStats:
    """`daily` = (date, iv_percent, trading_days_to_expiry) for earlier sessions, oldest first. Pure function."""
    if current_iv is None or not np.isfinite(current_iv) or current_iv <= 0:
        return IvStats(False, "the current implied volatility is not available")
    series = daily[-window_days:]
    n = len(series)
    if n < min_obs:
        return IvStats(False, f"only {n} sessions of recorded IV history; {min_obs} are needed (history is recorded from live "
                              "chain snapshots and has not built up yet)", current_iv, n, window_days,
                       series[0][0] if series else None, series[-1][0] if series else None)
    values = [v for _, v, _ in series]
    rank, pct = rank_percentile(values, current_iv)
    notes = []
    if rank is None:
        notes.append("IV rank is undefined: the window's IV never changed")
    c1 = current_iv - values[-1]
    c5 = current_iv - values[-5] if n >= 5 else None
    rank_t = pct_t = None
    n_t = 0
    if dte is not None:
        matched = [v for _, v, d in series if d is not None and abs(d - dte) <= tenor_tolerance]
        n_t = len(matched)
        if n_t >= min_obs:
            rank_t, pct_t = rank_percentile(matched, current_iv)
        else:
            notes.append(f"tenor-matched figures need {min_obs} sessions near {dte:.0f} trading days to expiry; have {n_t}")
    return IvStats(True, "", current_iv, n, window_days, series[0][0], series[-1][0], rank, pct, c1, c5, n_t, rank_t, pct_t, tuple(notes))


def load_daily_series(underlying: str, before: dt.datetime, window_days: int = WINDOW_DAYS,
                      expiry: str | None = None) -> list[tuple[dt.date, float, float | None]]:
    """The last observation of each trading date strictly before `before`'s date (oldest first). Empty on any error."""
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import IvObservation

        cutoff = dt.datetime.combine(before.date(), dt.time.min)
        with get_session() as session:
            q = session.query(IvObservation).filter(IvObservation.underlying == underlying, IvObservation.recorded_at < cutoff)
            if expiry:
                q = q.filter(IvObservation.expiry == expiry)
            rows = q.order_by(IvObservation.recorded_at).all()
            by_day: dict[dt.date, tuple[float, float | None]] = {}
            for r in rows:  # ascending, so the last write per date is the date's last observation
                by_day[r.recorded_at.date()] = (float(r.atm_iv), r.trading_days_to_expiry)
        days = sorted(by_day)[-window_days:]
        return [(d, by_day[d][0], by_day[d][1]) for d in days]
    except Exception as e:
        log_event("iv_history", f"Could not read IV history for {underlying}: {e}", level="WARNING")
        return []


def iv_stats(underlying: str, current_iv: float | None, asof: dt.datetime | None = None, dte: float | None = None,
             **kw) -> IvStats:
    """Rank/percentile of `current_iv` (percent) for `underlying` at `asof` using only earlier recorded sessions."""
    asof = asof or now_ist()
    if current_iv is None:
        return IvStats(False, "the current implied volatility is not available")
    return compute_iv_stats(load_daily_series(underlying, asof, kw.get("window_days", WINDOW_DAYS)), current_iv, dte, **kw)


def record_iv(chain, now: dt.datetime | None = None, interval_min: float = RECORD_INTERVAL_MIN,
              dte: float | None = None) -> bool:
    """Store the chain's ATM IV (throttled to one row per underlying/expiry per `interval_min`). Never raises; returns
    True when a row was written. A chain with no ATM IV records nothing."""
    try:
        iv = chain.atm_iv
        if iv is None or not np.isfinite(iv) or iv <= 0:
            return False
        now = now or chain.fetched_at or now_ist()
        key = (chain.underlying, chain.expiry)
        last = _last_recorded.get(key)
        if last is not None and (now - last).total_seconds() < interval_min * 60:
            return False
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import IvObservation

        with get_session() as session:
            exists = session.query(IvObservation).filter_by(underlying=chain.underlying, expiry=chain.expiry, recorded_at=now).first()
            if exists is None:
                session.add(IvObservation(underlying=chain.underlying, expiry=chain.expiry, recorded_at=now, atm_iv=float(iv),
                                          spot=chain.spot_price, atm_strike=chain.atm_strike, trading_days_to_expiry=dte))
        _last_recorded[key] = now
        return True
    except Exception as e:
        log_event("iv_history", f"Could not record IV for {getattr(chain, 'underlying', '?')}: {e}", level="WARNING")
        return False
