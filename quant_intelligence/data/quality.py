"""
Market-data quality validation.

Every dataset that enters the pipeline is checked for: missing candles,
duplicate timestamps, timezone consistency, zero/negative prices, abnormal
OHLC relationships (high < low, close outside [low, high], etc.), missing
volume, and staleness. The result feeds market_data_metadata and, at
decision time, the Market State Engine's data_quality_status — poor quality
data must lead to NO TRADE, never a silently-degraded decision.
"""
from __future__ import annotations

import datetime as dt
from quant_intelligence.utils.timeutil import now_ist
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_intelligence.utils.market_calendar import most_recent_expected_bar_time
from quant_intelligence.utils.market_profile import NSE, MarketProfile

SESSION_OPEN_MIN = NSE.open_min
SESSION_CLOSE_MIN = NSE.close_min
# More than this fraction of bars outside the session means the clock/timezone is wrong
# (a real misalignment), as opposed to a few stray pre/post-close prints.
MAX_OUT_OF_SESSION_FRACTION = 0.02

QUALITY_OK = "OK"
QUALITY_DEGRADED = "DEGRADED"
QUALITY_FAIL = "FAIL"

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ["open", "high", "low", "close"]

# Finer, human-facing classification derived from the report (the three-valued `status` is what gates trading).
CATEGORY_VALID = "VALID"
CATEGORY_VALID_WITH_WARNINGS = "VALID_WITH_WARNINGS"
CATEGORY_INCOMPLETE = "INCOMPLETE"
CATEGORY_STALE = "STALE"
CATEGORY_INVALID = "INVALID"
CATEGORY_UNAVAILABLE = "UNAVAILABLE"


@dataclass
class QualityReport:
    status: str
    issues: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_duplicates: int = 0
    n_missing_candles: int = 0
    n_ohlc_violations: int = 0
    n_zero_or_negative: int = 0
    n_missing_volume: int = 0
    is_stale: bool = False
    last_timestamp: dt.datetime | None = None
    n_nonfinite: int = 0  # rows with a NaN / infinite / non-numeric price
    n_bad_timestamps: int = 0  # rows whose timestamp cannot be parsed
    n_out_of_order: int = 0  # rows that arrived earlier than the row before them (the data is sorted afterwards)
    n_irregular_intervals: int = 0  # intra-session steps much shorter than the bar interval (wrong timeframe / duplicates)

    @property
    def category(self) -> str:
        """VALID, VALID_WITH_WARNINGS, INCOMPLETE (gaps, duplicates, missing volume), STALE, INVALID or UNAVAILABLE.
        Only OK data may drive a new trading decision (`status`); the category says why it is not OK."""
        if self.n_rows == 0:
            return CATEGORY_UNAVAILABLE
        if self.status == QUALITY_FAIL:
            return CATEGORY_INVALID
        if self.is_stale:
            return CATEGORY_STALE
        if self.n_missing_candles or self.n_duplicates or self.n_missing_volume:
            return CATEGORY_INCOMPLETE
        return CATEGORY_VALID_WITH_WARNINGS if self.issues else CATEGORY_VALID

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["last_timestamp"] = str(self.last_timestamp) if self.last_timestamp else None
        d["category"] = self.category
        return d


def out_of_session_mask(timestamps: pd.Series, profile: MarketProfile = NSE) -> pd.Series:
    ts = pd.to_datetime(timestamps)
    minutes = ts.dt.hour * 60 + ts.dt.minute
    return (minutes < profile.open_min) | (minutes >= profile.close_min)


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe_minutes: int,
    staleness_threshold_minutes: int = 30,
    now: dt.datetime | None = None,
    profile: MarketProfile = NSE,
) -> QualityReport:
    issues: list[str] = []
    n_rows = len(df)

    if n_rows == 0:
        return QualityReport(status=QUALITY_FAIL, issues=["no data rows"], n_rows=0)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:  # a verdict, not a KeyError
        return QualityReport(status=QUALITY_FAIL, issues=[f"missing required column(s): {', '.join(missing_columns)}"], n_rows=n_rows)

    parsed_ts = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad_timestamps = int(parsed_ts.isna().sum())
    if n_bad_timestamps > 0:
        issues.append(f"{n_bad_timestamps} rows with a missing or unparseable timestamp")
    n_out_of_order = int((parsed_ts.diff().dt.total_seconds() < 0).sum())
    if n_out_of_order > 0:
        issues.append(f"{n_out_of_order} rows arrived out of chronological order (sorted before use)")

    df = df.assign(timestamp=parsed_ts).dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return QualityReport(status=QUALITY_FAIL, issues=issues + ["no rows with a valid timestamp"], n_rows=n_rows,
                             n_bad_timestamps=n_bad_timestamps)

    # Prices must be finite numbers. Comparisons with NaN are False, so without this check a NaN close would pass
    # every test below and the row would be rated OK.
    prices = df[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    n_nonfinite = int((~np.isfinite(prices.to_numpy(dtype=float))).any(axis=1).sum())
    if n_nonfinite > 0:
        issues.append(f"{n_nonfinite} rows with a missing, non-numeric or infinite price")

    df = df.assign(**{c: prices[c] for c in PRICE_COLUMNS})
    n_duplicates = int(df["timestamp"].duplicated().sum())
    if n_duplicates > 0:
        issues.append(f"{n_duplicates} duplicate timestamps")

    # Missing candles: within each trading session, gaps larger than the expected
    # timeframe are flagged (holidays/weekends across days are expected and excluded).
    ts = pd.to_datetime(df["timestamp"])
    same_day = ts.dt.date == ts.dt.date.shift(1)
    deltas_minutes = ts.diff().dt.total_seconds() / 60.0
    expected = timeframe_minutes
    gap_mask = same_day & (deltas_minutes > expected * 1.5)
    n_missing_candles = int(gap_mask.sum())
    if n_missing_candles > 0:
        issues.append(f"{n_missing_candles} intra-session gaps larger than expected bar interval")
    # Steps well under the bar interval mean the data is not at the requested timeframe (or has near-duplicate stamps).
    n_irregular = int((same_day & (deltas_minutes > 0) & (deltas_minutes < expected * 0.5)).sum())
    if n_irregular > 0:
        issues.append(f"{n_irregular} intra-session steps much shorter than the {expected}-minute bar interval")

    n_out_of_session = 0
    if timeframe_minutes < 1440:
        n_out_of_session = int(out_of_session_mask(df["timestamp"], profile).sum())
        if n_out_of_session / n_rows > MAX_OUT_OF_SESSION_FRACTION:
            issues.append(f"{n_out_of_session} bars outside the {profile.label} session (timezone/data error)")
        else:
            n_out_of_session = 0  # a few stray pre/post-close prints; removed by clean_ohlcv

    zero_or_neg = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    n_zero_or_negative = int(zero_or_neg.sum())
    if n_zero_or_negative > 0:
        issues.append(f"{n_zero_or_negative} rows with zero/negative price")

    ohlc_violation = (
        (df["high"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
    )
    n_ohlc_violations = int(ohlc_violation.sum())
    if n_ohlc_violations > 0:
        issues.append(f"{n_ohlc_violations} rows with abnormal OHLC relationships")

    volume = pd.to_numeric(df["volume"], errors="coerce")
    n_missing_volume = int((volume.isna() | (volume < 0) | ~np.isfinite(volume.fillna(0))).sum())
    if n_missing_volume > 0:
        issues.append(f"{n_missing_volume} rows with missing/invalid volume")

    last_timestamp = ts.max()
    now = now or now_ist()
    reference = most_recent_expected_bar_time(now, profile)
    last_ts_py = last_timestamp.to_pydatetime()
    if last_ts_py.tzinfo is None:
        last_ts_py = last_ts_py.replace(tzinfo=reference.tzinfo)
    is_stale = bool((reference - last_ts_py) > dt.timedelta(minutes=staleness_threshold_minutes))
    if is_stale:
        behind = reference - last_ts_py
        age = f"{behind.days}d {behind.seconds // 3600}h" if behind.days else f"{behind.seconds // 3600}h {behind.seconds % 3600 // 60}m"
        issues.append(
            f"data is stale: last bar {pd.Timestamp(last_timestamp):%d %b %H:%M}, "
            f"{age} behind the {reference:%d %b %H:%M} IST market clock"
        )

    # Determine overall status.
    hard_fail = (n_zero_or_negative > 0 or n_ohlc_violations > 0 or n_rows < 5 or n_out_of_session > 0
                 or n_nonfinite > 0 or n_bad_timestamps > 0)
    degraded = n_duplicates > 0 or n_missing_candles > 0 or n_missing_volume > 0 or is_stale

    if hard_fail:
        status = QUALITY_FAIL
    elif degraded:
        status = QUALITY_DEGRADED
    else:
        status = QUALITY_OK

    return QualityReport(
        status=status,
        issues=issues,
        n_rows=n_rows,
        n_duplicates=n_duplicates,
        n_missing_candles=n_missing_candles,
        n_ohlc_violations=n_ohlc_violations,
        n_zero_or_negative=n_zero_or_negative,
        n_missing_volume=n_missing_volume,
        is_stale=is_stale,
        last_timestamp=last_timestamp.to_pydatetime() if isinstance(last_timestamp, pd.Timestamp) else last_timestamp,
        n_nonfinite=n_nonfinite,
        n_bad_timestamps=n_bad_timestamps,
        n_out_of_order=n_out_of_order,
        n_irregular_intervals=n_irregular,
    )


def clean_ohlcv(df: pd.DataFrame, timeframe_minutes: int | None = None, profile: MarketProfile = NSE) -> pd.DataFrame:
    """Deterministic cleanup: drop duplicate timestamps (keep last) and, for intraday
    data, bars outside the 09:15-15:30 session (stray pre/post-close prints); sort, reset index."""
    if timeframe_minutes is not None and timeframe_minutes < 1440 and len(df):
        df = df[~out_of_session_mask(df["timestamp"], profile)]
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if timeframe_minutes is not None and timeframe_minutes < 1440 and len(df):
        df = _drop_bad_opening_bars(df)
    return df


def _ohlc_violation_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["high"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
    )


def _drop_bad_opening_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Drop OHLC-inconsistent bars that are the FIRST bar of a session.

    Dhan occasionally reports an opening bar whose open lies outside its own high-low range
    (seen on MCX silver at 09:00 - the exchange's opening-call price). Dropping it loses one bar
    and leaves no intra-session gap (the next bar simply becomes the first). The same violation
    anywhere else in a session is still left in place so validate_ohlcv hard-fails it.
    """
    is_first = df.groupby(pd.to_datetime(df["timestamp"]).dt.date).cumcount() == 0
    drop = _ohlc_violation_mask(df) & is_first
    if drop.any():
        from quant_intelligence.utils.logging_utils import log_event

        log_event("data_quality", f"Dropped {int(drop.sum())} inconsistent session-opening bar(s)", level="WARNING")
    return df[~drop].reset_index(drop=True)
