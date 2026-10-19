"""
Market State Feature Engine.

Converts raw OHLCV bars into a structured, timestamped feature vector.
Every feature at row i is computed using ONLY data up to and including bar i
(rolling windows, shifts, expanding stats) - this is enforced by construction
(all pandas rolling/expanding/shift operations here are backward-looking).

Feature families implemented in this MVP (index/futures only - no options
chain / breadth data available from the synthetic/CSV sources, so those
fields are included as columns but populated as NaN with a clear
`has_derivatives_data` flag rather than being silently omitted, per the
"every feature must have a clear definition" requirement):

  Price/trend:       returns, trend_slope, momentum, market_structure
  Volatility:        atr, atr_percentile, realized_vol, vol_expansion
  Liquidity/volume:  relative_volume, volume_acceleration
  Session/time:      session_phase, time_since_open, opening_range, gap
  VWAP:              vwap, vwap_distance
  Derivatives:       futures_basis, oi, oi_change, pcr, iv, iv_percentile (NaN placeholders)
  Event:             expiry_proximity, event_risk_flag (heuristic placeholders)

Each feature column is documented in FEATURE_DEFINITIONS below.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from quant_intelligence.utils.market_profile import NSE, MarketProfile

FEATURE_DEFINITIONS: dict[str, str] = {
    "return_1": "1-bar log return: ln(close_t / close_{t-1})",
    "return_5": "5-bar log return",
    "return_20": "20-bar log return",
    "trend_slope": "Slope of a linear regression of close over the last 20 bars, normalized by price",
    "momentum_20": "Rate of change of close over the last 20 bars",
    "trend_z": "trend_slope divided by its own rolling standard deviation (500 bars, past bars only): trend strength relative to this instrument's usual",
    "momentum_z": "momentum_20 divided by its own rolling standard deviation (500 bars, past bars only)",
    "or_position": "Where the close sits in the opening range, (close - range low) / (range high - range low); <0 below, >1 above; NaN until the range exists",
    "prev_day_high": "High of the previous trading session",
    "prev_day_low": "Low of the previous trading session",
    "prev_day_range_pos": "Where the close sits in the previous session's range, (close - low) / (high - low); <0 below, >1 above",
    "market_structure": "Higher-high/higher-low (+1), lower-high/lower-low (-1), or mixed (0) over the last 10 bars",
    "atr_14": "Average True Range over 14 bars",
    "atr_pct_of_price": "ATR(14) as a percentage of current close",
    "atr_percentile_100": "Percentile rank of current ATR(14) within the trailing 100 bars",
    "realized_vol_20": "Annualized realized volatility of 1-bar returns over the last 20 bars",
    "vol_expansion": "Ratio of realized_vol_20 to its 100-bar rolling mean; >1 = expanding vol",
    "relative_volume": "Current bar volume divided by the rolling 20-bar average volume",
    "volume_acceleration": "Change in relative_volume vs. the previous bar",
    "vwap": "Session-cumulative volume-weighted average price (resets each session)",
    "vwap_distance_pct": "(close - vwap) / vwap, in percent",
    "opening_range_high": "High of the first 30 minutes of the session (set once, held for the session)",
    "opening_range_low": "Low of the first 30 minutes of the session",
    "gap_pct": "Overnight/opening gap: (session_open - prev_session_close) / prev_session_close",
    "time_since_open_min": "Minutes elapsed since session open",
    "session_phase": "OPEN (first 30min) / MID / CLOSE (last 30min) of the trading session",
    "futures_basis": "Futures - spot basis (NaN: no futures leg in current data sources)",
    "oi": "Open interest (NaN: not available from current data sources)",
    "oi_change_pct": "Change in open interest (NaN placeholder)",
    "pcr": "Put/Call ratio (NaN placeholder)",
    "iv": "Implied volatility (NaN placeholder)",
    "iv_percentile": "IV percentile rank (NaN placeholder)",
    "expiry_proximity_days": "Trading days to nearest weekly expiry (heuristic: next Thursday)",
    "has_derivatives_data": "False in this MVP - options/futures chain not wired up yet",
    "rsi_2": "2-period Relative Strength Index (Wilder), used for short-term mean reversion",
    "rsi_14": "14-period Relative Strength Index (Wilder)",
    "bb_mid_20": "20-bar simple moving average of close (Bollinger midline)",
    "bb_upper_20": "Bollinger upper band: bb_mid_20 + 2 * 20-bar rolling std of close",
    "bb_lower_20": "Bollinger lower band: bb_mid_20 - 2 * 20-bar rolling std of close",
    "bb_pct_b": "Position of close within the Bollinger bands: (close - lower) / (upper - lower)",
    "ema_20": "20-bar exponential moving average of close",
    "ema_50": "50-bar exponential moving average of close",
    "ema_200": "200-bar exponential moving average of close",
    "macd_line": "MACD line: EMA(12) - EMA(26) of close",
    "macd_signal": "9-bar EMA of the MACD line",
    "macd_hist": "MACD histogram: macd_line - macd_signal",
    "supertrend": "Supertrend indicator value (ATR(10), multiplier 3) computed on close",
    "supertrend_direction": "Supertrend trend direction: +1 uptrend (support below price), -1 downtrend (resistance above price)",
    "donchian_high_20": "Highest high over the trailing 20 bars, excluding the current bar (Donchian/Turtle entry channel)",
    "donchian_low_20": "Lowest low over the trailing 20 bars, excluding the current bar",
    "cpr_pivot": "Central Pivot Range pivot: (prev session high + low + close) / 3",
    "cpr_bc": "CPR bottom central: (prev session high + low) / 2",
    "cpr_tc": "CPR top central: 2 * cpr_pivot - cpr_bc",
    "camarilla_r3": "Camarilla resistance 3 from prior session: close + (high-low) * 1.1/4",
    "camarilla_r4": "Camarilla resistance 4 from prior session: close + (high-low) * 1.1/2",
    "camarilla_s3": "Camarilla support 3 from prior session: close - (high-low) * 1.1/4",
    "camarilla_s4": "Camarilla support 4 from prior session: close - (high-low) * 1.1/2",
    "is_inside_bar": "True when the current bar's high/low range is fully contained within the previous bar's range",
    "prev_bar_high": "High of the previous bar (helper for inside-bar breakout)",
    "prev_bar_low": "Low of the previous bar (helper for inside-bar breakout)",
}

DERIVATIVE_PLACEHOLDER_COLUMNS = ["futures_basis", "oi", "oi_change_pct", "pcr", "iv", "iv_percentile"]


def compute_features(df: pd.DataFrame, profile: MarketProfile = NSE) -> pd.DataFrame:
    """Compute the full feature vector for each bar. Input df must have OHLCV columns
    sorted ascending by timestamp. Returns a new DataFrame aligned to df's index with
    a `timestamp` column plus all feature columns.
    """
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    out = pd.DataFrame({"timestamp": df["timestamp"]})

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    log_close = np.log(close)
    out["return_1"] = log_close.diff(1)
    out["return_5"] = log_close.diff(5)
    out["return_20"] = log_close.diff(20)

    out["trend_slope"] = _rolling_slope(close, 20) / close
    out["momentum_20"] = close.pct_change(20)
    out["market_structure"] = _market_structure(high, low, window=10)
    # Instrument-relative versions: divide by the series' own recent spread so "strong" means strong for THIS instrument
    # (a fixed scaling treats a quiet index and a volatile commodity alike). Backward-looking window only.
    for src, dst in (("trend_slope", "trend_z"), ("momentum_20", "momentum_z")):
        spread = out[src].rolling(500, min_periods=60).std()
        out[dst] = out[src] / spread.where(spread > 0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    out["atr_14"] = atr14
    out["atr_pct_of_price"] = atr14 / close
    out["atr_percentile_100"] = atr14.rolling(100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    ret1 = out["return_1"]
    # Annualise by the number of bars in a year: 252 sessions x (session length / bar length). The bar length is read from
    # the data (it used to be hard-coded to 5 minutes, which mis-scales any other timeframe).
    bars_per_year = 252 * profile.session_minutes / infer_bar_minutes(df["timestamp"])
    realized_vol_20 = ret1.rolling(20, min_periods=20).std() * np.sqrt(bars_per_year)
    out["realized_vol_20"] = realized_vol_20
    out["vol_expansion"] = realized_vol_20 / realized_vol_20.rolling(100, min_periods=20).mean()

    avg_vol_20 = volume.rolling(20, min_periods=20).mean()
    out["relative_volume"] = volume / avg_vol_20
    out["volume_acceleration"] = out["relative_volume"].diff(1)

    session_date = df["timestamp"].dt.date
    session_group = session_date.astype(str)
    typical_price = (high + low + close) / 3
    cum_vol = volume.groupby(session_group).cumsum()
    cum_vol_price = (typical_price * volume).groupby(session_group).cumsum()
    out["vwap"] = cum_vol_price / cum_vol
    out["vwap_distance_pct"] = (close - out["vwap"]) / out["vwap"] * 100

    session_open_time = df.groupby(session_group)["timestamp"].transform("min")
    minutes_since_open = (df["timestamp"] - session_open_time).dt.total_seconds() / 60.0
    out["time_since_open_min"] = minutes_since_open

    or_window = df.assign(_sess=session_group, _mins=minutes_since_open)
    or_mask = or_window["_mins"] <= 30
    # Causal opening range: a RUNNING max/min of the bars seen so far inside the first 30 minutes, then held for the
    # rest of the session. (A whole-session max would expose the completed range to bars still inside the window.)
    or_high = high.where(or_mask).groupby(session_group).cummax()
    or_low = low.where(or_mask).groupby(session_group).cummin()
    out["opening_range_high"] = or_high.groupby(session_group).ffill()
    out["opening_range_low"] = or_low.groupby(session_group).ffill()
    or_width = (out["opening_range_high"] - out["opening_range_low"]).where(lambda w: w > 0)
    out["or_position"] = (close - out["opening_range_low"]) / or_width

    prev_session_close = close.groupby(session_group).transform("last").shift(1)
    session_open_price = open_.groupby(session_group).transform("first")
    first_bar_mask = minutes_since_open == minutes_since_open.groupby(session_group).transform("min")
    gap = (session_open_price - prev_session_close) / prev_session_close * 100
    out["gap_pct"] = np.where(first_bar_mask, gap, np.nan)
    out["gap_pct"] = pd.Series(out["gap_pct"]).groupby(session_group).transform(lambda s: s.ffill())

    def phase(m):
        if m <= 30:
            return "OPEN"
        if m >= profile.session_minutes - 30:  # last 30 min of the session
            return "CLOSE"
        return "MID"

    out["session_phase"] = minutes_since_open.apply(phase)

    out["rsi_2"] = _rsi(close, 2)
    out["rsi_14"] = _rsi(close, 14)

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    out["bb_mid_20"] = bb_mid
    out["bb_upper_20"] = bb_mid + 2 * bb_std
    out["bb_lower_20"] = bb_mid - 2 * bb_std
    band_width = out["bb_upper_20"] - out["bb_lower_20"]
    out["bb_pct_b"] = (close - out["bb_lower_20"]) / band_width.replace(0, np.nan)

    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    out["ema_20"] = ema20
    out["ema_50"] = ema50
    out["ema_200"] = ema200

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal

    supertrend, supertrend_dir = _supertrend(high, low, close, atr_period=10, multiplier=3.0)
    out["supertrend"] = supertrend
    out["supertrend_direction"] = supertrend_dir

    out["donchian_high_20"] = high.shift(1).rolling(20, min_periods=20).max()
    out["donchian_low_20"] = low.shift(1).rolling(20, min_periods=20).min()

    prev_session_high = high.groupby(session_group).transform("max").groupby(session_group).transform("last")
    prev_sess_high = high.groupby(session_group).transform("max")
    prev_sess_low = low.groupby(session_group).transform("min")
    prev_sess_close = close.groupby(session_group).transform("last")
    # Shift the prior session's H/L/C forward onto today's bars for pivot-style levels.
    sess_hlc = pd.DataFrame(
        {"_sess": session_group, "h": prev_sess_high, "l": prev_sess_low, "c": prev_sess_close}
    ).drop_duplicates("_sess")
    sess_hlc = sess_hlc.assign(
        pivot=(sess_hlc["h"] + sess_hlc["l"] + sess_hlc["c"]) / 3,
        bc=(sess_hlc["h"] + sess_hlc["l"]) / 2,
        rng=(sess_hlc["h"] - sess_hlc["l"]),
    )
    sess_hlc["tc"] = 2 * sess_hlc["pivot"] - sess_hlc["bc"]
    sess_hlc["r3"] = sess_hlc["c"] + sess_hlc["rng"] * 1.1 / 4
    sess_hlc["r4"] = sess_hlc["c"] + sess_hlc["rng"] * 1.1 / 2
    sess_hlc["s3"] = sess_hlc["c"] - sess_hlc["rng"] * 1.1 / 4
    sess_hlc["s4"] = sess_hlc["c"] - sess_hlc["rng"] * 1.1 / 2
    sess_hlc = sess_hlc.set_index("_sess").shift(1)  # prior session's levels apply to today

    session_index = pd.Series(session_group.values, index=out.index)
    out["cpr_pivot"] = session_index.map(sess_hlc["pivot"]).values
    out["cpr_bc"] = session_index.map(sess_hlc["bc"]).values
    out["cpr_tc"] = session_index.map(sess_hlc["tc"]).values
    out["camarilla_r3"] = session_index.map(sess_hlc["r3"]).values
    out["camarilla_r4"] = session_index.map(sess_hlc["r4"]).values
    out["camarilla_s3"] = session_index.map(sess_hlc["s3"]).values
    out["camarilla_s4"] = session_index.map(sess_hlc["s4"]).values
    out["prev_day_high"] = session_index.map(sess_hlc["h"]).values
    out["prev_day_low"] = session_index.map(sess_hlc["l"]).values
    prev_day_width = pd.Series(out["prev_day_high"] - out["prev_day_low"]).where(lambda w: w > 0)
    out["prev_day_range_pos"] = (close - out["prev_day_low"]) / prev_day_width
    del prev_session_high

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    out["prev_bar_high"] = prev_high
    out["prev_bar_low"] = prev_low
    out["is_inside_bar"] = (high < prev_high) & (low > prev_low)

    for col in DERIVATIVE_PLACEHOLDER_COLUMNS:
        out[col] = np.nan
    out["has_derivatives_data"] = False

    out["expiry_proximity_days"] = df["timestamp"].apply(_days_to_next_thursday)

    return out


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)  # only gains: RSI is 100
    # No movement at all (0 gains and 0 losses) is UNDEFINED, not "maximally overbought": returning 100 there would
    # make a flat or illiquid stretch look like a mean-reversion SHORT signal.
    return rsi.where(~((avg_loss == 0) & (avg_gain == 0)), np.nan)


def infer_bar_minutes(timestamps: pd.Series, default: float = 5.0) -> float:
    """Median spacing, in minutes, between consecutive bars of the same session (overnight gaps excluded)."""
    ts = pd.to_datetime(timestamps)
    gaps = ts.diff().dt.total_seconds() / 60.0
    intraday = gaps[(ts.dt.date == ts.shift(1).dt.date) & (gaps > 0)]
    minutes = float(intraday.median()) if len(intraday) else default
    return minutes if np.isfinite(minutes) and minutes > 0 else default


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, atr_period: int, multiplier: float):
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean()
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction = pd.Series(1, index=close.index)
    supertrend = pd.Series(np.nan, index=close.index)

    for i in range(1, len(close)):
        if pd.isna(atr.iloc[i]):
            continue
        prev_final_upper = final_upper.iloc[i - 1]
        prev_final_lower = final_lower.iloc[i - 1]

        if pd.isna(prev_final_upper) or basic_upper.iloc[i] < prev_final_upper or close.iloc[i - 1] > prev_final_upper:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_final_upper

        if pd.isna(prev_final_lower) or basic_lower.iloc[i] > prev_final_lower or close.iloc[i - 1] < prev_final_lower:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_final_lower

        if pd.isna(prev_final_upper):
            direction.iloc[i] = 1
        elif close.iloc[i] > prev_final_upper:
            direction.iloc[i] = 1
        elif close.iloc[i] < prev_final_lower:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        supertrend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    direction = direction.where(atr.notna())
    return supertrend, direction


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def slope(y):
        y = np.asarray(y)
        return ((x - x_mean) * (y - y.mean())).sum() / denom

    return series.rolling(window, min_periods=window).apply(slope, raw=True)


def _market_structure(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    rolling_high = high.rolling(window, min_periods=window)
    rolling_low = low.rolling(window, min_periods=window)
    is_higher_high = high == rolling_high.max()
    is_higher_low = low > rolling_low.min().shift(1)
    is_lower_high = high < rolling_high.max().shift(1)
    is_lower_low = low == rolling_low.min()

    structure = pd.Series(0, index=high.index)
    structure = structure.mask(is_higher_high & is_higher_low, 1)
    structure = structure.mask(is_lower_high & is_lower_low, -1)
    return structure


def _days_to_next_thursday(ts: dt.datetime) -> int:
    days_ahead = (3 - ts.weekday()) % 7  # Thursday = weekday 3
    return int(days_ahead)
