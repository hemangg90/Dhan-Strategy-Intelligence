"""Session VWAP features and an approximate volume profile.

VWAP is session-aware (resets each date) and uses the bar volume the data provides. Many index series carry no volume;
then every VWAP and volume-profile field is reported as UNAVAILABLE - the typical price is never substituted.

The volume profile is an APPROXIMATION: OHLCV has no volume-at-price, so each bar's volume is spread evenly over the
bar's high-low range across fixed price bins. POC/VAH/VAL from it show where time-and-volume-weighted trading happened,
not true traded volume per price. The result says so in `approximation`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

APPROXIMATION = "bar volume spread uniformly over each bar's high-low range (no volume-at-price in OHLCV)"


def _has_volume(df: pd.DataFrame) -> bool:
    if "volume" not in df.columns:
        return False
    v = pd.to_numeric(df["volume"], errors="coerce")
    return bool(v.notna().any() and float(v.fillna(0).sum()) > 0)


def vwap_features(df: pd.DataFrame, atr: float | None = None, slope_bars: int = 6, rejection_window: int = 12) -> dict:
    """VWAP picture at the last bar of `df`. Keys: available, reason, vwap, distance_pct, distance_atr, slope_atr_per_bar,
    crossings_session, rejections_recent (bars that wicked through VWAP and closed back on one side)."""
    out = {"available": False, "reason": "", "vwap": None, "distance_pct": None, "distance_atr": None,
           "slope_atr_per_bar": None, "crossings_session": None, "rejections_recent": None}
    needed = {"timestamp", "high", "low", "close", "volume"}
    if df is None or not needed <= set(getattr(df, "columns", [])) or len(df) < 2:
        out["reason"] = "price history with volume is not available"
        return out
    if not _has_volume(df):
        out["reason"] = "the instrument reports no volume, so VWAP is not defined"
        return out
    d = df.sort_values("timestamp").reset_index(drop=True)
    session = d["timestamp"].dt.date
    vol = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0)
    typical = (d["high"] + d["low"] + d["close"]) / 3.0
    cum_v = vol.groupby(session).cumsum()
    cum_pv = (typical * vol).groupby(session).cumsum()
    vwap = (cum_pv / cum_v.where(cum_v > 0)).to_numpy(float)
    t = len(d) - 1
    if not np.isfinite(vwap[t]):
        out["reason"] = "no volume traded yet in the current session"
        return out
    close = d["close"].to_numpy(float)
    high, low = d["high"].to_numpy(float), d["low"].to_numpy(float)
    out.update(available=True, vwap=float(vwap[t]), distance_pct=float((close[t] - vwap[t]) / vwap[t] * 100.0))
    if atr and atr > 0:
        out["distance_atr"] = float((close[t] - vwap[t]) / atr)
        k = min(slope_bars, t)
        prior = vwap[t - k] if k > 0 else np.nan
        if k > 0 and np.isfinite(prior) and session.iloc[t] == session.iloc[t - k]:
            out["slope_atr_per_bar"] = float((vwap[t] - prior) / k / atr)
    today = (session == session.iloc[t]).to_numpy()
    side = np.sign(close - vwap)
    side_today = side[today & np.isfinite(vwap)]
    side_today = side_today[side_today != 0]
    out["crossings_session"] = int(np.sum(side_today[1:] != side_today[:-1])) if len(side_today) > 1 else 0
    rej = 0
    for j in range(max(0, t - rejection_window + 1), t + 1):
        if not (np.isfinite(vwap[j]) and session.iloc[j] == session.iloc[t]):
            continue
        if close[j] > vwap[j] and low[j] < vwap[j]:
            rej += 1  # dipped below VWAP and closed back above it
        elif close[j] < vwap[j] and high[j] > vwap[j]:
            rej += 1  # pushed above VWAP and closed back below it
    out["rejections_recent"] = rej
    return out


def volume_profile(df: pd.DataFrame, bins: int = 24, value_area: float = 0.70, sessions: int = 1) -> dict:
    """Approximate profile over the last `sessions` sessions of `df`. Keys: available, reason, poc, vah, val, bins,
    approximation, sessions."""
    base = {"available": False, "reason": "", "poc": None, "vah": None, "val": None, "bins": bins,
            "approximation": APPROXIMATION, "sessions": sessions}
    needed = {"timestamp", "high", "low", "volume"}
    if df is None or not needed <= set(getattr(df, "columns", [])) or len(df) < 2:
        base["reason"] = "price history with volume is not available"
        return base
    if not _has_volume(df):
        base["reason"] = "the instrument reports no volume, so no profile can be built"
        return base
    d = df.sort_values("timestamp")
    keep = list(dict.fromkeys(d["timestamp"].dt.date.tolist()))[-sessions:]
    d = d[d["timestamp"].dt.date.isin(keep)]
    lo, hi = float(d["low"].min()), float(d["high"].max())
    if not hi > lo:
        base["reason"] = "the price range is zero"
        return base
    edges = np.linspace(lo, hi, bins + 1)
    profile = np.zeros(bins)
    vols = pd.to_numeric(d["volume"], errors="coerce").fillna(0).to_numpy(float)
    for h, l, v in zip(d["high"].to_numpy(float), d["low"].to_numpy(float), vols):
        if v <= 0:
            continue
        if h <= l:
            profile[min(int((l - lo) / (hi - lo) * bins), bins - 1)] += v
            continue
        overlap = np.clip(np.minimum(edges[1:], h) - np.maximum(edges[:-1], l), 0, None)
        total = overlap.sum()
        if total > 0:
            profile += v * overlap / total
    if profile.sum() <= 0:
        base["reason"] = "no volume in the window"
        return base
    centers = (edges[:-1] + edges[1:]) / 2.0
    poc = int(np.argmax(profile))
    chosen = {poc}
    covered = float(profile[poc])
    target = value_area * float(profile.sum())
    left, right = poc - 1, poc + 1
    while covered < target and (left >= 0 or right < bins):
        lv = float(profile[left]) if left >= 0 else -1.0
        rv = float(profile[right]) if right < bins else -1.0
        if rv >= lv:
            chosen.add(right)
            covered += max(rv, 0.0)
            right += 1
        else:
            chosen.add(left)
            covered += max(lv, 0.0)
            left -= 1
    base.update(available=True, poc=float(centers[poc]), vah=float(edges[max(chosen) + 1]), val=float(edges[min(chosen)]))
    return base
