"""Does a regime label separate what happens next? Descriptive check of the classifier against realised data.

For every bar the label was known at the close of that bar; this module looks at what happened over the NEXT `horizon`
bars of the same session (forward realised volatility, forward absolute move, forward signed return) and groups it by
label. It answers, per instrument: "after bars labelled HIGH_VOL, was the next stretch actually more volatile than
usual?" Trend labels are reported, not assumed to predict direction. Confidence intervals cluster by trading day.
Purely a diagnostic: nothing here feeds a decision, and with weeks of data most cells are small (n is reported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_intelligence.analogues.analogue_engine import _cluster_robust_stderr

DEFAULT_HORIZON = 12  # 12 five-minute bars = one hour


def regime_forward_stats(ohlcv: pd.DataFrame, regimes: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """One row per label plus an ALL row. `ohlcv` needs timestamp and close; `regimes` is `classify_series` output."""
    px = ohlcv[["timestamp", "close"]].sort_values("timestamp").reset_index(drop=True)
    log_close = np.log(px["close"].astype(float))
    one_bar = log_close.diff()
    forward_vol = one_bar.rolling(horizon).std().shift(-horizon)  # std of the next `horizon` one-bar returns
    forward_ret = log_close.shift(-horizon) - log_close
    day = px["timestamp"].dt.date
    same_session = day.shift(-horizon) == day  # the look-ahead window must not cross into another session
    frame = pd.DataFrame({
        "timestamp": px["timestamp"], "day": day,
        "fwd_vol": forward_vol.where(same_session), "fwd_ret": forward_ret.where(same_session),
    }).merge(regimes[["timestamp", "label"]], on="timestamp", how="inner").dropna(subset=["fwd_vol", "fwd_ret"])
    if frame.empty:
        return pd.DataFrame(columns=["regime", "n", "n_days", "fwd_vol", "fwd_vol_vs_all", "fwd_abs_move",
                                     "fwd_ret", "fwd_ret_ci_low", "fwd_ret_ci_high"])
    overall_vol = float(frame["fwd_vol"].mean())
    rows = []
    for name, g in [("ALL", frame)] + list(frame.groupby("label")):
        ret = g["fwd_ret"].to_numpy()
        se = _cluster_robust_stderr(ret, g["day"].to_numpy())
        mean = float(ret.mean())
        rows.append({
            "regime": name, "n": int(len(g)), "n_days": int(g["day"].nunique()),
            "fwd_vol": float(g["fwd_vol"].mean()), "fwd_vol_vs_all": float(g["fwd_vol"].mean() / overall_vol),
            "fwd_abs_move": float(np.abs(ret).mean()), "fwd_ret": mean,
            "fwd_ret_ci_low": None if np.isnan(se) else mean - 1.96 * se,
            "fwd_ret_ci_high": None if np.isnan(se) else mean + 1.96 * se,
        })
    return pd.DataFrame(rows)
