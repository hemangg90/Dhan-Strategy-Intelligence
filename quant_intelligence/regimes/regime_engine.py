"""
Regime Engine.

Transparent, rule-based regime classifier (no deep learning, no fitted model). Produces a normalised
SCORE distribution over regimes rather than a single hard label.

IMPORTANT: these are heuristic scores passed through a softmax so they sum to 1. They are NOT calibrated
probabilities: nothing has been fitted or validated against realised outcomes, and the regimes overlap (a market can be
trending AND high-volatility), which caps the largest share. The regime label is currently used for display, logging and
order tags only; it is not an input to the analogue search or the strategy ranking.

Regimes: TREND_UP, TREND_DOWN, RANGE, COMPRESSION, VOL_EXPANSION,
VOL_CONTRACTION, HIGH_VOL, LOW_VOL, EVENT_DRIVEN, LIQUIDITY_STRESS.

Method: each regime has a scoring rule built from already-computed features. Scores are converted into a distribution
via a softmax. Three things make it work across very different instruments (indices, stocks, MCX commodities):

  * Instrument-normalised trend inputs. When the feature engine supplies `trend_z` / `momentum_z` (the slope and the
    momentum divided by that instrument's own recent standard deviation, past bars only) the trend score uses them, so
    "strong trend" means strong FOR THIS INSTRUMENT. Without them the old fixed scalings are used.
  * Overlays only where they make sense. EVENT_DRIVEN needs a known weekly-expiry rule (the option underlyings);
    elsewhere it scores 0. LIQUIDITY_STRESS needs traded volume; without volume it scores 0.
  * Persistence. `classify_series` smooths the scores over time and only switches label after a challenger has led for a
    few bars, so the history does not flicker bar to bar. `classify_regime` (one bar) stays instantaneous.

Two axes are reported next to the label (`axes`): direction (UP / DOWN / NEUTRAL) and volatility (LOW / NORMAL / HIGH,
with EXPANDING / CONTRACTING / STEADY). `clarity` (0-1) says how decisive the label is: the margin between the top two
scores and whether the label agrees with the axes. It is a heuristic, like the scores, not a probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = [
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "COMPRESSION",
    "VOL_EXPANSION",
    "VOL_CONTRACTION",
    "HIGH_VOL",
    "LOW_VOL",
    "EVENT_DRIVEN",
    "LIQUIDITY_STRESS",
]
_IDX = {r: i for i, r in enumerate(REGIMES)}

TEMPERATURE = 1.5
Z_SCALE = 2.0  # a trend / momentum z-score of 2 maps to tanh(1) = 0.76 of full strength
DIRECTION_THRESHOLD = 0.25
# Persistence (classify_series)
SMOOTHING_ALPHA = 0.3  # EMA weight of the newest bar's scores
SWITCH_MARGIN = 0.3  # a challenger must lead the current label by this many score units ...
SWITCH_BARS = 4  # ... for this many consecutive bars before the label changes

# What each label implies about the two axes, for the agreement part of `clarity`.
_EXPECTED_AXES = {
    "TREND_UP": {"direction": "UP"},
    "TREND_DOWN": {"direction": "DOWN"},
    "RANGE": {"direction": "NEUTRAL"},
    "COMPRESSION": {"volatility": "LOW", "volatility_trend": "CONTRACTING"},
    "VOL_EXPANSION": {"volatility_trend": "EXPANDING"},
    "VOL_CONTRACTION": {"volatility_trend": "CONTRACTING"},
    "HIGH_VOL": {"volatility": "HIGH"},
    "LOW_VOL": {"volatility": "LOW"},
}


def _safe(x, default=0.0):
    if x is None:
        return default
    try:
        if np.isnan(x):
            return default
    except TypeError:
        return default
    return x


def _arr(values, default: float) -> np.ndarray:
    """Float array with missing / non-numeric entries replaced by `default`."""
    a = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(a), a, default)


def _has_expiry_rule(instrument: str | None) -> bool:
    """Whether the next-Thursday expiry heuristic means something for this instrument. Unknown instrument (None) keeps
    the historical behaviour (rule applies)."""
    if instrument is None:
        return True
    try:
        from quant_intelligence.config.settings import SETTINGS

        return instrument.upper() in {u.upper() for u in SETTINGS.option_underlyings}
    except Exception:
        return True


def _inputs(get, n: int) -> dict:
    """Pull the classifier inputs out of a dict-like (`get(name)` -> scalar or array-like of length n)."""
    def col(name, default):
        v = get(name)
        return _arr(np.full(n, np.nan) if v is None else v, default)

    tz = col("trend_z", np.nan)
    mz = col("momentum_z", np.nan)
    slope = col("trend_slope", 0.0)
    mom = col("momentum_20", 0.0)
    raw_volume = get("relative_volume")
    rel_vol = col("relative_volume", 1.0)
    return {
        "trend_strength": np.where(
            np.isfinite(tz) & np.isfinite(mz),
            0.6 * np.tanh(tz / Z_SCALE) + 0.4 * np.tanh(mz / Z_SCALE),
            0.6 * np.tanh(slope * 500) + 0.4 * np.tanh(mom * 20),  # legacy fixed scaling
        ),
        "atr_pct": np.clip(col("atr_percentile_100", 0.5), 0.0, 1.0),
        "vol_expansion": col("vol_expansion", 1.0),
        "rel_vol": rel_vol,
        "has_volume": np.isfinite(_arr(np.full(n, np.nan) if raw_volume is None else raw_volume, np.nan)),
        "structure": col("market_structure", 0.0),
        "expiry_days": col("expiry_proximity_days", 3.0),
    }


def _score_matrix(x: dict, has_expiry: bool) -> np.ndarray:
    """(n, 10) heuristic scores, columns in REGIMES order."""
    n = len(x["trend_strength"])
    ts, atr, ve, rv = x["trend_strength"], x["atr_pct"], x["vol_expansion"], x["rel_vol"]
    st, ex = x["structure"], x["expiry_days"]
    s = np.zeros((n, len(REGIMES)))
    s[:, _IDX["TREND_UP"]] = np.maximum(ts, 0) * 3 + (st > 0)
    s[:, _IDX["TREND_DOWN"]] = np.maximum(-ts, 0) * 3 + (st < 0)
    s[:, _IDX["RANGE"]] = (1 - np.abs(ts)) * 2 + (st == 0)
    s[:, _IDX["HIGH_VOL"]] = atr * 3
    s[:, _IDX["LOW_VOL"]] = (1 - atr) * 3
    s[:, _IDX["VOL_EXPANSION"]] = np.maximum(ve - 1, 0) * 4
    s[:, _IDX["VOL_CONTRACTION"]] = np.maximum(1 - ve, 0) * 4
    s[:, _IDX["COMPRESSION"]] = (1 - atr) * 2 + np.maximum(1 - ve, 0) * 2
    if has_expiry:  # near weekly expiry: proxy for scheduled event risk in Indian index derivatives
        s[:, _IDX["EVENT_DRIVEN"]] = np.maximum(0, 1 - ex / 3) * 2
    # Very low relative volume together with high volatility. Without traded volume this cannot be assessed -> 0.
    stress = (np.maximum(0, 1 - rv) * 1.5 + atr * 1.5) * (rv < 0.5) * x["has_volume"]
    s[:, _IDX["LIQUIDITY_STRESS"]] = stress
    return s


def _softmax(scores: np.ndarray) -> np.ndarray:
    z = (scores - scores.max(axis=1, keepdims=True)) / TEMPERATURE
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _axes(x: dict, i: int = 0) -> dict:
    ts, atr, ve = float(x["trend_strength"][i]), float(x["atr_pct"][i]), float(x["vol_expansion"][i])
    return {
        "direction": "UP" if ts > DIRECTION_THRESHOLD else "DOWN" if ts < -DIRECTION_THRESHOLD else "NEUTRAL",
        "volatility": "HIGH" if atr > 0.7 else "LOW" if atr < 0.3 else "NORMAL",
        "volatility_trend": "EXPANDING" if ve > 1.15 else "CONTRACTING" if ve < 0.87 else "STEADY",
    }


def _clarity(label: str, axes: dict, sorted_scores: np.ndarray) -> float:
    """0-1. Half is the margin between the top two scores (saturating at 0.5 score units), half is whether the label
    agrees with the axes (labels with no axis expectation, e.g. EVENT_DRIVEN, count as agreeing)."""
    margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
    expected = _EXPECTED_AXES.get(label, {})
    agree = 1.0 if all(axes.get(k) == v for k, v in expected.items()) else 0.0
    return 0.5 * min(max(margin, 0.0) / 0.5, 1.0) + 0.5 * agree


def classify_detail(features: dict, instrument: str | None = None) -> dict:
    """One bar. Returns label, probabilities, axes, clarity (see module docstring)."""
    x = _inputs(lambda name: [features[name]] if name in features and features[name] is not None else None, 1)
    scores = _score_matrix(x, _has_expiry_rule(instrument))
    probs = _softmax(scores)[0]
    label = REGIMES[int(np.argmax(probs))]
    axes = _axes(x)
    return {
        "label": label,
        "probabilities": {r: float(p) for r, p in zip(REGIMES, probs)},
        "axes": axes,
        "clarity": _clarity(label, axes, np.sort(scores[0])),
    }


def classify_regime(features: dict, instrument: str | None = None) -> tuple[str, dict[str, float]]:
    """Return (label, probabilities) for ONE bar. `features` is a MarketState.features dict (or equivalent); missing
    values take neutral defaults. Pass `instrument` so overlays that do not apply to it (weekly-expiry events for a
    stock or commodity) are switched off."""
    d = classify_detail(features, instrument)
    return d["label"], d["probabilities"]


def classify_series(features: pd.DataFrame, instrument: str | None = None) -> pd.DataFrame:
    """Regime for every bar with persistence. Causal: row i uses only rows <= i.

    Columns: timestamp, raw_label (instantaneous argmax), label (smoothed, with hysteresis), confidence (score share of the
    label shown), clarity, age_bars (bars since the label last changed), and the axes.
    Path dependent by design (the label depends on how it got there), so a very short window can start differently
    from a long one; the first bars of a series are a warm-up."""
    n = len(features)
    if n == 0:
        return pd.DataFrame(columns=["timestamp", "raw_label", "label", "confidence", "clarity", "age_bars",
                                     "direction", "volatility", "volatility_trend"])
    x = _inputs(lambda name: features[name].to_numpy() if name in features.columns else None, n)
    raw = _score_matrix(x, _has_expiry_rule(instrument))

    smooth = np.empty_like(raw)
    smooth[0] = raw[0]
    for i in range(1, n):
        smooth[i] = SMOOTHING_ALPHA * raw[i] + (1 - SMOOTHING_ALPHA) * smooth[i - 1]

    labels = np.empty(n, dtype=int)
    ages = np.zeros(n, dtype=int)
    current, challenger, streak, age = int(np.argmax(smooth[0])), -1, 0, 0
    for i in range(n):
        top = int(np.argmax(smooth[i]))
        if top != current and smooth[i, top] - smooth[i, current] > SWITCH_MARGIN:
            streak = streak + 1 if top == challenger else 1
            challenger = top
            if streak >= SWITCH_BARS:
                current, challenger, streak, age = top, -1, 0, 0
        else:
            challenger, streak = -1, 0
        labels[i] = current
        ages[i] = age
        age += 1

    probs = _softmax(smooth)
    out = pd.DataFrame({
        "timestamp": features["timestamp"].to_numpy() if "timestamp" in features.columns else np.arange(n),
        "raw_label": [REGIMES[j] for j in np.argmax(raw, axis=1)],
        "label": [REGIMES[j] for j in labels],
        "confidence": probs[np.arange(n), labels],  # share of the label that is shown (it may lag the argmax)
        "age_bars": ages,
    })
    out.attrs["probabilities"] = {r: float(p) for r, p in zip(REGIMES, probs[-1])}  # latest bar, smoothed scores
    axes = [_axes(x, i) for i in range(n)]
    out["direction"] = [a["direction"] for a in axes]
    out["volatility"] = [a["volatility"] for a in axes]
    out["volatility_trend"] = [a["volatility_trend"] for a in axes]
    srt = np.sort(smooth, axis=1)
    out["clarity"] = [_clarity(out["label"].iat[i], axes[i], srt[i]) for i in range(n)]
    return out


def transition_matrix(labels: pd.Series) -> pd.DataFrame:
    """Row-normalised counts of label(t) -> label(t+1) for the labels that occur (rows sum to 1)."""
    s = pd.Series(labels).reset_index(drop=True)
    pairs = pd.crosstab(s.iloc[:-1].to_numpy(), s.iloc[1:].to_numpy())
    return pairs.div(pairs.sum(axis=1).replace(0, np.nan), axis=0)


def regime_confidence(probabilities: dict[str, float]) -> float:
    return max(probabilities.values()) if probabilities else 0.0
