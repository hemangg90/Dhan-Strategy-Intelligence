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

Method: each regime has a scoring rule built from already-computed features
(trend_slope, momentum, atr_percentile, vol_expansion, relative_volume,
market_structure, expiry_proximity). Scores are converted into a probability
distribution via a softmax, a transparent way to turn several weak evidence signals into a relative ranking of
regimes. The label is the argmax; the full distribution is always stored, and "confidence" is simply the largest share.
"""
from __future__ import annotations

import numpy as np

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


def _safe(x, default=0.0):
    if x is None:
        return default
    try:
        if np.isnan(x):
            return default
    except TypeError:
        return default
    return x


def classify_regime(features: dict) -> tuple[str, dict[str, float]]:
    """Return (label, probabilities). `features` is a MarketState.features dict
    (or equivalent) with at least trend_slope, momentum_20, atr_percentile_100,
    vol_expansion, relative_volume, market_structure, expiry_proximity_days.
    """
    trend_slope = _safe(features.get("trend_slope"))
    momentum = _safe(features.get("momentum_20"))
    atr_pct = _safe(features.get("atr_percentile_100"), 0.5)
    vol_expansion = _safe(features.get("vol_expansion"), 1.0)
    rel_vol = _safe(features.get("relative_volume"), 1.0)
    structure = _safe(features.get("market_structure"))
    expiry_days = _safe(features.get("expiry_proximity_days"), 3)

    scores = {r: 0.0 for r in REGIMES}

    # Trend regimes: driven by slope, momentum and market structure agreement.
    trend_strength = np.tanh(trend_slope * 500) * 0.6 + np.tanh(momentum * 20) * 0.4
    scores["TREND_UP"] = max(trend_strength, 0) * 3 + (1 if structure > 0 else 0)
    scores["TREND_DOWN"] = max(-trend_strength, 0) * 3 + (1 if structure < 0 else 0)

    # Range: low trend strength, structure flat.
    scores["RANGE"] = (1 - abs(trend_strength)) * 2 + (1 if structure == 0 else 0)

    # Volatility regimes: from ATR percentile and vol_expansion ratio.
    scores["HIGH_VOL"] = atr_pct * 3
    scores["LOW_VOL"] = (1 - atr_pct) * 3
    scores["VOL_EXPANSION"] = max(vol_expansion - 1, 0) * 4
    scores["VOL_CONTRACTION"] = max(1 - vol_expansion, 0) * 4
    scores["COMPRESSION"] = (1 - atr_pct) * 2 + max(1 - vol_expansion, 0) * 2

    # Event-driven: near weekly expiry (proxy for scheduled event risk in Indian index derivatives).
    scores["EVENT_DRIVEN"] = max(0, (1 - expiry_days / 3)) * 2

    # Liquidity stress: very low relative volume combined with high volatility.
    scores["LIQUIDITY_STRESS"] = max(0, (1 - rel_vol)) * 1.5 + atr_pct * 1.5 if rel_vol < 0.5 else 0.0

    values = np.array([scores[r] for r in REGIMES])
    # Softmax with temperature: keeps distribution meaningfully spread rather than
    # collapsing to a near-one-hot vector, which would misrepresent classifier confidence.
    temperature = 1.5
    exp_scores = np.exp((values - values.max()) / temperature)
    probs = exp_scores / exp_scores.sum()

    prob_dict = {r: float(p) for r, p in zip(REGIMES, probs)}
    label = max(prob_dict, key=prob_dict.get)
    return label, prob_dict


def regime_confidence(probabilities: dict[str, float]) -> float:
    return max(probabilities.values()) if probabilities else 0.0
