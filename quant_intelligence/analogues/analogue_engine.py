"""
Historical Analogue Engine.

Given the current market state, find historical strategy_observations whose market-state features were most similar,
and compute conditional performance metrics from those analogues: "how has this strategy performed under conditions
like today's" instead of relying on unconditional (global) statistics. A transparent k-NN: every analogue used is
inspectable (timestamp, features, outcome).

Two methods, selected by an `AnalogueConfig` (default from SETTINGS.analogue_method):

LEGACY  six features, mean / standard-deviation scaling, Euclidean distance, the nearest `top_k` rows.

ROBUST  - more context: instrument-relative trend and momentum, gap, position in the opening range and in the previous
          day's range, on top of volatility and VWAP distance;
        - robust scaling (median / MAD, clipped at +-4) so one extreme value cannot dominate a distance;
        - a distance cutoff taken from the observations' own pairwise distance distribution, so "no good match" returns
          few analogues instead of the least-bad 30;
        - independence: at most `max_per_day` analogues per trading day (adjacent bars of one session are one piece of
          evidence, not thirty), and analogues of a similar time of day when enough exist;
        - kernel weights exp(-(d/h)^2): closer analogues count more; the effective sample size is reported;
        - an informativeness test (`conditioning_test`): do analogue sets for random reference observations differ
          from the strategy's overall mean as much as THIS set does? If not, the estimate falls back to the overall
          (unconditional) mean, which is then the better-supported number.

None of this can create evidence the history does not contain: with a few weeks of data the honest outcome is often
"not enough comparable days" or "analogues add no information". Both are reported, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COMPARISON_FEATURES = [
    "trend_slope",
    "momentum_20",
    "atr_percentile_100",
    "vol_expansion",
    "relative_volume",
    "vwap_distance_pct",
]
ROBUST_FEATURES = [
    "trend_z",
    "momentum_z",
    "atr_percentile_100",
    "vol_expansion",
    "relative_volume",
    "vwap_distance_pct",
    "gap_pct",
    "or_position",
    "prev_day_range_pos",
]
CONTEXT_COLUMNS = ["time_since_open_min"]
# Everything an observation row must carry for either method.
ALL_FEATURES = list(dict.fromkeys(COMPARISON_FEATURES + ROBUST_FEATURES + CONTEXT_COLUMNS))

MIN_SAMPLE_FOR_ANY_CONFIDENCE = 10
# Confidence is about how many INDEPENDENT trading days the analogues span, not how many rows:
# the nearest analogues are mostly adjacent bars of the same session (autocorrelated), so
# 30 rows from 2 days carry the evidence of ~2 observations, not 30.
MIN_DAYS_FOR_MEDIUM_CONFIDENCE = 6
MIN_DAYS_FOR_HIGH_CONFIDENCE = 15
MIN_EFFECTIVE_N_FOR_MEDIUM = 5.0  # kernel-weighted analogues: effective sample size needed for more than LOW
INFORMATIVE_P_MAX = 0.20  # conditioning test: above this p-value the analogues are treated as adding no information
CLIP_Z = 4.0


@dataclass(frozen=True)
class AnalogueConfig:
    name: str = "legacy"
    features: tuple = tuple(COMPARISON_FEATURES)
    weights: dict = field(default_factory=dict)  # per-feature multiplier on the squared difference (default 1)
    scaling: str = "zscore"  # "zscore" or "robust"
    distance: str = "euclid"  # "euclid" or "mahalanobis" (shrunk covariance)
    max_per_day: int | None = None
    cutoff_quantile: float | None = None  # distance cutoff = this quantile of the observations' pairwise distances
    kernel: bool = False
    phase_window_min: float | None = None  # restrict to similar time of day (minutes since open) when enough remain
    min_after_phase: int = 20
    test_information: bool = False


LEGACY = AnalogueConfig()
ROBUST = AnalogueConfig(
    name="robust",
    features=tuple(ROBUST_FEATURES),
    scaling="robust",
    max_per_day=2,
    cutoff_quantile=0.30,
    kernel=True,
    phase_window_min=90.0,
    test_information=True,
)
CONFIGS = {"legacy": LEGACY, "robust": ROBUST}


def default_config() -> AnalogueConfig:
    try:
        from quant_intelligence.config.settings import SETTINGS

        return CONFIGS.get(SETTINGS.analogue_method, LEGACY)
    except Exception:
        return LEGACY


# --------------------------------------------------------------------------------------------- index
class AnalogueIndex:
    """Observations in a scaled feature space, ready to answer distance queries. Built once per search; the
    informativeness test reuses it for its leave-one-out reference queries."""

    def __init__(self, observations: pd.DataFrame, current: dict, cfg: AnalogueConfig):
        self.cfg = cfg
        self.obs = observations
        cols, mats = [], []
        for c in cfg.features:
            if c not in observations.columns:
                continue
            col = pd.to_numeric(observations[c], errors="coerce")
            q = current.get(c)
            if q is None or not np.isfinite(q) or col.notna().sum() < 2 or not (col.std() > 0):
                continue
            cols.append(c)
            mats.append(col.to_numpy(dtype=float))
        self.cols = cols
        self.valid = bool(cols)
        if not self.valid:
            return
        M = np.column_stack(mats)
        q = np.array([float(current[c]) for c in cols])
        if cfg.scaling == "robust":
            center = np.nanmedian(M, axis=0)
            mad = np.nanmedian(np.abs(M - center), axis=0) * 1.4826
            std = np.nanstd(M, axis=0, ddof=1)
            scale = np.where(mad > 0, mad, std)
            M = np.where(np.isnan(M), center, M)  # a missing value is "typical" (z = 0)
        else:
            center = np.nanmean(M, axis=0)
            M = np.where(np.isnan(M), center, M)  # as before: fill with the mean, THEN take the spread
            scale = M.std(axis=0, ddof=1)
        Z, zq = (M - center) / scale, (q - center) / scale
        if cfg.scaling == "robust":
            Z, zq = np.clip(Z, -CLIP_Z, CLIP_Z), np.clip(zq, -CLIP_Z, CLIP_Z)
        w = np.sqrt(np.array([float(cfg.weights.get(c, 1.0)) for c in cols]))
        Z, zq = Z * w, zq * w
        if cfg.distance == "mahalanobis" and len(cols) > 1 and len(Z) > len(cols) + 2:
            cov = np.atleast_2d(np.cov(Z.T))
            shrunk = 0.7 * cov + 0.3 * np.eye(len(cols)) * float(np.mean(np.diag(cov)))
            L = np.linalg.cholesky(np.linalg.pinv(shrunk))
            Z, zq = Z @ L, zq @ L
        self.Z, self.zq = Z, zq

        ts = pd.to_datetime(observations["entry_timestamp"]) if "entry_timestamp" in observations.columns else None
        self.day = pd.factorize(ts.dt.date)[0] if ts is not None else np.arange(len(observations))
        self.minutes = (pd.to_numeric(observations["time_since_open_min"], errors="coerce").to_numpy(dtype=float)
                        if "time_since_open_min" in observations.columns else None)
        self.cutoff = self._pairwise_cutoff() if cfg.cutoff_quantile else None

    def _pairwise_cutoff(self) -> float | None:
        n = len(self.Z)
        if n < 5:
            return None
        sample = self.Z if n <= 250 else self.Z[np.random.default_rng(0).choice(n, 250, replace=False)]
        diff = sample[:, None, :] - sample[None, :, :]
        d = np.sqrt((diff**2).sum(axis=2))
        off_diag = d[~np.eye(len(sample), dtype=bool)]
        cut = float(np.quantile(off_diag, self.cfg.cutoff_quantile))
        return cut if cut > 0 else None

    def distances(self, q: np.ndarray | None = None) -> np.ndarray:
        return np.sqrt(((self.Z - (self.zq if q is None else q)) ** 2).sum(axis=1))

    def select(self, d: np.ndarray, top_k: int, minutes_now: float | None = None, exclude_day: int | None = None) -> np.ndarray:
        """Row positions of the chosen analogues, nearest first."""
        cfg = self.cfg
        allowed = np.ones(len(d), dtype=bool)
        if cfg.phase_window_min and self.minutes is not None and minutes_now is not None and np.isfinite(minutes_now):
            near = np.abs(self.minutes - minutes_now) <= cfg.phase_window_min  # NaN compares False
            if near.sum() >= cfg.min_after_phase:
                allowed &= near
        if self.cutoff is not None:
            allowed &= d <= self.cutoff
        if exclude_day is not None:
            allowed &= self.day != exclude_day
        chosen, per_day = [], {}
        for i in np.argsort(d, kind="stable"):
            if not allowed[i]:
                continue
            if cfg.max_per_day is not None:
                if per_day.get(self.day[i], 0) >= cfg.max_per_day:
                    continue
                per_day[self.day[i]] = per_day.get(self.day[i], 0) + 1
            chosen.append(i)
            if len(chosen) >= top_k:
                break
        return np.array(chosen, dtype=int)

    def kernel_weights(self, d: np.ndarray) -> np.ndarray:
        h = self.cutoff if self.cutoff else (float(np.median(d)) if len(d) else 1.0)
        h = h if h and h > 0 else 1.0
        return np.exp(-((d / h) ** 2))


def analogue_search(
    current_features: dict,
    observations: pd.DataFrame,
    top_k: int = 30,
    config: AnalogueConfig | None = None,
    as_of=None,
) -> tuple[pd.DataFrame, AnalogueIndex | None]:
    """Returns (analogues, index). `as_of`: ignore observations whose outcome was not yet known then (exit_timestamp
    later than `as_of`) when the column exists."""
    cfg = config or default_config()
    if len(observations) == 0:
        return observations.assign(similarity_score=pd.Series(dtype=float)), None
    obs = observations
    if as_of is not None and "exit_timestamp" in obs.columns:
        obs = obs[pd.to_datetime(obs["exit_timestamp"]) <= pd.Timestamp(as_of)]
        if len(obs) == 0:
            return obs.assign(similarity_score=pd.Series(dtype=float)), None
    index = AnalogueIndex(obs, current_features, cfg)
    if not index.valid:  # no feature carries information on both sides
        return obs.head(0).assign(similarity_score=pd.Series(dtype=float)), None
    d = index.distances()
    minutes_now = current_features.get("time_since_open_min")
    pos = index.select(d, top_k, minutes_now=float(minutes_now) if minutes_now is not None else None)
    picked = obs.iloc[pos].copy()
    if cfg.cutoff_quantile and index.cutoff:
        picked["similarity_score"] = np.clip(1 - d[pos] / index.cutoff, 0.0, 1.0)  # absolute: 0 = at the cutoff
    else:
        max_dist = d.max() if len(d) and d.max() > 0 else 1.0
        picked["similarity_score"] = 1 - d[pos] / max_dist
    picked["distance"] = d[pos]
    if cfg.kernel:
        picked["weight"] = index.kernel_weights(d)[pos]
    picked.attrs.update({"method": cfg.name, "features_used": list(index.cols), "cutoff": index.cutoff,
                         "n_observations": len(obs)})
    return picked, index


def find_analogues(
    current_features: dict,
    observations: pd.DataFrame,
    top_k: int = 30,
    config: AnalogueConfig | None = None,
    as_of=None,
) -> pd.DataFrame:
    """observations: DataFrame with a column per feature (flattened out of market_state_features) plus r_multiple,
    outcome, entry_timestamp (and exit_timestamp, time_since_open_min where available). Returns the chosen analogues
    with `similarity_score` (0-1, higher = closer), nearest first."""
    return analogue_search(current_features, observations, top_k, config, as_of)[0]


# ------------------------------------------------------------------------------ informativeness
def conditioning_test(
    index: AnalogueIndex,
    r_all: np.ndarray,
    observed_mean: float,
    top_k: int = 30,
    n_ref: int = 60,
    seed: int = 11,
    min_analogues: int = MIN_SAMPLE_FOR_ANY_CONFIDENCE,
) -> dict:
    """Do analogue sets differ from the strategy's overall mean more than chance explains?

    For random reference observations (excluding the reference's own trading day) build the analogue set exactly as
    for the live query, and record how far its (weighted) mean sits from the overall mean. The p-value is the share
    of those reference deviations at least as large as the live one. A large p-value means picking analogues gave
    no more separation than picking them for an arbitrary day: the conditioning adds no detectable information.
    Returns {"p_value": None, ...} when too few reference sets can be formed."""
    n = len(r_all)
    if n < 2 * min_analogues:
        return {"p_value": None, "n_reference": 0}
    overall = float(np.mean(r_all))
    rng = np.random.default_rng(seed)
    devs = []
    for j in rng.permutation(n)[:n_ref]:
        d = index.distances(index.Z[j])
        minutes_ref = float(index.minutes[j]) if index.minutes is not None and np.isfinite(index.minutes[j]) else None
        pos = index.select(d, top_k, minutes_now=minutes_ref, exclude_day=int(index.day[j]))
        if len(pos) < min_analogues:
            continue
        w = index.kernel_weights(d)[pos] if index.cfg.kernel else np.ones(len(pos))
        devs.append(abs(float(np.average(r_all[pos], weights=w)) - overall))
    if len(devs) < 15:
        return {"p_value": None, "n_reference": len(devs)}
    observed = abs(observed_mean - overall)
    p = (1 + sum(dv >= observed for dv in devs)) / (1 + len(devs))
    return {"p_value": float(p), "n_reference": len(devs)}


# ------------------------------------------------------------------------------------- metrics
def conditional_metrics_from_analogues(analogues: pd.DataFrame) -> dict:
    """Compute conditional expectancy statistics from a set of analogue observations. If the set carries kernel
    `weight`s, the mean, win rate and standard error are weighted and the effective sample size is reported."""
    n = len(analogues)
    if n < MIN_SAMPLE_FOR_ANY_CONFIDENCE:
        return {
            "sample_size": n,
            "confidence_label": "INSUFFICIENT_DATA",
            "expected_r": None,
            "prob_positive_return": None,
            "win_rate": None,
            "median_r": None,
        }

    r = analogues["r_multiple"].dropna()
    wins = r > 0
    weights = analogues.loc[r.index, "weight"].to_numpy(dtype=float) if "weight" in analogues.columns else None

    days = _trading_days(analogues.loc[r.index])
    n_days = int(days.nunique()) if days is not None else n  # no timestamps: treat rows as independent
    if n_days >= MIN_DAYS_FOR_HIGH_CONFIDENCE:
        confidence = "HIGH"
    elif n_days >= MIN_DAYS_FOR_MEDIUM_CONFIDENCE:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    out = {"sample_size": n, "n_days": n_days, "median_r": float(r.median())}
    day_arr = days.to_numpy() if days is not None else None
    if weights is not None and weights.sum() > 0:
        mean = float(np.average(r, weights=weights))
        n_eff = float(weights.sum() ** 2 / (weights**2).sum())
        if n_eff < MIN_EFFECTIVE_N_FOR_MEDIUM and confidence != "LOW":
            confidence = "LOW"
        se = _weighted_cluster_stderr(r.to_numpy(), weights, day_arr)
        out.update({
            "expected_r": mean,
            "prob_positive_return": float(np.average(wins.to_numpy(dtype=float), weights=weights)),
            "n_effective": n_eff,
        })
        lo, hi = _day_bootstrap_ci(r.to_numpy(), weights, day_arr)
        out.update({"expected_r_ci95": [lo, hi] if lo is not None else None})
    else:
        se = _cluster_robust_stderr(r.to_numpy(), day_arr)
        out.update({"expected_r": float(r.mean()), "prob_positive_return": float(wins.mean())})
    out["win_rate"] = out["prob_positive_return"]
    out["confidence_label"] = confidence
    out["expected_r_stderr"] = float(se) if not np.isnan(se) else None
    return out


def _trading_days(analogues: pd.DataFrame):
    if "entry_timestamp" not in analogues.columns:
        return None
    return pd.to_datetime(analogues["entry_timestamp"]).dt.date


def _cluster_robust_stderr(values: np.ndarray, days) -> float:
    """SE of the mean with observations clustered by trading day (each day's residuals may move
    together). With no cluster info, or one row per day, this is the ordinary SE."""
    n = len(values)
    if n < 2:
        return np.nan
    resid = values - values.mean()
    if days is None:
        return float(values.std(ddof=1) / np.sqrt(n))
    sums = pd.Series(resid).groupby(np.asarray(days)).sum().to_numpy()
    g = len(sums)
    if g < 2:
        return np.nan  # a single day tells us nothing about day-to-day variability
    return float(np.sqrt(g / (g - 1) * np.sum(sums**2)) / n)


def _weighted_cluster_stderr(values: np.ndarray, weights: np.ndarray, days) -> float:
    """Day-clustered SE of a weighted mean: sqrt(g/(g-1) * sum_days (sum_i w_i (r_i - m))^2) / sum(w)."""
    if len(values) < 2:
        return np.nan
    m = np.average(values, weights=weights)
    contrib = weights * (values - m)
    if days is None:
        sums = contrib
    else:
        sums = pd.Series(contrib).groupby(np.asarray(days)).sum().to_numpy()
    g = len(sums)
    if g < 2:
        return np.nan
    return float(np.sqrt(g / (g - 1) * np.sum(sums**2)) / weights.sum())


def _day_bootstrap_ci(values: np.ndarray, weights: np.ndarray, days, n_boot: int = 400, seed: int = 5):
    """95% interval of the weighted mean, resampling whole trading days. (None, None) with fewer than 3 days."""
    if days is None:
        return None, None
    frame = pd.DataFrame({"wr": weights * values, "w": weights, "d": np.asarray(days)}).groupby("d").sum()
    g = len(frame)
    if g < 3:
        return None, None
    idx = np.random.default_rng(seed).integers(0, g, size=(n_boot, g))
    wr, w = frame["wr"].to_numpy(), frame["w"].to_numpy()
    means = wr[idx].sum(axis=1) / np.maximum(w[idx].sum(axis=1), 1e-12)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
