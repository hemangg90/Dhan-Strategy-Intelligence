"""
Strategy Ranking & NO-TRADE Engine.

Transparent scoring: combines conditional expected R, probability of positive
return, confidence/sample quality, drawdown/risk character, and a cost
penalty into one visible score per strategy. Every component is exposed so
the UI can show exactly why a strategy ranked where it did - nothing is a
black box.

NO-TRADE is a first-class outcome, chosen when:
  - No strategy clears a minimum expected-edge bar after costs.
  - The best strategy's confidence is LOW/INSUFFICIENT_DATA.
  - The gap between the top two strategies is not practically significant.
  - Data quality is not OK.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CONFIDENCE_WEIGHTS = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3, "INSUFFICIENT_DATA": 0.0}

MIN_EXPECTED_R_AFTER_COSTS = 0.05  # minimum edge (in R) required to even consider trading
MIN_CONFIDENCE_TO_TRADE = "MEDIUM"
MIN_SCORE_GAP_FOR_SIGNIFICANCE = 0.08  # top-2 score gap below this => ambiguous, prefer NO TRADE
DRAWDOWN_R_CAP = 20.0  # a cumulative net-R drawdown of this size (or worse) takes the full penalty

# Uncertainty-aware scoring. The conditional edge is estimated from few, clustered analogues and the
# winner is picked out of ~18 strategies, so raw means are optimistic (winner's curse).
SHRINKAGE_TAU = 0.35  # prior sd (in R) of a strategy's true conditional edge around its own global mean
LOWER_BOUND_Z = 1.0  # rank on (shrunk mean - z * stderr)
TIE_Z = 1.0  # top two must differ by at least this many combined standard errors
MIN_INFORMATIVE_FOLDS = 3  # stability guard only applies with at least this many usable time folds
MIN_POSITIVE_FOLD_FRACTION = 0.5  # ... and the edge must be positive in at least this share of them

CONFIDENCE_RANK = {"INSUFFICIENT_DATA": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


@dataclass
class StrategyScore:
    strategy_name: str
    score: float
    expected_r: float | None
    prob_positive_return: float | None
    confidence_label: str
    sample_size: int
    max_drawdown: float | None
    eligible: bool
    ineligibility_reason: str | None = None
    components: dict = field(default_factory=dict)
    edge_mean: float | None = None  # shrunk conditional edge (R)
    edge_stderr: float | None = None  # its standard error (None when unknown)


def score_strategy(conditional_metrics: dict, max_drawdown: float | None, cost_penalty: float = 0.0) -> StrategyScore:
    """`max_drawdown` is in R units (worst fall of cumulative net R). `expected_r` in the conditional
    metrics is already net of costs, so `cost_penalty` is an optional extra haircut (default 0)."""
    name = conditional_metrics.get("strategy_name", "unknown")
    confidence = conditional_metrics.get("confidence_label", "INSUFFICIENT_DATA")
    sample_size = conditional_metrics.get("sample_size", 0)
    expected_r = conditional_metrics.get("expected_r")
    prob_pos = conditional_metrics.get("prob_positive_return")

    eligible = True
    reason = None

    if confidence == "INSUFFICIENT_DATA" or expected_r is None:
        eligible = False
        reason = f"Insufficient historical analogues ({sample_size}) to estimate conditional edge"
        return StrategyScore(name, 0.0, expected_r, prob_pos, confidence, sample_size, max_drawdown, eligible, reason)

    edge_mean, edge_se = _shrunk_edge(
        expected_r, conditional_metrics.get("expected_r_stderr"), conditional_metrics.get("global_expected_r")
    )
    conf_weight = CONFIDENCE_WEIGHTS.get(confidence, 0.0)
    drawdown_penalty = min(abs(max_drawdown) / DRAWDOWN_R_CAP, 1.0) if max_drawdown else 0.0

    # Rank on a lower bound of the shrunk edge, not the raw mean (raw mean when no stderr is known).
    expected_r_component = edge_mean - LOWER_BOUND_Z * edge_se if edge_se is not None else edge_mean
    prob_component = (prob_pos or 0.5) - 0.5  # center at 0

    score = (
        0.5 * expected_r_component
        + 0.3 * prob_component
        + 0.3 * conf_weight
        - 0.2 * drawdown_penalty
        - cost_penalty
    )

    folds = conditional_metrics.get("stability_folds", 0)
    pos_frac = conditional_metrics.get("stability_positive_frac")
    if edge_mean < MIN_EXPECTED_R_AFTER_COSTS:
        eligible = False
        reason = f"Shrunk conditional expected R ({edge_mean:.3f}) below minimum edge threshold ({MIN_EXPECTED_R_AFTER_COSTS})"
    elif folds >= MIN_INFORMATIVE_FOLDS and pos_frac is not None and pos_frac < MIN_POSITIVE_FOLD_FRACTION:
        eligible = False
        reason = f"Edge not stable over time: positive in only {pos_frac:.0%} of {folds} time folds"
    elif CONFIDENCE_RANK.get(confidence, 0) < CONFIDENCE_RANK.get(MIN_CONFIDENCE_TO_TRADE, 2):
        eligible = False
        reason = f"Confidence '{confidence}' below minimum required '{MIN_CONFIDENCE_TO_TRADE}'"

    return StrategyScore(
        strategy_name=name,
        score=score,
        expected_r=expected_r,
        prob_positive_return=prob_pos,
        confidence_label=confidence,
        sample_size=sample_size,
        max_drawdown=max_drawdown,
        eligible=eligible,
        ineligibility_reason=reason,
        edge_mean=edge_mean,
        edge_stderr=edge_se,
        components={
            "expected_r_component": 0.5 * expected_r_component,
            "shrunk_expected_r": edge_mean,
            "edge_stderr": edge_se,
            "prob_component": 0.3 * prob_component,
            "confidence_component": 0.3 * conf_weight,
            "drawdown_penalty": -0.2 * drawdown_penalty,
            "cost_penalty": -cost_penalty,
        },
    )


def _shrunk_edge(mean: float, se, prior_mean) -> tuple[float, float | None]:
    """Empirical-Bayes shrinkage of the conditional mean toward the strategy's own global mean.
    Returns (posterior mean, posterior stderr). Without a usable stderr there is nothing to weigh
    the prior against, so the raw mean is returned with stderr None."""
    if se is None or not np.isfinite(se) or se <= 0:
        return mean, None
    prior = 0.0 if prior_mean is None or not np.isfinite(prior_mean) else float(prior_mean)
    w = SHRINKAGE_TAU**2 / (SHRINKAGE_TAU**2 + se**2)
    return prior + w * (mean - prior), float(np.sqrt(w) * se)


@dataclass
class RankingDecision:
    ranked: list[StrategyScore]
    selected_strategy: str | None
    is_no_trade: bool
    reason: str
    tie_break: bool = False  # selected only because the top strategies were tied (see rank_and_select)
    tie_runner_up: str | None = None


TIE_BREAK_TAG = "TIE-BREAK"


def tradable(decision: "RankingDecision", mode: str = "PAPER") -> tuple[bool, str | None]:
    """Can this decision be traded in `mode` ("PAPER" or "LIVE")? Returns (ok, why_not).

    A NO TRADE decision never is. A tie-break decision is honoured only if TIE_BREAK_PAPER / TIE_BREAK_LIVE
    allows it for that mode (both default on)."""
    if decision.is_no_trade:
        return False, decision.reason
    if decision.tie_break:
        from quant_intelligence.config.settings import SETTINGS

        enabled = SETTINGS.tie_break_live if mode.upper() == "LIVE" else SETTINGS.tie_break_paper
        if not enabled:
            return False, f"Tie-break trading is turned off for {mode.lower()} trading. {decision.reason}"
    return True, None


def rank_and_select(
    scores: list[StrategyScore], data_quality_status: str = "OK", allow_tie_break: bool = False
) -> RankingDecision:
    """Pick the strategy to trade now, or NO TRADE.

    `allow_tie_break`: when the top two eligible strategies cannot be told apart statistically, select the
    top one anyway and flag it (`tie_break=True`) so callers tag the trade and size it down - instead of
    returning NO TRADE. Data quality not OK, no eligible strategy, or an ineligible leader still mean NO TRADE.
    """
    ranked = sorted(scores, key=lambda s: s.score, reverse=True)

    if data_quality_status != "OK":
        return RankingDecision(ranked, None, True, f"Data quality is {data_quality_status}: NO TRADE forced")

    eligible = [s for s in ranked if s.eligible]

    if not eligible:
        return RankingDecision(ranked, None, True, "No strategy clears minimum edge/confidence thresholds")

    top = eligible[0]

    def tied(runner_up: StrategyScore, detail: str) -> RankingDecision:
        if allow_tie_break:
            return RankingDecision(
                ranked, top.strategy_name, False,
                f"TIE-BREAK: {top.strategy_name} chosen over {runner_up.strategy_name} ({detail}). No strategy is "
                "clearly best, so this trade is tagged and sized down.",
                tie_break=True, tie_runner_up=runner_up.strategy_name,
            )
        return RankingDecision(ranked, None, True, f"Top strategies ({top.strategy_name} vs {runner_up.strategy_name}) are {detail}")

    if len(eligible) > 1:
        gap = top.score - eligible[1].score
        if gap < MIN_SCORE_GAP_FOR_SIGNIFICANCE:
            return tied(
                eligible[1],
                f"statistically/practically indistinguishable (score gap {gap:.3f} < {MIN_SCORE_GAP_FOR_SIGNIFICANCE})",
            )

    if len(eligible) > 1:
        runner_up = eligible[1]
        if None not in (top.edge_mean, top.edge_stderr, runner_up.edge_mean, runner_up.edge_stderr):
            se = float(np.hypot(top.edge_stderr, runner_up.edge_stderr))
            z = (top.edge_mean - runner_up.edge_mean) / se if se > 0 else float("inf")
            if z < TIE_Z:
                return tied(
                    runner_up,
                    f"not statistically distinguishable (edge difference {z:.2f} standard errors < {TIE_Z})",
                )

    return RankingDecision(ranked, top.strategy_name, False, f"{top.strategy_name} ranks highest on historical conditional expectancy (not a prediction)")
