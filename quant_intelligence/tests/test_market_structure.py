"""Swing confirmation timing, BOS/MSS, displacement, Fair Value Gaps and the no-look-ahead guarantee."""
import numpy as np
import pandas as pd
import pytest

from quant_intelligence.features.market_structure import (
    BEAR,
    BOS,
    BULL,
    EXPIRED,
    INVALIDATED,
    MITIGATED,
    MSS,
    PARTIAL,
    UNMITIGATED,
    StructureConfig,
    analyze_structure,
    detect_swings,
)


def _frame(rows, start="2030-01-02 09:15"):
    """rows: (open, high, low, close) per 5-minute bar."""
    stamps = pd.date_range(start, periods=len(rows), freq="5min")
    return pd.DataFrame({"timestamp": stamps, "open": [r[0] for r in rows], "high": [r[1] for r in rows],
                         "low": [r[2] for r in rows], "close": [r[3] for r in rows], "volume": 1000})


def _from_closes(closes, pad=0.5):
    rows, prev = [], closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + pad, min(prev, c) - pad, c))
        prev = c
    return _frame(rows)


FLAT = [(100.0, 100.5, 99.5, 100.0)] * 20  # 20 quiet bars so the ATR exists and is about 1


# ------------------------------------------------------------------ swings and their confirmation time
def test_a_swing_high_is_known_only_swing_bars_after_it():
    highs = [10, 11, 12, 13, 14, 20, 15, 14, 13, 12, 11]
    df = _frame([(h - 0.5, h, h - 1, h - 0.5) for h in highs])
    swings = [s for s in detect_swings(df, 3) if s.kind == "HIGH"]
    assert len(swings) == 1
    s = swings[0]
    assert (s.index, s.confirmed_index, s.price) == (5, 8, 20.0) and s.known_at == df["timestamp"][8]
    assert not [x for x in analyze_structure(df, as_of_index=7).swings if x.kind == "HIGH"]  # not yet known
    assert [x for x in analyze_structure(df, as_of_index=8).swings if x.kind == "HIGH"]


def test_a_tie_with_a_later_equal_high_still_gives_one_swing():
    highs = [10, 11, 12, 20, 20, 12, 11, 10, 9]
    df = _frame([(h - 0.5, h, h - 1, h - 0.5) for h in highs])
    assert len([s for s in detect_swings(df, 3) if s.kind == "HIGH"]) == 1


def test_the_config_is_validated():
    for bad in (dict(swing_bars=0), dict(break_on="tick"), dict(displacement_close_frac=0.9)):
        with pytest.raises(ValueError):
            StructureConfig(**bad)


# ------------------------------------------------------------------ BOS and MSS
CLOSES = [100, 101, 102, 103, 104, 103, 102, 101, 102, 103, 104, 106, 107, 106, 105, 104, 103, 102, 101, 100, 99]


def test_a_close_through_the_swing_high_is_a_bos_then_a_close_through_the_swing_low_against_it_is_an_mss():
    report = analyze_structure(_from_closes(CLOSES))
    first, second = report.events[0], report.events[1]
    assert (first.kind, first.direction, first.break_index, first.initial) == (BOS, BULL, 11, True)
    assert first.level == pytest.approx(104.5) and first.known_at == first.detected_at
    assert (second.kind, second.direction, second.break_index) == (MSS, BEAR, 19)
    assert report.trend == "DOWN"


def test_a_swing_is_broken_only_once():
    report = analyze_structure(_from_closes(CLOSES + [98, 97, 96]))
    levels = [(e.direction, e.swing_index) for e in report.events]
    assert len(levels) == len(set(levels))


def test_a_wick_break_does_not_count_when_breaks_need_a_close():
    closes = list(CLOSES[:11]) + [104.0, 103.0, 102.0, 101.0]
    df = _from_closes(closes)
    df.loc[11, "high"] = 106.0  # a wick through 104.5, closing back at 104
    assert not analyze_structure(df).events
    assert analyze_structure(df, StructureConfig(break_on="wick")).events


# ------------------------------------------------------------------ displacement
def _gap_frame(after):
    """20 flat bars, a base bar, a big up bar, a bar that leaves a gap above the base bar's high, then `after` bars."""
    rows = list(FLAT) + [(100.0, 100.5, 99.5, 100.0), (100.5, 104.0, 100.4, 103.8), (103.8, 105.0, 102.0, 104.5)]
    return _frame(rows + after)


def test_displacement_needs_a_big_body_closing_near_its_extreme():
    report = analyze_structure(_gap_frame([]))
    disp = [d for d in report.displacements if d.index == 21]
    assert disp and disp[0].direction == BULL and disp[0].body_atr >= 1.5
    weak = _frame(list(FLAT) + [(100.0, 103.0, 99.0, 100.5)])  # big range, small body
    assert not analyze_structure(weak).displacements


# ------------------------------------------------------------------ fair value gaps
def _fvg(after, **cfg):
    report = analyze_structure(_gap_frame(after), StructureConfig(**cfg))
    gaps = [g for g in report.fvgs if g.direction == BULL and g.index == 22]
    assert gaps
    return gaps[0]


def test_a_bullish_gap_is_created_with_its_zone_and_known_at_the_third_bar():
    gap = _fvg([])
    assert (gap.low, gap.high, gap.index) == (100.5, 102.0, 22) and gap.status == UNMITIGATED and gap.open


def test_a_gap_price_has_not_returned_to_stays_unmitigated():
    assert _fvg([(104.5, 105.5, 103.0, 105.0)]).status == UNMITIGATED


def test_a_gap_price_dips_into_is_partial_with_a_fill_fraction():
    gap = _fvg([(104.5, 105.0, 101.5, 104.0)])
    assert gap.status == PARTIAL and gap.fill_pct == pytest.approx((102.0 - 101.5) / 1.5)


def test_a_gap_whose_whole_zone_is_traded_through_is_mitigated():
    gap = _fvg([(104.5, 105.0, 100.4, 101.0)])
    assert gap.status == MITIGATED and gap.fill_pct == 1.0 and not gap.open


def test_a_close_through_the_far_side_invalidates_the_gap():
    gap = _fvg([(104.5, 105.0, 99.0, 100.0)])
    assert gap.status == INVALIDATED and gap.status_index == 23


def test_an_old_untouched_gap_expires():
    after = [(105.0, 105.5, 104.5, 105.0)] * 8
    assert _fvg(after, fvg_max_age_bars=5).status == EXPIRED


def test_a_gap_smaller_than_the_atr_filter_is_ignored():
    rows = list(FLAT) + [(100.0, 100.5, 99.5, 100.0), (100.5, 101.0, 100.4, 100.9), (100.9, 101.2, 100.6, 101.0)]  # 0.1 gap, ATR ~1
    report = analyze_structure(_frame(rows), StructureConfig(fvg_min_atr=0.5))
    assert not report.fvgs


# ------------------------------------------------------------------ no look-ahead
def _walk(n=300, seed=7):
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 0.6, n))
    rows, prev = [], closes[0]
    for c in closes:
        hi, lo = max(prev, c) + abs(rng.normal(0, 0.3)), min(prev, c) - abs(rng.normal(0, 0.3))
        rows.append((prev, hi, lo, c))
        prev = c
    return _frame(rows)


def _signature(report):
    return (report.trend,
            [(s.kind, s.index, s.confirmed_index, s.price) for s in report.swings],
            [(e.kind, e.direction, e.break_index, e.level) for e in report.events],
            [(d.direction, d.index) for d in report.displacements],
            [(g.direction, g.index, g.status, round(g.fill_pct, 6), g.status_index) for g in report.fvgs])


@pytest.mark.parametrize("t", [40, 90, 150, 211, 299])
def test_the_answer_at_a_bar_does_not_change_when_later_bars_change(t):
    df = _walk()
    base = _signature(analyze_structure(df, as_of_index=t))
    mutated = df.copy()
    rng = np.random.default_rng(99)
    for col in ("open", "high", "low", "close"):
        mutated.loc[t + 1:, col] = mutated.loc[t + 1:, col] + rng.normal(0, 25, len(mutated) - t - 1)
    assert _signature(analyze_structure(mutated, as_of_index=t)) == base
    assert _signature(analyze_structure(df.iloc[: t + 1])) == base


def test_every_event_is_known_no_earlier_than_it_happened():
    report = analyze_structure(_walk())
    assert all(s.known_at > s.timestamp for s in report.swings)
    assert all(e.known_at >= e.detected_at for e in report.events)
    assert all(e.break_index > e.swing_index for e in report.events)


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        analyze_structure(pd.DataFrame({"timestamp": [], "close": []}))
    with pytest.raises(ValueError):
        analyze_structure(_walk(), as_of_index=10_000)
