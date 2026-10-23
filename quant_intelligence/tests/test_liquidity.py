"""Liquidity zones and the sweep / acceptance / failed-breakout event classifier."""
import numpy as np
import pandas as pd
import pytest

from quant_intelligence.features.liquidity import (
    ACTIVE,
    BREAKOUT_ACCEPTED,
    BUY_SIDE,
    EXPIRED,
    FAILED_BREAKOUT,
    INVALIDATED,
    SELL_SIDE,
    SWEEP_CONFIRMED,
    SWEEP_UNCONFIRMED,
    UNRESOLVED,
    LiquidityConfig,
    analyze_liquidity,
)


def _frame(rows, start="2030-01-02 09:15"):
    stamps = pd.date_range(start, periods=len(rows), freq="5min")
    return pd.DataFrame({"timestamp": stamps, "open": [r[0] for r in rows], "high": [r[1] for r in rows],
                         "low": [r[2] for r in rows], "close": [r[3] for r in rows], "volume": 1000})


FLAT = [(100.0, 100.5, 99.5, 100.0)] * 20
# a ramp to a peak at 110.5 (bar 22) and a decline: the swing high is confirmed at bar 25
RAMP = [(100.0, 102.5, 101.5, 102.0), (102.0, 105.5, 104.5, 105.0), (105.0, 110.5, 109.5, 110.0),
        (110.0, 108.5, 107.5, 108.0), (108.0, 107.5, 106.5, 107.0), (107.0, 106.5, 105.5, 106.0), (106.0, 106.8, 106.0, 106.5)]
LEVEL = 110.5


def _build(tail):
    return _frame(FLAT + RAMP + tail)


def _event(df, **kw):
    report = analyze_liquidity(df, cfg=LiquidityConfig(**kw))
    swing = [e for e in report.events if e.source == "SWING_HIGH" and abs(e.level - LEVEL) < 1e-9]
    return report, (swing[0] if swing else None)


def test_a_confirmed_swing_high_becomes_a_buy_side_zone_that_stands_until_breached():
    report, event = _event(_build([]))
    zone = [z for z in report.zones if z.source == "SWING_HIGH" and z.level == LEVEL]
    assert zone and zone[0].side == BUY_SIDE and zone[0].known_index == 25
    assert event is None and zone[0] in report.standing


def test_a_wick_above_the_high_that_closes_back_inside_is_a_confirmed_buy_side_sweep():
    df = _build([(106.5, 111.5, 108.0, 109.0)])  # wick to 111.5, close 109.0
    _, ev = _event(df)
    assert ev.kind == SWEEP_CONFIRMED and ev.side == BUY_SIDE and ev.status == ACTIVE
    assert ev.extreme == 111.5 and ev.known_index == ev.breach_index == 27 and "not a signal" in ev.note


def test_two_closes_above_the_level_are_an_accepted_breakout():
    df = _build([(106.5, 112.0, 110.0, 111.8), (111.8, 113.0, 111.5, 112.5)])
    _, ev = _event(df)
    assert ev.kind == BREAKOUT_ACCEPTED and ev.known_index == 28


def test_a_close_above_that_does_not_hold_is_a_failed_breakout():
    df = _build([(106.5, 112.0, 110.0, 111.5), (111.5, 111.8, 109.0, 109.5)])
    _, ev = _event(df)
    assert ev.kind == FAILED_BREAKOUT and ev.known_index == 28


def test_a_wick_with_no_rejection_after_the_window_is_an_unconfirmed_sweep():
    tail = [(106.5, 111.0, 110.0, 110.4)] + [(110.4, 110.45, 110.2, 110.3)] * 3
    _, ev = _event(_build(tail), reclaim_atr=2.0)
    assert ev.kind == SWEEP_UNCONFIRMED and ev.known_index == 30


def test_a_breach_on_the_latest_bar_is_unresolved():
    _, ev = _event(_build([(106.5, 111.0, 110.0, 110.4)]), reclaim_atr=2.0)
    assert ev.kind == UNRESOLVED


def test_a_breach_smaller_than_the_tolerance_is_not_an_episode():
    df = _build([(106.5, 110.51, 109.0, 109.5)])  # 0.01 above the level, tolerance is 0.05 ATR
    _, ev = _event(df)
    assert ev is None


def test_the_event_is_invalidated_by_a_later_close_beyond_its_extreme():
    df = _build([(106.5, 111.5, 108.0, 109.0), (109.0, 109.5, 108.0, 108.5), (108.5, 113.0, 108.0, 112.5)])
    _, ev = _event(df)
    assert ev.kind == SWEEP_CONFIRMED and ev.status == INVALIDATED


def test_the_event_expires_after_its_valid_window():
    quiet = [(108.0, 108.5, 107.5, 108.0)] * 15
    df = _build([(106.5, 111.5, 108.0, 109.0)] + quiet)
    _, ev = _event(df, event_valid_bars=12)
    assert ev.status == EXPIRED


def test_a_sweep_of_equal_lows_is_a_sell_side_event():
    # two swing lows at 90.0 (bars 22 and 32) then a wick under them that closes back above
    lows = [(100.0, 100.5, 99.5, 100.0)] * 20
    pattern = [(100, 101, 95, 96), (96, 97, 92, 93), (93, 94, 90.0, 91), (91, 95, 91.2, 94), (94, 97, 93, 96), (96, 99, 95, 98),
               (98, 100, 97, 99), (99, 100, 96, 97), (97, 98, 94, 95), (95, 96, 92, 93), (93, 94, 90.0, 91), (91, 95, 91.2, 94),
               (94, 97, 93, 96), (96, 99, 95, 98), (98, 100, 97, 99), (99, 100, 98, 99.5)]
    df = _frame(lows + pattern + [(99.5, 99.6, 88.0, 92.5)])
    report = analyze_liquidity(df)
    eq = [e for e in report.events if e.source == "EQUAL_LOWS"]
    assert eq and eq[0].side == SELL_SIDE and eq[0].kind == SWEEP_CONFIRMED and eq[0].level == 90.0


def test_prior_session_and_opening_range_zones_exist_with_the_right_known_time():
    day1 = _frame([(100.0, 101.0 + 0.1 * i, 99.0 - 0.05 * i, 100.0 + 0.01 * i) for i in range(40)], "2030-01-02 09:15")
    day2 = _frame([(100.0, 100.6 + 0.02 * i, 99.4, 100.0 + 0.01 * i) for i in range(30)], "2030-01-03 09:15")
    report = analyze_liquidity(pd.concat([day1, day2], ignore_index=True))
    sources = {z.source: z for z in report.zones}
    for name in ("PDH", "PDL", "OR_HIGH", "OR_LOW"):
        assert name in sources, name
    assert sources["PDH"].level == pytest.approx(float(day1["high"].max())) and sources["PDH"].known_index == 40
    assert sources["OR_HIGH"].known_index == 40 + 6  # 30-minute opening range = six 5-minute bars


def test_the_opening_range_is_not_a_zone_before_it_is_complete():
    day1 = _frame([(100.0, 101.0, 99.0, 100.0)] * 20, "2030-01-02 09:15")
    day2 = _frame([(100.0, 100.5, 99.5, 100.0)] * 4, "2030-01-03 09:15")  # only 20 minutes in
    report = analyze_liquidity(pd.concat([day1, day2], ignore_index=True))
    assert "OR_HIGH" not in {z.source for z in report.zones}


def test_no_atr_means_no_zones_and_no_events():
    df = _frame([(100.0, 100.5, 99.5, 100.0)] * 5)
    report = analyze_liquidity(df)
    assert report.zones == [] and report.events == []


# ------------------------------------------------------------------ no look-ahead
def _walk(n=300, seed=3):
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 0.6, n))
    rows, prev = [], closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + abs(rng.normal(0, 0.3)), min(prev, c) - abs(rng.normal(0, 0.3)), c))
        prev = c
    return _frame(rows)


def _signature(report):
    return ([(z.source, z.side, z.level, z.known_index) for z in report.zones],
            [(e.kind, e.side, e.level, e.source, e.breach_index, e.known_index, e.extreme, e.status) for e in report.events])


@pytest.mark.parametrize("t", [60, 120, 199, 260])
def test_the_liquidity_picture_at_a_bar_does_not_depend_on_later_bars(t):
    df = _walk()
    base = _signature(analyze_liquidity(df, as_of_index=t))
    mutated = df.copy()
    rng = np.random.default_rng(5)
    for col in ("open", "high", "low", "close"):
        mutated.loc[t + 1:, col] = mutated.loc[t + 1:, col] + rng.normal(0, 30, len(mutated) - t - 1)
    assert _signature(analyze_liquidity(mutated, as_of_index=t)) == base


def test_an_event_is_never_known_before_its_breach_and_its_zone():
    report = analyze_liquidity(_walk())
    known = {(z.source, z.level): z.known_index for z in report.zones}
    for e in report.events:
        assert e.known_index >= e.breach_index >= known[(e.source, e.level)]
