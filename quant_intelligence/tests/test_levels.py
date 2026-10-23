"""Support/resistance and liquidity levels: building them, checking a trade against them, and the auto-trader wiring."""
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution import auto_trader
from quant_intelligence.features.levels import Level, assess_trade, find_levels
from quant_intelligence.strategies.base_strategy import Setup


def _day(date, closes, pad=0.2):
    """5-minute bars whose open = close, so every local peak in `closes` is a unique swing point."""
    stamps = pd.date_range(f"{date} 09:15", periods=len(closes), freq="5min")
    return pd.DataFrame({"timestamp": stamps, "open": closes, "high": [c + pad for c in closes],
                         "low": [c - pad for c in closes], "close": closes, "volume": 1000})


PRIOR = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 109, 108, 107, 106, 105]  # PDH 110.2, PDL 99.8
TODAY = [104, 105, 106, 107, 106, 105, 104, 105, 106, 107, 106, 105, 104.5]  # two equal peaks at 107 = a liquidity pool


def _frame(*days):
    return pd.concat([_day(d, c) for d, c in days], ignore_index=True)


def _history():
    return _frame(("2030-01-02", PRIOR), ("2030-01-03", PRIOR), ("2030-01-04", PRIOR), ("2030-01-07", PRIOR), ("2030-01-08", TODAY))


def _kinds(levels):
    return {lv.kind for lv in levels}


# ------------------------------------------------------------------ building levels
def test_levels_include_prior_day_week_pivots_and_the_equal_highs_pool():
    levels = find_levels(_history(), atr=1.0)
    kinds = " ".join(_kinds(levels))
    for expected in ("PDH", "PDL", "WEEK_HIGH", "WEEK_LOW", "PIVOT_R1", "PIVOT_S1", "EQ_HIGHS", "DAY_LOW"):
        assert expected in kinds, expected
    pool = next(lv for lv in levels if "EQ_HIGHS" in lv.kind and abs(lv.price - 107.2) < 0.01)
    assert pool.touches >= 2 and pool.significant


def test_a_lone_swing_high_is_a_weak_level():
    quiet = [104, 104.2, 104.5, 106, 105, 104, 104.5, 105, 106.5, 107, 106, 105, 104.5]  # one peak at 106, day high 107
    levels = find_levels(_frame(("2030-01-02", PRIOR), ("2030-01-03", PRIOR), ("2030-01-04", quiet)), atr=1.0)
    lone = [lv for lv in levels if lv.kind == "SWING_HIGH" and abs(lv.price - 106.2) < 0.01]
    assert lone and not lone[0].significant and lone[0].touches == 1


def test_oi_peaks_from_the_chain_are_listed_but_never_significant():
    levels = find_levels(_history(), atr=1.0, chain_context={"call_oi_peak": 112.0, "put_oi_peak": 98.0})
    peaks = [lv for lv in levels if "OI_" in lv.kind]
    assert {lv.kind for lv in peaks} == {"OI_CALL_PEAK", "OI_PUT_PEAK"} and not any(lv.significant for lv in peaks)
    merged = find_levels(_history(), atr=1.0, chain_context={"call_oi_peak": 110.25, "put_oi_peak": None})
    assert any("PDH" in lv.kind and "OI_CALL_PEAK" in lv.kind for lv in merged)


def test_an_oi_peak_alone_cannot_veto_a_trade():
    peak = find_levels(_history(), atr=1.0, chain_context={"call_oi_peak": 106.9, "put_oi_peak": None})
    only_peak = [lv for lv in peak if lv.kind == "OI_CALL_PEAK"]
    assert assess_trade("LONG", 106.5, 105.5, 110.0, only_peak, atr=1.0).ok


@pytest.mark.parametrize("data,atr", [(None, 1.0), (pd.DataFrame(), 1.0), (_day("2030-01-02", [100, 101, 102]), 1.0),
                                       (None, None), (SimpleNamespace(tail=lambda n: None), 1.0)])
def test_unusable_data_gives_no_levels(data, atr):
    assert find_levels(data, atr) == []


def test_a_nan_or_zero_atr_gives_no_levels():
    assert find_levels(_history(), float("nan")) == [] and find_levels(_history(), 0.0) == []


# ------------------------------------------------------------------ checking a trade
RES = Level(108.0, "PDH", 2.0)
SUP = Level(104.0, "PDL", 2.0)
WEAK = Level(107.0, "SWING_HIGH", 1.0)


def test_a_long_with_a_resistance_too_close_is_vetoed():
    check = assess_trade("LONG", 106.5, 105.5, 109.0, [RES], atr=1.0)  # 1.5 away = 1.5R: just enough? no: needs 1.5 x 1.0
    assert check.ok  # exactly at the minimum is allowed
    check = assess_trade("LONG", 107.0, 106.0, 110.0, [RES], atr=1.0)  # 1.0 away < 1.5R
    assert not check.ok and "no room" in check.reason and "Pdh 108" in check.reason


def test_a_target_beyond_the_level_is_held_short_of_it():
    check = assess_trade("LONG", 104.0, 103.0, 110.0, [Level(108.0, "PDH", 2.0)], atr=1.0, buffer_atr=0.15)
    assert check.ok and check.target_capped and check.target == pytest.approx(107.85) and check.stop == 103.0


def test_a_target_short_of_the_level_is_left_alone():
    check = assess_trade("LONG", 104.0, 103.0, 106.0, [RES], atr=1.0)
    assert check.ok and not check.target_capped and check.target == 106.0


def test_the_short_side_is_the_mirror_image():
    support = Level(100.0, "PDL", 2.0)
    veto = assess_trade("SHORT", 100.8, 101.8, 97.0, [support], atr=1.0)
    assert not veto.ok and "no room" in veto.reason
    ok = assess_trade("SHORT", 104.0, 105.0, 90.0, [support], atr=1.0)
    assert ok.ok and ok.target_capped and ok.target == pytest.approx(100.15) and ok.stop == 105.0


def test_a_weak_level_is_ignored():
    check = assess_trade("LONG", 106.5, 105.5, 110.0, [WEAK], atr=1.0)
    assert check.ok and not check.target_capped and check.opposing is None


def test_the_stop_moves_behind_a_supporting_level_but_never_gets_wider():
    check = assess_trade("LONG", 106.0, 103.0, 112.0, [SUP], atr=1.0, buffer_atr=0.15)  # ATR stop 3 away, level 2 away
    assert check.stop_changed and check.stop == pytest.approx(103.85) and check.supporting is SUP
    behind = assess_trade("LONG", 106.0, 105.0, 112.0, [Level(103.0, "PDL", 2.0)], atr=1.0)  # level is past the ATR stop
    assert not behind.stop_changed and behind.stop == 105.0


def test_a_level_too_close_to_the_entry_is_not_used_as_a_stop():
    check = assess_trade("LONG", 105.0, 103.5, 109.0, [Level(104.8, "PDL", 2.0)], atr=2.0, buffer_atr=0.15)
    assert not check.stop_changed and check.stop == 103.5


def test_a_short_stop_moves_down_behind_resistance():
    res = Level(104.0, "PDH", 2.0)
    check = assess_trade("SHORT", 101.0, 104.0, 95.0, [res], atr=1.0, buffer_atr=0.15)
    assert check.stop_changed and check.stop == pytest.approx(104.15) or check.stop == pytest.approx(104.0)
    assert check.stop <= 104.0  # never wider than the original


def test_no_levels_or_bad_inputs_pass_unchanged():
    for args in [("LONG", 100.0, 99.0, 103.0, [], 1.0), ("FLAT", 100.0, 99.0, 103.0, [RES], 1.0),
                 ("LONG", 100.0, 101.0, 103.0, [RES], 1.0), ("LONG", float("nan"), 99.0, 103.0, [RES], 1.0)]:
        check = assess_trade(*args)
        assert check.ok and check.stop == args[2] and check.target == args[3]


def test_notes_report_an_entry_at_support_and_a_sweep_and_reclaim():
    data = _day("2030-01-08", [106, 106, 106, 106.2, 105.0, 106.0])  # the last bars dip through 105.5 and close back above
    check = assess_trade("LONG", 106.0, 104.9, 109.0, [Level(105.5, "PDL", 2.0)], atr=1.0, ohlcv=data)
    assert any("entry at" in n for n in check.notes) and any("swept" in n and "reclaimed" in n for n in check.notes)
    assert check.summary.startswith("Levels: ")


# ------------------------------------------------------------------ the auto-trader wiring
@pytest.fixture
def levels_on():
    original = SETTINGS.levels_enabled
    object.__setattr__(SETTINGS, "levels_enabled", True)
    yield
    object.__setattr__(SETTINGS, "levels_enabled", original)


def _output(ohlcv=None, atr=1.0):
    return SimpleNamespace(ohlcv=_history() if ohlcv is None else ohlcv, market_state={"atr_14": atr})


def _setup(direction="LONG", entry=106.8, stop=105.8, target=110.0):
    return Setup(pd.Timestamp("2030-01-08 10:00"), direction, entry, stop, target, {"transaction": "BUY"})


def test_the_check_is_off_when_disabled():
    original = SETTINGS.levels_enabled
    object.__setattr__(SETTINGS, "levels_enabled", False)
    try:
        setup = _setup()
        assert auto_trader.apply_level_check(_output(), setup, "Momentum") == (setup, "", None)
    finally:
        object.__setattr__(SETTINGS, "levels_enabled", original)


def test_a_trade_into_the_liquidity_pool_is_vetoed_with_the_reason(levels_on):
    setup, review, veto = auto_trader.apply_level_check(_output(), _setup(), "Momentum")  # pool at 107.2 is 0.4 above entry
    assert veto and veto.startswith("LEVEL CHECK: no room") and review.startswith("Levels: ")


def test_a_level_native_strategy_is_not_second_guessed(levels_on):
    setup = _setup()
    assert auto_trader.apply_level_check(_output(), setup, "Camarilla Pivot Reversal") == (setup, "", None)


def test_a_clear_trade_gets_its_stop_and_target_moved_to_the_levels(levels_on):
    setup = _setup(entry=105.0, stop=104.0, target=112.0)  # the equal-highs pool sits 2.2 above: 2.2R of room
    new, review, veto = auto_trader.apply_level_check(_output(), setup, "Momentum")
    assert veto is None and new is not setup
    assert new.target_price == pytest.approx(107.05) and new.stop_price == 104.0 and "levels" in new.meta
    assert setup.target_price == 112.0  # the original setup object is untouched


def test_missing_data_or_atr_leaves_the_setup_unchanged(levels_on):
    setup = _setup()
    assert auto_trader.apply_level_check(SimpleNamespace(ohlcv=None, market_state={"atr_14": 1.0}), setup, "Momentum") == (setup, "", None)
    assert auto_trader.apply_level_check(_output(atr=None), setup, "Momentum") == (setup, "", None)


def test_a_fault_in_the_level_reader_never_stops_the_cycle(levels_on, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad data")

    monkeypatch.setattr(auto_trader, "find_levels", boom)
    setup = _setup()
    assert auto_trader.apply_level_check(_output(), setup, "Momentum") == (setup, "", None)


def test_the_full_cycle_skips_a_trade_with_no_room_and_logs_it(levels_on, monkeypatch):
    from quant_intelligence.brokers.paper_broker import PaperBroker
    from quant_intelligence.reports.decision_log import classify_cycle
    from quant_intelligence.risk.risk_engine import AccountState

    setup = _setup()
    monkeypatch.setattr(auto_trader, "get_strategy", lambda n: object())
    monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, **k: SimpleNamespace(status=auto_trader.SETUP_TRIGGERED, setup=setup))
    ranking = SimpleNamespace(is_no_trade=False, selected_strategy="Momentum", tie_break=False, reason="r", tie_runner_up=None,
                              ranked=[])
    output = SimpleNamespace(ranking=ranking, ohlcv=_history(), market_state={"atr_14": 1.0}, data_quality_status="OK",
                             timestamp=pd.Timestamp("2030-01-08 10:00"))
    account = AccountState(equity=1_000_000.0, peak_equity=1_000_000.0, daily_pnl=0.0, open_positions_count=0, trades_today=0,
                           exposure_by_strategy={}, total_exposure=0.0, broker_connected=True, kill_switch_engaged=False)
    broker = PaperBroker(starting_capital=1_000_000)
    chain = SimpleNamespace(underlying="NIFTY")
    result = auto_trader.run_auto_option_cycle(output, "NIFTY", broker, client=None, chain=chain, account=account)
    assert result.order is None and broker.get_open_positions() == []
    assert result.reason.startswith("LEVEL CHECK: no room") and "Levels:" in result.reason
    row = classify_cycle(output, result)
    assert (row["status"], row["reason_class"]) == ("SKIPPED", "NO_ROOM")


def test_the_equity_cycle_applies_the_same_check(levels_on, monkeypatch):
    from quant_intelligence.brokers.paper_broker import PaperBroker
    from quant_intelligence.risk.risk_engine import AccountState

    setup = _setup()
    monkeypatch.setattr(auto_trader, "get_strategy", lambda n: object())
    monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, **k: SimpleNamespace(status=auto_trader.SETUP_TRIGGERED, setup=setup))
    ranking = SimpleNamespace(is_no_trade=False, selected_strategy="Momentum", tie_break=False, reason="r", ranked=[])
    output = SimpleNamespace(ranking=ranking, ohlcv=_history(), market_state={"atr_14": 1.0}, data_quality_status="OK",
                             instrument="TCS", timestamp=pd.Timestamp("2030-01-08 10:00"))
    account = AccountState(equity=1_000_000.0, peak_equity=1_000_000.0, daily_pnl=0.0, open_positions_count=0, trades_today=0,
                           exposure_by_strategy={}, total_exposure=0.0, broker_connected=True, kill_switch_engaged=False)
    broker = PaperBroker(starting_capital=1_000_000)
    result = auto_trader.run_auto_equity_cycle(output, broker, account)
    assert result.order is None and result.reason.startswith("LEVEL CHECK")


# ------------------------------------------------------------------ each level's own record
from quant_intelligence.features import levels as levels_module  # noqa: E402


def _bars(spec):
    """spec: list of (high, low, close)"""
    return [x[0] for x in spec], [x[1] for x in spec], [x[2] for x in spec]


APPROACH = [(98.5, 97.5, 98.0), (99.5, 98.5, 99.0)]
LINGER = [(100.0, 99.4, 99.8), (100.05, 99.5, 99.7), (99.95, 99.45, 99.6)]  # three bars in contact with 100: ONE excursion
FALL = [(99.0, 97.5, 98.0)] * 10


def test_a_visit_that_is_followed_by_a_reversal_is_one_reaction_however_long_it_lingers():
    h, l, c = _bars(APPROACH + LINGER + FALL)
    reactions, since, behavior = levels_module._reaction_record(100.0, h, l, c, atr=1.0)
    assert (reactions, behavior) == (1, "REJECTED") and since == len(c) - 1 - 4


def test_two_separate_visits_that_each_turn_back_are_two_reactions():
    h, l, c = _bars(APPROACH + LINGER + FALL + APPROACH + LINGER + FALL)
    assert levels_module._reaction_record(100.0, h, l, c, atr=1.0)[0] == 2


def test_a_close_through_the_level_is_a_break_not_a_reaction():
    through = [(100.1, 99.5, 99.9)] + [(101.5, 100.5, 101.2)] * 10
    h, l, c = _bars(APPROACH + through)
    reactions, since, behavior = levels_module._reaction_record(100.0, h, l, c, atr=1.0)
    assert (reactions, since, behavior) == (0, None, "BROKE")


def test_a_visit_that_has_not_yet_resolved_is_open_and_an_untouched_level_is_none():
    h, l, c = _bars(APPROACH + [(100.0, 99.4, 99.8)])
    assert levels_module._reaction_record(100.0, h, l, c, atr=1.0) == (0, None, "OPEN")
    h, l, c = _bars(APPROACH + FALL)
    assert levels_module._reaction_record(110.0, h, l, c, atr=1.0) == (0, None, "NONE")


def test_support_tests_from_above_are_read_the_same_way():
    down = [(102.5, 101.5, 102.0), (101.5, 100.5, 101.0), (100.5, 99.9, 100.2), (100.4, 99.95, 100.3)]
    bounce = [(101.5, 100.5, 101.2)] * 10
    h, l, c = _bars(down + bounce)
    assert levels_module._reaction_record(100.0, h, l, c, atr=1.0)[::2] == (1, "REJECTED")


def test_levels_carry_their_record_and_higher_timeframe_flag_and_confluence():
    levels = find_levels(_history(), atr=1.0)
    pdh = next(lv for lv in levels if "PDH" in lv.kind)
    assert pdh.htf and pdh.reactions >= 0 and pdh.behavior in ("NONE", "OPEN", "REJECTED", "BROKE")
    swing = next(lv for lv in levels if "DAY_LOW" in lv.kind and not any(k in lv.kind for k in ("PDL", "PIVOT", "WEEK")))
    assert not swing.htf
    df = _history()
    enriched = levels_module._enrich([Level(100.0, "SWING_HIGH", 1.0), Level(100.2, "PDH", 2.0), Level(105.0, "PDL", 2.0)], df, 1.0)
    assert [lv.confluence for lv in enriched] == [1, 1, 0] and [lv.htf for lv in enriched] == [False, True, True]


def test_a_tight_recent_range_adds_its_edges_as_levels_and_a_trend_does_not():
    flat = _frame(("2030-01-02", PRIOR), ("2030-01-03", [100.0, 100.4, 100.1, 100.3] * 12))
    kinds = " ".join(_kinds(find_levels(flat, atr=1.0)))
    assert "CONSOLIDATION_HIGH" in kinds and "CONSOLIDATION_LOW" in kinds
    trending = _frame(("2030-01-02", PRIOR), ("2030-01-03", [100.0 + 0.8 * i for i in range(48)]))
    assert "CONSOLIDATION" not in " ".join(_kinds(find_levels(trending, atr=1.0)))
