"""IV history, OI/PCR features, execution quality, expiry intelligence and the OptionsContext."""
import datetime as dt
import json

import numpy as np
import pytest

from quant_intelligence.database.db import get_session, init_db
from quant_intelligence.database.models import IvObservation
from quant_intelligence.intelligence.options_context import build_options_context
from quant_intelligence.intelligence.status import CRITICAL, OK, WARNING
from quant_intelligence.options import iv_history
from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow
from quant_intelligence.options.contracts import OptionQuote, quotes_from_chain
from quant_intelligence.options.execution_quality import ExecutionLimits, assess_execution_quality
from quant_intelligence.options.expiry import decay_and_gamma_profile, expiry_info
from quant_intelligence.options.iv_history import MIN_OBSERVATIONS, compute_iv_stats, rank_percentile, record_iv
from quant_intelligence.options.oi_features import oi_features

NOW = dt.datetime(2030, 1, 7, 10, 0)  # Monday, mid-session
EXPIRY = "2030-01-31"
LIMITS = ExecutionLimits(max_spread_pct=8.0, max_quote_age_sec=120.0, min_premium=1.0, min_top_qty_lots=1.0)


def _row(strike, **kw):
    base = dict(strike=float(strike), ce_ltp=100.0, ce_oi=1000, ce_oi_change=50, ce_volume=300, ce_iv=18.0, ce_delta=0.5,
                ce_security_id=f"{strike}C", pe_ltp=90.0, pe_oi=1200, pe_oi_change=-20, pe_volume=250, pe_iv=19.0, pe_delta=-0.5,
                pe_security_id=f"{strike}P", ce_bid=99.0, ce_ask=101.0, pe_bid=89.0, pe_ask=91.0, ce_bid_qty=500, ce_ask_qty=400,
                pe_bid_qty=300, pe_ask_qty=350, ce_prev_close=95.0, pe_prev_close=92.0)
    base.update(kw)
    return StrikeRow(**base)


def _chain(strikes=(23900, 23950, 24000, 24050, 24100), spot=24000.0, rows=None, **meta):
    meta.setdefault("fetched_at", NOW)
    meta.setdefault("lot_size", 75)
    meta.setdefault("exchange_segment", "NSE_FNO")
    return ChainSnapshot("NIFTY", EXPIRY, spot, rows if rows is not None else [_row(s) for s in strikes], **meta)


def _quote(**kw):
    base = dict(underlying="NIFTY", exchange_segment="NSE_FNO", security_id="1", option_type="CE", strike=24000.0, expiry=EXPIRY,
                lot_size=75, bid=99.0, ask=101.0, bid_qty=300.0, ask_qty=300.0, ltp=100.0, volume=500.0, oi=5000.0, oi_change=0.0,
                iv=18.0, delta=0.5, gamma=None, theta=None, vega=None, prev_close=95.0, quote_time=NOW, source="t", unavailable=())
    base.update(kw)
    return OptionQuote(**base)


# ------------------------------------------------------------------ IV rank and percentile
def test_rank_and_percentile_have_precise_definitions():
    history = [10.0, 12.0, 14.0, 16.0, 20.0]
    rank, pct = rank_percentile(history, 15.0)
    assert rank == pytest.approx(50.0) and pct == pytest.approx(60.0)  # (15-10)/(20-10); 3 of 5 strictly below
    assert rank_percentile(history, 25.0) == (100.0, 100.0) and rank_percentile(history, 5.0) == (0.0, 0.0)
    assert rank_percentile([10.0, 10.0], 12.0)[0] is None  # a flat window has no rank
    assert rank_percentile([], 12.0) == (None, None)


def _daily(n, start=dt.date(2029, 6, 1), value=lambda i: 15 + (i % 10), dte=lambda i: 5 + (i % 5)):
    return [(start + dt.timedelta(days=i), float(value(i)), float(dte(i))) for i in range(n)]


def test_too_little_history_is_unavailable_with_the_reason_never_estimated():
    stats = compute_iv_stats(_daily(MIN_OBSERVATIONS - 1), 18.0)
    assert not stats.available and stats.iv_rank is None and stats.iv_percentile is None
    assert f"only {MIN_OBSERVATIONS - 1} sessions" in stats.reason
    assert not compute_iv_stats([], 18.0).available and not compute_iv_stats(_daily(60), float("nan")).available


def test_enough_history_gives_rank_percentile_and_changes():
    daily = _daily(60)
    stats = compute_iv_stats(daily, 18.0)
    values = [v for _, v, _ in daily]
    assert stats.available and stats.n_obs == 60 and stats.first_observation == daily[0][0]
    assert stats.iv_rank == pytest.approx((18 - min(values)) / (max(values) - min(values)) * 100)
    assert stats.iv_percentile == pytest.approx(np.mean(np.array(values) < 18) * 100)
    assert stats.change_1d == pytest.approx(18 - values[-1]) and stats.change_5d == pytest.approx(18 - values[-5])


def test_the_window_keeps_only_the_newest_sessions():
    daily = _daily(400, value=lambda i: 40.0 if i < 100 else 15.0 + (i % 5))
    stats = compute_iv_stats(daily, 17.0, window_days=252)
    assert stats.n_obs == 252 and stats.iv_rank < 100  # the old 40% era fell out of the window


def test_tenor_matched_figures_use_only_observations_near_todays_maturity():
    stats = compute_iv_stats(_daily(120), 18.0, dte=6.0, tenor_tolerance=1.0)
    assert stats.n_tenor and stats.n_tenor < stats.n_obs
    few = compute_iv_stats(_daily(40), 18.0, dte=6.0, tenor_tolerance=0.1)
    assert few.available and few.iv_rank_tenor is None and any("tenor-matched" in n for n in few.notes)


@pytest.fixture
def iv_db():
    init_db()
    with get_session() as s:
        s.query(IvObservation).delete()
    iv_history._last_recorded.clear()
    yield
    with get_session() as s:
        s.query(IvObservation).delete()
    iv_history._last_recorded.clear()


def test_recording_is_throttled_and_never_uses_a_missing_iv(iv_db):
    chain = _chain()
    assert record_iv(chain, NOW, dte=17.0) is True
    assert record_iv(chain, NOW + dt.timedelta(minutes=5)) is False  # inside the throttle window
    assert record_iv(chain, NOW + dt.timedelta(minutes=45)) is True
    blank = _chain(rows=[_row(24000, ce_iv=None, pe_iv=None)])
    assert record_iv(blank, NOW + dt.timedelta(hours=3)) is False
    with get_session() as s:
        assert s.query(IvObservation).count() == 2


def test_history_read_back_excludes_the_decision_day_and_the_future(iv_db):
    chain = _chain()
    for day in range(1, 6):  # five earlier sessions, two snapshots each: the later one is the day's value
        for hour, shift in ((10, 0), (14, 2)):
            snap = _chain(rows=[_row(24000, ce_iv=18.0 + day + shift, pe_iv=18.0 + day + shift)])
            record_iv(snap, dt.datetime(2030, 1, day, hour, 0), interval_min=0, dte=10.0)
    series = iv_history.load_daily_series("NIFTY", dt.datetime(2030, 1, 4, 11, 0))
    assert [d for d, _, _ in series] == [dt.date(2030, 1, 1), dt.date(2030, 1, 2), dt.date(2030, 1, 3)]  # not the 4th, not the 5th
    assert [v for _, v, _ in series] == [21.0, 22.0, 23.0]
    assert iv_history.load_daily_series("NIFTY", dt.datetime(2030, 1, 1, 9, 0)) == []
    assert not iv_history.iv_stats("NIFTY", 20.0, dt.datetime(2030, 1, 4, 11, 0)).available


# ------------------------------------------------------------------ OI and PCR
def test_pcr_totals_changes_and_window():
    f = oi_features(_chain(), window=1)
    assert f.total_call_oi == 5000 and f.total_put_oi == 6000
    assert f.pcr_oi == pytest.approx(1.2) and f.pcr_volume == pytest.approx(1250 / 1500)
    assert f.pcr_oi_window == pytest.approx(1.2) and f.call_oi_change == 250 and f.put_oi_change == -100 and f.oi_change_coverage == 1.0
    assert f.call_oi_to_volume == pytest.approx(1500 / 5000) and f.unavailable == ()


def test_zero_denominators_are_unavailable_not_zero_or_infinite():
    rows = [_row(s, ce_oi=0, ce_volume=0) for s in (23900, 24000, 24100)]
    f = oi_features(_chain(rows=rows))
    assert f.pcr_oi is None and f.pcr_volume is None and {"pcr_oi", "pcr_volume"} <= set(f.unavailable)
    assert f.call_concentration is None and f.call_oi_peak is None and any("PCR (OI)" in n for n in f.notes)
    empty = oi_features(ChainSnapshot("NIFTY", EXPIRY, 24000.0, []))
    assert empty.pcr_oi is None and empty.unavailable == ("all",)


def test_missing_oi_change_lowers_coverage_and_flags_the_totals():
    rows = [_row(23900, ce_oi_change=None, pe_oi_change=None), _row(24000), _row(24100, ce_oi_change=None)]
    f = oi_features(_chain(rows=rows))
    assert f.oi_change_coverage == pytest.approx(3 / 6) and "oi_change_totals" in f.unavailable


def test_concentration_peaks_and_hhi():
    rows = [_row(23900, ce_oi=100), _row(24000, ce_oi=8000), _row(24100, ce_oi=100)]
    f = oi_features(_chain(rows=rows), top_n=1, atr=100.0)
    assert f.call_concentration == pytest.approx(8000 / 8200) and f.call_hhi == pytest.approx((8000 / 8200) ** 2 + 2 * (100 / 8200) ** 2)
    assert f.call_oi_peak["strike"] == 24000 and f.call_oi_peak["distance_pct"] == 0 and f.call_oi_peak["distance_atr"] == 0
    assert "resistance" not in str(f).lower() and "support" not in str(f).lower()  # peaks are named, never interpreted


def test_the_change_in_concentration_needs_previous_open_interest():
    # today 24000 holds 8000; yesterday it held 2000 (change +6000), so concentration rose
    rows = [_row(23900, ce_oi=100, ce_oi_change=0), _row(24000, ce_oi=8000, ce_oi_change=6000), _row(24100, ce_oi=100, ce_oi_change=0)]
    f = oi_features(_chain(rows=rows), top_n=1)
    assert f.call_concentration_change == pytest.approx(8000 / 8200 - 2000 / 2200)
    gone = oi_features(_chain(rows=[_row(s, ce_oi_change=None, pe_oi_change=None) for s in (23900, 24000, 24100)]))
    assert gone.call_concentration_change is None and "concentration_change" in gone.unavailable


# ------------------------------------------------------------------ execution quality
def test_a_liquid_contract_is_eligible_and_fills_conservatively():
    eq = assess_execution_quality(_quote(), NOW, "BUY", 150, LIMITS)
    assert eq.eligible and eq.mid == 100.0 and eq.rel_spread_pct == pytest.approx(2.0)
    assert eq.entry_fill == pytest.approx(101.05) and eq.exit_fill == pytest.approx(98.95)  # ask+tick, bid-tick: never mid or last
    assert eq.round_trip_costs > 0 and eq.breakeven_move_pct > 0 and eq.quantity == 150 and eq.assumptions


def test_a_sell_fills_at_the_bid_minus_slippage():
    eq = assess_execution_quality(_quote(), NOW, "SELL", 75, LIMITS)
    assert eq.entry_fill == pytest.approx(98.95) and eq.exit_fill == pytest.approx(101.05)


@pytest.mark.parametrize("kw,text", [
    (dict(bid=None), "no two-sided quote"),
    (dict(ask=None), "no two-sided quote"),
    (dict(bid=0.0), "no bid"),
    (dict(bid=101.0, ask=99.0), "crossed quote"),
    (dict(bid=80.0, ask=120.0), "spread 40.0%"),
    (dict(bid=0.4, ask=0.6), "below the minimum"),
    (dict(ask_qty=20.0), "units (0.3 lots) quoted on the ask"),
    (dict(volume=0.0), "no volume"),
    (dict(oi=0.0), "no open interest"),
    (dict(quote_time=NOW - dt.timedelta(seconds=300)), "stale"),
    (dict(quote_time=None), "no quote time"),
    (dict(quote_time=NOW + dt.timedelta(seconds=60)), "in the future"),
])
def test_each_failure_is_a_stated_hard_rejection(kw, text):
    eq = assess_execution_quality(_quote(**kw), NOW, "BUY", None, LIMITS)
    assert not eq.eligible and any(text in r for r in eq.reasons), eq.reasons


def test_an_order_larger_than_the_quoted_size_pays_extra_impact():
    small = assess_execution_quality(_quote(ask_qty=300.0), NOW, "BUY", 150, LIMITS)
    big = assess_execution_quality(_quote(ask_qty=300.0), NOW, "BUY", 900, LIMITS)
    assert big.entry_fill > small.entry_fill and any("exceeds the 300 quoted" in w for w in big.warnings)


def test_unreported_size_and_volume_are_warnings_not_invented():
    eq = assess_execution_quality(_quote(ask_qty=None, volume=None, oi=None), NOW, "BUY", None, LIMITS)
    assert eq.eligible and {"quoted size is not reported", "volume is not reported", "open interest is not reported"} <= set(eq.warnings)


# ------------------------------------------------------------------ expiry
def test_expiry_counts_trading_time_and_flags_the_risk_band():
    info = expiry_info("NIFTY", EXPIRY, NOW, [EXPIRY, "2030-01-10", "2030-02-28"])
    assert info.available and info.risk_band == "NORMAL" and info.is_monthly is True and not info.is_expiry_day
    assert info.sessions_remaining == 19 and info.trading_days == pytest.approx(18.9, abs=0.1)  # today's rest + 18 whole sessions
    weekly = expiry_info("NIFTY", "2030-01-09", NOW, [EXPIRY, "2030-01-09"])
    assert weekly.is_monthly is False and weekly.risk_band == "ELEVATED"


def test_expiry_day_and_near_expiry_bands():
    morning = expiry_info("NIFTY", "2030-01-07", dt.datetime(2030, 1, 7, 10, 0))
    assert morning.is_expiry_day and "EXPIRY_DAY" in morning.flags and "NEAR_EXPIRY" in morning.flags
    assert morning.risk_band == "HIGH" and 0 < morning.minutes_to_close <= 330
    last_hour = expiry_info("NIFTY", "2030-01-07", dt.datetime(2030, 1, 7, 14, 45))
    assert last_hour.risk_band == "EXTREME"
    eve = expiry_info("NIFTY", "2030-01-08", dt.datetime(2030, 1, 7, 15, 0))
    assert eve.risk_band == "ELEVATED" and not eve.is_expiry_day  # 1.06 trading days: just past the HIGH band


def test_expired_invalid_and_calendar_conflicting_expiries_are_reported():
    assert not expiry_info("NIFTY", "2030-01-04", NOW).available and "expired" in expiry_info("NIFTY", "2030-01-04", NOW).reason
    assert not expiry_info("NIFTY", "2030-01-07", dt.datetime(2030, 1, 7, 15, 45)).available
    assert not expiry_info("NIFTY", "garbage", NOW).available
    sat = expiry_info("NIFTY", "2030-01-12", NOW)  # a Saturday
    assert not sat.available and "EXPIRY_ON_NON_TRADING_DAY" in sat.flags


def test_decay_and_gamma_profile():
    p = decay_and_gamma_profile(premium=100.0, spot=24000.0, gamma=0.0005, theta_per_day=-8.0)
    assert p["theta_pct_of_premium_per_day"] == pytest.approx(-8.0) and p["delta_change_per_1pct_move"] == pytest.approx(0.12)
    assert decay_and_gamma_profile(0.0, 24000.0, 0.1, -1.0)["theta_pct_of_premium_per_day"] is None


# ------------------------------------------------------------------ the options context
def test_a_valid_chain_builds_a_full_context(iv_db):
    ctx = build_options_context(_chain(), now=NOW, realized_vol=0.15, atr=80.0, expected_lot_size=75, expected_strike_step=50,
                                listed_expiries=[EXPIRY], limits=LIMITS)
    assert ctx.underlying == "NIFTY" and ctx.atm_strike == 24000 and ctx.atm_iv == pytest.approx(18.5)
    assert ctx.info_cutoff == ctx.timestamp == NOW and len(ctx.quotes) == 10 and ctx.chain_hash
    assert ctx.atm_greeks["CE"].available and ctx.atm_greeks["PE"].available
    assert ctx.iv_vs_realized["available"] and ctx.iv_vs_realized["iv_minus_rv"] == pytest.approx(0.185 - 0.15)
    assert ctx.liquidity["eligible"] == 10 and ctx.skew["available"] and ctx.oi.pcr_oi == pytest.approx(1.2)
    assert ctx.expiry_info.available and ctx.quote(24000.0, "CE").strike == 24000
    # no recorded history yet: rank and percentile are named as unavailable, and that is a warning not a failure
    assert not ctx.iv.available and ctx.status.level == WARNING and {"iv_rank", "iv_percentile"} <= set(ctx.status.unavailable)


def test_the_context_is_reproducible_and_json_safe(iv_db):
    a = build_options_context(_chain(), now=NOW, limits=LIMITS).as_record()
    b = build_options_context(_chain(), now=NOW, limits=LIMITS).as_record()
    assert a == b and json.loads(json.dumps(a))["chain_hash"] == a["chain_hash"]
    changed = build_options_context(_chain(rows=[_row(s, ce_oi=999) for s in (23900, 23950, 24000, 24050, 24100)]), now=NOW, limits=LIMITS)
    assert changed.chain_hash != a["chain_hash"]


def test_a_chain_that_fails_validation_gives_a_critical_empty_context(iv_db):
    stale = build_options_context(_chain(fetched_at=NOW - dt.timedelta(seconds=900)), now=NOW, limits=LIMITS)
    assert stale.status.level == CRITICAL and "old" in stale.status.reason and stale.quotes == () and stale.atm_strike is None
    wrong_lot = build_options_context(_chain(), now=NOW, expected_lot_size=50, limits=LIMITS)
    assert wrong_lot.status.level == CRITICAL and "lot size" in wrong_lot.status.reason


def test_a_near_expiry_chain_adds_an_expiry_warning(iv_db):
    chain = ChainSnapshot("NIFTY", "2030-01-07", 24000.0, [_row(s) for s in (23900, 24000, 24100)], fetched_at=NOW, lot_size=75)
    ctx = build_options_context(chain, now=NOW, limits=LIMITS)
    assert ctx.status.ok and "gamma/theta" in ctx.status.reason and ctx.expiry_info.risk_band == "HIGH"


def test_term_structure_and_missing_realized_vol(iv_db):
    other = _chain()
    other.expiry = "2030-02-28"
    ctx = build_options_context(_chain(), now=NOW, other_expiries={"2030-02-28": other}, limits=LIMITS)
    assert [e for e, _ in ctx.term_structure] == [EXPIRY, "2030-02-28"]
    assert not ctx.iv_vs_realized["available"]
    only = build_options_context(_chain(), now=NOW, limits=LIMITS)
    assert len(only.term_structure) == 1


def test_a_context_needs_an_underlying():
    chain = _chain()
    chain.underlying = ""
    with pytest.raises(ValueError):
        build_options_context(chain, now=NOW, limits=LIMITS)


# ------------------------------------------------------------------ options features for the analogue query
def test_options_features_for_the_analogue_query_include_only_what_is_available(iv_db):
    from quant_intelligence.intelligence.context_features import options_context_features

    ctx = build_options_context(_chain(), now=NOW, realized_vol=0.15, limits=LIMITS)
    feats = options_context_features(ctx)
    assert feats["atm_iv"] == pytest.approx(18.5) and feats["pcr_oi"] == pytest.approx(1.2) and feats["dte_trading_days"] > 17
    assert feats["iv_minus_rv"] == pytest.approx(0.185 - 0.15) and "atm_spread_pct" in feats
    assert "iv_rank" not in feats  # no recorded history yet: absent, not zero
    stale = build_options_context(_chain(fetched_at=NOW - dt.timedelta(seconds=900)), now=NOW, limits=LIMITS)
    assert options_context_features(stale) == {}
