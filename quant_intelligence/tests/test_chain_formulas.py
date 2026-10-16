"""Option-chain formulas against hand-computed answers (no network, no broker)."""
import pytest

from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow


def _row(strike, ce_oi=None, pe_oi=None, ce_iv=None, pe_iv=None, ce_vol=None, pe_vol=None):
    return StrikeRow(strike, None, ce_oi, None, ce_vol, ce_iv, None, None, None, pe_oi, None, pe_vol, pe_iv, None, None)


def _chain(spot=111.0):
    # CE OI 10/20/30 (sum 60), PE OI 36/24/12 (sum 72) at strikes 100/110/120
    return ChainSnapshot("TEST", "2026-10-29", spot, [
        _row(100, 10, 36, ce_iv=12, pe_iv=18), _row(110, 20, 24, ce_iv=15, pe_iv=17), _row(120, 30, 12, ce_iv=14, pe_iv=16),
    ])


def test_total_pcr_is_put_open_interest_over_call_open_interest():
    assert _chain().total_pcr == pytest.approx(72 / 60)


def test_pcr_at_a_strike_and_its_undefined_cases():
    assert _chain().strikes[0].pcr_at_strike == pytest.approx(36 / 10)
    assert _row(100, ce_oi=0, pe_oi=5).pcr_at_strike is None  # no call OI: undefined, not infinity
    assert _row(100, ce_oi=None, pe_oi=5).pcr_at_strike is None
    assert ChainSnapshot("T", "e", 100.0, [_row(100, 0, 5)]).total_pcr is None


def test_max_pain_minimises_total_option_holder_payout():
    # Hand calculation: payout at 100 = 480, at 110 = 220, at 120 = 400 -> 110
    assert _chain().max_pain_strike == 110
    assert ChainSnapshot("T", "e", 100.0, []).max_pain_strike is None


def test_max_pain_ties_resolve_to_the_lowest_strike_deterministically():
    flat = ChainSnapshot("T", "e", 100.0, [_row(100, 10, 10), _row(110, 10, 10)])
    # payout at 100 = (110-100)*10 = 100; at 110 = (110-100)*10 = 100 -> tie -> first (lowest)
    assert flat.max_pain_strike == 100


def test_atm_strike_atm_iv_and_skew():
    chain = _chain(spot=111.0)
    assert chain.atm_strike == 110
    assert chain.atm_iv == pytest.approx((15 + 17) / 2)
    assert chain.iv_skew == pytest.approx(18 - 14)  # put IV one strike below ATM minus call IV one strike above


def test_skew_is_undefined_without_both_sides_or_enough_strikes():
    assert ChainSnapshot("T", "e", 100.0, [_row(100, 1, 1, 10, 10), _row(110, 1, 1, 10, 10)]).iv_skew is None
    edge = ChainSnapshot("T", "e", 100.0, [_row(100, 1, 1, 10, 10), _row(110, 1, 1, 10, 10), _row(120, 1, 1, 10, 10)], )
    assert edge.atm_strike == 100 and edge.iv_skew is None  # nothing below the ATM strike


def test_unusual_volume_flags_a_strike_over_two_standard_deviations_above_the_mean_ratio():
    # volume/OI ratios: five at 1.0 and one at 10.0 -> mean 2.5, population std 3.354, z(10) = 2.236 > 2
    rows = [_row(100, 100, 100, ce_vol=100, pe_vol=100), _row(110, 100, 100, ce_vol=100, pe_vol=100),
            _row(120, 100, 100, ce_vol=100, pe_vol=1000)]
    assert ChainSnapshot("T", "e", 110.0, rows).unusual_volume_strikes == [120]
    assert ChainSnapshot("T", "e", 110.0, rows[:1]).unusual_volume_strikes == []  # fewer than 4 observations
    even = [_row(k, 100, 100, ce_vol=50, pe_vol=50) for k in (100, 110, 120)]
    assert ChainSnapshot("T", "e", 110.0, even).unusual_volume_strikes == []  # no dispersion, nothing unusual
