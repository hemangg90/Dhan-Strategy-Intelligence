"""Option-specific risk controls: exposure measurement and the fail-closed limits."""
from types import SimpleNamespace

import pytest

from quant_intelligence.risk.options_risk import check_option_trade, exposure_from_positions, position_exposure, trade_exposure

EQUITY = 1_000_000.0
GREEKS = {"delta": 0.5, "gamma": 0.0005, "vega": 20.0, "theta": -15.0, "spot": 24000.0}


def _limits(**kw):
    base = dict(max_order_lots=0, min_trading_days_to_expiry=0.0, max_premium_at_risk_pct=0.0, max_underlying_premium_pct=0.0,
                max_net_delta_pct=0.0, max_gamma_pct=0.0, max_vega_pct=0.0, max_theta_pct=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _pos(qty=75, transaction="BUY", price=100.0, greeks=GREEKS, underlying="NIFTY"):
    return {"option_type": "CE", "transaction": transaction, "entry_price": price, "quantity": qty, "underlying": underlying,
            "entry_greeks": greeks}


def _option(greeks=GREEKS, days=5.0, underlying="NIFTY"):
    return {"underlying": underlying, "greeks": greeks, "trading_days_to_expiry": days}


# ------------------------------------------------------------------ measuring
def test_position_exposure_follows_the_definitions_and_the_sign():
    e = position_exposure(_pos())
    assert e["delta_notional"] == pytest.approx(0.5 * 75 * 24000) and e["gamma_1pct"] == pytest.approx(0.0005 * 75 * 24000**2 * 0.01)
    assert e["vega"] == pytest.approx(1500.0) and e["theta"] == pytest.approx(-1125.0)
    short = position_exposure(_pos(transaction="SELL"))
    assert short["delta_notional"] == -e["delta_notional"] and short["theta"] == -e["theta"]  # a writer collects decay


def test_missing_or_non_finite_greeks_are_unknown_not_zero():
    assert position_exposure(_pos(greeks=None)) is None
    assert position_exposure(_pos(greeks={**GREEKS, "delta": float("nan")})) is None
    assert position_exposure(_pos(greeks={"delta": 0.5})) is None


def test_the_account_exposure_sums_positions_and_counts_unknowns():
    book = [_pos(), _pos(qty=150, underlying="BANKNIFTY"), _pos(greeks=None), {"option_type": None, "quantity": 10}]
    ex = exposure_from_positions(book)
    assert ex["n_option_positions"] == 3 and ex["unknown_positions"] == 1
    assert ex["premium_at_risk"] == pytest.approx(100.0 * (75 + 150 + 75)) and ex["by_underlying"]["BANKNIFTY"] == pytest.approx(15000.0)
    assert ex["vega"] == pytest.approx(20.0 * (75 + 150))
    assert exposure_from_positions([])["unknown_positions"] == 0


def test_written_options_add_greeks_but_not_premium_at_risk():
    ex = exposure_from_positions([_pos(transaction="SELL")])
    assert ex["premium_at_risk"] == 0.0 and ex["theta"] > 0


# ------------------------------------------------------------------ limits
def _check(limits, ex=None, option=None, qty=75, price=100.0, tx="BUY", lot=75):
    return check_option_trade(ex if ex is not None else exposure_from_positions([]), option or _option(), qty, price, tx, EQUITY, limits, lot)


def test_with_every_limit_off_nothing_is_vetoed_and_nothing_needs_greeks():
    checks, why = _check(_limits(), option={"underlying": "NIFTY", "greeks": None, "trading_days_to_expiry": None})
    assert why is None and checks["order_size_within_limit"]


def test_the_maximum_order_size():
    assert _check(_limits(max_order_lots=10), qty=75 * 10)[1] is None
    checks, why = _check(_limits(max_order_lots=10), qty=75 * 11)
    assert "11 lots exceeds" in why and not checks["order_size_within_limit"]


def test_expiry_constraint_applies_to_new_long_options_and_fails_closed_when_unknown():
    limits = _limits(min_trading_days_to_expiry=0.5)
    assert _check(limits, option=_option(days=0.4))[1] == "Only 0.40 trading days to expiry (minimum 0.5 for a new long option)"
    assert _check(limits, option=_option(days=0.6))[1] is None
    assert "unknown" in _check(limits, option=_option(days=None))[1]
    assert _check(limits, option=_option(days=0.1), tx="SELL")[1] is None  # the constraint is about buying decay


def test_premium_at_risk_total_and_per_underlying():
    limits = _limits(max_premium_at_risk_pct=1.0, max_underlying_premium_pct=0.8)
    ex = exposure_from_positions([_pos(qty=75, price=100.0)])  # 7,500 = 0.75% already in NIFTY
    assert "Premium at risk would be 1.50%" in _check(limits, ex=ex, price=100.0)[1]  # the total limit is checked first
    only_underlying = _limits(max_premium_at_risk_pct=5.0, max_underlying_premium_pct=0.8)
    assert "Premium at risk in NIFTY would be 1.50%" in _check(only_underlying, ex=ex, price=100.0)[1]
    other = _check(limits, ex=ex, option=_option(underlying="BANKNIFTY"), price=100.0)  # 0.75% + 0.75% = 1.5% total
    assert "Premium at risk would be 1.50%" in other[1]
    small = _check(limits, ex=ex, option=_option(underlying="BANKNIFTY"), qty=75, price=20.0)
    assert small[1] is None and small[0]["premium_at_risk_within_limit"]


@pytest.mark.parametrize("limit,text", [
    (dict(max_net_delta_pct=5.0), "Net delta"),
    (dict(max_gamma_pct=5.0), "Gamma"),
    (dict(max_vega_pct=0.1), "Vega"),
    (dict(max_theta_pct=0.1), "Theta"),
])
def test_each_greek_limit_vetoes_when_the_post_trade_exposure_exceeds_it(limit, text):
    checks, why = _check(_limits(**limit))
    assert why is not None and why.startswith(text) and not checks[f"{next(iter(limit))[4:-4]}_within_limit"]


def test_greek_limits_are_compared_after_adding_the_open_book():
    one = exposure_from_positions([_pos()])
    two = exposure_from_positions([_pos(), _pos()])
    limits = _limits(max_vega_pct=0.2)  # one position is 0.15%, two are 0.30%
    assert _check(limits, ex=exposure_from_positions([]))[1] is None
    assert "Vega exposure would be 0.30%" in _check(limits, ex=one)[1]
    assert "Vega exposure would be 0.45%" in _check(limits, ex=two)[1]


def test_a_short_option_offsets_long_delta():
    long_book = exposure_from_positions([_pos(qty=300)])
    limits = _limits(max_net_delta_pct=100.0)  # 0.5*300*24000 = 3.6m = 360% of equity already: over
    assert "Net delta" in _check(limits, ex=long_book, qty=75)[1]
    hedged = exposure_from_positions([_pos(qty=300), _pos(qty=300, transaction="SELL")])
    assert _check(limits, ex=hedged, qty=75)[1] is None  # 0.5*75*24000 = 0.9m = 90%


def test_unknown_greeks_fail_closed_only_when_a_greek_limit_is_on():
    limits = _limits(max_vega_pct=5.0)
    checks, why = _check(limits, option=_option(greeks=None))
    assert "Greeks of the proposed option are unavailable" in why and not checks["trade_greeks_known"]
    ex = exposure_from_positions([_pos(greeks=None)])
    assert "no stored Greeks" in _check(limits, ex=ex)[1]
    assert _check(_limits(), ex=ex, option=_option(greeks=None))[1] is None


def test_non_positive_equity_fails_closed():
    assert "Equity is not positive" in check_option_trade({}, _option(), 75, 100.0, "BUY", 0.0, _limits(), 75)[1]
    assert trade_exposure(_option(), 75, "BUY")["vega"] == pytest.approx(1500.0)
