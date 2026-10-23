"""Normalised contracts, chain validation and Greeks (against textbook references and finite differences)."""
import datetime as dt

import pytest

from quant_intelligence.config.expiry_rules import time_to_expiry_years
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.intelligence.status import CRITICAL, OK, WARNING
from quant_intelligence.options import black_scholes as bs
from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow, parse_option_chain
from quant_intelligence.options.contracts import OptionQuote, quotes_from_chain, validate_chain
from quant_intelligence.options.greeks import MIN_T_TRADING_DAYS, compute_greeks, rates_for
from quant_intelligence.utils.market_profile import MCX, NSE

NOW = dt.datetime(2030, 1, 7, 10, 0)  # a Monday, mid-session
EXPIRY = "2030-01-31"


# ------------------------------------------------------------------ textbook references
def test_black_scholes_matches_published_values():
    # S=K=100, T=1, r=5%, sigma=20%, no dividend: widely tabulated reference values
    args = (100.0, 100.0, 1.0, 0.05, 0.0, 0.20)
    assert bs.price(*args, "CALL") == pytest.approx(10.4506, abs=1e-4)
    assert bs.price(*args, "PUT") == pytest.approx(5.5735, abs=1e-4)
    g = bs.greeks(*args, "CALL")
    assert g["delta"] == pytest.approx(0.63683, abs=1e-5) and g["gamma"] == pytest.approx(0.018762, abs=1e-6)
    assert g["vega"] == pytest.approx(37.524, abs=1e-3) and g["theta"] == pytest.approx(-6.41405, abs=1e-4)
    assert bs.rho(*args, "CALL") == pytest.approx(53.2325, abs=1e-3)
    assert bs.rho(*args, "PUT") == pytest.approx(-41.8905, abs=1e-3)
    # Hull's worked example: S=42, K=40, r=10%, sigma=20%, T=0.5
    assert bs.price(42, 40, 0.5, 0.10, 0.0, 0.20, "CALL") == pytest.approx(4.7594, abs=1e-3)
    assert bs.price(42, 40, 0.5, 0.10, 0.0, 0.20, "PUT") == pytest.approx(0.8086, abs=1e-3)


@pytest.mark.parametrize("kind", ["CALL", "PUT"])
def test_greeks_agree_with_finite_differences_of_the_price(kind):
    S, K, T, r, q, sigma = 105.0, 100.0, 0.4, 0.06, 0.012, 0.23
    price = lambda **kw: bs.price(kw.get("S", S), K, kw.get("T", T), kw.get("r", r), q, kw.get("sigma", sigma), kind)  # noqa: E731
    h = 1e-3 * S
    delta = (price(S=S + h) - price(S=S - h)) / (2 * h)
    gamma = (price(S=S + h) - 2 * price() + price(S=S - h)) / h**2
    vega = (price(sigma=sigma + 1e-4) - price(sigma=sigma - 1e-4)) / 2e-4
    theta = -(price(T=T + 1e-5) - price(T=T - 1e-5)) / 2e-5
    rho = (price(r=r + 1e-5) - price(r=r - 1e-5)) / 2e-5
    g = bs.greeks(S, K, T, r, q, sigma, kind)
    assert g["delta"] == pytest.approx(delta, abs=1e-5) and g["gamma"] == pytest.approx(gamma, rel=1e-3)
    assert g["vega"] == pytest.approx(vega, rel=1e-4) and g["theta"] == pytest.approx(theta, rel=1e-3)
    assert bs.rho(S, K, T, r, q, sigma, kind) == pytest.approx(rho, rel=1e-4)


def test_put_call_parity_holds_with_carry():
    S, K, T, r, q, sigma = 100.0, 95.0, 0.5, 0.06, 0.02, 0.25
    lhs = bs.price(S, K, T, r, q, sigma, "CALL") - bs.price(S, K, T, r, q, sigma, "PUT")
    assert lhs == pytest.approx(S * 2.718281828**(-q * T) - K * 2.718281828**(-r * T), abs=1e-4)


# ------------------------------------------------------------------ the contract schema
def _row(strike, **kw):
    base = dict(strike=float(strike), ce_ltp=10.0, ce_oi=1000, ce_oi_change=50, ce_volume=300, ce_iv=18.0, ce_delta=0.5,
                ce_security_id=f"{strike}C", pe_ltp=9.0, pe_oi=1200, pe_oi_change=-20, pe_volume=250, pe_iv=19.0, pe_delta=-0.5,
                pe_security_id=f"{strike}P", ce_bid=9.9, ce_ask=10.1, pe_bid=8.9, pe_ask=9.1, ce_bid_qty=500, ce_ask_qty=400,
                pe_bid_qty=300, pe_ask_qty=350, ce_prev_close=9.0, pe_prev_close=9.5)
    base.update(kw)
    return StrikeRow(**base)


def _chain(strikes=(23900, 23950, 24000, 24050, 24100), spot=24000.0, **meta):
    meta.setdefault("fetched_at", NOW)
    meta.setdefault("lot_size", 75)
    meta.setdefault("exchange_segment", "NSE_FNO")
    return ChainSnapshot("NIFTY", EXPIRY, spot, [_row(s) for s in strikes], **meta)


def test_a_chain_flattens_to_one_quote_per_contract_with_metadata():
    quotes = quotes_from_chain(_chain())
    assert len(quotes) == 10 and {q.option_type for q in quotes} == {"CE", "PE"}
    q = next(x for x in quotes if x.strike == 24000 and x.option_type == "PE")
    assert (q.bid, q.ask, q.bid_qty, q.oi, q.oi_change, q.lot_size, q.exchange_segment, q.quote_time) == (8.9, 9.1, 300, 1200, -20, 75, "NSE_FNO", NOW)
    assert q.mid == pytest.approx(9.0) and q.has_two_sided_quote and q.unavailable == ("gamma", "theta", "vega")


def test_missing_fields_are_named_not_filled():
    chain = _chain()
    chain.strikes[0] = _row(23900, ce_bid=None, ce_ask=None, ce_iv=None, ce_delta=None)
    q = next(x for x in quotes_from_chain(chain) if x.strike == 23900 and x.option_type == "CE")
    assert q.bid is None and q.iv is None and {"bid", "ask", "iv", "delta"} <= set(q.unavailable) and q.mid is None


def test_a_zero_or_crossed_quote_has_no_mid():
    chain = _chain()
    chain.strikes[0] = _row(23900, ce_bid=0.0, ce_ask=5.0, pe_bid=5.0, pe_ask=4.0)
    quotes = {(x.strike, x.option_type): x for x in quotes_from_chain(chain)}
    assert quotes[(23900.0, "CE")].mid is None and quotes[(23900.0, "PE")].mid is None


def test_the_parser_carries_depth_prices_and_greeks():
    raw = {"data": {"last_price": 100.0, "oc": {"100.000000": {
        "ce": {"last_price": 5.0, "oi": 10, "previous_oi": 8, "top_bid_price": 4.9, "top_ask_price": 5.1, "top_bid_quantity": 75,
               "top_ask_quantity": 150, "previous_close_price": 4.0, "security_id": "1",
               "greeks": {"delta": 0.5, "gamma": 0.01, "theta": -1.2, "vega": 3.0}},
        "pe": {"last_price": 5.0, "oi": 10, "security_id": "2"}}}}}
    row = parse_option_chain("NIFTY", EXPIRY, raw).strikes[0]
    assert (row.ce_bid_qty, row.ce_ask_qty, row.ce_gamma, row.ce_theta, row.ce_vega) == (75, 150, 0.01, -1.2, 3.0)
    assert row.pe_gamma is None and row.pe_bid_qty is None


# ------------------------------------------------------------------ chain validation
def test_a_good_chain_validates_clean():
    status = validate_chain(_chain(), now=NOW, expected_lot_size=75, expected_strike_step=50)
    assert status.level == OK


@pytest.mark.parametrize("chain,kwargs,text", [
    (ChainSnapshot("NIFTY", EXPIRY, 24000.0, []), {}, "no strikes"),
    (ChainSnapshot("NIFTY", EXPIRY, 0.0, [_row(24000)]), {}, "underlying price"),
    (ChainSnapshot("NIFTY", "garbage", 24000.0, [_row(24000)]), {}, "not a valid date"),
    (ChainSnapshot("NIFTY", "2029-12-31", 24000.0, [_row(24000)]), {}, "already passed"),
    (ChainSnapshot("NIFTY", EXPIRY, 24000.0, [_row(24000), _row(24000)]), {}, "duplicate strikes"),
    (_chain(), {"expected_lot_size": 50}, "lot size 75 differs"),
    (_chain(lot_size=None), {"expected_lot_size": 75}, "no lot size"),
    (_chain(), {"expected_strike_step": 100}, "strike step 50"),
    (_chain(fetched_at=NOW - dt.timedelta(seconds=500)), {}, "500s old"),
    (_chain(fetched_at=NOW + dt.timedelta(seconds=60)), {}, "in the future"),
])
def test_a_bad_chain_is_critical_with_the_reason(chain, kwargs, text):
    status = validate_chain(chain, now=NOW, **kwargs)
    assert status.level == CRITICAL and text in status.reason


def test_warnings_for_gaps_missing_snapshot_time_and_unquoted_contracts():
    gap = validate_chain(_chain(strikes=(23900, 23950, 24100, 24150)), now=NOW)
    assert gap.level == WARNING and "gap" in gap.reason
    no_time = validate_chain(_chain(fetched_at=None), now=NOW)
    assert no_time.level == WARNING and "quote_time" in no_time.unavailable
    chain = _chain()
    chain.strikes = [_row(s, ce_bid=0.0, ce_ask=0.0, pe_bid=0.0, pe_ask=0.0) for s in (23900, 23950, 24000, 24050, 24100)]
    assert "two-sided quote" in validate_chain(chain, now=NOW).reason


# ------------------------------------------------------------------ Greeks of a quoted contract
def _quote(strike=24000.0, kind="CE", underlying="NIFTY", expiry=EXPIRY, **kw):
    base = dict(underlying=underlying, exchange_segment="NSE_FNO", security_id="1", option_type=kind, strike=strike, expiry=expiry,
                lot_size=75, bid=None, ask=None, bid_qty=None, ask_qty=None, ltp=None, volume=100, oi=1000, oi_change=0, iv=None,
                delta=None, gamma=None, theta=None, vega=None, prev_close=None, quote_time=NOW, source="test", unavailable=())
    base.update(kw)
    return OptionQuote(**base)


def _fair(strike, kind, sigma, spot, underlying="NIFTY", expiry=EXPIRY, now=NOW):
    profile = MCX if underlying in ("CRUDEOIL", "NATURALGAS", "CRUDEOILM") else NSE
    r, q, _ = rates_for(underlying)
    t = time_to_expiry_years(now, dt.date.fromisoformat(expiry), profile, SETTINGS.trading_days_per_year)
    return bs.price(spot, strike, t, r, q, sigma, "CALL" if kind == "CE" else "PUT"), t, r, q


def test_the_solved_volatility_round_trips_and_the_greeks_follow():
    spot = 24000.0
    price, t, r, q = _fair(24100.0, "CE", 0.18, spot)
    g = compute_greeks(_quote(24100.0, ltp=price), spot, NOW)
    assert g.available and g.model == "BSM" and g.price_basis == "ltp" and g.iv_source == "solved_from_price"
    assert g.iv == pytest.approx(0.18, abs=1e-6) and g.t_years == pytest.approx(t)
    ref = bs.greeks(spot, 24100.0, t, r, q, 0.18, "CALL")
    assert g.delta == pytest.approx(ref["delta"]) and g.gamma == pytest.approx(ref["gamma"])
    assert g.vega == pytest.approx(ref["vega"] / 100) and g.theta == pytest.approx(ref["theta"] / SETTINGS.trading_days_per_year)
    assert g.theta < 0 and g.vega > 0 and g.gamma > 0 and 0 < g.delta < 1 and g.rho > 0


def test_a_put_has_negative_delta_and_negative_rho():
    price, *_ = _fair(23900.0, "PE", 0.2, 24000.0)
    g = compute_greeks(_quote(23900.0, "PE", ltp=price), 24000.0, NOW)
    assert g.available and -1 < g.delta < 0 and g.rho < 0


def test_a_two_sided_quote_prices_off_the_mid():
    price, *_ = _fair(24000.0, "CE", 0.2, 24000.0)
    g = compute_greeks(_quote(bid=price - 1, ask=price + 1, ltp=price + 30), 24000.0, NOW)
    assert g.price_basis == "mid" and g.iv == pytest.approx(0.2, abs=1e-6)


def test_commodity_options_use_black_76_with_the_forward_as_spot():
    expiry = "2030-01-25"
    price, t, r, q = _fair(6000.0, "CE", 0.35, 6050.0, "CRUDEOIL", expiry)
    assert q == r
    g = compute_greeks(_quote(6000.0, underlying="CRUDEOIL", expiry=expiry, ltp=price), 6050.0, NOW)
    assert g.available and g.model == "BLACK76" and g.iv == pytest.approx(0.35, abs=1e-6)


def test_greeks_are_unavailable_not_zero_when_inputs_are_missing_or_unsafe():
    assert "no usable price" in compute_greeks(_quote(), 24000.0, NOW).reason
    assert "expired" in compute_greeks(_quote(ltp=5.0, expiry="2029-12-20"), 24000.0, NOW).reason
    soon = dt.datetime(2030, 1, 31, 15, 29, 30)
    near = compute_greeks(_quote(ltp=5.0), 24000.0, soon)
    assert not near.available and "too close to expiry" in near.reason and near.trading_days < MIN_T_TRADING_DAYS
    assert "underlying price" in compute_greeks(_quote(ltp=5.0), float("nan"), NOW).reason
    assert "strike" in compute_greeks(_quote(strike=-5.0, ltp=5.0), 24000.0, NOW).reason
    assert "expiry" in compute_greeks(_quote(ltp=5.0, expiry="soon"), 24000.0, NOW).reason
    assert "option type" in compute_greeks(_quote(kind="XX", ltp=5.0), 24000.0, NOW).reason


def test_a_price_below_intrinsic_falls_back_to_the_provider_iv_or_is_unavailable():
    deep = _quote(23000.0, ltp=100.0)  # intrinsic is about 1000: no volatility explains 100
    assert not compute_greeks(deep, 24000.0, NOW).available
    fallback = compute_greeks(_quote(23000.0, ltp=100.0, iv=20.0), 24000.0, NOW)
    assert fallback.available and fallback.iv_source == "provider" and fallback.iv == pytest.approx(0.20)


def test_disagreement_with_the_provider_delta_is_flagged():
    price, *_ = _fair(24000.0, "CE", 0.2, 24000.0)
    agree = compute_greeks(_quote(ltp=price, delta=0.53), 24000.0, NOW)
    off = compute_greeks(_quote(ltp=price, delta=0.20), 24000.0, NOW)
    assert not agree.provider_disagrees and off.provider_disagrees and off.provider_delta_gap > 0.05


def test_position_exposure_scales_by_quantity_and_sign():
    price, *_ = _fair(24000.0, "CE", 0.2, 24000.0)
    g = compute_greeks(_quote(ltp=price), 24000.0, NOW)
    long_ = g.scaled(150)
    short = g.scaled(-150)
    assert long_["delta"] == pytest.approx(g.delta * 150) and short["vega"] == pytest.approx(-long_["vega"])
    with pytest.raises(ValueError):
        compute_greeks(_quote(), 24000.0, NOW).scaled(10)
