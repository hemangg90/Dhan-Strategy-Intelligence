"""Best-strike ranking: filters, scoring, BUY vs SELL profiles, missing data, and the OPTION_SELECTION switch."""
import dataclasses

import pytest

from quant_intelligence.config import settings as settings_module
from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow, parse_option_chain
from quant_intelligence.options.option_selector import OptionSelectionError, choose_contract, select_contract
from quant_intelligence.options.strike_ranker import rank_strikes, select_best_contract
from quant_intelligence.ui.chain_views import ranked_frame


def _row(strike, ce_delta, pe_delta, ce_ltp=50.0, pe_ltp=50.0, oi=10_000, vol=5_000, spread=0.02, theta=-4.0):
    def bid_ask(ltp):
        return ltp * (1 - spread / 2), ltp * (1 + spread / 2)

    ce_b, ce_a = bid_ask(ce_ltp)
    pe_b, pe_a = bid_ask(pe_ltp)
    return StrikeRow(
        strike=strike, ce_ltp=ce_ltp, ce_oi=oi, ce_oi_change=0, ce_volume=vol, ce_iv=15.0, ce_delta=ce_delta,
        ce_security_id=f"{strike:g}CE", pe_ltp=pe_ltp, pe_oi=oi, pe_oi_change=0, pe_volume=vol, pe_iv=15.0,
        pe_delta=pe_delta, pe_security_id=f"{strike:g}PE", ce_bid=ce_b, ce_ask=ce_a, ce_theta=theta,
        pe_bid=pe_b, pe_ask=pe_a, pe_theta=theta)


def _chain():
    rows = [
        _row(22800, 0.85, -0.15, ce_ltp=300),
        _row(22900, 0.70, -0.25, ce_ltp=210),
        _row(23000, 0.55, -0.45, ce_ltp=130),
        _row(23100, 0.45, -0.55, ce_ltp=90),
        _row(23200, 0.30, -0.70, ce_ltp=60),
        _row(23300, 0.15, -0.85, ce_ltp=35),
    ]
    return ChainSnapshot(underlying="NIFTY", expiry="2026-10-29", spot_price=23050.0, strikes=rows)


def test_a_buy_prefers_the_strike_nearest_the_target_delta():
    ranked = rank_strikes(_chain(), "LONG", "BUY", lot_size=75)
    assert ranked[0].strike == 23100 and ranked[0].option_type == "CE"  # delta 0.45 = the BUY target
    by_strike = {r.strike: r for r in ranked}
    assert not by_strike[22800].eligible and "delta" in by_strike[22800].notes[0]  # deep ITM: outside the band
    assert not by_strike[23300].eligible and [r.strike for r in ranked if r.eligible][0] == 23100


def test_a_sell_prefers_lower_delta_and_a_put_uses_the_pe_side():
    ranked = rank_strikes(_chain(), "SHORT", "SELL")
    assert ranked[0].option_type == "PE" and abs(ranked[0].delta) <= 0.35
    assert ranked[0].strike == 22900  # |delta| 0.25 is closest to the SELL target of 0.20 among eligible puts


def test_a_wide_spread_or_dead_strike_is_filtered_out():
    chain = _chain()
    wide = _row(23100, 0.45, -0.55, ce_ltp=90, spread=0.30)  # 30% spread
    dead = _row(23000, 0.55, -0.45, ce_ltp=130, oi=0, vol=0)
    chain.strikes[3], chain.strikes[2] = wide, dead
    by_strike = {r.strike: r for r in rank_strikes(chain, "LONG", "BUY")}
    assert not by_strike[23100].eligible and "too wide" in by_strike[23100].notes[0]
    assert not by_strike[23000].eligible and "open interest" in by_strike[23000].notes[0]
    assert rank_strikes(chain, "LONG", "BUY")[0].eligible


def test_liquidity_breaks_a_tie_between_equal_strikes():
    chain = _chain()
    chain.strikes[3] = _row(23100, 0.45, -0.55, ce_ltp=90, oi=500, vol=50)  # same delta/spread, thin
    chain.strikes[2] = _row(23000, 0.45, -0.55, ce_ltp=130, oi=50_000, vol=40_000)
    assert rank_strikes(chain, "LONG", "BUY")[0].strike == 23000


def test_max_outlay_excludes_dear_strikes_for_a_buy_only():
    ranked = rank_strikes(_chain(), "LONG", "BUY", lot_size=75, max_outlay=7_000)  # 90 x 75 = 6,750 fits; 130 x 75 does not
    by_strike = {r.strike: r for r in ranked}
    assert by_strike[23100].eligible and not by_strike[23000].eligible and "cap" in by_strike[23000].notes[0]
    sell = {r.strike: r for r in rank_strikes(_chain(), "SHORT", "SELL", lot_size=75, max_outlay=1)}
    assert sell[22900].eligible  # premium received is not an outlay


def test_missing_quote_data_scores_neutral_and_is_noted_not_guessed():
    chain = _chain()
    row = chain.strikes[3]
    chain.strikes[3] = dataclasses.replace(row, ce_bid=None, ce_ask=None, ce_theta=None)
    r = next(x for x in rank_strikes(chain, "LONG", "BUY") if x.strike == 23100)
    assert r.eligible and r.parts["spread"] == 0.5 and r.parts["decay"] == 0.5
    assert "no spread data" in r.notes and "no decay data" in r.notes and r.spread_pct is None


def test_select_best_contract_returns_a_tradeable_contract_or_explains():
    contract = select_best_contract(_chain(), "LONG", "BUY", lot_size=75)
    assert (contract.security_id, contract.strike, contract.transaction, contract.lot_size) == ("23100CE", 23100, "BUY", 75)
    with pytest.raises(OptionSelectionError, match="No CE strike passes"):
        select_best_contract(_chain(), "LONG", "BUY", lot_size=75, max_outlay=100)


def test_bad_input_is_rejected_like_select_contract():
    with pytest.raises(OptionSelectionError):
        rank_strikes(_chain(), "UP", "BUY")
    with pytest.raises(OptionSelectionError):
        rank_strikes(ChainSnapshot("NIFTY", "x", 1.0, []), "LONG", "BUY")


def test_choose_contract_follows_the_setting(monkeypatch):
    chain = _chain()
    off = dataclasses.replace(settings_module.SETTINGS, option_selection="OFFSET")
    best = dataclasses.replace(settings_module.SETTINGS, option_selection="BEST")
    monkeypatch.setattr(settings_module, "SETTINGS", off)
    # Spot 23050 is equidistant from 23000 and 23100: the offset rule takes 23000, the ranker prefers delta 0.45 at 23100.
    assert choose_contract(chain, "LONG", "BUY", 0, 75).strike == select_contract(chain, "LONG", "BUY", 0, 75).strike == 23000
    monkeypatch.setattr(settings_module, "SETTINGS", best)
    assert choose_contract(chain, "LONG", "BUY", 0, 75).strike == 23100
    assert choose_contract(chain, "LONG", "BUY", 0, 75, mode="OFFSET").strike == 23000  # explicit mode wins


def test_option_selection_defaults_to_offset_and_only_best_enables_the_ranker():
    assert settings_module.SETTINGS.option_selection == "OFFSET"


def test_parser_keeps_bid_ask_and_theta():
    raw = {"data": {"last_price": 100.0, "oc": {"100.000000": {
        "ce": {"last_price": 5.0, "oi": 10, "volume": 3, "top_bid_price": 4.9, "top_ask_price": 5.1,
               "greeks": {"delta": 0.5, "theta": -1.2}, "security_id": 7},
        "pe": {"last_price": 5.0}}}}}
    row = parse_option_chain("NIFTY", "2026-10-29", raw).strikes[0]
    assert (row.ce_bid, row.ce_ask, row.ce_theta) == (4.9, 5.1, -1.2) and row.pe_bid is None


def test_ranked_frame_lists_rank_score_and_reasons():
    df = ranked_frame(rank_strikes(_chain(), "LONG", "BUY"))
    assert df.iloc[0]["rank"] == 1 and df.iloc[0]["strike"] == 23100 and df.iloc[0]["score"] > 0
    rejected = df[df["strike"] == 22800].iloc[0]
    assert rejected["score"] is None or rejected["score"] != rejected["score"]  # None/NaN: not scored
    assert "delta" in rejected["notes"]


def test_option_chain_page_renders_the_ranking(monkeypatch):
    from unittest.mock import MagicMock

    from streamlit.testing.v1 import AppTest


    at = AppTest.from_file(str(__import__("pathlib").Path(settings_module.__file__).resolve().parent.parent
                               / "pages" / "19_Option_Chain.py"), default_timeout=60)
    at.session_state["dhan_api_client"] = MagicMock(is_configured=MagicMock(return_value=True))
    at.session_state["option_chain"] = _chain()
    at.session_state["oc_underlying"] = "NIFTY"
    at.run()
    assert not at.exception
    assert any("Best:" in s.value and "23100" in s.value for s in at.success)
    assert at.dataframe and len(at.dataframe[0].value) == 6
