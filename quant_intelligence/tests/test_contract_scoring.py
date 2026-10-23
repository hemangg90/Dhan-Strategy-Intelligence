"""Chain-scored contract selection: tradeability filters, OI build-up, walls, and the auto-trader wiring."""
import pytest

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.options.chain_analytics import ChainSnapshot, StrikeRow, parse_option_chain
from quant_intelligence.options.contract_scoring import (
    LONG_BUILDUP,
    LONG_UNWINDING,
    NEUTRAL,
    SHORT_BUILDUP,
    SHORT_COVERING,
    UNKNOWN,
    classify_buildup,
    select_best_contract,
)
from quant_intelligence.options.option_selector import OptionSelectionError


def _row(strike, spot=100.0, **kw):
    """A healthy strike: both legs liquid, tight, delta falling away from the money."""
    ce_delta = max(0.05, min(0.95, 0.5 - (strike - spot) / 40.0))
    base = dict(
        strike=float(strike),
        ce_ltp=max(spot - strike, 0) + 5.0, ce_oi=10_000, ce_oi_change=0, ce_volume=5_000, ce_iv=15.0, ce_delta=ce_delta,
        ce_security_id=f"{strike}CE", ce_bid=None, ce_ask=None, ce_prev_close=None,
        pe_ltp=max(strike - spot, 0) + 5.0, pe_oi=10_000, pe_oi_change=0, pe_volume=5_000, pe_iv=15.0, pe_delta=-(1 - ce_delta),
        pe_security_id=f"{strike}PE", pe_bid=None, pe_ask=None, pe_prev_close=None,
    )
    base.update(kw)
    return StrikeRow(**base)


def _chain(rows=None, spot=100.0):
    rows = rows if rows is not None else [_row(k, spot) for k in (90, 95, 100, 105, 110)]
    return ChainSnapshot(underlying="NIFTY", expiry="2030-01-31", spot_price=spot, strikes=rows)


def _walled(rows):
    """Put the heaviest call and put OI far from the money so the OI-wall part does not decide a test about something else."""
    return rows + [_row(120, ce_oi=100_000), _row(80, pe_oi=100_000)]


def _quote(price, spread_pct):
    half = price * spread_pct / 200.0
    return price - half, price + half


# ------------------------------------------------------------------ build-up reading
@pytest.mark.parametrize("ltp,prev,oi,chg,expected", [
    (12.0, 10.0, 1100, 100, LONG_BUILDUP),
    (8.0, 10.0, 1100, 100, SHORT_BUILDUP),
    (12.0, 10.0, 900, -100, SHORT_COVERING),
    (8.0, 10.0, 900, -100, LONG_UNWINDING),
    (12.0, 10.0, 1001, 1, NEUTRAL),  # 0.1% OI move is noise
    (10.0, 10.0, 1100, 100, NEUTRAL),  # price unchanged
    (None, 10.0, 1100, 100, UNKNOWN),
    (12.0, None, 1100, 100, UNKNOWN),
    (12.0, 10.0, 1100, None, UNKNOWN),
])
def test_buildup_reading(ltp, prev, oi, chg, expected):
    assert classify_buildup(ltp, prev, oi, chg) == expected


# ------------------------------------------------------------------ choosing the strike
def test_a_strike_with_long_buildup_beats_an_otherwise_equal_one():
    rows = [_row(k) for k in (95, 100, 105)]
    rows[2] = _row(105, ce_prev_close=3.0, ce_oi_change=1500, ce_oi=11_500)  # price up on rising OI at 105
    rows[1] = _row(100, ce_prev_close=7.0, ce_oi_change=1500, ce_oi=11_500)  # price down on rising OI at 100: writers
    choice = select_best_contract(_chain(_walled(rows)), "LONG", "BUY", 50)
    assert choice.contract.strike == 105 and choice.scored.buildup == LONG_BUILDUP
    by_strike = {s.strike: s for s in choice.ranking}
    assert by_strike[100].buildup == SHORT_BUILDUP and by_strike[100].score < choice.scored.score


def test_long_goes_to_calls_and_short_to_puts():
    assert select_best_contract(_chain(), "LONG", "BUY").contract.option_type == "CE"
    assert select_best_contract(_chain(), "SHORT", "BUY").contract.option_type == "PE"


def test_the_contract_carries_the_chain_values():
    contract = select_best_contract(_chain(), "LONG", "BUY", 75).contract
    assert contract.lot_size == 75 and contract.transaction == "BUY" and contract.underlying == "NIFTY"
    assert contract.security_id == f"{int(contract.strike)}CE" and contract.premium > 0 and contract.delta is not None


def test_a_wide_spread_strike_is_rejected_and_a_tight_one_wins():
    rows = [_row(k) for k in (95, 100, 105)]
    bid, ask = _quote(5.0, 30.0)
    rows[1] = _row(100, ce_bid=bid, ce_ask=ask)  # ATM, but a 30% spread
    bid, ask = _quote(5.0, 2.0)
    rows[2] = _row(105, ce_bid=bid, ce_ask=ask)
    choice = select_best_contract(_chain(_walled(rows)), "LONG", "BUY")
    assert choice.contract.strike == 105
    assert (100.0, "spread too wide (30.0%)") in [(k, v) for k, v in choice.rejected]


def test_thin_open_interest_no_volume_no_bid_and_low_premium_are_rejected():
    rows = [
        _row(95, ce_oi=100),  # far below the chain median
        _row(100, ce_volume=0),
        _row(105, ce_bid=0.0, ce_ask=5.0),
        _row(110, ce_ltp=0.5),
        _row(90),
    ]
    choice = select_best_contract(_chain(rows), "LONG", "BUY")
    reasons = dict(choice.rejected)
    assert reasons[95.0] == "thin open interest" and reasons[100.0] == "no volume today"
    assert reasons[105.0] == "no bid" and reasons[110.0].startswith("premium too low")
    assert choice.contract.strike == 90


def test_delta_outside_the_band_is_rejected():
    rows = [_row(90, ce_delta=0.9), _row(100), _row(110, ce_delta=0.1)]
    choice = select_best_contract(_chain(rows), "LONG", "BUY")
    assert {k for k, v in choice.rejected} == {90.0, 110.0} and choice.contract.strike == 100


def test_a_call_beyond_the_heaviest_call_oi_strike_is_nudged_down():
    rows = [_row(k) for k in (100, 105, 110)]
    rows[1] = _row(105, ce_oi=60_000)  # the most call OI sits at 105
    choice = select_best_contract(_chain(rows), "LONG", "BUY")
    parts = {s.strike: s.parts["oi_peak"] for s in choice.ranking}
    assert choice.context["call_oi_peak"] == 105
    assert parts[110] < parts[105] and parts[100] == parts[105] == 1.0


def test_a_put_below_the_heaviest_put_oi_strike_is_nudged_down():
    rows = [_row(k) for k in (90, 95, 100)]
    rows[1] = _row(95, pe_oi=60_000)  # the most put OI at 95
    choice = select_best_contract(_chain(rows), "SHORT", "BUY")
    parts = {s.strike: s.parts["oi_peak"] for s in choice.ranking}
    assert choice.context["put_oi_peak"] == 95 and parts[90] < parts[95]


def test_the_oi_peak_part_carries_only_a_small_weight():
    from quant_intelligence.options.contract_scoring import WEIGHTS

    assert WEIGHTS["oi_peak"] <= 0.05 and abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_cheaper_iv_is_preferred_when_buying_and_richer_when_selling():
    buy_rows = [_row(95, ce_iv=10.0), _row(100, ce_iv=15.0), _row(105, ce_iv=25.0)]
    buy = {s.strike: s.parts["iv"] for s in select_best_contract(_chain(_walled(buy_rows)), "LONG", "BUY").ranking}
    assert buy[95] > buy[100] > buy[105]
    sell_rows = [_row(100), _row(105, ce_iv=10.0), _row(110, ce_iv=15.0), _row(115, ce_iv=25.0)]  # inside the 0.10-0.45 write band
    sell = {s.strike: s.parts["iv"] for s in select_best_contract(_chain(_walled(sell_rows)), "LONG", "SELL").ranking}
    assert sell[115] > sell[110] > sell[105]


def test_only_the_window_around_atm_is_considered():
    original = SETTINGS.option_window_strikes
    object.__setattr__(SETTINGS, "option_window_strikes", 1)
    try:
        rows = [_row(k) for k in (80, 85, 90, 95, 100, 105, 110, 115, 120)]
        choice = select_best_contract(_chain(rows), "LONG", "BUY")
        assert choice.considered == 3 and {s.strike for s in choice.ranking} <= {95.0, 100.0, 105.0}
    finally:
        object.__setattr__(SETTINGS, "option_window_strikes", original)


def test_nothing_tradeable_fails_closed_with_the_reasons():
    rows = [_row(k, ce_volume=0) for k in (95, 100, 105)]
    with pytest.raises(OptionSelectionError) as err:
        select_best_contract(_chain(rows), "LONG", "BUY")
    assert "NO SUITABLE CONTRACT" in str(err.value) and "no tradeable CE" in str(err.value) and "3 x no volume today" in str(err.value)


def test_missing_bid_ask_and_delta_do_not_reject_but_score_lower():
    complete = _row(100, ce_bid=_quote(5.0, 1.0)[0], ce_ask=_quote(5.0, 1.0)[1])
    missing = _row(105, ce_delta=None)
    choice = select_best_contract(_chain([complete, missing]), "LONG", "BUY")
    by_strike = {s.strike: s for s in choice.ranking}
    assert 105.0 in by_strike and by_strike[105].parts["spread"] < by_strike[100].parts["spread"]


def test_invalid_inputs_and_an_empty_chain_raise():
    with pytest.raises(OptionSelectionError):
        select_best_contract(_chain(), "SIDEWAYS", "BUY")
    with pytest.raises(OptionSelectionError):
        select_best_contract(_chain(), "LONG", "HOLD")
    with pytest.raises(OptionSelectionError):
        select_best_contract(ChainSnapshot("NIFTY", "2030-01-31", 100.0, []), "LONG", "BUY")


def test_the_summary_names_the_pick_and_its_reasons():
    text = select_best_contract(_chain(), "LONG", "BUY").summary
    assert text.startswith("Strike pick: ") and "score" in text and "PCR" in text and "of 5 considered" in text


# ------------------------------------------------------------------ parsing the extra quote fields
def test_the_parser_reads_bid_ask_and_previous_close():
    raw = {"data": {"last_price": 100.0, "oc": {"100.000000": {
        "ce": {"last_price": 5.0, "oi": 10, "previous_oi": 8, "top_bid_price": 4.9, "top_ask_price": 5.1,
               "previous_close_price": 4.0, "security_id": "1", "greeks": {"delta": 0.5}},
        "pe": {"last_price": 5.0, "oi": 10, "previous_oi": 12, "security_id": "2"},
    }}}}
    row = parse_option_chain("NIFTY", "2030-01-31", raw).strikes[0]
    assert (row.ce_bid, row.ce_ask, row.ce_prev_close) == (4.9, 5.1, 4.0) and row.pe_bid is None and row.ce_oi_change == 2


# ------------------------------------------------------------------ the auto-trader uses it
def _auto_cycle(monkeypatch, chain, smart=True):
    from types import SimpleNamespace

    from quant_intelligence.brokers.paper_broker import PaperBroker
    from quant_intelligence.execution import auto_trader
    from quant_intelligence.risk.risk_engine import AccountState

    monkeypatch.setattr(auto_trader, "get_strategy", lambda n: object())
    setup = SimpleNamespace(direction="LONG", meta={"transaction": "BUY"}, entry_price=100.0, stop_price=98.0, target_price=104.0)
    monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, **k: SimpleNamespace(status=auto_trader.SETUP_TRIGGERED, setup=setup))
    monkeypatch.setattr(auto_trader, "get_underlying_info", lambda u: {"lot_size": 50})
    object.__setattr__(SETTINGS, "option_selection", "SCORED" if smart else "OFFSET")
    ranking = SimpleNamespace(is_no_trade=False, selected_strategy="Momentum", tie_break=False, reason="r", tie_runner_up=None)
    output = SimpleNamespace(ranking=ranking, ohlcv=SimpleNamespace(tail=lambda n: None),
                             market_state=SimpleNamespace(get=lambda k, d=None: None), data_quality_status="OK")
    account = AccountState(equity=1_000_000.0, peak_equity=1_000_000.0, daily_pnl=0.0, open_positions_count=0, trades_today=0,
                           exposure_by_strategy={}, total_exposure=0.0, broker_connected=True, kill_switch_engaged=False)
    broker = PaperBroker(starting_capital=1_000_000)
    return broker, auto_trader.run_auto_option_cycle(output, "NIFTY", broker, client=None, chain=chain, account=account)


@pytest.fixture
def restore_smart_flag():
    original = SETTINGS.option_selection
    yield
    object.__setattr__(SETTINGS, "option_selection", original)


def test_the_auto_cycle_trades_the_scored_strike_and_says_why(monkeypatch, restore_smart_flag):
    rows = _walled([_row(95), _row(100), _row(105, ce_prev_close=3.0, ce_oi_change=1500, ce_oi=11_500)])
    broker, result = _auto_cycle(monkeypatch, _chain(rows))
    assert result.order is not None and result.order.status == "FILLED"
    assert broker.get_open_positions()[0]["strike"] == 105.0  # the fixed ATM rule would have taken 100
    assert "Strike pick: 105 CE" in result.reason and result.contract_review.startswith("Strike pick: 105 CE")


def test_the_fixed_strike_rule_still_works_when_smart_selection_is_off(monkeypatch, restore_smart_flag):
    rows = [_row(95), _row(100), _row(105, ce_prev_close=3.0, ce_oi_change=1500, ce_oi=11_500)]
    broker, result = _auto_cycle(monkeypatch, _chain(rows), smart=False)
    assert broker.get_open_positions()[0]["strike"] == 100.0 and result.contract_review == ""


def test_a_dead_chain_places_no_order(monkeypatch, restore_smart_flag):
    rows = [_row(k, ce_volume=0) for k in (95, 100, 105)]
    broker, result = _auto_cycle(monkeypatch, _chain(rows))
    assert result.order is None and broker.get_open_positions() == []
    assert result.reason.startswith("Could not resolve a tradeable option: NO SUITABLE CONTRACT: no tradeable CE")
