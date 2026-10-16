from quant_intelligence.risk.risk_engine import AccountState, ProposedTrade, evaluate_trade


def _account(**overrides):
    base = dict(
        equity=1_000_000,
        peak_equity=1_000_000,
        daily_pnl=0.0,
        open_positions_count=0,
        trades_today=0,
        exposure_by_strategy={},
        total_exposure=0.0,
        broker_connected=True,
        kill_switch_engaged=False,
    )
    base.update(overrides)
    return AccountState(**base)


def _trade(**overrides):
    base = dict(
        strategy_name="Momentum",
        direction="LONG",
        entry_price=100.0,
        stop_price=99.0,
        target_price=103.0,
        quantity=50,
        relative_volume=1.0,
        data_quality_status="OK",
    )
    base.update(overrides)
    return ProposedTrade(**base)


def test_kill_switch_vetoes_everything():
    decision = evaluate_trade(_account(kill_switch_engaged=True), _trade())
    assert not decision.approved
    assert "kill switch" in decision.reason.lower()


def test_disconnected_broker_vetoes():
    decision = evaluate_trade(_account(broker_connected=False), _trade())
    assert not decision.approved


def test_bad_data_quality_vetoes():
    decision = evaluate_trade(_account(), _trade(data_quality_status="FAIL"))
    assert not decision.approved


def test_daily_loss_limit_vetoes():
    decision = evaluate_trade(_account(daily_pnl=-50_000), _trade())  # 5% loss > 3% default limit
    assert not decision.approved
    assert "daily loss" in decision.reason.lower()


def test_max_trades_per_day_vetoes():
    decision = evaluate_trade(_account(trades_today=10), _trade())
    assert not decision.approved


def test_risk_per_trade_too_large_vetoes():
    # Risk = |100-99|*50 = 50 => 0.005% of equity, should pass. Make stop far away to exceed 1% limit.
    decision = evaluate_trade(_account(), _trade(stop_price=50.0, quantity=1000))
    assert not decision.approved
    assert "risk" in decision.reason.lower()


def test_valid_trade_is_approved():
    decision = evaluate_trade(_account(), _trade())
    assert decision.approved
    assert decision.decision_id.startswith("RISK-")


def test_illiquid_market_vetoes():
    decision = evaluate_trade(_account(), _trade(relative_volume=0.1))
    assert not decision.approved
    assert "liquidity" in decision.reason.lower()


import datetime as dt

import pytest

NOW = dt.datetime(2026, 10, 15, 11, 0)


@pytest.mark.parametrize("overrides,fragment", [
    (dict(stop_price=100.0), "zero stop distance"),
    (dict(stop_price=101.0), "not below entry"),
    (dict(direction="SHORT"), "not above entry"),
    (dict(target_price=99.0), "target"),
    (dict(entry_price=0.0), "entry price"),
    (dict(entry_price=float("nan")), "entry price"),
    (dict(stop_price=-1.0), "stop price"),
    (dict(quantity=0), "quantity"),
    (dict(quantity=-5), "quantity"),
    (dict(quantity=7, lot_size=5), "lot size"),
    (dict(direction="FLAT"), "direction"),
])
def test_invalid_order_parameters_always_veto(overrides, fragment):
    decision = evaluate_trade(_account(), _trade(**overrides), now=NOW)
    assert not decision.approved
    assert decision.reason.startswith("Invalid order")
    assert fragment in decision.reason
    assert decision.checks["order_parameters_valid"] is False


def test_a_whole_lot_multiple_is_valid():
    assert evaluate_trade(_account(), _trade(quantity=50, lot_size=25), now=NOW).approved


def test_stale_quote_vetoes_independently_of_the_data_quality_label():
    old = NOW - dt.timedelta(minutes=45)
    decision = evaluate_trade(_account(), _trade(data_quality_status="OK", quote_time=old), now=NOW)
    assert not decision.approved and "stale" in decision.reason.lower()


def test_fresh_quote_passes_and_future_timestamp_is_rejected():
    assert evaluate_trade(_account(), _trade(quote_time=NOW - dt.timedelta(minutes=2)), now=NOW).approved
    future = evaluate_trade(_account(), _trade(quote_time=NOW + dt.timedelta(minutes=30)), now=NOW)
    assert not future.approved and "stale" in future.reason.lower()


def test_wide_spread_vetoes():
    decision = evaluate_trade(_account(), _trade(bid=95.0, ask=105.0), now=NOW)
    assert not decision.approved and "spread" in decision.reason.lower()


def test_slippage_larger_than_half_the_stop_distance_vetoes():
    # spread 0.9 on a 100 price is 0.9% (allowed) but half-spread 0.45 is 45% of... use a tighter stop
    decision = evaluate_trade(_account(), _trade(stop_price=99.8, bid=99.55, ask=100.45), now=NOW)
    assert not decision.approved and "slippage" in decision.reason.lower()


def test_missing_or_crossed_quote_fails_closed():
    assert not evaluate_trade(_account(), _trade(bid=99.9), now=NOW).approved
    crossed = evaluate_trade(_account(), _trade(bid=100.2, ask=100.0), now=NOW)
    assert not crossed.approved and "crossed" in crossed.reason.lower()


def test_tight_spread_is_approved():
    assert evaluate_trade(_account(), _trade(bid=99.95, ask=100.05), now=NOW).approved


def test_the_veto_is_final_whatever_the_ranker_said():
    """The engine takes no ranking input: a selected, top-ranked or tie-break strategy is vetoed the same way."""
    for name in ("Momentum", "TIE-BREAK: Momentum", "Opening Range Breakout"):
        assert not evaluate_trade(_account(), _trade(strategy_name=name, stop_price=100.0), now=NOW).approved


def test_every_veto_has_a_reason():
    for acct, trade in [
        (_account(kill_switch_engaged=True), _trade()),
        (_account(), _trade(data_quality_status="DEGRADED")),
        (_account(trades_today=99), _trade()),
    ]:
        decision = evaluate_trade(acct, trade)
        assert decision.reason and len(decision.reason) > 0
