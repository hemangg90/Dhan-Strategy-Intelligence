"""Tie-break trading: when the top strategies are statistically tied but the top one is eligible, trade it
anyway - tagged and sized down - for paper and (manually confirmed) live."""
import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from quant_intelligence.brokers.base_broker import OrderRequest
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.config import settings as settings_module
from quant_intelligence.execution import auto_trader
from quant_intelligence.ranking.ranking_engine import TIE_BREAK_TAG, rank_and_select, score_strategy, tradable
from quant_intelligence.risk.risk_engine import AccountState


def _m(name, mean, se=0.2, folds=0, pos=None, conf="HIGH"):
    return {"strategy_name": name, "confidence_label": conf, "sample_size": 30, "n_days": 20, "expected_r": mean,
            "prob_positive_return": 0.55, "expected_r_stderr": se, "global_expected_r": 0.0,
            "stability_folds": folds, "stability_positive_frac": pos}


def _tied_scores():
    # both eligible, edges 0.30 vs 0.25 with se 0.2 each -> z ~ 0.18 << 1: not distinguishable
    return [score_strategy(_m("Momentum", 0.30), -2.0), score_strategy(_m("VWAP Mean Reversion", 0.25), -2.0)]


# ---------------------------------------------------------------- ranking rules
def test_without_tie_break_a_tie_is_still_no_trade():
    d = rank_and_select(_tied_scores())
    assert d.is_no_trade and not d.tie_break and "distinguishable" in d.reason


def test_with_tie_break_the_top_eligible_strategy_is_selected_and_flagged():
    d = rank_and_select(_tied_scores(), allow_tie_break=True)
    assert not d.is_no_trade and d.tie_break
    assert d.selected_strategy == "Momentum" and d.tie_runner_up == "VWAP Mean Reversion"
    assert d.reason.startswith("TIE-BREAK:") and "VWAP Mean Reversion" in d.reason


def test_a_clear_winner_is_not_a_tie_break():
    scores = [score_strategy(_m("A", 0.9, se=0.05), -2.0), score_strategy(_m("B", 0.2, se=0.05), -2.0)]
    d = rank_and_select(scores, allow_tie_break=True)
    assert d.selected_strategy == "A" and not d.tie_break and not d.is_no_trade


def test_tie_break_never_overrides_safety_conditions():
    # data quality not OK -> still NO TRADE
    assert rank_and_select(_tied_scores(), "DEGRADED", allow_tie_break=True).is_no_trade
    # nothing eligible -> still NO TRADE
    weak = [score_strategy(_m("A", -0.2), -2.0), score_strategy(_m("B", -0.3), -2.0)]
    assert rank_and_select(weak, allow_tie_break=True).is_no_trade
    # an unstable leader is ineligible, so a tie with it is not a trade
    unstable = [score_strategy(_m("A", 0.5, folds=4, pos=0.0), -2.0), score_strategy(_m("B", 0.4, folds=4, pos=0.0), -2.0)]
    assert rank_and_select(unstable, allow_tie_break=True).is_no_trade


def test_tradable_honours_the_per_mode_switches(monkeypatch):
    d = rank_and_select(_tied_scores(), allow_tie_break=True)
    # defaults: paper trades tie-breaks, live does NOT (no demonstrated edge, so it is never sent live unless enabled)
    assert tradable(d, "PAPER") == (True, None)
    ok, why = tradable(d, "LIVE")
    assert not ok and "turned off for live" in why

    both_on = dataclasses.replace(settings_module.SETTINGS, tie_break_live=True, tie_break_paper=True)
    monkeypatch.setattr(settings_module, "SETTINGS", both_on)
    assert tradable(d, "PAPER") == (True, None) and tradable(d, "LIVE") == (True, None)

    paper_off = dataclasses.replace(settings_module.SETTINGS, tie_break_paper=False)
    monkeypatch.setattr(settings_module, "SETTINGS", paper_off)
    ok, why = tradable(d, "PAPER")
    assert not ok and "turned off for paper" in why

    no_trade = rank_and_select(_tied_scores())
    assert tradable(no_trade, "PAPER")[0] is False


# ---------------------------------------------------------------- sizing
def _account(equity=1_000_000.0):
    return AccountState(equity=equity, peak_equity=equity, daily_pnl=0.0, open_positions_count=0, trades_today=0,
                        exposure_by_strategy={}, total_exposure=0.0, broker_connected=True, kill_switch_engaged=False)


def test_tie_break_sizing_halves_lots_but_never_to_zero():
    full = auto_trader.size_position(_account(), 50.0, 40.0, 100)
    half = auto_trader.size_position(_account(), 50.0, 40.0, 100, size_factor=0.5)
    assert full == 900 and half == 400  # 9 lots -> 4 lots
    assert auto_trader.size_position(_account(), 50.0, 40.0, 100, size_factor=0.01) == 100  # floor of one lot
    assert auto_trader.size_position(_account(), 50.0, 50.0, 100, size_factor=0.5) == 0  # nothing to size stays zero


# ---------------------------------------------------------------- the paper path end to end
def _cycle(monkeypatch, tie_break, tag_expected):
    broker = PaperBroker(starting_capital=1_000_000)
    setup = SimpleNamespace(direction="LONG", meta={"transaction": "BUY"})
    contract = SimpleNamespace(lot_size=100, transaction="BUY", security_id="777", option_type="CE", strike=22700.0,
                               expiry="2030-01-01", trading_symbol="NIFTY-X")
    premium = SimpleNamespace(entry_price=50.0, stop_price=40.0, target_price=70.0)
    monkeypatch.setattr(auto_trader, "get_strategy", lambda n: object())
    monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, **k: SimpleNamespace(status=auto_trader.SETUP_TRIGGERED, setup=setup))
    monkeypatch.setattr(auto_trader, "get_underlying_info", lambda u: {"lot_size": 100})
    monkeypatch.setattr(auto_trader, "choose_contract", lambda *a, **k: contract)
    monkeypatch.setattr(auto_trader, "translate_setup", lambda *a, **k: premium)
    ranking = SimpleNamespace(is_no_trade=False, selected_strategy="Momentum", tie_break=tie_break, reason="r",
                              tie_runner_up="VWAP")
    output = SimpleNamespace(ranking=ranking, ohlcv=SimpleNamespace(tail=lambda n: None),
                             market_state=SimpleNamespace(get=lambda k, d=None: None), data_quality_status="OK")
    chain = SimpleNamespace(underlying="NIFTY")
    result = auto_trader.run_auto_option_cycle(output, "NIFTY", broker, client=None, chain=chain, account=_account())
    return broker, result


def test_a_tie_break_cycle_places_a_smaller_tagged_order(monkeypatch):
    broker, result = _cycle(monkeypatch, tie_break=True, tag_expected=TIE_BREAK_TAG)
    assert result.order is not None and result.order.status == "FILLED"
    assert result.tag == TIE_BREAK_TAG and "TIE-BREAK" in result.reason and "x0.5" in result.reason
    pos = broker.get_open_positions()[0]
    assert pos["tag"] == TIE_BREAK_TAG and pos["quantity"] == 400  # 9 lots x 0.5 -> 4 lots of 100


def test_a_normal_cycle_is_untagged_and_full_size(monkeypatch):
    broker, result = _cycle(monkeypatch, tie_break=False, tag_expected=None)
    pos = broker.get_open_positions()[0]
    assert pos["tag"] is None and pos["quantity"] == 900 and result.tag is None


def test_tie_break_off_for_paper_means_no_order(monkeypatch):
    monkeypatch.setattr(auto_trader, "SETTINGS", dataclasses.replace(settings_module.SETTINGS, tie_break_paper=False))
    monkeypatch.setattr(settings_module, "SETTINGS", dataclasses.replace(settings_module.SETTINGS, tie_break_paper=False))
    broker, result = _cycle(monkeypatch, tie_break=True, tag_expected=None)
    assert result.order is None and "turned off for paper" in result.reason and broker.get_open_positions() == []


def test_the_tag_is_persisted_on_the_order_and_position_rows(monkeypatch):
    from quant_intelligence.database.db import get_session, init_db
    from quant_intelligence.database.models import Order, Position

    init_db()
    broker, _ = _cycle(monkeypatch, tie_break=True, tag_expected=TIE_BREAK_TAG)
    pid = broker.get_open_positions()[0]["position_id"]
    with get_session() as session:
        assert session.query(Position).filter_by(position_id=pid).one().tag == TIE_BREAK_TAG
        assert session.query(Order).filter(Order.tag == TIE_BREAK_TAG).count() >= 1


# ---------------------------------------------------------------- live: tag reaches Dhan, gates unchanged
def test_a_live_tie_break_order_carries_the_tag_to_dhans_order_book(monkeypatch):
    from quant_intelligence.brokers import dhan_broker as module

    live = dataclasses.replace(settings_module.SETTINGS, trading_mode="LIVE",
                               trading_live_confirm="YES_I_UNDERSTAND_THE_RISK",
                               dhan_client_id="1000000001", dhan_access_token="tok")
    monkeypatch.setattr(module, "SETTINGS", live)
    broker = module.DhanBroker()
    broker.client = MagicMock()
    broker.client.place_order.return_value = {"orderId": "9", "orderStatus": "PENDING"}

    def order(tag):
        return OrderRequest(strategy_name="Momentum", instrument="NIFTY", direction="LONG", quantity=65, price=50.0,
                            decision_id="RISK-abc123def456", security_id="777", exchange_segment="NSE_FNO",
                            transaction_type="BUY", option_type="CE", strike=22700.0, expiry="2030-01-01", lot_size=65, tag=tag)

    from quant_intelligence.risk.risk_engine import _note_approval

    _note_approval("RISK-abc123def456")  # a live order must carry a fresh risk-engine approval (single use)
    broker.place_order(order(TIE_BREAK_TAG))
    assert broker.client.place_order.call_args.args[0]["correlationId"] == "TB-RISK-abc123def456"
    _note_approval("RISK-abc123def456")
    broker.place_order(order(None))
    assert broker.client.place_order.call_args.args[0]["correlationId"] == "RISK-abc123def456"


def test_the_live_gate_still_blocks_a_tie_break_order_when_live_is_not_authorised():
    from quant_intelligence.brokers.dhan_broker import DhanBroker

    broker = DhanBroker()
    broker.client = MagicMock()
    ack = broker.place_order(OrderRequest(strategy_name="Momentum", instrument="NIFTY", direction="LONG", quantity=65,
                                          security_id="777", tag=TIE_BREAK_TAG))
    assert ack.status == "REJECTED" and broker.client.place_order.call_count == 0


def test_tie_break_is_not_sent_live_by_default():
    """A tie-break pick has no demonstrated edge: the safe default keeps it out of live trading."""
    import importlib
    import os

    assert "TIE_BREAK_LIVE" not in os.environ or os.environ["TIE_BREAK_LIVE"].lower() in ("0", "false", "no", "")
    assert settings_module.SETTINGS.tie_break_live is False
