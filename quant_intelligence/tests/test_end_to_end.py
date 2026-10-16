"""End-to-end workflow, deterministic and unconditional.

data -> quality gate -> features -> regime -> analogues -> strategy evaluation -> ranking -> setup -> independent risk
evaluation -> paper execution -> persistence -> restart recovery -> reporting.

Real components throughout (real quality gate, features, ranking, risk engine, paper broker, database). Two things are
controlled: the candles are seeded synthetic data (production has no synthetic fallback), and the moment a setup
triggers is forced so the execution stages always run. No broker or network is involved.
"""
import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import Order, Position, RiskEvent
from quant_intelligence.execution import auto_trader
from quant_intelligence.execution.engine import TradingEngine
from quant_intelligence.execution.position_monitor import monitor_positions
from quant_intelligence.ranking.ranking_engine import RankingDecision, rank_and_select
from quant_intelligence.reports import performance
from quant_intelligence.research.pipeline import run_pipeline
from quant_intelligence.risk.risk_engine import AccountState
from quant_intelligence.strategies.base_strategy import Setup
from quant_intelligence.ui.position_views import positions_frame, trade_stats
from quant_intelligence.utils.market_calendar import most_recent_expected_bar_time
from quant_intelligence.utils.market_profile import NSE
from quant_intelligence.utils.timeutil import now_ist


@pytest.fixture(scope="module")
def fresh_output(tmp_path_factory):
    """A real pipeline run over FRESH seeded candles (ending at the latest expected bar, so the freshness gate passes)."""
    from quant_intelligence.data import data_manager as dm_module
    from quant_intelligence.data.data_manager import DataManager
    from quant_intelligence.data_adapters.synthetic import SyntheticAdapter

    mp = pytest.MonkeyPatch()
    mp.setattr(dm_module, "CACHE_DIR", tmp_path_factory.mktemp("cache"))
    mp.setattr(DataManager, "_fetch", lambda self, i, tf, s, e, prefer: ("test", SyntheticAdapter(seed=11).get_ohlcv(i, tf, s, e)))
    try:
        end = most_recent_expected_bar_time(now_ist(), NSE).replace(tzinfo=None) + dt.timedelta(minutes=5)
        yield run_pipeline("NIFTY", "5min", end - dt.timedelta(days=45), end)
    finally:
        mp.undo()


def _account(**kw):
    base = dict(equity=1_000_000.0, peak_equity=1_000_000.0, daily_pnl=0.0, open_positions_count=0, trades_today=0)
    base.update(kw)
    return AccountState(**base)


def _force_trade(monkeypatch, output, setup, bar_age_minutes=1.0):
    """Select Momentum and make it trigger `setup`; everything downstream is real. The bar is given a controlled age so the
    order-time freshness check does not depend on when the test happens to run (it vetoes bars older than 15 minutes)."""
    output.timestamp = pd.Timestamp(now_ist().replace(tzinfo=None)) - pd.Timedelta(minutes=bar_age_minutes)
    output.ranking = RankingDecision(ranked=[], selected_strategy="Momentum", is_no_trade=False, reason="forced for the test")
    monkeypatch.setattr(auto_trader, "get_strategy", lambda name: object())
    monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, **k: SimpleNamespace(status=auto_trader.SETUP_TRIGGERED, setup=setup))


def _price(output):
    return float(output.ohlcv["close"].iloc[-1])


def test_the_research_chain_runs_on_fresh_data_and_is_internally_consistent(fresh_output):
    out = fresh_output
    assert out.data_quality_status in ("OK", "DEGRADED")  # a verdict was reached, by the real gate
    assert out.market_state is not None and out.regime_label
    assert 0.0 <= out.regime_confidence <= 1.0 and abs(sum(out.regime_probabilities.values()) - 1.0) < 1e-9
    assert len(out.strategy_intel) >= 10 and all(si.score is not None for si in out.strategy_intel)
    ranking = out.ranking
    assert (ranking.selected_strategy is None) == ranking.is_no_trade
    eligible = [s for s in ranking.ranked if s.eligible]
    if out.data_quality_status != "OK":
        assert ranking.is_no_trade  # bad data can never produce a trade
    if not eligible:
        assert ranking.is_no_trade  # nothing eligible, nothing selected


def test_a_full_trade_lifecycle_is_executed_persisted_closed_recovered_and_reported(monkeypatch, fresh_output):
    out = fresh_output
    price = _price(out)
    setup = Setup(out.timestamp, "LONG", price, price - 20.0, price + 40.0)
    _force_trade(monkeypatch, out, setup)
    out.data_quality_status = "OK"
    broker, account = PaperBroker(starting_capital=1_000_000), _account()

    result = auto_trader.run_auto_equity_cycle(out, broker, account)

    # ---- risk approved, order placed with the approval attached
    assert result.order is not None and result.order.status == "FILLED", result.reason
    assert result.risk_decision_id and result.risk_decision_id.startswith("RISK-")
    pos = broker.get_open_positions()[0]
    assert pos["decision_id"] == result.risk_decision_id and pos["quantity"] > 0
    assert pos["quantity"] * abs(setup.entry_price - setup.stop_price) <= account.equity * 0.01  # inside the risk budget

    # ---- persisted: order, position and the approving risk event all exist and agree
    with get_session() as s:
        order_row = s.query(Order).filter_by(decision_id=result.risk_decision_id).one()
        risk_row = s.query(RiskEvent).filter_by(decision_id=result.risk_decision_id).one()
        pos_row = s.query(Position).filter_by(position_id=pos["position_id"]).one()
    assert risk_row.event_type == "APPROVED" and order_row.status == "FILLED" and pos_row.status == "OPEN"
    assert (order_row.instrument, order_row.quantity) == (pos_row.instrument, pos_row.quantity)

    # ---- a bar that reaches the target closes it with costs booked
    bar = pd.Series({"open": price + 5, "high": price + 41.0, "low": price + 4, "close": price + 38})
    events = monitor_positions(broker, bar)
    assert [e["type"] for e in events] == ["TARGET_HIT"]
    closed = events[0]["position"]
    assert closed["gross_pnl"] > 0 and closed["costs"] > 0
    assert closed["net_pnl"] == pytest.approx(closed["gross_pnl"] - closed["costs"])
    assert broker.cash == pytest.approx(1_000_000 + closed["net_pnl"])  # cash accounting agrees with the trade record

    # ---- the stored row has the same numbers (nothing lost or doubled on the way to the database)
    with get_session() as s:
        stored = s.query(Position).filter_by(position_id=pos["position_id"]).one()
    assert (stored.gross_pnl, stored.costs, stored.net_pnl) == pytest.approx((closed["gross_pnl"], closed["costs"], closed["net_pnl"]))
    assert stored.status == "CLOSED" and stored.exit_reason == "TARGET"

    # ---- restart: a fresh engine rebuilds the book from the database with identical figures
    engine = TradingEngine()
    engine.restore_from_db()
    restored = engine.broker.positions[pos["position_id"]]
    assert restored["status"] == "CLOSED" and restored["net_pnl"] == pytest.approx(closed["net_pnl"])
    assert restored["costs"] == pytest.approx(closed["costs"])

    # ---- reporting reads the same booked numbers: no second deduction of charges
    frame = positions_frame([stored])
    row = frame.iloc[0]
    assert row["gross_pnl"] == pytest.approx(closed["gross_pnl"]) and row["charges"] == pytest.approx(closed["costs"])
    assert row["net_pnl"] == pytest.approx(closed["net_pnl"])
    stats = trade_stats(frame)
    assert stats["trades"] == 1 and stats["net"] == pytest.approx(closed["net_pnl"]) and stats["wins"] == 1
    assert len(performance.closed_only(frame)) == 1


def test_a_risk_veto_stops_the_trade_before_any_order_exists(monkeypatch, fresh_output):
    out = fresh_output
    price = _price(out)
    bad = Setup(out.timestamp, "LONG", price, price + 10.0, price + 40.0)  # a LONG stop ABOVE the entry
    _force_trade(monkeypatch, out, bad)
    out.data_quality_status = "OK"
    broker = PaperBroker(starting_capital=1_000_000)
    with get_session() as s:
        orders_before = s.query(Order).count()

    result = auto_trader.run_auto_equity_cycle(out, broker, _account())

    assert result.order is None and "RISK ENGINE VETO" in result.reason
    assert broker.get_positions() == []
    with get_session() as s:
        assert s.query(Order).count() == orders_before  # no order was created, let alone filled
        veto = s.query(RiskEvent).filter_by(decision_id=result.risk_decision_id).one()
    assert veto.event_type == "VETO" and "stop" in veto.reason.lower()


@pytest.mark.parametrize("account_kw,fragment", [
    (dict(kill_switch_engaged=True), "kill switch"),
    (dict(daily_pnl=-60_000.0), "daily loss"),
    (dict(equity=float("nan")), "not a valid number"),  # refused at sizing, before the risk engine is even asked
    (dict(broker_connected=False), "disconnected"),
])
def test_mandatory_risk_controls_block_a_good_setup(monkeypatch, fresh_output, account_kw, fragment):
    out = fresh_output
    price = _price(out)
    _force_trade(monkeypatch, out, Setup(out.timestamp, "LONG", price, price - 20.0, price + 40.0))
    out.data_quality_status = "OK"
    broker = PaperBroker(starting_capital=1_000_000)
    result = auto_trader.run_auto_equity_cycle(out, broker, _account(**account_kw))
    assert result.order is None and fragment in result.reason.lower() and broker.get_positions() == []


def test_a_high_ranking_cannot_override_bad_data(fresh_output):
    from quant_intelligence.ranking.ranking_engine import score_strategy

    metrics = {"strategy_name": "A", "confidence_label": "HIGH", "sample_size": 500, "expected_r": 2.0,
               "prob_positive_return": 0.95, "expected_r_stderr": 0.05, "global_expected_r": 1.0}
    for status in ("DEGRADED", "FAIL"):
        decision = rank_and_select([score_strategy(metrics, -1.0)], data_quality_status=status)
        assert decision.is_no_trade and decision.selected_strategy is None


def test_stale_data_stops_the_cycle_before_a_setup_is_even_looked_for(monkeypatch, fresh_output):
    out = fresh_output
    called = []
    monkeypatch.setattr(auto_trader, "detect_setup", lambda *a, **k: called.append(1))
    out.ranking = RankingDecision(ranked=[], selected_strategy=None, is_no_trade=True, reason="Data quality is DEGRADED: NO TRADE forced")
    result = auto_trader.run_auto_equity_cycle(out, PaperBroker(), _account())
    assert result.order is None and result.reason.startswith("NO TRADE") and called == []


def test_an_old_bar_is_vetoed_at_order_time_even_though_the_ranker_selected_it(monkeypatch, fresh_output):
    """The risk engine's freshness check is independent of the ranker and of the research-time quality gate."""
    out = fresh_output
    price = _price(out)
    _force_trade(monkeypatch, out, Setup(out.timestamp, "LONG", price, price - 20.0, price + 40.0), bar_age_minutes=40.0)
    out.data_quality_status = "OK"
    broker = PaperBroker(starting_capital=1_000_000)
    result = auto_trader.run_auto_equity_cycle(out, broker, _account())
    assert result.order is None and "stale" in result.reason.lower() and broker.get_positions() == []
