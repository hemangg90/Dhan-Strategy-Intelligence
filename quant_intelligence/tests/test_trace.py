"""The decision -> risk -> order -> fill -> position -> outcome trace, and the rejected-decision reader."""
import datetime as dt
import uuid

import pytest

from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import Fill, Order, Position, RiskEvent, ScanDecision
from quant_intelligence.reports import trace as tr
from quant_intelligence.ui.trace_view import timeline_frame, trace_summary_frame

T0 = dt.datetime(2026, 10, 16, 10, 0)  # naive IST


def _ids():
    u = uuid.uuid4().hex[:8]
    return f"RISK-{u}", f"PAPER-{u}", f"POS-{u}", f"NIFTY{u}"


def _seed(*, complete=True, legacy=False, closed=True, instrument=None):
    decision_id, order_id, position_id, inst = _ids()
    inst = instrument or inst
    with get_session() as s:
        if complete:
            s.add(ScanDecision(decided_at=T0 + dt.timedelta(minutes=5), bar_ts=T0, instrument=inst, strategy="ORB", status="FILLED",
                               reason_class="FILLED", reason="filled", expected_r=0.25, confidence="MEDIUM",
                               decision_id=decision_id, order_id=order_id))
            # risk events are stored in UTC: 10:05 IST is 04:35 UTC
            s.add(RiskEvent(timestamp=dt.datetime(2026, 10, 16, 4, 35), decision_id=decision_id, event_type="APPROVED",
                            reason="All risk checks passed", details={"daily_loss_within_limit": True, "risk_multiplier": 1.0}))
            s.add(Order(order_id=order_id, decision_id=decision_id, timestamp=T0 + dt.timedelta(minutes=5), instrument=inst,
                        strategy_name="ORB", direction="LONG", quantity=50, order_type="MARKET", price=100.0, status="FILLED", mode="PAPER"))
            s.add(Fill(order_id=order_id, timestamp=T0 + dt.timedelta(minutes=5), fill_price=100.05, quantity=50, slippage=0.05))
        s.add(Position(
            position_id=position_id, instrument=inst, strategy_name="ORB", direction="LONG", quantity=50, entry_price=100.05,
            stop_price=99.05, target_price=102.05, status="CLOSED" if closed else "OPEN", opened_at=T0 + dt.timedelta(minutes=5),
            closed_at=T0 + dt.timedelta(minutes=45) if closed else None, exit_price=102.05 if closed else None,
            net_pnl=(1500.0 if legacy else 1400.0) if closed else None, gross_pnl=None if legacy else 1500.0,
            costs=None if legacy else 100.0, exit_reason="TARGET" if closed else None, mode="PAPER",
            order_id=order_id if complete else None, decision_id=decision_id, expected_r=0.25, confidence="MEDIUM", regime="RANGE"))
    return decision_id, order_id, position_id, inst


def test_full_chain_is_joined_in_one_clock_with_expected_vs_realised_r():
    decision_id, order_id, _, inst = _seed()
    with get_session() as s:
        [t] = tr.trade_trace(s, instrument=inst)
    assert t["decision_id"] == decision_id and t["order_id"] == order_id and t["missing_links"] == []
    # net 1400 on a risk of |100.05 - 99.05| * 50 = 50  ->  28 R
    assert t["realised_r"] == pytest.approx(28.0) and t["r_basis"] == "net"
    assert t["r_difference"] == pytest.approx(28.0 - 0.25) and t["label"] == "SIMULATED"
    times = [e["time"] for e in t["timeline"]]
    assert times == sorted(times)  # one chronological clock
    stages = [e["stage"] for e in t["timeline"]]
    assert stages[0] == "Signal bar" and stages[-1].startswith("Exit: TARGET")
    risk_event = next(e for e in t["timeline"] if e["stage"].startswith("Risk engine"))
    assert risk_event["time"] == dt.datetime(2026, 10, 16, 10, 5)  # UTC converted to IST
    assert "Fill assumption" in stages and "Order (simulated)" in stages


def test_legacy_row_is_reported_as_gross_and_never_called_net():
    *_, inst = _seed(legacy=True)
    with get_session() as s:
        [t] = tr.trade_trace(s, instrument=inst)
    assert t["r_basis"].startswith("gross") and t["net_pnl"] is None and t["gross_pnl"] == 1500.0


def test_missing_links_are_listed_not_hidden():
    *_, inst = _seed(complete=False)
    with get_session() as s:
        [t] = tr.trade_trace(s, instrument=inst)
    assert set(t["missing_links"]) == {"scan decision", "risk approval", "order", "fill"}


def test_open_position_has_no_realised_r():
    *_, inst = _seed(closed=False)
    with get_session() as s:
        [t] = tr.trade_trace(s, instrument=inst)
    assert t["realised_r"] is None and t["r_difference"] is None and t["status"] == "OPEN"


def test_orphans_found_for_approval_without_position_and_position_without_approval():
    decision_id = f"RISK-{uuid.uuid4().hex[:8]}"
    with get_session() as s:
        s.add(RiskEvent(decision_id=decision_id, event_type="APPROVED", reason="ok", details={}))
    *_, inst = _seed(complete=False)
    with get_session() as s:
        found = tr.orphans(s)
    assert any(o["decision_id"] == decision_id for o in found["approved_without_position"])
    assert any(o["instrument"] == inst for o in found["position_without_approval"])


def test_a_complete_trade_is_not_an_orphan():
    decision_id, order_id, position_id, _ = _seed()
    with get_session() as s:
        found = tr.orphans(s)
    assert all(o["decision_id"] != decision_id for o in found["approved_without_position"])
    assert all(o["position_id"] != position_id for o in found["position_without_approval"])
    assert all(o["order_id"] != order_id for o in found["filled_order_without_position"])


def test_rejected_decisions_exclude_fills_and_filter_by_reason():
    inst = f"X{uuid.uuid4().hex[:6]}"
    with get_session() as s:
        for i, (status, code) in enumerate([("NO_TRADE", "TIED"), ("VETOED", "DAILY_LOSS"), ("FILLED", "FILLED"), ("WAITING", "WAITING_FOR_SETUP")]):
            s.add(ScanDecision(decided_at=T0 + dt.timedelta(minutes=i), bar_ts=T0 + dt.timedelta(minutes=5 * i), instrument=inst,
                               status=status, reason_class=code, reason=code.lower()))
    with get_session() as s:
        rows = [r for r in tr.rejected_decisions(s, limit=1000) if r["instrument"] == inst]
        only = [r for r in tr.rejected_decisions(s, limit=1000, reason_class="DAILY_LOSS") if r["instrument"] == inst]
    assert {r["status"] for r in rows} == {"NO_TRADE", "VETOED", "WAITING"}
    assert [r["status"] for r in only] == ["VETOED"]


def test_view_frames_have_the_expected_columns():
    *_, inst = _seed()
    with get_session() as s:
        traces = tr.trade_trace(s, instrument=inst)
    assert {"Expected R", "Realised R", "R basis", "Missing links", "Type"} <= set(trace_summary_frame(traces).columns)
    assert list(timeline_frame(traces[0]["timeline"]).columns) == ["Time (IST)", "Step", "What happened"]
