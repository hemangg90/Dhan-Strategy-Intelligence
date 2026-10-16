"""History tools: CSV import (restore) and the SQLite -> durable database migration script."""
import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quant_intelligence.database import db as db_module
from quant_intelligence.database.models import DailyReport, Fill, Order, Position, RiskEvent
from quant_intelligence.reports import history_io
from quant_intelligence.ui.position_views import positions_frame


def _position(pid, **kw):
    base = dict(position_id=pid, instrument="NIFTY", underlying="NIFTY", strategy_name="ORB", direction="LONG", quantity=75,
                entry_price=120.0, stop_price=96.0, target_price=168.0, status="CLOSED", opened_at=dt.datetime(2026, 10, 5, 10, 0),
                closed_at=dt.datetime(2026, 10, 5, 10, 40), exit_price=150.0, net_pnl=2250.0, exit_reason="TARGET", mode="PAPER",
                security_id="777", option_type="CE", strike=22500.0, expiry="2026-10-27", transaction="BUY", tag="TIE-BREAK",
                order_id=f"O-{pid}", decision_id=f"D-{pid}", expected_r=0.4, confidence="HIGH", regime="RANGE")
    base.update(kw)
    return Position(**base)


# ---------------------------------------------------------------- CSV import (restore)
def _download_csv(positions: list[Position]) -> pd.DataFrame:
    """What the Trade history tab downloads: the positions frame without `held`."""
    rows = [{c.name: getattr(p, c.name) for c in Position.__table__.columns} for p in positions]
    return positions_frame(rows).drop(columns=["held"])


def test_a_download_round_trips_through_import_and_stays_idempotent():
    from quant_intelligence.database.db import get_session, init_db

    init_db()
    df = _download_csv([_position("IMP-1"), _position("IMP-2", strategy_name="VWAP", exit_price=100.0, net_pnl=-1500.0, exit_reason="STOP")])
    csv_roundtrip = pd.read_csv(pd.io.common.StringIO(df.to_csv(index=False)))  # through a real CSV file
    first = history_io.import_positions(csv_roundtrip)
    assert first == {"inserted": 2, "skipped_existing": 0, "invalid": []}
    again = history_io.import_positions(csv_roundtrip)  # the same file twice never duplicates a trade
    assert again["inserted"] == 0 and again["skipped_existing"] == 2

    with get_session() as s:
        row = s.query(Position).filter_by(position_id="IMP-2").one()
        assert (row.strategy_name, row.status, row.exit_reason, row.option_type, row.transaction) == ("VWAP", "CLOSED", "STOP", "CE", "BUY")
        assert row.quantity == 75 and row.entry_price == 120.0 and row.mode == "PAPER" and row.direction == "LONG"
        assert row.net_pnl == pytest.approx(-1500.0)  # the broker's booked (gross) P&L, not the after-charges figure
        assert row.opened_at == dt.datetime(2026, 10, 5, 10, 0) and row.expected_r == 0.4 and row.regime == "RANGE"


def test_import_rejects_bad_rows_and_files_without_hurting_good_ones():
    df = _download_csv([_position("IMP-3")])
    bad = pd.concat([df, df.assign(position_id="IMP-4", status="WEIRD"), df.assign(position_id=None)], ignore_index=True)
    result = history_io.import_positions(bad)
    assert result["inserted"] == 1 and len(result["invalid"]) == 2
    assert any("unknown status" in why for _, why in result["invalid"]) and any("required" in why for _, why in result["invalid"])

    wrong = history_io.import_positions(pd.DataFrame({"foo": [1], "bar": [2]}))
    assert wrong["inserted"] == 0 and "missing column" in wrong["invalid"][0][1]


def test_an_open_position_in_a_file_is_restored_as_stale_never_as_live():
    from quant_intelligence.database.db import get_session

    row = _position("IMP-5", status="OPEN", exit_price=None, closed_at=None, net_pnl=None, exit_reason=None)
    history_io.import_positions(_download_csv([row]))
    with get_session() as s:
        got = s.query(Position).filter_by(position_id="IMP-5").one()
        assert got.status == "STALE" and got.exit_reason == "STALE_ON_RESTART"


def test_a_put_is_restored_as_a_short_underlying_view():
    from quant_intelligence.database.db import get_session

    history_io.import_positions(_download_csv([_position("IMP-6", option_type="PE", direction="SHORT")]))
    with get_session() as s:
        assert s.query(Position).filter_by(position_id="IMP-6").one().direction == "SHORT"


# ---------------------------------------------------------------- migration script
def _script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "migrate_db.py"
    spec = importlib.util.spec_from_file_location("migrate_db", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def engines(tmp_path):
    source_url = "sqlite:///" + (tmp_path / "source.db").as_posix()
    src = create_engine(source_url)
    db_module.init_db(src)
    with sessionmaker(bind=src)() as s:
        s.add(_position("REAL-1"))
        s.add(_position("REAL-2", strategy_name="VWAP"))
        s.add(_position("FAKE-1", strategy_name="Momentum", security_id=None, entry_price=100.05, underlying=None, option_type=None))
        s.add(Order(order_id="OR-1", timestamp=dt.datetime(2026, 10, 5, 10), instrument="NIFTY", strategy_name="ORB", status="FILLED", price=120.0, security_id="777", mode="PAPER"))
        s.add(Order(order_id="OR-FAKE", timestamp=dt.datetime(2026, 10, 5, 10), instrument="NIFTY", strategy_name="Momentum", status="FILLED", price=100.0, security_id=None, mode="PAPER"))
        s.add(Fill(order_id="OR-1", timestamp=dt.datetime(2026, 10, 5, 10), fill_price=120.05, quantity=75, slippage=0.05))
        s.add(Fill(order_id="OR-FAKE", timestamp=dt.datetime(2026, 10, 5, 10), fill_price=100.0, quantity=75, slippage=0.0))
        s.add(RiskEvent(decision_id="RISK-1", timestamp=dt.datetime(2026, 10, 5, 4), event_type="VETO", reason="x"))
        s.add(DailyReport(report_date=dt.date(2026, 10, 5), scope="ALL", content_json={"a": 1}, content_markdown="# r"))
        s.commit()
    target = create_engine("sqlite:///" + (tmp_path / "target.db").as_posix())
    return source_url, target


def _count(engine, model):
    with sessionmaker(bind=engine)() as s:
        return s.query(model).count()


def test_dry_run_reports_what_would_be_copied_and_changes_nothing(engines):
    source_url, target = engines
    rep = _script().migrate(source_url, target, apply=False)
    assert rep["positions"] == {"found": 2, "new": 2} and rep["orders"] == {"found": 1, "new": 1}  # fixtures excluded
    assert rep["fills"]["new"] == 1 and rep["risk_events"]["new"] == 1 and rep["daily_reports"]["new"] == 1
    from sqlalchemy import inspect

    assert inspect(target).get_table_names() == []  # a dry run does not even create tables


def test_apply_copies_history_skips_fixtures_and_is_idempotent(engines):
    source_url, target = engines
    mod = _script()
    first = mod.migrate(source_url, target, apply=True)
    assert first["positions"]["new"] == 2
    assert (_count(target, Position), _count(target, Order), _count(target, Fill), _count(target, RiskEvent), _count(target, DailyReport)) == (2, 1, 1, 1, 1)
    with sessionmaker(bind=target)() as s:
        got = s.query(Position).filter_by(position_id="REAL-1").one()
        assert got.expected_r == 0.4 and got.tag == "TIE-BREAK" and got.order_id == "O-REAL-1"
        assert s.query(Position).filter_by(position_id="FAKE-1").count() == 0
        assert s.query(DailyReport).one().content_json == {"a": 1}
    again = mod.migrate(source_url, target, apply=True)
    assert all(v["new"] == 0 for k, v in again.items() if k != "fills")  # nothing new the second time
    assert _count(target, Position) == 2 and _count(target, Fill) == 1  # and nothing duplicated


def test_migration_upgrades_an_old_target_that_lacks_the_new_columns(engines, tmp_path):
    source_url, _ = engines
    old = create_engine("sqlite:///" + (tmp_path / "old.db").as_posix())
    with old.begin() as conn:  # a database from before expected_r/regime/order_id existed
        conn.exec_driver_sql("CREATE TABLE positions (id INTEGER PRIMARY KEY, position_id VARCHAR(64) UNIQUE NOT NULL, instrument VARCHAR(32) NOT NULL, strategy_name VARCHAR(64))")
    _script().migrate(source_url, old, apply=True)
    with sessionmaker(bind=old)() as s:
        assert s.query(Position).filter_by(position_id="REAL-1").one().regime == "RANGE"


def test_the_script_refuses_to_copy_a_file_onto_itself(monkeypatch, capsys, tmp_path):
    mod = _script()
    from quant_intelligence.database import db as live_db

    src = Path(live_db.get_engine().url.database)
    monkeypatch.setattr(sys, "argv", ["migrate_db", "--source", str(src)])
    if not src.exists():
        pytest.skip("test database file is created lazily")
    assert mod.main() == 1 and "source file itself" in capsys.readouterr().out


# ---------------------------------------------------------------- the history filter must not crash on any data
def test_default_statuses_only_offers_what_exists():
    from quant_intelligence.ui.position_views import default_statuses

    assert default_statuses(["CLOSED"]) == ["CLOSED"]  # the crash: STALE was pre-selected but not an option
    assert default_statuses(["CLOSED", "OPEN", "STALE"]) == ["CLOSED", "STALE"]
    assert default_statuses(["OPEN"]) == [] and default_statuses([]) == []


def test_the_history_tab_renders_when_every_trade_is_closed():
    from streamlit.testing.v1 import AppTest

    from quant_intelligence.database.db import get_session, init_db

    init_db()
    with get_session() as s:
        s.add(_position("HISTPAGE-1"))  # a CLOSED trade; there is no STALE row in this database state
    page = Path(__file__).resolve().parents[1] / "pages" / "08_Positions_and_Orders.py"
    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_a_current_convention_row_round_trips_with_its_booked_costs():
    """Rows closed by the current paper broker (gross, costs, net) restore exactly, not as re-estimated legacy rows."""
    from quant_intelligence.database.db import get_session, init_db

    init_db()
    booked = _position("IMP-BOOKED", gross_pnl=-1500.0, costs=61.5, net_pnl=-1561.5, costs_model="OPTION_PREMIUM", exit_price=100.0)
    frame = _download_csv([booked])
    assert bool(frame["charges_booked"].iloc[0]) and frame["charges"].iloc[0] == pytest.approx(61.5)
    csv_roundtrip = pd.read_csv(pd.io.common.StringIO(frame.to_csv(index=False)))
    assert history_io.import_positions(csv_roundtrip)["inserted"] == 1
    with get_session() as s:
        row = s.query(Position).filter_by(position_id="IMP-BOOKED").one()
    assert (row.gross_pnl, row.costs, row.net_pnl) == pytest.approx((-1500.0, 61.5, -1561.5))
    again = positions_frame([row]).iloc[0]  # and the page shows the same figures it showed before the export
    assert (again["gross_pnl"], again["charges"], again["net_pnl"]) == pytest.approx((-1500.0, 61.5, -1561.5))
