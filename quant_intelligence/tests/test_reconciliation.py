"""Position reconciliation: paper (memory vs database) and live (expected vs broker), fail-closed on unreadable sources."""
import pytest

from quant_intelligence.brokers.base_broker import OrderRequest
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.brokers.reconciliation import dhan_book, local_book, reconcile_books, reconcile_live, reconcile_paper
from quant_intelligence.database.db import get_session, init_db
from quant_intelligence.database.models import Position


def test_identical_books_agree_and_differences_are_listed_exactly():
    assert reconcile_books({"A": 75, "B": -150}, {"A": 75, "B": -150}, "T").ok
    report = reconcile_books({"A": 75, "B": -150, "C": 10}, {"A": 75, "B": -75, "D": 5}, "T")
    assert not report.ok
    assert report.missing_remotely == [{"id": "C", "local_qty": 10}] and report.unexpected_remotely == [{"id": "D", "remote_qty": 5}]
    assert report.quantity_mismatches == [{"id": "B", "local_qty": -150, "remote_qty": -75}]
    assert not report.status.ok and "mismatch" in report.reason and report.summary.startswith("T: ")


def test_flat_positions_are_ignored_and_tolerance_applies():
    assert reconcile_books({"A": 0, "B": 5}, {"B": 5, "C": 0}, "T").ok
    assert not reconcile_books({"A": 100}, {"A": 99}, "T").ok and reconcile_books({"A": 100}, {"A": 99}, "T", tolerance=1).ok


def test_the_broker_book_comes_from_security_id_and_net_quantity():
    raw = [{"securityId": "123", "netQty": 75}, {"securityId": "456", "netQty": -150}, {"securityId": "789", "netQty": 0},
           {"securityId": "123", "netQty": 25}]
    assert dhan_book(raw) == {"123": 100.0, "456": -150.0, "789": 0.0}
    assert dhan_book(None) == {}


def test_the_local_book_signs_by_transaction_and_ignores_closed_positions():
    positions = [{"security_id": "1", "transaction": "BUY", "quantity": 75, "status": "OPEN"},
                 {"security_id": "2", "transaction": "SELL", "quantity": 75, "status": "OPEN"},
                 {"security_id": "3", "transaction": "BUY", "quantity": 75, "status": "CLOSED"},
                 {"instrument": "TCS", "direction": "SHORT", "quantity": 10, "status": "OPEN"}]
    assert local_book(positions) == {"1": 75.0, "2": -75.0, "TCS": -10.0}


class FakeClient:
    def __init__(self, raw=None, error=None):
        self.raw, self.error = raw, error

    def get_positions(self):
        if self.error:
            raise self.error
        return self.raw


def test_live_reconciliation_compares_the_expected_book_with_the_brokers():
    assert reconcile_live(FakeClient([{"securityId": "1", "netQty": 75}]), {"1": 75.0}).ok
    report = reconcile_live(FakeClient([]), {"1": 75.0})
    assert not report.ok and report.missing_remotely


def test_an_unreachable_broker_fails_reconciliation_instead_of_reading_as_flat():
    report = reconcile_live(FakeClient(error=RuntimeError("timeout")), {})
    assert not report.ok and not report.status.ok and "not 'flat'" in report.reason


@pytest.fixture
def clean_positions():
    init_db()
    with get_session() as s:
        s.query(Position).delete()
    yield
    with get_session() as s:
        s.query(Position).delete()


def _open(broker, strike=24000.0):
    order = OrderRequest(strategy_name="Momentum", instrument="NIFTY", direction="LONG", quantity=75, price=100.0, stop_price=90.0,
                         target_price=120.0, decision_id=f"d{strike}", security_id=str(int(strike)), option_type="CE", strike=strike,
                         expiry="2030-01-31", lot_size=75, transaction_type="BUY")
    ack = broker.place_order(order, market_price=100.0)
    assert ack.status == "FILLED"


def test_a_paper_book_that_matches_the_database_reconciles(clean_positions):
    broker = PaperBroker(starting_capital=1_000_000)
    _open(broker, 24000.0)
    _open(broker, 24100.0)
    assert reconcile_paper(broker).ok


def test_a_position_that_never_reached_the_database_is_detected(clean_positions):
    broker = PaperBroker(starting_capital=1_000_000)
    _open(broker)
    with get_session() as s:
        s.query(Position).delete()  # the save failed (or was lost): memory and database now disagree
    report = reconcile_paper(broker)
    assert not report.ok and len(report.missing_remotely) == 1


def test_a_quantity_changed_in_memory_only_is_detected(clean_positions):
    broker = PaperBroker(starting_capital=1_000_000)
    _open(broker)
    next(iter(broker.positions.values()))["quantity"] = 150
    report = reconcile_paper(broker)
    assert not report.ok and report.quantity_mismatches


def test_a_database_row_the_book_does_not_know_is_detected(clean_positions):
    broker = PaperBroker(starting_capital=1_000_000)
    _open(broker)
    broker.positions.clear()
    report = reconcile_paper(broker)
    assert not report.ok and len(report.unexpected_remotely) == 1


def test_an_existing_database_gets_the_entry_greeks_column_without_losing_rows(tmp_path):
    """Older databases have a positions table without `entry_greeks`; start-up must add it and keep the saved rows."""
    from sqlalchemy import create_engine, inspect, text

    engine = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE positions (id INTEGER PRIMARY KEY, position_id VARCHAR(64) UNIQUE NOT NULL, instrument VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO positions (position_id, instrument) VALUES ('P1', 'NIFTY')"))
    init_db(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("positions")}
    assert "entry_greeks" in cols
    with engine.connect() as conn:
        assert conn.execute(text("SELECT position_id, entry_greeks FROM positions")).fetchall() == [("P1", None)]
