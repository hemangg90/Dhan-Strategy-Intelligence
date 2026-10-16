"""Failure injection for persistence: a failed write is never silent, a broken file never becomes data."""
import pandas as pd
import pytest

from quant_intelligence.brokers import paper_broker
from quant_intelligence.brokers.base_broker import OrderRequest
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.data import data_manager as dm


def _order(decision_id=None):
    return OrderRequest(strategy_name="X", instrument="NIFTY", direction="LONG", quantity=65, price=100.0,
                        option_type="CE", transaction_type="BUY", security_id="1", decision_id=decision_id)


def test_a_failed_database_write_is_logged_and_the_simulation_continues(monkeypatch):
    from quant_intelligence.database import db

    logged = []
    monkeypatch.setattr(paper_broker, "log_event", lambda component, message, **kw: logged.append((kw.get("level"), message)))
    monkeypatch.setattr(db, "get_session", lambda: (_ for _ in ()).throw(OSError("database is locked")))
    broker = PaperBroker(starting_capital=1_000_000)
    ack = broker.place_order(_order(), market_price=100.0)
    assert ack.status == "FILLED" and len(broker.get_open_positions()) == 1  # the in-memory book is intact
    errors = [m for level, m in logged if level == "ERROR"]
    assert any("Could not persist" in m for m in errors), "a lost row must be visible in the logs"


def test_the_same_approved_decision_cannot_open_two_positions():
    broker = PaperBroker(starting_capital=1_000_000)
    first = broker.place_order(_order("RISK-aaaa11112222"), market_price=100.0)
    second = broker.place_order(_order("RISK-aaaa11112222"), market_price=100.0)
    assert first.status == "FILLED" and second.status == "REJECTED" and "Duplicate" in second.reject_reason
    assert len(broker.get_open_positions()) == 1
    assert broker.place_order(_order("RISK-bbbb33334444"), market_price=100.0).status == "FILLED"  # a new decision is fine


def test_closing_twice_books_the_pnl_once():
    broker = PaperBroker(starting_capital=1_000_000)
    broker.place_order(_order(), market_price=100.0)
    pid = broker.get_open_positions()[0]["position_id"]
    first = broker.close_position(pid, 110.0, "TARGET")
    cash_after_first = broker.cash
    assert first is not None and broker.close_position(pid, 90.0, "STOP") is None
    assert broker.cash == cash_after_first  # a repeated exit event cannot move cash again


def test_cash_and_closed_pnl_stay_consistent_over_many_trades():
    broker = PaperBroker(starting_capital=1_000_000)
    exits = [110.0, 95.0, 100.0, 130.0, 70.0]
    for price in exits:
        broker.place_order(_order(), market_price=100.0)
        broker.close_position(broker.get_open_positions()[0]["position_id"], price, "T")
    closed = [p for p in broker.get_positions() if p["status"] == "CLOSED"]
    assert broker.cash == pytest.approx(1_000_000 + sum(p["net_pnl"] for p in closed))
    assert all(p["net_pnl"] == pytest.approx(p["gross_pnl"] - p["costs"]) for p in closed)


@pytest.mark.parametrize("content", [b"", b"PAR1 truncated", b"not a parquet file at all"])
def test_a_corrupt_or_empty_cache_file_is_unreadable_not_data(tmp_path, content):
    path = tmp_path / "X_5min.parquet"
    path.write_bytes(content)
    assert dm._read_cache(path) is None


def test_an_interrupted_cache_write_leaves_the_previous_file_intact_and_no_temp_files(tmp_path, monkeypatch):
    path = tmp_path / "X_5min.parquet"
    good = pd.DataFrame({"timestamp": pd.date_range("2026-10-16 09:15", periods=5, freq="5min"), "close": [1.0] * 5})
    dm._write_cache(good, path)

    def broken_to_parquet(self, target, *a, **kw):
        open(target, "wb").write(b"half a fi")  # the process "dies" mid-write
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", broken_to_parquet)
    with pytest.raises(OSError):
        dm._write_cache(good.assign(close=[9.0] * 5), path)
    monkeypatch.undo()
    assert pd.read_parquet(path)["close"].tolist() == [1.0] * 5  # the old, complete file survived
    assert [p.name for p in tmp_path.iterdir()] == ["X_5min.parquet"]  # and the temp file was cleaned up


def test_a_cache_file_is_stamped_with_what_it_holds_and_a_mismatch_is_refused(tmp_path):
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-10-16 09:15", periods=5, freq="5min"), "close": [1.0] * 5})
    nifty = tmp_path / "NIFTY_5min.parquet"
    dm._write_cache(frame, nifty)
    assert dm._read_cache(nifty) is not None and pd.read_parquet(nifty).attrs["identity"] == "NIFTY_5min"
    banknifty = tmp_path / "BANKNIFTY_5min.parquet"
    nifty.rename(banknifty)  # someone copies the wrong file into place
    assert dm._read_cache(banknifty) is None, "NIFTY candles must never be served as BANKNIFTY"
    other_timeframe = tmp_path / "NIFTY_15min.parquet"
    banknifty.rename(other_timeframe)
    assert dm._read_cache(other_timeframe) is None


def test_a_legacy_cache_file_without_a_stamp_is_still_readable(tmp_path):
    legacy = tmp_path / "TCS_5min.parquet"
    pd.DataFrame({"timestamp": pd.date_range("2026-10-16 09:15", periods=5, freq="5min"), "close": [1.0] * 5}).to_parquet(legacy, index=False)
    assert dm._read_cache(legacy) is not None  # files written before stamping existed keep working (and are stamped on next write)
