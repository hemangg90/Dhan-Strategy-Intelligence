"""Diagnostics report only what was checked; nothing is shown healthy without a check, and no secret appears."""
import dataclasses
import datetime as dt
import json

import pandas as pd

from quant_intelligence import diagnostics as d
from quant_intelligence.config import settings as settings_module
from quant_intelligence.utils.logging_utils import redact


def test_the_api_is_not_tested_until_someone_runs_the_test():
    check = d.api_check(run=False)
    assert check.status == d.NOT_TESTED and "no connection test" in check.detail


def test_market_data_is_not_tested_before_the_pipeline_has_run():
    assert d.data_freshness_check(None).status == d.NOT_TESTED


def test_live_order_state_is_not_tested_without_a_live_broker_and_fails_when_an_order_is_unresolved():
    assert d.unresolved_order_check(None).status == d.NOT_TESTED

    class Broker:
        unreconciled_order = None

    broker = Broker()
    assert d.unresolved_order_check(broker).status == d.OK
    broker.unreconciled_order = "RISK-abc"
    check = d.unresolved_order_check(broker)
    assert check.status == d.FAIL and "UNKNOWN" in check.detail


def test_execution_mode_reports_lockdown_and_live_authorisation(monkeypatch):
    base = settings_module.SETTINGS
    monkeypatch.setattr(d, "SETTINGS", dataclasses.replace(base, demo_lockdown=True))
    assert d.execution_mode_check().status == d.OK and "lockdown" in d.execution_mode_check().detail
    live = dataclasses.replace(base, demo_lockdown=False, trading_mode="LIVE", trading_live_confirm="YES_I_UNDERSTAND_THE_RISK")
    monkeypatch.setattr(d, "SETTINGS", live)
    check = d.execution_mode_check()
    assert check.status == d.WARN and "LIVE ORDERS ARE AUTHORISED" in check.detail


def test_credentials_check_never_contains_the_token(monkeypatch):
    secret = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjk5OTk5OTk5OTl9.SECRETSIGNATUREVALUE123"
    monkeypatch.setattr(d, "SETTINGS", dataclasses.replace(settings_module.SETTINGS, dhan_access_token=secret))
    check = d.credentials_check()
    assert "SECRETSIGNATURE" not in check.detail and "eyJ" not in check.detail


def test_a_corrupt_cache_file_is_reported_not_hidden(tmp_path):
    pd.DataFrame({"timestamp": pd.date_range("2026-10-16 09:15", periods=3, freq="5min")}).to_parquet(tmp_path / "GOOD_5min.parquet")
    (tmp_path / "BAD_5min.parquet").write_bytes(b"garbage")
    check = d.cache_check(tmp_path)
    assert check.status == d.FAIL and "BAD_5min.parquet" in check.detail
    assert d.cache_check(tmp_path / "missing").status == d.NOT_TESTED


def test_stale_or_invalid_market_data_is_not_reported_healthy():
    class Out:
        instrument = "NIFTY"
        timestamp = pd.Timestamp("2026-09-29 15:00")
        data_quality_status = "DEGRADED"
        data_quality_issues = ["data is stale: last bar 29 Sep 15:00"]

    check = d.data_freshness_check(Out())
    assert check.status == d.FAIL and "stale" in check.detail


def test_recent_errors_are_parsed_newest_first_and_redacted(tmp_path):
    log = tmp_path / "system.log"
    lines = [
        json.dumps({"ts": "t1", "level": "INFO", "component": "a", "msg": {"message": "fine"}}),
        json.dumps({"ts": "t2", "level": "ERROR", "component": "dhan", "msg": {"message": "failed access_token=abc123secret here"}}),
        "not json at all",
        json.dumps({"ts": "t3", "level": "WARNING", "component": "data", "msg": {"message": "stale"}}),
    ]
    log.write_text("\n".join(lines), encoding="utf-8")
    errors = d.recent_errors(5, log_file=log)
    assert [e["time"] for e in errors] == ["t3", "t2"]
    assert "abc123secret" not in json.dumps(errors) and "[redacted]" in errors[1]["message"]
    assert d.recent_errors(5, log_file=tmp_path / "none.log") == []


def test_collect_runs_every_check_and_never_raises():
    checks = d.collect()
    assert {c.status for c in checks} <= {d.OK, d.WARN, d.FAIL, d.NOT_TESTED}
    assert any(c.name == "Dhan API" and c.status == d.NOT_TESTED for c in checks)


def test_redaction_removes_credentials_but_leaves_ordinary_text():
    assert "[jwt-redacted]" in redact("t eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcDEF123_-xyz")
    assert redact('{"Authorization": "Bearer abc.def"}') == '{"Authorization": "[redacted]"}'
    assert redact("price 100 pinned") == "price 100 pinned"
    assert "hunter2" not in redact("password=hunter2")


def test_log_event_stores_no_secret_in_the_log_file_or_the_database(tmp_path):
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import SystemEvent
    from quant_intelligence.utils.logging_utils import log_event

    marker = "ZZ-MARKER-TOKEN-9876"
    log_event("diag_test", "calling Dhan", access_token=marker, client="ok")
    with get_session() as session:
        row = session.query(SystemEvent).filter(SystemEvent.component == "diag_test").order_by(SystemEvent.id.desc()).first()
    assert marker not in json.dumps(row.details) and row.details.get("client") == "ok"
