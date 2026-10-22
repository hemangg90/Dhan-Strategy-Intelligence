"""Permanent history and the daily report: Postgres-ready database layer, analytics maths, the decision log, the
report builder and its storage, the EOD job, and an end-to-end paper trade through to a stored report."""
import dataclasses
import datetime as dt
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_intelligence.config import settings as settings_module
from quant_intelligence.database import db as db_module
from quant_intelligence.reports import decision_log, eod, eod_job
from quant_intelligence.reports import performance as perf
from quant_intelligence.ui.position_views import positions_frame


# ---------------------------------------------------------------- 1. database layer
def test_postgres_urls_are_normalised_for_sqlalchemy():
    n = db_module.normalise_database_url
    assert n("postgres://u:p@host/db?sslmode=require") == "postgresql+psycopg2://u:p@host/db?sslmode=require"
    assert n("postgresql://u:p@host/db") == "postgresql+psycopg2://u:p@host/db"
    assert n("postgresql+psycopg2://u:p@host/db") == "postgresql+psycopg2://u:p@host/db"
    assert n("  sqlite:///x.db ") == "sqlite:///x.db" and n("") == ""


def test_engine_options_fit_the_backend():
    assert db_module.engine_options("sqlite:///x.db") == {"connect_args": {"check_same_thread": False}}
    pg = db_module.engine_options("postgresql+psycopg2://u:p@h/db")
    assert pg["pool_pre_ping"] is True and pg["pool_recycle"] <= 300 and "connect_args" not in pg


def test_backend_and_durability_never_expose_credentials():
    assert db_module.backend_name("postgresql+psycopg2://user:SECRET@h/db") == "postgresql"
    assert db_module.backend_name("sqlite:///x.db") == "sqlite"
    assert db_module.is_durable("postgresql+psycopg2://u:p@h/db") and not db_module.is_durable("sqlite:///x.db")
    assert "SECRET" not in db_module.backend_name("postgresql://user:SECRET@h/db")


def test_every_table_compiles_for_postgres_and_a_new_column_migrates():
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    from quant_intelligence.database.models import Base

    dialect = postgresql.dialect()
    ddl = {t.name: str(CreateTable(t).compile(dialect=dialect)) for t in Base.metadata.sorted_tables}
    assert {"positions", "orders", "fills", "scan_decisions", "daily_reports", "vol_forecasts"} <= set(ddl)
    assert "expected_r FLOAT" in ddl["positions"] or "expected_r DOUBLE PRECISION" in ddl["positions"]
    assert "JSON" in ddl["daily_reports"] and "UNIQUE (report_date, scope)" in ddl["daily_reports"]
    col = Base.metadata.tables["positions"].c.regime
    assert col.type.compile(dialect=dialect) == "VARCHAR(32)"  # what _sync_schema would ALTER TABLE ... ADD COLUMN


# ---------------------------------------------------------------- helpers: trades
def _row(pid, strategy="Momentum", entry=100.0, exit_=110.0, stop=90.0, qty=100, day="2026-10-05", hour=10, status="CLOSED",
         tag=None, expected_r=None, instrument="NIFTY", option_type="CE"):
    opened = pd.Timestamp(f"{day} {hour:02d}:00:00").to_pydatetime()
    return {
        "position_id": pid, "instrument": instrument, "underlying": instrument, "strategy_name": strategy, "direction": "LONG",
        "quantity": qty, "entry_price": entry, "stop_price": stop, "target_price": entry + 20, "status": status,
        "opened_at": opened, "closed_at": (opened + dt.timedelta(minutes=30)) if status == "CLOSED" else None,
        "exit_price": exit_ if status == "CLOSED" else None,
        "net_pnl": (exit_ - entry) * qty if status == "CLOSED" else None, "exit_reason": "TARGET" if exit_ > entry else "STOP",
        "security_id": "111", "option_type": option_type, "strike": 22500.0, "expiry": "2026-10-27", "transaction": "BUY",
        "tag": tag, "expected_r": expected_r,
    }


def _frame(rows):
    return perf.with_r(positions_frame(rows))


def _synthetic(strategy, rs, start="2026-09-01"):
    """A closed-trade frame with exact R multiples `rs` (for verdict maths), one trade per day."""
    n = len(rs)
    days = pd.bdate_range(start, periods=n)
    net = np.array(rs, float) * 1000.0
    return pd.DataFrame({
        "position_id": [f"S{i}" for i in range(n)], "strategy": strategy, "instrument": "NIFTY", "status": "CLOSED",
        "net_pnl": net, "gross_pnl": net + 10, "charges": 10.0, "r_multiple": np.array(rs, float),
        "closed_at": days + pd.Timedelta(hours=11), "opened_at": days + pd.Timedelta(hours=10),
        "held": pd.Timedelta(minutes=30), "date": [d.date() for d in days], "tag": None, "expected_r": np.nan,
        "invested": 5000.0, "market": "NSE", "qty": 100, "entry": 50.0, "stop": 40.0, "option_type": "CE",
        "exit_reason": "TARGET", "hour": 10, "weekday": "Monday", "tag_label": "Normal",
    })


# ---------------------------------------------------------------- 3. analytics
def test_wilson_interval_matches_the_textbook_value():
    lo, hi = perf.wilson_interval(8, 10)
    assert lo == pytest.approx(49.0, abs=0.3) and hi == pytest.approx(94.3, abs=0.3)
    assert all(math.isnan(x) for x in perf.wilson_interval(0, 0))
    lo0, hi0 = perf.wilson_interval(0, 5)
    assert lo0 == 0.0 and 40 < hi0 < 50  # never a degenerate 0%-0% interval


def test_r_multiple_is_net_pnl_over_premium_risked():
    df = _frame([_row("A", entry=100, exit_=110, stop=90, qty=100), _row("B", entry=100, exit_=90, stop=90, qty=100)])
    a, b = df.iloc[0], df.iloc[1]
    assert a["r_multiple"] == pytest.approx(a["net_pnl"] / (10 * 100))
    assert a["r_multiple"] < 1.0 and b["r_multiple"] < -1.0  # charges make a full stop-out slightly worse than -1R
    assert a["hour"] == 10 and a["date"] == dt.date(2026, 10, 5) and a["tag_label"] == "Normal"
    sell = _frame([dict(_row("C"), transaction="SELL", stop_price=110.0)])
    assert sell.iloc[0]["r_multiple"] == pytest.approx(sell.iloc[0]["net_pnl"] / (10 * 100))  # risk is |entry - stop|


@pytest.mark.parametrize("n,avg,t,code", [
    (29, 0.9, 5.0, "TOO_FEW"),  # a great average on 29 trades is still too few
    (40, 0.5, 3.1, "EDGE"), (40, 0.1, 0.6, "POSITIVE_NS"), (40, -0.1, -0.6, "NO_EDGE"), (40, -0.6, -3.5, "LOSING"),
    (40, None, None, "TOO_FEW"),
])
def test_verdict_only_claims_what_the_sample_supports(n, avg, t, code):
    assert perf.verdict(n, avg, t)[0] == code


def test_summarise_gives_mean_r_standard_error_and_verdict():
    rs = [1.0, -1.0] * 20 + [1.0] * 10  # 50 trades, mean R = 10/50 = 0.2
    s = perf.summarise(_synthetic("S", rs))
    assert s["trades"] == 50 and s["avg_r"] == pytest.approx(0.2) and s["win_rate"] == pytest.approx(60.0)
    assert s["r_se"] == pytest.approx(np.std(rs, ddof=1) / math.sqrt(50)) and s["t_stat"] == pytest.approx(0.2 / s["r_se"])
    assert s["verdict"] == "POSITIVE_NS" and s["win_ci_low"] < 60 < s["win_ci_high"]
    strong = perf.summarise(_synthetic("S", [0.5 + 0.2 * ((-1) ** i) for i in range(40)]))
    assert strong["verdict"] == "EDGE" and strong["t_stat"] > 2
    empty = perf.summarise(_synthetic("S", [])) if False else perf.summarise(_synthetic("S", [0.0]).iloc[0:0])
    assert empty["trades"] == 0 and empty["verdict"] == "TOO_FEW" and empty["avg_r"] is None


def test_strategy_table_ranks_proven_strategies_first_and_never_crowns_a_small_sample():
    big_good = _synthetic("Steady", [0.4 + 0.5 * ((-1) ** i) for i in range(40)])
    big_bad = _synthetic("Leaky", [-0.3 + 0.5 * ((-1) ** i) for i in range(40)])
    tiny_lucky = _synthetic("Lucky", [3.0, 3.0, 3.0])
    table = perf.strategy_table(pd.concat([big_good, big_bad, tiny_lucky], ignore_index=True))
    assert list(table["strategy"]) == ["Steady", "Leaky", "Lucky"]  # 40-trade strategies first, by mean R
    assert table.set_index("strategy").loc["Lucky", "verdict"] == "TOO_FEW"
    lead = perf.leader(table)
    assert lead["name"] == "Steady" and "statistically significant" in lead["text"]


def test_leader_text_for_the_weak_evidence_cases():
    assert "No closed trades" in perf.leader(pd.DataFrame())["text"]
    small = perf.strategy_table(_synthetic("Lucky", [3.0, 3.0, 3.0]))
    t = perf.leader(small)
    assert t["name"] is None and "can be called best" in t["text"] and "noise" in t["text"]
    mid = perf.strategy_table(_synthetic("Meh", [0.2 + 0.9 * ((-1) ** i) for i in range(40)]))
    assert perf.leader(mid)["name"] is None and "No strategy has a statistically significant edge" in perf.leader(mid)["text"]


def test_breakdowns_daily_series_and_unresolved():
    rows = [_row("A", day="2026-10-05", hour=9, exit_=110), _row("B", day="2026-10-05", hour=10, exit_=95, tag="TIE-BREAK"),
            _row("C", day="2026-10-06", hour=10, exit_=90, instrument="CRUDEOIL"), _row("D", status="STALE", day="2026-10-06"),
            _row("E", status="OPEN", day="2026-10-07")]
    df = _frame(rows)
    by_tag = perf.breakdown(df, "tag_label").set_index("tag_label")
    assert by_tag.loc["TIE-BREAK", "trades"] == 1 and by_tag.loc["Normal", "trades"] == 2
    assert set(perf.breakdown(df, "market")["market"]) == {"NSE", "MCX"}
    assert perf.breakdown(df, "hour").set_index("hour").loc[9, "net_pnl"] > 0

    daily = perf.daily_series(df)
    assert list(daily["trades"]) == [2, 1] and daily["cumulative"].iloc[-1] == pytest.approx(daily["net_pnl"].sum())
    assert (daily["drawdown"] <= 0).all()
    un = perf.unresolved(df)
    assert un["stale"] == 1 and un["open"] == 1 and un["dates"] == ["2026-10-06"] and un["invested"] > 0
    assert len(perf.closed_only(df)) == 3  # STALE and OPEN rows never count as results


def test_expected_vs_realised_flags_an_underperforming_strategy():
    rows = [_row(f"P{i}", strategy="Overrated", expected_r=1.0, exit_=(110 if i % 2 else 90)) for i in range(10)]
    rows += [_row("Q", strategy="NoExpectation")]  # older trades have no expectation: not compared
    gap = perf.expected_vs_realised(_frame(rows)).set_index("strategy")
    assert list(gap.index) == ["Overrated"] and gap.loc["Overrated", "expected_r"] == 1.0
    assert gap.loc["Overrated", "gap"] < -0.8 and gap.loc["Overrated", "gap_t"] < -2


# ---------------------------------------------------------------- 4. decision log
def _output(strategy="Momentum", no_trade=False, reason="r", tie=False, ts="2031-03-03 10:00"):
    score = SimpleNamespace(strategy_name=strategy, expected_r=0.25, confidence_label="HIGH")
    ranking = SimpleNamespace(is_no_trade=no_trade, selected_strategy=None if no_trade else strategy, reason=reason,
                              tie_break=tie, ranked=[score])
    return SimpleNamespace(ranking=ranking, timestamp=pd.Timestamp(ts), regime_label="RANGE")


def _result(**kw):
    base = dict(instrument="NIFTY", strategy_name="Momentum", setup_status=None, order=None, risk_decision_id=None,
                reason="", tag=None)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("reason,code", [
    ("Data quality is DEGRADED: NO TRADE forced", "DATA_QUALITY"),
    ("Top strategies (A vs B) are not distinguishable", "TIED"),
    ("No strategy clears minimum edge/confidence thresholds", "NO_EDGE"), ("something else", "OTHER"),
])
def test_no_trade_reasons_are_classified(reason, code):
    assert decision_log.classify_no_trade(reason) == code


def test_cycle_outcomes_are_classified_into_one_status():
    c = decision_log.classify_cycle
    assert c(_output(no_trade=True, reason="Data quality is FAIL"), _result())["status"] == "NO_TRADE"
    filled = c(_output(), _result(order=SimpleNamespace(status="FILLED", order_id="O1"), risk_decision_id="RISK-1"))
    assert filled["status"] == "FILLED" and filled["order_id"] == "O1" and filled["expected_r"] == 0.25 and filled["confidence"] == "HIGH"
    assert c(_output(), _result(order=SimpleNamespace(status="REJECTED", order_id="O2")))["status"] == "REJECTED"
    veto = c(_output(), _result(reason="RISK ENGINE VETO (R): Daily loss 5% exceeds limit 3%"))
    assert veto["status"] == "VETOED" and veto["reason_class"] == "DAILY_LOSS"
    assert c(_output(), _result(setup_status="CHAIN_ERROR", reason="chain down"))["status"] == "CHAIN_ERROR"
    assert c(_output(), _result(reason="Skipped: already have an open position on NIFTY."))["status"] == "SKIPPED"
    assert c(_output(), _result(setup_status="WAITING_FOR_SETUP", reason="Waiting for setup."))["status"] == "WAITING"
    assert c(_output(tie=True), _result(setup_status="WAITING_FOR_SETUP"))["tie_break"] is True


def _decision_rows(instrument="DLTEST"):
    from quant_intelligence.database.db import get_session
    from quant_intelligence.database.models import ScanDecision

    with get_session() as s:
        return [(r.status, r.order_id) for r in s.query(ScanDecision).filter_by(instrument=instrument).all()]


def test_one_row_per_bar_and_the_strongest_outcome_wins():
    from quant_intelligence.database.db import init_db

    init_db()
    decision_log._seen.clear()
    out = _output(ts="2031-04-04 10:05")
    rs = lambda **kw: _result(instrument="DLTEST", **kw)  # noqa: E731
    decision_log.log_cycle(out, rs(setup_status="WAITING_FOR_SETUP"))
    assert [s for s, _ in _decision_rows()] == ["WAITING"]
    decision_log.log_cycle(out, rs(setup_status="WAITING_FOR_SETUP"))  # a repeat scan of the same bar: nothing new
    decision_log.log_cycle(out, rs(order=SimpleNamespace(status="FILLED", order_id="O9"), risk_decision_id="RISK-9"))
    assert _decision_rows() == [("FILLED", "O9")]
    decision_log.log_cycle(out, rs(setup_status="WAITING_FOR_SETUP"))  # a weaker outcome never downgrades it
    assert _decision_rows() == [("FILLED", "O9")]
    decision_log.log_cycle(_output(ts="2031-04-04 10:10"), rs(setup_status="WAITING_FOR_SETUP"))  # next bar: new row
    assert len(_decision_rows()) == 2
    decision_log._seen.clear()  # a restart: the database still holds the stronger row, so it is not downgraded
    decision_log.log_cycle(out, rs(setup_status="WAITING_FOR_SETUP"))
    assert ("FILLED", "O9") in _decision_rows()


def test_logging_failures_never_reach_the_trading_cycle():
    decision_log.log_cycle(SimpleNamespace(), SimpleNamespace(instrument="X"))  # missing attributes: swallowed
    decision_log.log_chain_error("X", SimpleNamespace(), "boom")


def test_old_decisions_are_pruned():
    from quant_intelligence.database.db import get_session, init_db
    from quant_intelligence.database.models import ScanDecision

    init_db()
    with get_session() as s:
        s.add(ScanDecision(decided_at=dt.datetime(2000, 1, 1), bar_ts=dt.datetime(2000, 1, 1, 10), instrument="OLDONE", status="NO_TRADE"))
    assert decision_log.prune(90) >= 1
    assert _decision_rows("OLDONE") == []


# ---------------------------------------------------------------- 5. report
def _seed(monkeypatch, positions, decisions=(), orders=(), risk=(), events=()):
    data = {"positions": positions_frame(positions), "decisions": list(decisions), "orders": list(orders),
            "risk": list(risk), "events": list(events)}
    monkeypatch.setattr(eod, "_load", lambda report_date, scope: data)


def _trades_over_days():
    rows = []
    for i in range(6):
        rows.append(_row(f"M{i}", strategy="Momentum", day="2026-10-05", hour=10, exit_=112 if i % 3 else 92, expected_r=0.5))
    rows += [_row("V1", strategy="VWAP", day="2026-10-05", hour=11, exit_=105), _row("T1", strategy="Donchian", day="2026-10-02", exit_=95, tag="TIE-BREAK"),
             _row("S1", strategy="Donchian", day="2026-10-01", status="STALE")]
    return rows


def test_report_structure_numbers_and_honest_wording(monkeypatch):
    decisions = [{"status": "NO_TRADE", "reason_class": "TIED", "strategy": None, "tie_break": False, "reason": "x"}] * 5 + [
        {"status": "FILLED", "reason_class": "FILLED", "strategy": "Momentum", "tie_break": False, "reason": ""},
        {"status": "VETOED", "reason_class": "TRADE_RISK", "strategy": "VWAP", "tie_break": True, "reason": ""}]
    _seed(monkeypatch, _trades_over_days(), decisions,
          orders=[{"status": "FILLED", "reject_reason": None, "strategy": "Momentum"}, {"status": "REJECTED", "reject_reason": "Insufficient funds", "strategy": "VWAP"}],
          risk=[{"event_type": "VETO", "reason": "Trade risk 4% exceeds max"}], events=[{"component": "data_keeper", "level": "WARNING", "message": "paused"}])
    rep = eod.build_report(dt.date(2026, 10, 5), "ALL", now=dt.datetime(2026, 10, 5, 16, 0))
    j = rep["content_json"]
    json.dumps(j)  # fully JSON-serialisable (NaN/inf/Timestamps cleaned)
    today = j["summary"]["today"]
    assert today["trades"] == 7 and j["summary"]["previous"]["date"] == "2026-10-02" and j["summary"]["cumulative"]["trades"] == 8
    assert j["funnel"]["bars_evaluated"] == 7 and j["funnel"]["signals"] == 2 and j["funnel"]["filled"] == 1 and j["funnel"]["tie_break"] == 1
    assert j["funnel"]["no_trade_reasons"][0] == {"reason": "TIED", "count": 5}
    assert j["orders"]["rejected"] == 1 and j["orders"]["reject_reasons"][0]["reason"] == "Insufficient funds"
    assert j["health"]["unresolved"]["stale"] == 1 and j["health"]["warnings"] == 1
    by = {r["strategy"]: r for r in j["strategies"]["all_time"]}
    assert by["Momentum"]["trades"] == 6 and by["Momentum"]["verdict"] == "TOO_FEW"
    assert j["leader"]["name"] is None and "can be called best" in j["leader"]["text"]
    text = " ".join(j["narrative"])
    assert "7 trade(s) closed today" in text and "stood aside" in text and "tie-break" in text and "left unresolved" in text
    md = rep["content_markdown"]
    for heading in ("# Daily report - 2026-10-05", "## How things went", "## Strategy leaderboard", "## Signals vs trades",
                    "## Realised vs backtest-expected R", "## System and data health", "## Caveats"):
        assert heading in md
    assert "Too few trades to conclude" in md and "TIED" in md


def test_empty_history_still_produces_a_clear_report(monkeypatch):
    _seed(monkeypatch, [])
    rep = eod.build_report(dt.date(2026, 10, 5), "ALL")
    text = " ".join(rep["content_json"]["narrative"])
    assert "No trade closed today" in text and "No scan decisions were recorded" in text and "No closed trades yet" in text
    assert "# Daily report" in rep["content_markdown"]
    with pytest.raises(ValueError):
        eod.build_report(dt.date(2026, 10, 5), "FOREX")


def test_reports_are_stored_replaced_listed_and_loaded():
    d = dt.date(2031, 5, 5)
    r1 = {"content_json": {"summary": {"today": {"trades": 3, "net": 1500.0, "win_rate": 66.0}, "cumulative": {"net": 4000.0}}}, "content_markdown": "# one"}
    eod.save_report(d, "ALL", r1)
    r2 = {"content_json": {"summary": {"today": {"trades": 4, "net": -200.0, "win_rate": 25.0}, "cumulative": {"net": 3800.0}}}, "content_markdown": "# two"}
    eod.save_report(d, "ALL", r2)  # same date + scope: replaced, not duplicated
    eod.save_report(d - dt.timedelta(days=1), "ALL", r1)
    eod.save_report(d, "NSE", r1)
    assert eod.load_report(d, "ALL")["content_markdown"] == "# two" and eod.load_report(d, "MCX") is None
    listing = [r for r in eod.list_reports("ALL") if r["date"] in (d, d - dt.timedelta(days=1))]
    assert [r["date"] for r in listing] == [d, d - dt.timedelta(days=1)]  # newest first, one row per date
    assert listing[0]["trades"] == 4 and listing[0]["net_pnl"] == -200.0 and listing[0]["cumulative"] == 3800.0


# ---------------------------------------------------------------- 5b. EOD job
@pytest.fixture
def job_env(monkeypatch):
    monkeypatch.setattr(eod_job, "SETTINGS", dataclasses.replace(settings_module.SETTINGS, eod_report_enabled=True, eod_force_square_off=True))
    keys = {"NSE": dt.date(2026, 10, 7), "MCX": dt.date(2026, 10, 6)}
    monkeypatch.setattr(eod_job, "session_key", lambda now, market: keys[market])
    made, squared = [], []
    monkeypatch.setattr(eod, "generate_and_save", lambda d, s: made.append((d, s)))
    monkeypatch.setattr(eod_job.EodJob, "force_square_off", staticmethod(lambda market: squared.append(market) or 2))
    monkeypatch.setattr(eod_job.EodJob, "_prune", staticmethod(lambda decision_log: None))
    return keys, made, squared


def test_the_job_builds_each_report_once_and_refreshes_all_after_the_later_close(job_env):
    keys, made, squared = job_env
    job = eod_job.EodJob()
    assert job.run_round() == 3 and sorted(s for _, s in made) == ["ALL", "MCX", "NSE"] and sorted(squared) == ["MCX", "NSE"]
    assert job.run_round() == 0  # nothing new: no repeat
    keys["MCX"] = dt.date(2026, 10, 7)  # MCX finishes its session for the same date, after midnight
    assert job.run_round() == 2 and [s for _, s in made[3:]] == ["MCX", "ALL"]  # NSE is not redone, ALL is refreshed
    assert job.status()["last_report"][1] == "ALL" and job.status()["error"] is None


def test_the_job_is_failure_isolated_and_retries_later(job_env, monkeypatch):
    keys, made, _ = job_env
    calls = []

    def flaky(d, s):
        calls.append(s)
        if s == "NSE" and len(calls) == 1:
            raise RuntimeError("db down")
        made.append((d, s))

    monkeypatch.setattr(eod, "generate_and_save", flaky)
    job = eod_job.EodJob(retry_after_sec=1.0)
    job.run_round()
    assert sorted(s for _, s in made) == ["ALL", "MCX"] and "db down" in job.status()["error"]  # others still ran
    assert job.run_round() == 0  # paused for the retry delay
    import time as _t
    _t.sleep(1.2)
    assert job.run_round() == 1 and ("NSE" in [s for _, s in made])


def test_forced_square_off_failure_still_produces_the_report(job_env, monkeypatch):
    _, made, _ = job_env
    monkeypatch.setattr(eod_job.EodJob, "force_square_off", staticmethod(lambda market: (_ for _ in ()).throw(RuntimeError("no prices"))))
    eod_job.EodJob().run_round()
    assert len(made) == 3  # the report is still stored (it lists the open positions)


def test_the_job_is_off_when_disabled(job_env, monkeypatch):
    monkeypatch.setattr(eod_job, "SETTINGS", dataclasses.replace(settings_module.SETTINGS, eod_report_enabled=False))
    assert eod_job.EodJob().run_round() == 0


def test_no_forced_square_off_when_switched_off(job_env, monkeypatch):
    _, made, squared = job_env
    monkeypatch.setattr(eod_job, "SETTINGS", dataclasses.replace(settings_module.SETTINGS, eod_report_enabled=True, eod_force_square_off=False))
    eod_job.EodJob().run_round()
    assert squared == [] and len(made) == 3


def test_session_keys_wait_for_the_close_plus_settle_time():
    assert eod_job.session_key(dt.datetime(2026, 10, 7, 11, 0), "NSE") is None
    assert eod_job.session_key(dt.datetime(2026, 10, 7, 15, 45), "NSE") == dt.date(2026, 10, 7)
    assert eod_job.session_key(dt.datetime(2026, 10, 8, 0, 20), "MCX") == dt.date(2026, 10, 7)


# ---------------------------------------------------------------- 2 + end to end
def test_a_paper_trade_flows_into_history_the_decision_log_and_the_report(monkeypatch):
    from quant_intelligence.brokers.paper_broker import PaperBroker
    from quant_intelligence.database.db import get_session, init_db
    from quant_intelligence.database.models import Order, Position
    from quant_intelligence.execution import auto_trader
    from quant_intelligence.ranking.ranking_engine import StrategyScore
    from quant_intelligence.risk.risk_engine import AccountState
    from quant_intelligence.utils.timeutil import now_ist

    init_db()
    decision_log._seen.clear()
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
    score = StrategyScore("E2E-Strategy", 0.5, 0.35, 0.6, "HIGH", 80, -2.0, True)
    ranking = SimpleNamespace(is_no_trade=False, selected_strategy="E2E-Strategy", tie_break=False, reason="r", tie_runner_up=None, ranked=[score])
    bar = pd.Timestamp(now_ist().replace(second=0, microsecond=0))
    output = SimpleNamespace(ranking=ranking, ohlcv=SimpleNamespace(tail=lambda n: None), timestamp=bar, regime_label="TREND_UP",
                             market_state=SimpleNamespace(get=lambda k, d=None: None), data_quality_status="OK")
    account = AccountState(equity=1_000_000.0, peak_equity=1_000_000.0, daily_pnl=0.0, open_positions_count=0, trades_today=0,
                           exposure_by_strategy={}, total_exposure=0.0, broker_connected=True, kill_switch_engaged=False)
    result = auto_trader.run_auto_option_cycle(output, "NIFTY", broker, client=None, chain=SimpleNamespace(underlying="NIFTY"), account=account)
    assert result.order.status == "FILLED"

    pos = broker.get_open_positions()[0]
    assert pos["expected_r"] == 0.35 and pos["confidence"] == "HIGH" and pos["regime"] == "TREND_UP"
    assert pos["order_id"] == result.order.order_id and pos["decision_id"] == result.risk_decision_id
    broker.close_position(pos["position_id"], 62.0, "TARGET")

    with get_session() as s:  # permanent history: the row carries what the system expected when it entered
        row = s.query(Position).filter_by(position_id=pos["position_id"]).one()
        assert (row.status, row.expected_r, row.regime, row.order_id) == ("CLOSED", 0.35, "TREND_UP", result.order.order_id)
        assert s.query(Order).filter_by(order_id=result.order.order_id).count() == 1
    assert ("FILLED", result.order.order_id) in _decision_rows("NIFTY")

    rep = eod.generate_and_save(now_ist().date(), "ALL")
    j = rep["content_json"]
    mine = {r["strategy"]: r for r in j["strategies"]["all_time"]}["E2E-Strategy"]
    assert mine["trades"] == 1 and mine["verdict"] == "TOO_FEW" and mine["net_pnl"] > 0
    assert any(r["strategy"] == "E2E-Strategy" and r["expected_r"] == 0.35 for r in j["expected_vs_realised"])
    assert j["funnel"]["filled"] >= 1
    assert eod.load_report(now_ist().date(), "ALL")["content_markdown"].startswith("# Daily report")
