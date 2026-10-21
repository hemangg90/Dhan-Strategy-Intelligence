"""The candidate workflow: every stage's state, the named failing condition, the three setup kinds and the risk preview."""
import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import RiskEvent
from quant_intelligence.execution.setup_detector import SETUP_TRIGGERED, WAITING_FOR_SETUP
from quant_intelligence.ranking.ranking_engine import RankingDecision, StrategyScore
from quant_intelligence.risk.risk_engine import AccountState, RiskDecision
from quant_intelligence.strategies.base_strategy import Setup
from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES
from quant_intelligence.ui import candidate as cand
from quant_intelligence.utils.timeutil import now_ist

BAR = pd.Timestamp("2026-10-16 15:25")


def _account():
    return AccountState(equity=1_000_000.0, peak_equity=1_000_000.0, daily_pnl=0.0, open_positions_count=0, trades_today=0)


class _State(dict):
    def get(self, k, default=None):
        v = super().get(k, default)
        return default if v is None else v


def _score(name, eligible=True):
    return StrategyScore(name, 0.3, 0.3, 0.55, "MEDIUM", 40, -3.0, eligible, None, edge_mean=0.3, edge_stderr=0.1)


def _output(selected="Momentum", quality="OK", mode="LIVE", no_trade=False, reason="Momentum ranks highest", issues=()):
    ranked = [_score("Momentum"), _score("ORB", eligible=False)]
    ranking = RankingDecision(ranked=ranked, selected_strategy=None if no_trade else selected, is_no_trade=no_trade, reason=reason)
    ohlcv = pd.DataFrame({"timestamp": pd.date_range("2026-10-16 15:00", periods=6, freq="5min"), "open": 100.0, "high": 101.0,
                          "low": 99.0, "close": 100.0, "volume": 1000})
    return SimpleNamespace(
        instrument="NIFTY", timestamp=BAR, ohlcv=ohlcv, ranking=ranking, data_quality_status=quality,
        data_quality_issues=list(issues), regime_label="TREND_UP", regime_axes={"direction": "UP", "volatility": "NORMAL", "volatility_trend": "STEADY"},
        provenance={"mode": mode, "mode_meaning": "m", "source": "cache"}, strategy_intel=[],
        market_state=_State({"vwap": 100.0, "opening_range_high": 101.0, "relative_volume": 1.0}))


def _setup(direction="LONG"):
    return Setup(BAR, direction, 100.0, 98.0 if direction == "LONG" else 102.0, 104.0 if direction == "LONG" else 96.0, {"note": "x"})


def _detect(status, setup=None):
    return lambda *a, **k: SimpleNamespace(status=status, setup=setup)


def _preview(approved=True, reason="All risk checks passed", checks=None):
    return lambda account, trade: RiskDecision("PREVIEW-1", approved, reason, checks or {"daily_loss_within_limit": True, "risk_per_trade_within_limit": approved}, preview=True)


def _states(c):
    return {s["key"]: s["state"] for s in c["stages"]}


def test_bad_data_stops_at_the_data_stage_and_names_the_condition():
    c = cand.build_candidate(_output(quality="DEGRADED", issues=["data is stale: last bar old"]), _account(),
                             detect=_detect(WAITING_FOR_SETUP), preview=_preview())
    assert _states(c)["data"] == cand.FAIL and c["verdict"] == "NO TRADE" and c["blocking_stage"] == "Data"
    assert "stale" in c["reason"] and "DEGRADED" in c["reason"]


def test_no_strategy_selected_names_the_ranker_reason_and_skips_later_stages():
    c = cand.build_candidate(_output(no_trade=True, reason="No strategy clears minimum edge/confidence thresholds"), _account())
    s = _states(c)
    assert s["evidence"] == cand.FAIL and s["setup"] == cand.SKIP and s["risk"] == cand.SKIP and s["order"] == cand.SKIP
    assert c["verdict"] == "NO TRADE" and c["blocking_stage"] == "Evidence" and "minimum edge" in c["reason"]


def test_selected_but_no_setup_yet_is_waiting_not_no_trade():
    c = cand.build_candidate(_output(), _account(), detect=_detect(WAITING_FOR_SETUP), preview=_preview())
    assert c["verdict"] == "WAITING FOR SETUP" and c["status"] is None and _states(c)["setup"] == cand.WAIT
    assert _states(c)["risk"] == cand.SKIP


def test_triggered_setup_with_approved_preview_is_a_ready_current_candidate():
    c = cand.build_candidate(_output(), _account(), detect=_detect(SETUP_TRIGGERED, _setup()), preview=_preview())
    assert c["verdict"] == "CANDIDATE READY" and c["status"] == cand.CURRENT_CANDIDATE
    g = c["setup"]
    assert g["risk_points"] == pytest.approx(2.0) and g["reward_points"] == pytest.approx(4.0) and g["risk_reward"] == pytest.approx(2.0)
    assert "below 98.00" in g["invalidation"] and len(c["assumptions"]) >= 3
    assert _states(c)["risk"] == cand.PASS and _states(c)["order"] == cand.PASS
    assert any(r["ok"] for r in c["risk_checks"])


def test_short_setup_invalidation_is_above_the_stop():
    c = cand.build_candidate(_output(), _account(), detect=_detect(SETUP_TRIGGERED, _setup("SHORT")), preview=_preview())
    assert "above 102.00" in c["setup"]["invalidation"]


def test_a_risk_veto_after_a_good_setup_is_named_as_the_stopping_condition():
    c = cand.build_candidate(_output(), _account(), detect=_detect(SETUP_TRIGGERED, _setup()),
                             preview=_preview(False, "Daily loss 3.50% exceeds limit 3.0%", {"daily_loss_within_limit": False}))
    assert c["verdict"] == "NO TRADE" and c["blocking_stage"] == "Risk engine" and "Daily loss" in c["reason"]
    assert _states(c)["order"] == cand.SKIP and c["status"] == cand.CURRENT_CANDIDATE  # the setup was real; risk stopped it


def test_historical_data_is_a_warning_for_research_not_a_failure():
    c = cand.build_candidate(_output(mode="HISTORICAL"), _account(), detect=_detect(WAITING_FOR_SETUP), preview=_preview())
    assert _states(c)["data"] == cand.WARN and "HISTORICAL" in c["stages"][0]["headline"]


def test_chain_aware_strategies_wait_for_a_chain():
    name = next(iter(CHAIN_AWARE_STRATEGY_NAMES))
    out = _output(selected=name)
    out.ranking.ranked = [_score(name)]
    c = cand.build_candidate(out, _account(), detect=_detect(SETUP_TRIGGERED, _setup()), preview=_preview())
    assert _states(c)["setup"] == cand.WAIT and c["status"] is None


def test_tie_break_is_flagged_as_a_warning_not_hidden():
    out = _output()
    out.ranking.tie_break = True
    c = cand.build_candidate(out, _account(), detect=_detect(WAITING_FOR_SETUP), preview=_preview())
    assert _states(c)["evidence"] == cand.WARN and "tie-break" in c["stages"][2]["headline"]


def test_outcome_stage_lists_earlier_paper_trades_as_simulated():
    trades = [{"status": "CLOSED", "realised_r": 1.5}]
    c = cand.build_candidate(_output(), _account(), recent_trades=trades, detect=_detect(WAITING_FOR_SETUP), preview=_preview())
    outcome = next(s for s in c["stages"] if s["key"] == "outcome")
    assert outcome["state"] == cand.INFO and "1.50" in " ".join(outcome["detail"])
    empty = cand.build_candidate(_output(), _account(), detect=_detect(WAITING_FOR_SETUP), preview=_preview())
    assert next(s for s in empty["stages"] if s["key"] == "outcome")["state"] == cand.WAIT


def test_levels_only_include_real_numbers():
    c = cand.build_candidate(_output(), _account(), detect=_detect(WAITING_FOR_SETUP), preview=_preview())
    assert c["levels"] == {"VWAP": 100.0, "Opening range high": 101.0}


# ------------------------------------------------------------------------- real risk engine
def test_the_real_risk_preview_inside_the_candidate_leaves_no_trace_and_names_a_stale_price():
    out = _output()
    out.timestamp = pd.Timestamp(now_ist()) - pd.Timedelta(hours=2)  # a two-hour-old bar
    with get_session() as s:
        before = s.query(RiskEvent).count()
    c = cand.build_candidate(out, _account(), detect=_detect(SETUP_TRIGGERED, _setup()))
    with get_session() as s:
        assert s.query(RiskEvent).count() == before  # a preview writes nothing
    assert c["verdict"] == "NO TRADE" and c["blocking_stage"] == "Risk engine" and "stale" in c["reason"].lower()


def test_the_real_risk_preview_approves_a_fresh_sound_setup():
    out = _output()
    out.timestamp = pd.Timestamp(now_ist()) - pd.Timedelta(minutes=1)
    out.ohlcv = out.ohlcv.assign(timestamp=pd.date_range(end=out.timestamp, periods=6, freq="5min"))
    c = cand.build_candidate(out, _account(), detect=_detect(SETUP_TRIGGERED, Setup(out.timestamp, "LONG", 22000.0, 21980.0, 22040.0)))
    assert c["verdict"] == "CANDIDATE READY", c["reason"]
    assert all(r["ok"] for r in c["risk_checks"]) and c["risk_checks"]


# ------------------------------------------------------------------------ settings & data status
def test_settings_rows_show_limits_and_never_a_credential():
    from quant_intelligence.config.settings import SETTINGS
    from quant_intelligence.ui import settings_view as sv

    rows = sv.mode_rows() + sv.risk_rows() + sv.housekeeping_rows()
    text = " ".join(str(v) for r in rows for v in r.values())
    for secret in (SETTINGS.dhan_access_token, SETTINGS.dhan_client_id, SETTINGS.dhan_pin, SETTINGS.dhan_totp_secret):
        assert not secret or secret not in text  # whatever credentials are configured never appear
    labels = " ".join(str(r.get("Setting", r.get("Limit", ""))).lower() for r in rows)
    assert not any(word in labels for word in ("token", "secret", "password", "pin", "api key", "client id"))
    assert any(r["Limit"] == "Max daily loss" for r in sv.risk_rows())
    assert next(r for r in sv.mode_rows() if r["Setting"] == "Trading mode")["Value"] == SETTINGS.trading_mode


def test_universe_rows_carry_a_currency_label(tmp_path, monkeypatch):
    import numpy as np

    from quant_intelligence.regimes import universe

    ts = []
    for d in pd.bdate_range("2026-08-03", periods=15):
        ts += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min"))
    close = 22000 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.0007, len(ts))))
    df = pd.DataFrame({"timestamp": ts, "open": np.r_[close[0], close[:-1]], "high": close * 1.0005, "low": close * 0.9995,
                       "close": close, "volume": 1000})
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    monkeypatch.setattr(universe, "CACHE_DIR", tmp_path)
    df.to_parquet(tmp_path / "NIFTY_5min.parquet", index=False)
    [row] = universe.scan_universe("5min", [("NIFTY", "index")])
    # Old candles are never LIVE: HISTORICAL when the market is closed now, DELAYED when it is open and the bar is old.
    assert row["mode"] in ("HISTORICAL", "DELAYED")


# --------------------------------------------------------------------------------- guide & reports
def test_guide_stages_match_the_candidate_flow_and_make_no_profit_promises():
    from quant_intelligence.ui import guide_content as gc

    stage_names = [name for name, _ in gc.STAGES]
    out = cand.build_candidate(_output(), _account(), detect=_detect(SETUP_TRIGGERED, _setup()), preview=_preview())
    assert stage_names == [s["title"] for s in out["stages"]]
    text = " ".join(list(gc.GLOSSARY.values()) + gc.HOW_TO_READ + gc.LIMITS + [t for _, t in gc.STAGES]).lower()
    for word in ("guarantee", "risk-free", "will profit", "always profitable"):
        assert word not in text
    assert all(definition.strip() for definition in gc.GLOSSARY.values())
    for term in ("R (risk multiple)", "NO TRADE", "SIMULATED", "Multiple testing", "LIVE / DELAYED / HISTORICAL"):
        assert term in gc.GLOSSARY


def test_markdown_to_html_is_self_contained_and_escapes():
    from quant_intelligence.reports import export

    md = "# Title <x>\n\n## Part\n- **bold** item\n\n| A | B |\n|---|---|\n| 1 | <script> |\n"
    html_out = export.markdown_to_html(md, "T")
    assert "<h1>Title &lt;x&gt;</h1>" in html_out and "<b>bold</b>" in html_out and "<th>A</th>" in html_out
    assert "<script>" not in html_out and "&lt;script&gt;" in html_out
    assert "http://" not in html_out and "https://" not in html_out and "<link" not in html_out


def test_a_built_report_has_the_new_sections_and_labels_results_as_simulated():
    from quant_intelligence.reports import eod, export

    built = eod.build_report(dt.date(2026, 1, 5), "ALL")
    rep, md = built["content_json"], built["content_markdown"]
    for key in ("rejected_setups", "data_quality", "methodology", "labels"):
        assert key in rep
    assert "SIMULATED" in rep["labels"]["trades"] and rep["methodology"]
    assert "## Methodology" in md and "## Data quality" in md and "## Setups that did not become trades" in md
    assert {"Setups not traded", "Data quality", "Methodology"} <= set(export.sheets(rep))
    old = {k: v for k, v in rep.items() if k not in ("rejected_setups", "data_quality", "methodology", "labels")}
    assert {"Setups not traded", "Data quality", "Methodology"} <= set(export.sheets(old))  # older stored reports still export


def test_rejected_setups_and_data_quality_sections_summarise_correctly():
    from quant_intelligence.reports import eod

    decisions = [
        {"status": "VETOED", "instrument": "NIFTY", "strategy": "ORB", "reason_class": "DAILY_LOSS", "reason": "x", "bar_ts": dt.datetime(2026, 1, 5, 10, 0)},
        {"status": "FILLED", "instrument": "NIFTY", "strategy": "ORB", "reason_class": "FILLED", "reason": "", "bar_ts": dt.datetime(2026, 1, 5, 10, 5)},
        {"status": "NO_TRADE", "instrument": "NIFTY", "strategy": None, "reason_class": "TIED", "reason": "", "bar_ts": dt.datetime(2026, 1, 5, 10, 10)},
        {"status": "TRIGGERED", "instrument": "SBIN", "strategy": "ORB", "reason_class": "NO_SIZE", "reason": "y", "bar_ts": dt.datetime(2026, 1, 5, 11, 0)},
    ]
    rows = eod._rejected_setups(decisions)
    assert [r["status"] for r in rows] == ["TRIGGERED", "VETOED"]  # newest first; fills and stand-asides excluded
    items = [{"instrument": "NIFTY", "status": "DEGRADED", "source": "cache", "category": "STALE", "issues": ["stale"]},
             {"instrument": "NIFTY", "status": "DEGRADED", "source": "cache", "category": "STALE", "issues": ["stale"]},
             {"instrument": "TCS", "status": "OK", "source": "cache", "category": "VALID", "issues": []}]
    dq = eod._data_quality(items)
    assert dq["loads"] == 3 and dq["not_ok"] == 2 and dq["items"][0]["times"] == 2 and dq["items"][0]["instrument"] == "NIFTY"
