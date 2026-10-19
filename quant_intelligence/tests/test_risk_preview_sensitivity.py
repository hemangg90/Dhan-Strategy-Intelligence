"""preview_trade (real verdict, no trace) and parameter sensitivity."""
import numpy as np
import pandas as pd
import pytest

from quant_intelligence.database.db import get_session
from quant_intelligence.database.models import RiskEvent
from quant_intelligence.features.feature_engine import compute_features
from quant_intelligence.research import sensitivity
from quant_intelligence.risk import risk_engine as re_
from quant_intelligence.risk.risk_engine import AccountState, ProposedTrade, consume_approval, evaluate_trade, preview_trade


def _account(**overrides):
    base = dict(equity=1_000_000, peak_equity=1_000_000, daily_pnl=0.0, open_positions_count=0, trades_today=0)
    base.update(overrides)
    return AccountState(**base)


def _trade(**overrides):
    base = dict(strategy_name="Momentum", direction="LONG", entry_price=100.0, stop_price=99.0, target_price=103.0,
                quantity=50, relative_volume=1.0, data_quality_status="OK")
    base.update(overrides)
    return ProposedTrade(**base)


def _events() -> int:
    with get_session() as s:
        return s.query(RiskEvent).count()


CASES = [
    (dict(), dict()),                                              # approved
    (dict(kill_switch_engaged=True), dict()),                      # kill switch
    (dict(daily_pnl=-50_000), dict()),                             # daily loss
    (dict(trades_today=10), dict()),                               # trade count
    (dict(), dict(data_quality_status="DEGRADED")),                # data quality
    (dict(), dict(quantity=0)),                                    # invalid order
    (dict(), dict(entry_price=100.0, stop_price=50.0, quantity=5000)),  # too much risk
    (dict(broker_connected=False), dict()),
]


@pytest.mark.parametrize("account_kw, trade_kw", CASES)
def test_preview_gives_the_same_verdict_as_the_real_evaluation(account_kw, trade_kw):
    preview = preview_trade(_account(**account_kw), _trade(**trade_kw))
    real = evaluate_trade(_account(**account_kw), _trade(**trade_kw))
    assert preview.approved == real.approved and preview.reason == real.reason and preview.checks == real.checks
    assert preview.preview is True and real.preview is False


def test_preview_writes_no_audit_row_and_issues_no_usable_approval():
    before = _events()
    approved = preview_trade(_account(), _trade())
    vetoed = preview_trade(_account(kill_switch_engaged=True), _trade())
    assert approved.approved and not vetoed.approved
    assert _events() == before  # nothing recorded, neither approval nor veto
    assert consume_approval(approved.decision_id) is False  # a broker cannot spend a preview


def test_a_real_evaluation_after_a_preview_is_still_audited_and_approved():
    preview_trade(_account(), _trade())
    before = _events()
    real = evaluate_trade(_account(), _trade())
    assert real.approved and _events() == before + 1
    assert consume_approval(real.decision_id) is True


def test_preview_is_fail_closed_on_error():
    bad = _account(equity=float("nan"))
    d = preview_trade(bad, _trade())
    assert d.approved is False and d.preview is True


def test_preview_flag_is_reset_even_if_the_evaluation_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(re_, "_evaluate_trade", boom)
    assert preview_trade(_account(), _trade()).approved is False
    assert re_._PREVIEW.get() is False


# --------------------------------------------------------------------------------- sensitivity
def _candles(n_days=14, seed=5):
    rng = np.random.default_rng(seed)
    ts = []
    for d in pd.bdate_range("2026-08-03", periods=n_days):
        ts += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min"))
    close = 22000 * np.exp(np.cumsum(rng.normal(0, 0.0007, len(ts))))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.0005, len(ts))) * close
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close, "volume": rng.integers(1000, 5000, len(ts))})


def test_sensitivity_varies_each_numeric_parameter_and_is_deterministic():
    ohlcv = _candles()
    feats = compute_features(ohlcv)
    from quant_intelligence.strategies.registry import STRATEGY_CLASSES

    name = next(n for n, c in STRATEGY_CLASSES.items() if "stop_atr_mult" in c.default_parameters and n not in
                {"PCR Contrarian", "IV Crush Writer", "OI Buildup"})
    one = sensitivity.parameter_sensitivity(name, ohlcv, feats, "NIFTY", 50)
    two = sensitivity.parameter_sensitivity(name, ohlcv, feats, "NIFTY", 50)
    assert one == two
    numeric = [k for k, v in one["base"]["parameters"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    assert {r["parameter"] for r in one["rows"]} <= set(numeric) and one["rows"]
    assert all(r["factor"] in (0.8, 1.2) for r in one["rows"])


def test_sensitivity_leaves_the_strategy_defaults_untouched_and_reports_invalid_settings():
    from quant_intelligence.strategies.registry import STRATEGY_CLASSES

    name, cls = next((n, c) for n, c in STRATEGY_CLASSES.items() if "stop_atr_mult" in c.default_parameters)
    before = dict(cls.default_parameters)
    sensitivity.parameter_sensitivity(name, _candles(8), compute_features(_candles(8)), "NIFTY", 50)
    assert cls.default_parameters == before


def test_sign_survival_logic(monkeypatch):
    results = iter([
        {"n_trades": 20, "mean_net_r": 0.2, "note": None},                         # base
        {"n_trades": 20, "mean_net_r": 0.1, "note": None}, {"n_trades": 20, "mean_net_r": 0.3, "note": None},
    ])
    monkeypatch.setattr(sensitivity, "_run", lambda *a, **k: next(results))

    class S:
        parameters = {"x": 10.0, "flag": True, "name": "a"}

    monkeypatch.setattr(sensitivity, "get_strategy", lambda *a, **k: S())
    out = sensitivity.parameter_sensitivity("S", None, None, "NIFTY", 50)
    assert [r["parameter"] for r in out["rows"]] == ["x", "x"] and out["sign_survives"] is True

    results2 = iter([{"n_trades": 20, "mean_net_r": 0.2, "note": None}, {"n_trades": 20, "mean_net_r": -0.1, "note": None},
                     {"n_trades": 20, "mean_net_r": 0.3, "note": None}])
    monkeypatch.setattr(sensitivity, "_run", lambda *a, **k: next(results2))
    assert sensitivity.parameter_sensitivity("S", None, None, "NIFTY", 50)["sign_survives"] is False


def test_too_few_trades_gives_no_sign_verdict(monkeypatch):
    results = iter([{"n_trades": 2, "mean_net_r": 0.2, "note": None}] * 3)
    monkeypatch.setattr(sensitivity, "_run", lambda *a, **k: next(results))

    class S:
        parameters = {"x": 10.0}

    monkeypatch.setattr(sensitivity, "get_strategy", lambda *a, **k: S())
    assert sensitivity.parameter_sensitivity("S", None, None, "NIFTY", 50)["sign_survives"] is None
