"""Editable risk limits: bounds, loosening rules, persistence, audit and fail-closed behaviour."""
import dataclasses
import math

import pytest

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.database.db import get_session, init_db
from quant_intelligence.database.models import RiskEvent, RiskLimitOverride
from quant_intelligence.risk import limit_overrides as LO


@pytest.fixture(autouse=True)
def clean_limits():
    """Every test starts and ends with the environment's limits and no saved overrides."""
    init_db()
    start = LO.current()
    with get_session() as s:
        s.query(RiskLimitOverride).delete()
        s.query(RiskEvent).filter(RiskEvent.event_type == "LIMIT_CHANGE").delete()
    yield
    for field, value in start.items():
        object.__setattr__(SETTINGS.risk, field, value)
    with get_session() as s:
        s.query(RiskLimitOverride).delete()
        s.query(RiskEvent).filter(RiskEvent.event_type == "LIMIT_CHANGE").delete()


def _risk():
    return SETTINGS.risk


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("spec", LO.EDITABLE, ids=lambda s: s.field)
def test_every_field_enforces_its_range(spec):
    ok, errors = LO.validate({spec.field: spec.hi})
    assert ok == {spec.field: spec.hi} or errors == ["Portfolio exposure must be at least the strategy exposure"]
    _, errors = LO.validate({spec.field: spec.hi * 1.01 + 1})
    assert errors, "above the ceiling must be rejected"
    _, errors = LO.validate({spec.field: spec.lo / 2 if spec.lo / 2 > 0 else -1})
    assert errors, "below the floor must be rejected"
    clean, errors = LO.validate({spec.field: 0})
    assert (not errors) == spec.off_allowed  # 0 only where the code treats it as "off"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "abc", None, True, [1]])
def test_non_numbers_are_rejected(bad):
    clean, errors = LO.validate({"max_daily_loss_pct": bad})
    assert errors and not clean


def test_integer_fields_need_whole_numbers_and_unknown_fields_are_refused():
    assert LO.validate({"max_trades_per_day": 2.5})[1]
    assert LO.validate({"max_trades_per_day": 3.0})[0] == {"max_trades_per_day": 3}
    assert LO.validate({"profit_lock_trigger_pct": 1.0})[1]  # not editable from the app


def test_portfolio_exposure_may_not_be_below_strategy_exposure():
    _, errors = LO.validate({"max_strategy_exposure_pct": 30.0, "max_portfolio_exposure_pct": 20.0})
    assert any("Portfolio exposure" in e for e in errors)


# ------------------------------------------------------------------ direction
def test_direction_for_every_kind_of_limit():
    assert LO.direction("max_daily_loss_pct", 1.5, 1.0) == LO.TIGHTER
    assert LO.direction("max_daily_loss_pct", 1.5, 2.0) == LO.LOOSER
    assert LO.direction("max_daily_loss_pct", 1.5, 1.5) is None
    assert LO.direction("loss_streak_pause_min", 60, 120) == LO.TIGHTER  # a longer pause is stricter
    assert LO.direction("loss_streak_pause_min", 60, 30) == LO.LOOSER
    assert LO.direction("max_open_positions", 4, 0) == LO.LOOSER  # switching a cap off is the loosest move
    assert LO.direction("max_open_positions", 0, 4) == LO.TIGHTER
    assert LO.direction("loss_streak_pause", 3, 2) == LO.TIGHTER  # pauses sooner


# ------------------------------------------------------------------ applying
def test_tightening_applies_at_once_without_confirmation_and_is_used_by_the_engine():
    new = round(_risk().max_risk_per_trade_pct / 2, 4)
    result = LO.apply({"max_risk_per_trade_pct": new})
    assert result.ok and _risk().max_risk_per_trade_pct == new
    assert result.applied["max_risk_per_trade_pct"]["direction"] == LO.TIGHTER
    from quant_intelligence.execution import auto_trader
    from quant_intelligence.risk.risk_engine import AccountState  # noqa: F401

    assert auto_trader.SETTINGS.risk.max_risk_per_trade_pct == new  # the same live object every read site uses


def test_loosening_needs_the_tick_and_a_reason_and_changes_nothing_without_them():
    before = _risk().max_daily_loss_pct
    target = min(before + 0.5, 5.0)
    assert not LO.apply({"max_daily_loss_pct": target}).ok
    assert not LO.apply({"max_daily_loss_pct": target}, confirm_loosen=True).ok  # no reason
    assert not LO.apply({"max_daily_loss_pct": target}, reason="ok", confirm_loosen=True).ok  # too short
    assert _risk().max_daily_loss_pct == before
    result = LO.apply({"max_daily_loss_pct": target}, reason="wider day for expiry", confirm_loosen=True)
    assert result.ok and _risk().max_daily_loss_pct == target


def test_loosening_is_refused_while_the_kill_switch_is_engaged_but_tightening_is_not():
    before = _risk().max_trades_per_day
    assert not LO.apply({"max_trades_per_day": min(before + 1, 30)}, reason="more trades", confirm_loosen=True,
                        kill_switch=True).ok
    assert LO.apply({"max_trades_per_day": max(before - 1, 1)}, kill_switch=True).ok


def test_nothing_changes_under_demo_lockdown(monkeypatch):
    from quant_intelligence.risk import limit_overrides

    monkeypatch.setattr(limit_overrides, "SETTINGS", dataclasses.replace(SETTINGS, demo_lockdown=True))
    result = limit_overrides.apply({"max_trades_per_day": 1})
    assert not result.ok and "lockdown" in result.errors[0]


def test_an_invalid_value_rejects_the_whole_change():
    before = LO.current()
    result = LO.apply({"max_daily_loss_pct": 1.0, "max_trades_per_day": 999})
    assert not result.ok and LO.current() == before


# ------------------------------------------------------------------ persistence and audit
def test_a_change_is_saved_audited_and_reloaded_after_a_restart():
    LO.apply({"max_daily_loss_pct": 0.5, "max_trades_per_day": 2}, reason="tighter week")
    with get_session() as s:
        saved = {r.field: r.value for r in s.query(RiskLimitOverride).all()}
        events = s.query(RiskEvent).filter(RiskEvent.event_type == "LIMIT_CHANGE").all()
    assert saved == {"max_daily_loss_pct": 0.5, "max_trades_per_day": 2.0}
    assert len(events) == 1 and events[0].details["changes"]["max_daily_loss_pct"]["new"] == 0.5
    assert events[0].details["reason"] == "tighter week"
    # simulate a restart: back to the environment values, then reload
    for field, value in LO.BASELINE.items():
        object.__setattr__(SETTINGS.risk, field, value)
    applied = LO.load_persisted()
    assert applied == {"max_daily_loss_pct": 0.5, "max_trades_per_day": 2}
    assert _risk().max_daily_loss_pct == 0.5 and _risk().max_trades_per_day == 2
    assert LO.recent_changes()[0]["Reason"] == "tighter week"


def test_saved_values_that_are_no_longer_valid_are_skipped_on_load():
    with get_session() as s:
        s.add(RiskLimitOverride(field="max_daily_loss_pct", value=50.0, reason="x"))  # above the ceiling
        s.add(RiskLimitOverride(field="not_a_limit", value=1.0, reason="x"))
    before = LO.current()
    assert LO.load_persisted() == {} and LO.current() == before


def test_reset_restores_the_defaults_and_forgets_the_saved_values():
    LO.apply({"max_daily_loss_pct": 0.5})
    result = LO.reset_to_defaults(reason="back to normal", confirm_loosen=True)
    assert result.ok and _risk().max_daily_loss_pct == LO.BASELINE["max_daily_loss_pct"]
    with get_session() as s:
        assert s.query(RiskLimitOverride).count() == 0
    assert not LO.reset_to_defaults().ok or LO.current() == LO.BASELINE  # already at defaults: nothing to loosen


def test_a_change_that_cannot_be_recorded_is_not_applied(monkeypatch):
    before = LO.current()

    def broken(*a, **k):
        raise RuntimeError("database down")

    monkeypatch.setattr(LO, "_store", broken)
    result = LO.apply({"max_daily_loss_pct": 0.5})
    assert not result.ok and "not applied" in result.errors[0] and LO.current() == before


def test_rows_report_which_limits_differ_from_the_defaults():
    LO.apply({"max_trades_per_day": max(_risk().max_trades_per_day - 1, 1)})
    changed = {r["field"] for r in LO.rows() if r["changed"]}
    assert changed == {"max_trades_per_day"}
    assert all(math.isfinite(r["current"]) for r in LO.rows())
