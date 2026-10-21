"""NSE and MCX are separate paper accounts with their own limits, drawdown, kill switch and audit trail."""
import pytest

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.database.db import get_session, init_db
from quant_intelligence.database.models import Position, RiskEvent, RiskLimitOverride
from quant_intelligence.execution import engine as engine_module
from quant_intelligence.execution.auto_trader import size_position
from quant_intelligence.execution.engine import ENGINE, MCX_ENGINE, engine_for, engine_for_market
from quant_intelligence.risk import limit_overrides as LO
from quant_intelligence.risk.risk_engine import AccountState, ProposedTrade, evaluate_trade, limits_for, preview_trade
from quant_intelligence.utils.market_profile import MCX, NSE


@pytest.fixture(autouse=True)
def clean_limits(tmp_path, monkeypatch):
    """Every test starts and ends with the environment's limits for both markets and no saved overrides."""
    init_db()
    start = {m: LO.current(m) for m in LO.MARKETS}
    monkeypatch.setattr(engine_module, "_PEAK_RESET_FILE", tmp_path / "peak.json")

    def wipe():
        with get_session() as s:
            s.query(RiskLimitOverride).delete()
            s.query(RiskEvent).filter(RiskEvent.event_type == "LIMIT_CHANGE").delete()

    wipe()
    yield
    for market, values in start.items():
        for field, value in values.items():
            object.__setattr__(LO._limits(market), field, value)
    wipe()


def _account(market="NSE", **kw):
    base = dict(equity=1_000_000, peak_equity=1_000_000, daily_pnl=0.0, open_positions_count=0, trades_today=0,
                exposure_by_strategy={}, total_exposure=0.0, broker_connected=True, kill_switch_engaged=False, market=market)
    base.update(kw)
    return AccountState(**base)


def _trade(**kw):
    base = dict(strategy_name="Momentum", direction="LONG", entry_price=100.0, stop_price=99.0, target_price=103.0,
                quantity=50, relative_volume=1.0, data_quality_status="OK")
    base.update(kw)
    return ProposedTrade(**base)


# ------------------------------------------------------------------ separate objects, separate engines
def test_the_two_markets_have_separate_limit_objects_and_engines():
    assert limits_for("NSE") is SETTINGS.risk and limits_for("MCX") is SETTINGS.risk_mcx
    assert SETTINGS.risk is not SETTINGS.risk_mcx
    assert limits_for(None) is SETTINGS.risk and limits_for("mcx") is SETTINGS.risk_mcx
    assert ENGINE is not MCX_ENGINE and ENGINE.broker is not MCX_ENGINE.broker and ENGINE.lock is not MCX_ENGINE.lock
    assert ENGINE.profile is NSE and MCX_ENGINE.profile is MCX
    assert engine_module.COMMODITY_RUNNER is MCX_ENGINE


def test_an_instrument_trades_in_its_own_markets_account():
    assert engine_for("CRUDEOIL") is MCX_ENGINE and engine_for("NATURALGAS") is MCX_ENGINE
    assert engine_for("NIFTY") is ENGINE and engine_for("TCS") is ENGINE and engine_for(None) is ENGINE
    assert engine_for_market("MCX") is MCX_ENGINE and engine_for_market("NSE") is ENGINE


def test_each_account_state_names_its_market_and_has_its_own_counters():
    nse = engine_module.TradingEngine()
    mcx = engine_module.TradingEngine("commodities", MCX)
    nse.add_trade(2)
    mcx.add_pnl(-500.0)
    a, b = nse.account_state(), mcx.account_state()
    assert (a.market, b.market) == ("NSE", "MCX")
    assert a.trades_today == 2 and b.trades_today == 0
    assert a.daily_pnl == 0.0 and b.daily_pnl == -500.0
    nse.kill_switch = True
    assert nse.account_state().kill_switch_engaged and not mcx.account_state().kill_switch_engaged


# ------------------------------------------------------------------ limits are chosen by the account's market
def test_a_limit_change_in_one_market_does_not_touch_the_other():
    nse_before = LO.current("NSE")
    assert LO.apply({"max_trades_per_day": 1}, market="MCX").ok
    assert LO.current("MCX")["max_trades_per_day"] == 1 and LO.current("NSE") == nse_before
    assert SETTINGS.risk.max_trades_per_day == nse_before["max_trades_per_day"]


def test_the_risk_engine_vetoes_with_the_accounts_own_market_limits():
    LO.apply({"max_trades_per_day": 1}, market="MCX")
    busy = dict(trades_today=1)
    assert not preview_trade(_account("MCX", **busy), _trade()).approved  # MCX allows 1 trade a day
    assert "trades" in preview_trade(_account("MCX", **busy), _trade()).reason.lower()
    assert preview_trade(_account("NSE", **busy), _trade()).approved  # NSE's limit is unchanged


def test_sizing_uses_the_accounts_own_market_budget():
    base = size_position(_account("NSE"), 100.0, 90.0, 1)
    assert size_position(_account("MCX"), 100.0, 90.0, 1) == base  # same starting limits, same size
    LO.apply({"max_risk_per_trade_pct": 0.05}, market="MCX")  # a tighter MCX budget
    assert size_position(_account("MCX"), 100.0, 90.0, 1) < base
    assert size_position(_account("NSE"), 100.0, 90.0, 1) == base


def test_each_market_has_its_own_kill_switch_in_the_account_it_is_checked_against():
    assert not evaluate_trade(_account("MCX", kill_switch_engaged=True), _trade()).approved
    assert preview_trade(_account("NSE", kill_switch_engaged=False), _trade()).approved


# ------------------------------------------------------------------ saved limits, audit trail
def test_saved_limits_are_kept_per_market_and_reloaded_into_the_right_market():
    LO.apply({"max_daily_loss_pct": 0.5}, market="NSE")
    LO.apply({"max_daily_loss_pct": 0.75}, market="MCX", reason="x")
    with get_session() as s:
        keys = {r.field: r.value for r in s.query(RiskLimitOverride).all()}
    assert keys == {"max_daily_loss_pct": 0.5, "MCX:max_daily_loss_pct": 0.75}  # NSE rows keep their original key
    for market in LO.MARKETS:
        for field, value in LO.BASELINES[market].items():
            object.__setattr__(LO._limits(market), field, value)
    LO.load_persisted()
    assert SETTINGS.risk.max_daily_loss_pct == 0.5 and SETTINGS.risk_mcx.max_daily_loss_pct == 0.75


def test_resetting_one_market_keeps_the_others_saved_values():
    LO.apply({"max_daily_loss_pct": 0.5}, market="NSE")
    LO.apply({"max_daily_loss_pct": 0.75}, market="MCX")
    assert LO.reset_to_defaults(market="MCX", confirm_loosen=True, reason="back to normal").ok
    assert LO.current("MCX") == LO.BASELINES["MCX"] and SETTINGS.risk.max_daily_loss_pct == 0.5
    with get_session() as s:
        assert {r.field for r in s.query(RiskLimitOverride).all()} == {"max_daily_loss_pct"}


def test_the_limit_change_log_is_filtered_by_market():
    LO.apply({"max_trades_per_day": 3}, market="NSE")
    LO.apply({"max_trades_per_day": 2}, market="MCX")
    nse, mcx = LO.recent_changes(market="NSE"), LO.recent_changes(market="MCX")
    assert len(nse) == 1 and "3" in nse[0]["Change"] and len(mcx) == 1 and "2" in mcx[0]["Change"]


def test_a_risk_decision_is_audited_under_its_own_market():
    with get_session() as s:
        s.query(RiskEvent).delete()
    evaluate_trade(_account("MCX", kill_switch_engaged=True), _trade())
    evaluate_trade(_account("NSE", kill_switch_engaged=True), _trade())
    with get_session() as s:
        markets = [e.details.get("market") for e in s.query(RiskEvent).order_by(RiskEvent.id).all()]
    assert markets == ["MCX", "NSE"]
    from quant_intelligence.ui.risk_control import _event_market

    with get_session() as s:
        assert [_event_market(e) for e in s.query(RiskEvent).order_by(RiskEvent.id).all()] == ["MCX", "NSE"]


# ------------------------------------------------------------------ the drawdown peak
def test_the_drawdown_peak_reset_is_stored_per_market(tmp_path):
    nse = engine_module.TradingEngine()
    mcx = engine_module.TradingEngine("commodities", MCX)
    nse.broker.cash = 900_000.0
    nse.reset_drawdown_peak()
    assert engine_module._load_peak_reset("NSE")["equity"] == 900_000.0
    assert engine_module._load_peak_reset("MCX") is None  # resetting NSE does not reset MCX
    mcx.broker.cash = 800_000.0
    mcx.reset_drawdown_peak()
    assert engine_module._load_peak_reset("MCX")["equity"] == 800_000.0
    assert engine_module._load_peak_reset("NSE")["equity"] == 900_000.0


# ------------------------------------------------------------------ restart recovery keeps the books apart
def test_a_restart_gives_each_account_only_its_own_markets_trades():
    import datetime as dt

    from quant_intelligence.utils.timeutil import today_ist

    today = dt.datetime.combine(today_ist(), dt.time(9, 30))

    def row(pid, instrument, pnl):
        return Position(position_id=pid, instrument=instrument, underlying=instrument, strategy_name="t", direction="LONG",
                        quantity=1, entry_price=100.0, stop_price=90.0, target_price=120.0, status="CLOSED", opened_at=today,
                        closed_at=today, net_pnl=pnl, exit_price=101.0, mode="PAPER", security_id="1", option_type="CE",
                        strike=1.0, expiry="2099-01-01", transaction="BUY")

    with get_session() as s:
        s.query(Position).delete()
        s.add_all([row("N1", "NIFTY", 1000.0), row("M1", "CRUDEOIL", -400.0), row("M2", "NATURALGAS", -100.0)])
    nse = engine_module.TradingEngine()
    mcx = engine_module.TradingEngine("commodities", MCX)
    nse.restore_from_db()
    mcx.restore_from_db()
    assert nse.trades_today == 1 and nse.daily_pnl == 1000.0 and nse.broker.cash == nse.broker.capital + 1000.0
    assert mcx.trades_today == 2 and mcx.daily_pnl == -500.0 and mcx.broker.cash == mcx.broker.capital - 500.0
    with get_session() as s:
        s.query(Position).delete()


def test_the_capital_of_each_account_comes_from_its_own_setting():
    assert ENGINE.broker.capital == SETTINGS.paper_starting_capital
    assert MCX_ENGINE.broker.capital == SETTINGS.paper_starting_capital_mcx
