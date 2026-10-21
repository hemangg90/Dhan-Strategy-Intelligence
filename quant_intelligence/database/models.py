"""
SQLAlchemy ORM models for the Quant Intelligence System.

Designed for SQLite in local dev (zero setup) and a straightforward migration
path to PostgreSQL later (no SQLite-specific types are used).
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Instrument(Base):
    __tablename__ = "instruments"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), unique=True, nullable=False)
    exchange = Column(String(16), default="NSE")
    segment = Column(String(16), default="INDEX")  # INDEX / FUT / OPT / EQ
    lot_size = Column(Integer, default=50)
    tick_size = Column(Float, default=0.05)
    created_at = Column(DateTime, default=utcnow)


class MarketDataMetadata(Base):
    __tablename__ = "market_data_metadata"

    id = Column(Integer, primary_key=True)
    instrument = Column(String(32), nullable=False)
    timeframe = Column(String(16), nullable=False)
    source = Column(String(32), nullable=False)  # csv, synthetic, dhan
    start_ts = Column(DateTime)
    end_ts = Column(DateTime)
    n_rows = Column(Integer)
    quality_status = Column(String(16))  # OK / DEGRADED / FAIL
    quality_report = Column(JSON)
    fetched_at = Column(DateTime, default=utcnow)


class MarketState(Base):
    __tablename__ = "market_states"

    id = Column(Integer, primary_key=True)
    instrument = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    features = Column(JSON, nullable=False)  # full feature vector snapshot
    data_quality_status = Column(String(16), default="OK")
    created_at = Column(DateTime, default=utcnow)


class RegimeState(Base):
    __tablename__ = "regime_states"

    id = Column(Integer, primary_key=True)
    instrument = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    label = Column(String(32), nullable=False)
    probabilities = Column(JSON, nullable=False)  # {regime_name: prob}
    confidence = Column(Float)
    created_at = Column(DateTime, default=utcnow)


class StrategyDefinition(Base):
    __tablename__ = "strategy_definitions"

    id = Column(Integer, primary_key=True)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(Text)
    parameters = Column(JSON)
    required_features = Column(JSON)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)


class StrategyObservation(Base):
    """One historical (backtested or live) occurrence of a strategy setup/trade."""

    __tablename__ = "strategy_observations"

    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(64), nullable=False, index=True)
    instrument = Column(String(32), nullable=False)
    entry_timestamp = Column(DateTime, nullable=False, index=True)
    exit_timestamp = Column(DateTime)
    market_state_features = Column(JSON, nullable=False)
    regime_label = Column(String(32))
    direction = Column(String(8))  # LONG / SHORT
    entry_price = Column(Float)
    stop_price = Column(Float)
    target_price = Column(Float)
    exit_price = Column(Float)
    r_multiple = Column(Float)
    holding_period_bars = Column(Integer)
    mfe = Column(Float)
    mae = Column(Float)
    gross_pnl = Column(Float)
    commissions = Column(Float)
    fees = Column(Float)
    slippage = Column(Float)
    net_pnl = Column(Float)
    outcome = Column(String(16))  # WIN / LOSS / SCRATCH
    source = Column(String(16), default="backtest")  # backtest / paper / live
    run_id = Column(String(64))  # links back to a backtest_runs.run_id
    created_at = Column(DateTime, default=utcnow)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True)
    run_id = Column(String(64), unique=True, nullable=False)
    strategy_name = Column(String(64), nullable=False)
    instrument = Column(String(32), nullable=False)
    timeframe = Column(String(16))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    split = Column(String(16))  # IN_SAMPLE / VALIDATION / OUT_OF_SAMPLE / WALK_FORWARD
    parameters = Column(JSON)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=utcnow)


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"

    id = Column(Integer, primary_key=True)
    run_id = Column(String(64), ForeignKey("backtest_runs.run_id"), nullable=False, index=True)
    entry_timestamp = Column(DateTime, nullable=False)
    exit_timestamp = Column(DateTime)
    direction = Column(String(8))
    entry_price = Column(Float)
    stop_price = Column(Float)
    target_price = Column(Float)
    exit_price = Column(Float)
    r_multiple = Column(Float)
    net_pnl = Column(Float)
    outcome = Column(String(16))


class StrategyMetrics(Base):
    """Aggregated (global or conditional) performance metrics for a strategy."""

    __tablename__ = "strategy_metrics"

    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(64), nullable=False, index=True)
    instrument = Column(String(32), nullable=False)
    condition_type = Column(String(16))  # GLOBAL / REGIME / ANALOGUE
    condition_key = Column(String(64))  # e.g. regime label, or 'current_state'
    sample_size = Column(Integer)
    win_rate = Column(Float)
    prob_positive_return = Column(Float)
    expected_r = Column(Float)
    median_r = Column(Float)
    avg_r = Column(Float)
    payoff_ratio = Column(Float)
    sharpe = Column(Float)
    sortino = Column(Float)
    max_drawdown = Column(Float)
    avg_mfe = Column(Float)
    avg_mae = Column(Float)
    avg_holding_bars = Column(Float)
    cost_adjusted_expectancy = Column(Float)
    confidence_label = Column(String(24))  # HIGH / MEDIUM / LOW / INSUFFICIENT_DATA
    computed_at = Column(DateTime, default=utcnow)


class AnalogueMatch(Base):
    __tablename__ = "analogue_matches"

    id = Column(Integer, primary_key=True)
    query_timestamp = Column(DateTime, nullable=False, index=True)
    instrument = Column(String(32), nullable=False)
    strategy_name = Column(String(64), nullable=False)
    observation_id = Column(Integer, ForeignKey("strategy_observations.id"))
    similarity_score = Column(Float)
    created_at = Column(DateTime, default=utcnow)


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    decision_id = Column(String(64), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    instrument = Column(String(32), nullable=False)
    strategy_name = Column(String(64), nullable=False)
    direction = Column(String(8))
    entry_price = Column(Float)
    stop_price = Column(Float)
    target_price = Column(Float)
    setup_status = Column(String(24))  # WAITING / TRIGGERED / EXPIRED
    created_at = Column(DateTime, default=utcnow)


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    order_id = Column(String(64), unique=True, nullable=False)
    decision_id = Column(String(64))
    timestamp = Column(DateTime, nullable=False)
    instrument = Column(String(32), nullable=False)
    strategy_name = Column(String(64))
    direction = Column(String(8))
    quantity = Column(Integer)
    order_type = Column(String(16))  # MARKET / LIMIT / SL / SL-M
    price = Column(Float)
    status = Column(String(24))  # PENDING/FILLED/REJECTED/CANCELLED
    reject_reason = Column(Text)
    mode = Column(String(8))  # PAPER / LIVE
    broker = Column(String(16))
    security_id = Column(String(32))
    exchange_segment = Column(String(16))
    product_type = Column(String(16))
    option_type = Column(String(4))  # CE / PE
    strike = Column(Float)
    expiry = Column(String(16))
    lot_size = Column(Integer)
    tag = Column(String(24))  # e.g. TIE-BREAK


class Fill(Base):
    __tablename__ = "fills"

    id = Column(Integer, primary_key=True)
    order_id = Column(String(64), ForeignKey("orders.order_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    fill_price = Column(Float)
    quantity = Column(Integer)
    slippage = Column(Float)


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    position_id = Column(String(64), unique=True, nullable=False)
    instrument = Column(String(32), nullable=False)
    strategy_name = Column(String(64))
    direction = Column(String(8))
    quantity = Column(Integer)
    entry_price = Column(Float)
    stop_price = Column(Float)
    target_price = Column(Float)
    status = Column(String(16))  # OPEN / CLOSED / STALE (left open by an earlier day, never closed)
    opened_at = Column(DateTime)
    closed_at = Column(DateTime)
    exit_price = Column(Float)
    # P&L convention. When `costs` is set (rows closed by the current paper broker): `gross_pnl` is the price P&L,
    # `costs` the brokerage/taxes/exit slippage, and `net_pnl` = gross_pnl - costs. When `costs` is NULL (legacy rows
    # and older exports) `net_pnl` holds the GROSS booked P&L and charges are estimated for display.
    net_pnl = Column(Float)
    gross_pnl = Column(Float)
    costs = Column(Float)
    costs_model = Column(String(24))  # OPTION_PREMIUM / BROKERAGE_ONLY
    exit_reason = Column(String(32))  # STOP / TARGET / EOD_SQUARE_OFF / MANUAL / STALE_ON_RESTART
    mode = Column(String(8))
    underlying = Column(String(32))
    security_id = Column(String(32))
    option_type = Column(String(4))  # CE / PE
    strike = Column(Float)
    expiry = Column(String(16))
    transaction = Column(String(8))  # BUY / SELL
    tag = Column(String(24))  # e.g. TIE-BREAK
    # What the system knew when it entered, so a trade can later be compared with its expectation.
    order_id = Column(String(64))  # the entry order
    decision_id = Column(String(64))  # the risk decision that approved it
    expected_r = Column(Float)  # the ranker's conditional net expected R for the strategy at entry
    confidence = Column(String(24))  # HIGH / MEDIUM / LOW ...
    regime = Column(String(32))  # market regime at entry


class RiskEvent(Base):
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=utcnow)
    decision_id = Column(String(64))
    event_type = Column(String(32))  # VETO / WARNING / KILL_SWITCH / APPROVED
    reason = Column(Text)
    details = Column(JSON)


class RiskLimitOverride(Base):
    """A risk limit changed from the app (risk/limit_overrides.py). One row per field; reloaded at start."""
    __tablename__ = "risk_limit_overrides"

    field = Column(String(48), primary_key=True)
    value = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=utcnow)
    reason = Column(Text)


class SystemEvent(Base):
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=utcnow)
    component = Column(String(32))
    level = Column(String(16))  # INFO / WARNING / ERROR
    message = Column(Text)
    details = Column(JSON)


class ResearchReport(Base):
    __tablename__ = "research_reports"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=utcnow)
    instrument = Column(String(32))
    report_type = Column(String(32))
    content_json = Column(JSON)
    content_markdown = Column(Text)


class VolForecastRow(Base):
    """A volatility forecast stored by the daily vol job (read, never computed, by trading cycles)."""

    __tablename__ = "vol_forecasts"

    id = Column(Integer, primary_key=True)
    instrument = Column(String(32), nullable=False, index=True)
    model = Column(String(16), nullable=False)  # EWMA / GARCH / GJR / EGARCH / HAR-RV
    asof = Column(DateTime)  # data used up to and including this date
    horizon_days = Column(Integer, default=1)
    sigma_1d = Column(Float)  # daily volatility, decimal (0.01 = 1%)
    sigma_horizon = Column(Float)  # total volatility over horizon_days, decimal
    selected = Column(Boolean, default=False)  # the model chosen for this instrument
    diagnostics = Column(JSON)  # params, or {"fallback": "EWMA", "reason": ...}
    created_at = Column(DateTime, default=utcnow)


class VolModelScore(Base):
    """Out-of-sample score of one model for one instrument (one evaluation run = one evaluated_at)."""

    __tablename__ = "vol_model_scores"

    id = Column(Integer, primary_key=True)
    instrument = Column(String(32), nullable=False, index=True)
    model = Column(String(16), nullable=False)
    evaluated_at = Column(DateTime, default=utcnow, index=True)
    n_obs = Column(Integer)
    qlike = Column(Float)  # lower is better
    mse = Column(Float)
    mz_alpha = Column(Float)
    mz_beta = Column(Float)
    mz_r2 = Column(Float)
    mz_p = Column(Float)  # Mincer-Zarnowitz joint test of alpha=0, beta=1
    dm_stat = Column(Float)  # Diebold-Mariano statistic vs EWMA (positive = better than EWMA)
    dm_p = Column(Float)  # one-sided p-value that the model beats EWMA
    selected = Column(Boolean, default=False)
    details = Column(JSON)


class ScanDecision(Base):
    """What the system decided for one instrument on one closed bar (strongest outcome wins). Feeds the
    signals-vs-trades funnel in the daily report."""

    __tablename__ = "scan_decisions"
    __table_args__ = (UniqueConstraint("instrument", "bar_ts", name="uq_scan_decision_bar"),)

    id = Column(Integer, primary_key=True)
    decided_at = Column(DateTime, nullable=False, index=True)  # naive IST wall clock
    bar_ts = Column(DateTime, nullable=False)  # the closed bar the decision was made on
    instrument = Column(String(32), nullable=False, index=True)
    market = Column(String(8))  # NSE / MCX
    strategy = Column(String(64))
    status = Column(String(16))  # NO_TRADE / SKIPPED / WAITING / NO_SETUP / CHAIN_ERROR / TRIGGERED / VETOED / REJECTED / FILLED
    reason_class = Column(String(32))  # short code: TIED, NO_EDGE, DATA_QUALITY, TRADE_RISK, ...
    reason = Column(Text)
    tie_break = Column(Boolean, default=False)
    expected_r = Column(Float)
    confidence = Column(String(24))
    decision_id = Column(String(64))
    order_id = Column(String(64))
    tag = Column(String(24))


class DailyReport(Base):
    """The end-of-day report for one date (re-generating the same date replaces it)."""

    __tablename__ = "daily_reports"
    __table_args__ = (UniqueConstraint("report_date", "scope", name="uq_daily_report"),)

    id = Column(Integer, primary_key=True)
    report_date = Column(Date, nullable=False, index=True)
    scope = Column(String(8), nullable=False, default="ALL")  # ALL / NSE / MCX
    generated_at = Column(DateTime, default=utcnow)
    content_json = Column(JSON)
    content_markdown = Column(Text)
