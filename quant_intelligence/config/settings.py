"""
Central configuration for the Quant Intelligence System.

Everything reads from environment variables (loaded from .env via python-dotenv).
Sensible, conservative defaults are provided so the app runs safely out of the box
in PAPER mode with SQLite, even if .env is missing entirely.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"
LOGS_DIR = Path(os.environ["LOGS_DIR"]) if os.environ.get("LOGS_DIR") else PROJECT_ROOT / "logs"  # tests point this at a temp dir
DATA_CACHE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "y")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


DEFAULT_MAX_RISK_PER_TRADE_PCT = 0.2  # was 1.0: ~Rs 8,000-9,000 lost per stopped trade on Rs 10 lakh
DEFAULT_MAX_DAILY_LOSS_PCT = 1.5  # was 3.0


@dataclass(frozen=True)
class RiskLimits:
    # What ONE stopped trade may cost, as % of equity. Sizing divides this budget by the stop distance, so it fixes
    # the rupee loss of a stop-out whatever the stop width: 0.2% is Rs 2,000 on Rs 10 lakh (sizing uses 90% of it,
    # about Rs 1,800). An instrument whose single lot already risks more than the budget does not trade.
    max_risk_per_trade_pct: float = _float_env("MAX_RISK_PER_TRADE_PCT", DEFAULT_MAX_RISK_PER_TRADE_PCT)
    # Ceiling on the capital (premium x quantity) committed to ONE trade, as % of equity. Risk-based
    # sizing alone lets a tight stop produce a huge quantity; 0 disables the cap.
    max_capital_per_trade_pct: float = _float_env("MAX_CAPITAL_PER_TRADE_PCT", 5.0)
    max_daily_loss_pct: float = _float_env("MAX_DAILY_LOSS_PCT", DEFAULT_MAX_DAILY_LOSS_PCT)
    max_strategy_exposure_pct: float = _float_env("MAX_STRATEGY_EXPOSURE_PCT", 10.0)
    max_portfolio_exposure_pct: float = _float_env("MAX_PORTFOLIO_EXPOSURE_PCT", 20.0)
    max_trades_per_day: int = _int_env("MAX_TRADES_PER_DAY", 6)
    max_drawdown_pct: float = _float_env("MAX_DRAWDOWN_PCT", 8.0)

    # Drawdown control (see risk_engine.risk_multiplier and evaluate_trade). A count/percentage of 0
    # disables that check.
    # Size throttle: full size below DD_THROTTLE_START_PCT drawdown, shrinking linearly to
    # DD_THROTTLE_FLOOR x size at MAX_DRAWDOWN_PCT (where the hard veto takes over).
    dd_throttle_start_pct: float = _float_env("DD_THROTTLE_START_PCT", 3.0)
    dd_throttle_floor: float = _float_env("DD_THROTTLE_FLOOR", 0.25)
    # Half size after this many consecutive losing trades today, or once half the daily loss limit is gone.
    loss_streak_throttle_after: int = _int_env("LOSS_STREAK_THROTTLE_AFTER", 2)
    max_open_positions: int = _int_env("MAX_OPEN_POSITIONS", 4)
    # Correlated instruments (config/risk_groups.py): at most this many same-direction positions per
    # group, and at most this much open risk per group as % of equity.
    max_group_positions: int = _int_env("MAX_GROUP_POSITIONS", 2)
    max_group_risk_pct: float = _float_env("MAX_GROUP_RISK_PCT", 2.0)
    # After this many consecutive losers today, no new entries for LOSS_STREAK_PAUSE_MIN minutes.
    loss_streak_pause: int = _int_env("LOSS_STREAK_PAUSE", 3)
    loss_streak_pause_min: int = _int_env("LOSS_STREAK_PAUSE_MIN", 60)
    # A strategy that has lost this many trades today is done for the day.
    strategy_max_losses_per_day: int = _int_env("STRATEGY_MAX_LOSSES_PER_DAY", 2)
    # Once the day's P&L has reached PROFIT_LOCK_TRIGGER_PCT of equity, stop opening trades if it gives
    # back PROFIT_LOCK_GIVEBACK (fraction) of that day's peak.
    profit_lock_trigger_pct: float = _float_env("PROFIT_LOCK_TRIGGER_PCT", 1.5)
    profit_lock_giveback: float = _float_env("PROFIT_LOCK_GIVEBACK", 0.5)


@dataclass(frozen=True)
class CostAssumptions:
    # Brokerage: Dhan charges a flat Rs 20 per executed order for equity and commodity F&O
    # (https://dhan.co/pricing/).
    brokerage_per_order_inr: float = _float_env("BROKERAGE_PER_ORDER_INR", 20.0)
    stt_rate: float = _float_env("STT_RATE", 0.0005)  # legacy; the option cost model below is used
    slippage_ticks: int = _int_env("SLIPPAGE_TICKS", 1)
    tick_size: float = _float_env("TICK_SIZE", 0.05)

    # Statutory charges on OPTIONS premium. These are set by the exchanges/government and are the
    # same at every broker; values from the published charges table at https://zerodha.com/charges/
    # (Dhan's own pricing page lists only brokerage, GST and the SEBI fee). Verify against your
    # contract note and override via environment variables if they differ.
    opt_stt_sell_rate_nse: float = _float_env("OPT_STT_SELL_RATE_NSE", 0.0015)  # 0.15% of sell premium
    opt_stt_sell_rate_mcx: float = _float_env("OPT_STT_SELL_RATE_MCX", 0.0005)  # 0.05% of sell premium
    opt_txn_rate_nse: float = _float_env("OPT_TXN_RATE_NSE", 0.0003553)  # 0.03553% of premium turnover
    opt_txn_rate_mcx: float = _float_env("OPT_TXN_RATE_MCX", 0.000418)  # 0.0418% of premium turnover
    sebi_rate: float = _float_env("SEBI_RATE", 0.000001)  # Rs 10 per crore of turnover
    stamp_buy_rate: float = _float_env("STAMP_BUY_RATE", 0.00003)  # 0.003% of buy-side premium
    gst_rate: float = _float_env("GST_RATE", 0.18)  # on brokerage + exchange charges + SEBI fee

    # The backtest only has the UNDERLYING's prices, so the option premium traded is an assumption:
    # premium per unit ~= OPTION_PREMIUM_PCT % of the underlying price, and the option moves
    # OPTION_DELTA points per underlying point. Costs and R are expressed on that basis.
    assumed_option_premium_pct: float = _float_env("OPTION_PREMIUM_PCT", 1.5)
    assumed_option_delta: float = _float_env("OPTION_DELTA", 0.5)


@dataclass(frozen=True)
class DhanLimits:
    """Client-side budget for Dhan API calls. Dhan's published limits apply to the whole ACCOUNT:
    orders 10/s, data (candles) 5/s, quote (LTP) 1/s, non-trading 20/s, option chain 1 per 3 s
    (https://dhanhq.co/docs/v2/). Minimum spacings below keep ~30% headroom under each."""

    min_interval_quote: float = _float_env("DHAN_MIN_INTERVAL_QUOTE_SEC", 1.25)  # limit 1/s
    min_interval_data: float = _float_env("DHAN_MIN_INTERVAL_DATA_SEC", 0.35)  # limit 5/s
    min_interval_non_trading: float = _float_env("DHAN_MIN_INTERVAL_NONTRADING_SEC", 0.1)  # limit 20/s
    min_interval_orders: float = _float_env("DHAN_MIN_INTERVAL_ORDERS_SEC", 0.17)  # limit 10/s
    min_interval_option_chain: float = _float_env("DHAN_MIN_INTERVAL_CHAIN_SEC", 3.5)  # limit 1 per 3 s

    # HTTP 429 handling for read calls: retry with backoff; after `breaker_threshold` consecutive 429s
    # stop sending reads for a short cooldown (doubling up to the max) so the account is not blocked.
    max_attempts: int = _int_env("DHAN_429_MAX_ATTEMPTS", 3)
    backoff_sec: float = _float_env("DHAN_429_BACKOFF_SEC", 6.0)
    breaker_threshold: int = _int_env("DHAN_BREAKER_THRESHOLD", 3)
    breaker_cooldown_sec: float = _float_env("DHAN_BREAKER_COOLDOWN_SEC", 30.0)
    breaker_max_cooldown_sec: float = _float_env("DHAN_BREAKER_MAX_COOLDOWN_SEC", 120.0)
    # After Dhan rejects the credentials (HTTP 401), send NOTHING with that same token for this long.
    # Retrying a rejected token on every cycle for every instrument is itself abusive; a new token
    # (changed in the sidebar/secrets) is picked up immediately.
    auth_failure_cooldown_sec: float = _float_env("DHAN_AUTH_FAILURE_COOLDOWN_SEC", 300.0)

    # Shared caches: every tab and both background runners reuse one response within the TTL.
    quote_cache_ttl_sec: float = _float_env("DHAN_QUOTE_CACHE_TTL_SEC", 2.0)
    chain_cache_ttl_sec: float = _float_env("DHAN_CHAIN_CACHE_TTL_SEC", 15.0)


@dataclass(frozen=True)
class Settings:
    trading_mode: str = os.getenv("TRADING_MODE", "PAPER").strip().upper()
    trading_live_confirm: str = os.getenv("TRADING_LIVE_CONFIRM", "").strip()
    # Presentation lock: forces paper mode whatever TRADING_MODE / TRADING_LIVE_CONFIRM say, and hides
    # credentials in the UI. Set DEMO_LOCKDOWN=1 for demos and screen shares.
    demo_lockdown: bool = _bool_env("DEMO_LOCKDOWN", False)

    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{(DATA_CACHE_DIR / 'quant_intelligence.db').as_posix()}"
    )

    dhan_client_id: str = os.getenv("DHAN_CLIENT_ID", "")
    dhan_access_token: str = os.getenv("DHAN_ACCESS_TOKEN", "")
    dhan_base_url: str = os.getenv("DHAN_BASE_URL", "https://api.dhan.co/v2")

    # Options trading (see quant_intelligence/options/)
    option_underlyings: tuple[str, ...] = tuple(
        s.strip().upper() for s in os.getenv("OPTION_UNDERLYINGS", "NIFTY,BANKNIFTY,FINNIFTY,MIDCPNIFTY,SENSEX").split(",") if s.strip()
    )
    option_moneyness_offset: int = _int_env("OPTION_MONEYNESS_OFFSET", 0)  # 0=ATM, +N = N strikes OTM
    # How the strike is chosen: OFFSET = ATM +/- OPTION_MONEYNESS_OFFSET; BEST = rank the whole chain on delta,
    # bid-ask spread, liquidity and decay (options/strike_ranker.py). Anything else falls back to OFFSET.
    option_selection: str = "BEST" if os.getenv("OPTION_SELECTION", "").strip().upper() == "BEST" else "OFFSET"
    option_expiry_preference: str = os.getenv("OPTION_EXPIRY_PREFERENCE", "NEAREST").strip().upper()
    option_product_type: str = os.getenv("OPTION_PRODUCT_TYPE", "INTRADAY").strip().upper()

    default_instrument: str = os.getenv("DEFAULT_INSTRUMENT", "NIFTY")
    default_timeframe: str = os.getenv("DEFAULT_TIMEFRAME", "5min")

    paper_starting_capital: float = _float_env("PAPER_STARTING_CAPITAL", 1_000_000.0)  # the NSE paper account
    # The MCX paper account is separate from NSE's. Defaults to the same size.
    paper_starting_capital_mcx: float = _float_env("PAPER_STARTING_CAPITAL_MCX", _float_env("PAPER_STARTING_CAPITAL", 1_000_000.0))

    # Tie-break: when the best two strategies are statistically tied but the top one is eligible, trade it
    # anyway - tagged "TIE-BREAK" and sized down - instead of standing aside. Live trading stays manual
    # (confirmation + risk engine + double gate); this only decides whether the tied pick is offered.
    tie_break_paper: bool = _bool_env("TIE_BREAK_PAPER", True)
    # Off by default: a tie-break pick has no demonstrated edge, so it is never sent to a live account unless explicitly enabled.
    tie_break_live: bool = _bool_env("TIE_BREAK_LIVE", False)
    tie_break_size_factor: float = _float_env("TIE_BREAK_SIZE_FACTOR", 0.5)

    # Background data keeper: refreshes every watchlist instrument's candles on its own (data/data_keeper.py).
    data_keeper_enabled: bool = _bool_env("DATA_KEEPER", True)
    data_keeper_interval_sec: int = _int_env("DATA_KEEPER_INTERVAL_SEC", 60)

    # End-of-day report (reports/eod_job.py): built by the data keeper after each market closes.
    eod_report_enabled: bool = _bool_env("EOD_REPORT", True)
    eod_force_square_off: bool = _bool_env("EOD_FORCE_SQUARE_OFF", True)  # close paper positions still open after the close
    log_retention_days: int = _int_env("LOG_RETENTION_DAYS", 30)  # system_events / market_data_metadata
    decision_retention_days: int = _int_env("DECISION_RETENTION_DAYS", 90)  # scan_decisions

    # Stop tightness: multiplies the DEFAULT stop_atr_mult and target_atr_mult of every ATR-based strategy (stop and
    # target together, so reward:risk is unchanged). 1.0 = as designed; 0.5 = stops and targets half as far. Values
    # given explicitly (Strategy Library / Backtest Lab) are never scaled. Clamped to 0.2-3.0.
    stop_atr_scale: float = min(max(_float_env("STOP_ATR_SCALE", 1.0), 0.2), 3.0)

    # Heatmap mover selection (execution/mover_selection.py): when on, each auto paper trader scans only the N biggest
    # gainers and N biggest losers per group, and an up-mover takes only bullish setups, a down-mover only bearish.
    # Off = scan everything as before. The page "NSE Heatmap" can also switch it on for the running process.
    mover_selection_enabled: bool = _bool_env("MOVER_SELECTION_ENABLED", False)
    mover_top_n: int = min(max(_int_env("MOVER_TOP_N", 5), 1), 15)
    mover_min_abs_pct: float = max(_float_env("MOVER_MIN_ABS_PCT", 0.3), 0.0)
    mover_max_age_minutes: float = max(_float_env("MOVER_MAX_AGE_MINUTES", 20.0), 1.0)
    # "nse": NSE stocks are every F&O stock on NSE's live heatmap (falls back to the watchlist when NSE is unreachable);
    # "watchlist": only the 15 watchlist stocks, read from the candle cache.
    mover_universe: str = "watchlist" if os.getenv("MOVER_UNIVERSE", "nse").strip().lower() == "watchlist" else "nse"
    mover_nse_refresh_minutes: float = max(_float_env("MOVER_NSE_REFRESH_MINUTES", 2.0), 0.5)  # how often NSE is asked

    # Volatility layer (quant_intelligence/volatility/). Every behaviour change is OFF by default; the pure
    # computation modules (estimators, Black-Scholes) are always importable.
    vol_models_enabled: bool = _bool_env("VOL_MODELS_ENABLED", False)
    vol_features_in_analogues: bool = _bool_env("VOL_FEATURES_IN_ANALOGUES", False)
    # "legacy" (6 features, z-score, plain nearest 30) or "robust" (see docs/REGIME_AND_ANALOGUES.md). Chosen on the walk-forward
    # evaluation, not on taste: the default only changes if the evidence there supports it.
    analogue_method: str = os.getenv("ANALOGUE_METHOD", "legacy").strip().lower()
    vol_premium_model_in_backtest: bool = _bool_env("VOL_PREMIUM_MODEL_IN_BACKTEST", False)
    vol_target_sizing: bool = _bool_env("VOL_TARGET_SIZING", False)
    vol_risk_check: bool = _bool_env("VOL_RISK_CHECK", False)
    vol_iv_gate: bool = _bool_env("VOL_IV_GATE", False)
    vol_refit_every_days: int = _int_env("VOL_REFIT_EVERY_DAYS", 5)  # GARCH/HAR parameter re-estimation schedule
    vol_history_years: int = _int_env("VOL_HISTORY_YEARS", 5)  # daily history requested for the vol models
    vol_min_train_days: int = _int_env("VOL_MIN_TRAIN_DAYS", 500)
    vol_min_eval_days: int = _int_env("VOL_MIN_EVAL_DAYS", 250)
    vol_dm_alpha: float = _float_env("VOL_DM_ALPHA", 0.10)  # a model must beat EWMA at this one-sided level
    risk_free_rate: float = _float_env("RISK_FREE_RATE", 0.065)  # continuous, annual; India ~6-7%
    dividend_yield: float = _float_env("DIVIDEND_YIELD", 0.012)  # index/stock carry; MCX uses q = r
    trading_days_per_year: int = _int_env("TRADING_DAYS_PER_YEAR", 252)

    # Token keeper (brokers/dhan_auth.py): renews the 24h Dhan token before it expires. An expired token has to be
    # entered by hand.
    token_keeper_enabled: bool = _bool_env("TOKEN_KEEPER", True)
    token_renew_before_hours: float = _float_env("TOKEN_RENEW_BEFORE_HOURS", 8.0)

    # Automation
    auto_trade_refresh_seconds: int = _int_env("AUTO_TRADE_REFRESH_SECONDS", 60)
    ltp_refresh_seconds: int = _int_env("LTP_REFRESH_SECONDS", 5)
    # Close each market's open paper positions shortly before its close (see execution/square_off.py).
    eod_square_off: bool = _bool_env("EOD_SQUARE_OFF", True)
    eod_square_off_minutes: int = _int_env("EOD_SQUARE_OFF_MINUTES", 5)

    # NSE heatmap-sourced stock universe (see data_adapters/nse_heatmap.py)
    heatmap_index: str = os.getenv("HEATMAP_INDEX", "NIFTY 50")
    heatmap_refresh_minutes: int = _int_env("HEATMAP_REFRESH_MINUTES", 15)
    heatmap_universe_top_n: int = _int_env("HEATMAP_UNIVERSE_TOP_N", 200)

    risk: RiskLimits = field(default_factory=RiskLimits)  # NSE limits
    # MCX has its own limit set (separate instance, same starting values). Edited on the MCX Risk Control page.
    risk_mcx: RiskLimits = field(default_factory=RiskLimits)
    costs: CostAssumptions = field(default_factory=CostAssumptions)
    dhan: DhanLimits = field(default_factory=DhanLimits)

    @property
    def is_live_mode(self) -> bool:
        return self.trading_mode == "LIVE" and not self.demo_lockdown

    @property
    def live_mode_fully_authorized(self) -> bool:
        """LIVE mode requires BOTH the mode flag AND an explicit confirmation phrase.

        This is a deliberate double-gate: a single environment variable
        (which could be flipped accidentally) is never enough to enable
        real order flow.
        """
        return (
            not self.demo_lockdown
            and self.is_live_mode
            and self.trading_live_confirm == "YES_I_UNDERSTAND_THE_RISK"
        )


SETTINGS = Settings()
