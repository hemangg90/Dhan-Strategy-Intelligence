"""
Event-driven backtesting engine.

Design principles (per project spec):
  - No look-ahead: strategies compute setups using only data up to bar i
    (enforced by the feature engine and strategy implementations).
  - No overlapping trades per strategy: once a trade is open, subsequent
    setups are ignored until it closes (sequential, realistic capital use).
  - Realistic costs: brokerage + STT-style fee + slippage are subtracted from
    every trade's gross P&L to produce net P&L.
  - Train/validation/out-of-sample separation and walk-forward are supported
    via `run_backtest(..., date_range=...)` being called on disjoint windows;
    `walk_forward` automates rolling that across multiple windows.

This is intentionally NOT a vectorized backtest - trades are simulated one
at a time via BaseStrategy.simulate_trade, which walks forward bar-by-bar
from entry, to avoid the classic vectorized-backtest bug of applying a
signal and a fill in the same timestep.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_intelligence.analytics.metrics import annualised_sharpe, max_drawdown, payoff_ratio, profit_factor
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.strategies.base_strategy import BaseStrategy, TradeResult, setup_geometry_ok


@dataclass
class BacktestResult:
    run_id: str
    strategy_name: str
    instrument: str
    split: str
    trades: list[TradeResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


MIN_PREMIUM = 0.05  # an option cannot trade below one tick


def execution_conventions(premium_model_used: bool) -> dict:
    """How this backtest turns signals into trades and money. Stored with every run so a result is never read without it.
    Nothing here is an observed fill: signals come from UNDERLYING prices and the option economics are modelled."""
    return {
        "basis": "UNDERLYING_PRICE_SIGNALS_WITH_MODELLED_OPTION_ECONOMICS",
        "signal_timing": "signal on a completed bar; entry assumed at that bar's close (no latency, no next-bar fill)",
        "same_bar_stop_and_target": "stop assumed to be hit first (conservative; OHLC cannot order events inside a bar)",
        "gap_through_stop": "filled at the worse of the stop and the bar open",
        "gap_through_target": "filled at the target (a better open is ignored)",
        "max_holding": "time exit at the close of the last allowed bar",
        "open_at_data_end": "excluded from statistics (no final outcome)",
        "invalid_setup_geometry": "skipped (stop or target on the wrong side of the entry)",
        "overlap": "one open trade per strategy at a time",
        "option_premium": ("reconstructed with a Black-Scholes style model" if premium_model_used
                           else "assumed: a fixed percentage of the underlying price, constant delta"),
        "costs": "modelled brokerage, STT, exchange, SEBI, stamp duty, GST and slippage on that assumed premium",
        "not_observed": "no historical option prices or fills are used; results are not evidence of options profitability",
    }


def market_for(instrument: str | None) -> str:
    """'NSE' or 'MCX' - decides which statutory option charges apply."""
    from quant_intelligence.utils.market_profile import profile_for

    return profile_for(instrument).name


def _apply_costs(
    trade: TradeResult, entry_price: float, quantity: int, market: str = "NSE"
) -> tuple[float, float, float, float]:
    """Return (commissions, fees, slippage, net_pnl) for trading this setup as an OPTION.

    Strategies are researched on the underlying but traded as options, so the cost model follows
    the option: brokerage per order, then STT (sell-side premium), exchange charge, SEBI fee and stamp
    duty on PREMIUM turnover, plus GST - not on the underlying's notional value, which would
    overcharge by an order of magnitude. Premium is estimated as a fixed % of the underlying price and
    moves by `delta` per underlying point (see CostAssumptions). Treated as a bought option: bought at
    entry, sold at exit. `net_pnl` is the option P&L (delta x underlying move x quantity) after costs.

    commissions = brokerage; fees = STT + exchange + SEBI + stamp + GST; slippage in premium units.
    """
    c = SETTINGS.costs
    delta = c.assumed_option_delta
    model = (trade.setup.meta or {}).get("bs_premium")  # set by run_backtest when VOL_PREMIUM_MODEL_IN_BACKTEST
    if model:
        entry_premium, exit_premium = model["entry"], model["exit"]
        option_pnl_per_unit = exit_premium - entry_premium
    else:
        entry_premium = max(c.assumed_option_premium_pct / 100.0 * entry_price, MIN_PREMIUM)
        exit_premium = max(entry_premium + delta * trade.gross_pnl, MIN_PREMIUM)
        # P&L follows the (floored) premiums, so a bought option can never lose more than its entry premium.
        option_pnl_per_unit = exit_premium - entry_premium
    buy_turnover = entry_premium * quantity
    sell_turnover = exit_premium * quantity
    turnover = buy_turnover + sell_turnover

    stt_rate = c.opt_stt_sell_rate_mcx if market == "MCX" else c.opt_stt_sell_rate_nse
    txn_rate = c.opt_txn_rate_mcx if market == "MCX" else c.opt_txn_rate_nse
    brokerage = c.brokerage_per_order_inr * 2  # entry + exit
    stt = sell_turnover * stt_rate
    txn = turnover * txn_rate
    sebi = turnover * c.sebi_rate
    stamp = buy_turnover * c.stamp_buy_rate
    gst = c.gst_rate * (brokerage + txn + sebi)

    commissions = brokerage
    fees = stt + txn + sebi + stamp + gst
    slippage_amount = c.slippage_ticks * c.tick_size * quantity * 2  # entry + exit
    net_pnl = option_pnl_per_unit * quantity - commissions - fees - slippage_amount
    return commissions, fees, slippage_amount, net_pnl


def net_r_multiple(trade: TradeResult, quantity: int, market: str = "NSE") -> float:
    """Net-of-cost R: option P&L after all costs / amount risked (delta x stop distance x quantity).
    This, not the gross `TradeResult.r_multiple`, is what a strategy actually earns."""
    net_pnl = _apply_costs(trade, trade.setup.entry_price, quantity, market)[3]
    model = (trade.setup.meta or {}).get("bs_premium")
    if model:
        risk = model["risk"] * quantity  # premium lost if the underlying reaches the stop
    else:
        risk = SETTINGS.costs.assumed_option_delta * abs(trade.setup.entry_price - trade.setup.stop_price) * quantity
    return net_pnl / risk if risk > 0 else 0.0


def max_drawdown_in_r(net_r: np.ndarray) -> float | None:
    """Worst peak-to-trough fall of cumulative net R (negative number; None with no trades)."""
    if len(net_r) == 0:
        return None
    return max_drawdown(np.cumsum(net_r))  # measured from a starting equity of 0


def run_backtest(
    strategy: BaseStrategy,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    instrument: str,
    split: str = "IN_SAMPLE",
    quantity: int = 50,
    max_holding_bars: int = 75,
    persist: bool = True,
) -> BacktestResult:
    run_id = f"BT-{uuid.uuid4().hex[:12]}"

    setups = strategy.generate_historical_setups(ohlcv, features)
    ts_to_idx = {ts: i for i, ts in enumerate(ohlcv["timestamp"])}
    feat_by_ts = features.set_index("timestamp")

    trades: list[TradeResult] = []
    next_available_idx = 0
    invalid_setups = 0
    premium_model = _premium_model(ohlcv, instrument)

    for setup in setups:
        entry_idx = ts_to_idx.get(setup.timestamp)
        if entry_idx is None or entry_idx < next_available_idx:
            continue  # skip overlapping setups while a trade would still be open

        if not setup_geometry_ok(setup):
            invalid_setups += 1  # e.g. the target is already on the wrong side of the entry: not a tradable setup
            continue

        result = strategy.simulate_trade(setup, ohlcv, entry_idx, max_holding_bars=max_holding_bars)
        if premium_model is not None:
            priced = premium_model.price(result)
            if priced:
                result.setup.meta = {**(result.setup.meta or {}), "bs_premium": priced}
        trades.append(result)

        exit_idx = ts_to_idx.get(result.exit_timestamp, entry_idx + max_holding_bars)
        next_available_idx = exit_idx + 1

    # A trade still open when the data ends has no final outcome: counting it as closed at the last price would feed a
    # partial result into the statistics (and, live, into the analogues). Drop it and say so.
    resolved = [t for t in trades if t.exit_reason != "DATA_END"]
    run = BacktestResult(run_id=run_id, strategy_name=strategy.name, instrument=instrument, split=split, trades=resolved)
    market = market_for(instrument)
    run.metrics = _compute_run_metrics(resolved, quantity, market)
    run.metrics["n_unresolved_excluded"] = len(trades) - len(resolved)
    run.metrics["conventions"] = execution_conventions(premium_model is not None)
    run.metrics["n_invalid_setups_skipped"] = invalid_setups

    if persist:
        _persist_backtest(run, strategy, ohlcv, features, feat_by_ts, quantity, market)
    return run


def _premium_model(ohlcv: pd.DataFrame, instrument: str):
    """Black-Scholes premium model when VOL_PREMIUM_MODEL_IN_BACKTEST is on; None otherwise or if it cannot be
    built (never raises into the backtest - the fixed assumption is the fallback)."""
    if not SETTINGS.vol_premium_model_in_backtest or len(ohlcv) == 0:
        return None
    try:
        from quant_intelligence.options.backtest_premium import BacktestPremiumModel

        return BacktestPremiumModel(ohlcv, instrument)
    except Exception as e:
        from quant_intelligence.utils.logging_utils import log_event

        log_event("backtest", f"Premium model unavailable for {instrument}, using the fixed assumption: {e}", level="WARNING")
        return None


MIN_DAYS_FOR_SHARPE = 10


def daily_sharpe_sortino(trades: list[TradeResult], net_rs: np.ndarray) -> tuple[float, float]:
    """Sharpe and Sortino from DAILY net R (sum of the R of trades closed that day), annualised by sqrt(252).

    Annualising per-trade R by sqrt(252) assumes one trade a day and overstates the ratio for strategies that
    trade more often. Fewer than MIN_DAYS_FOR_SHARPE trading days, or no variance, gives NaN (not enough evidence)."""
    nan = float("nan")
    if len(trades) == 0:
        return nan, nan
    days = pd.Series(net_rs).groupby(pd.to_datetime([t.exit_timestamp for t in trades]).normalize()).sum()
    if len(days) < MIN_DAYS_FOR_SHARPE:
        return nan, nan
    mean = float(days.mean())
    shared = annualised_sharpe(days.to_numpy(), 252, MIN_DAYS_FOR_SHARPE)
    sharpe = nan if shared is None else shared
    downside_dev = float(np.sqrt(np.mean(np.minimum(days.to_numpy(), 0.0) ** 2)))
    sortino = mean / downside_dev * np.sqrt(252) if downside_dev > 0 else nan
    return float(sharpe), float(sortino)


def _compute_run_metrics(trades: list[TradeResult], quantity: int, market: str = "NSE") -> dict:
    if not trades:
        return {"n_trades": 0, "status": "NO_TRADES"}

    r_multiples = np.array([t.r_multiple for t in trades])
    net_pnls = []
    for t in trades:
        _, _, _, net_pnl = _apply_costs(t, t.setup.entry_price, quantity, market)
        net_pnls.append(net_pnl)
    net_pnls = np.array(net_pnls)

    wins = r_multiples > 0
    win_rate = float(wins.mean())
    # Payoff = avg win / |avg loss| over genuine winners and losers (a scratch at exactly 0R is neither).
    payoff = payoff_ratio(r_multiples)
    payoff_value = float("nan") if payoff is None else payoff

    net_rs = np.array([net_r_multiple(t, quantity, market) for t in trades])
    equity_curve = np.cumsum(net_pnls)
    max_dd = max_drawdown(equity_curve)  # from the starting equity, so early losses count

    sharpe, sortino = daily_sharpe_sortino(trades, net_rs)

    return {
        "n_trades": len(trades),
        "win_rate": win_rate,
        "prob_positive_return": float((net_pnls > 0).mean()),
        "expected_r": float(r_multiples.mean()),
        "median_r": float(np.median(r_multiples)),
        "avg_r": float(r_multiples.mean()),
        "payoff_ratio": payoff_value,
        "profit_factor": profit_factor(net_pnls),  # net of costs; None when there is no losing trade
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,  # rupees (display)
        "max_drawdown_r": max_drawdown_in_r(net_rs),  # R units (used by the ranker)
        "net_expected_r": float(net_rs.mean()),
        "avg_mfe": float(np.mean([t.mfe for t in trades])),
        "avg_mae": float(np.mean([t.mae for t in trades])),
        "avg_holding_bars": float(np.mean([t.holding_period_bars for t in trades])),
        "net_pnl_total": float(net_pnls.sum()),
        "cost_adjusted_expectancy": float(net_pnls.mean()),
    }


def _persist_backtest(
    run: BacktestResult, strategy: BaseStrategy, ohlcv, features, feat_by_ts, quantity: int, market: str = "NSE"
) -> None:
    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import BacktestRun, BacktestTrade, StrategyObservation

        with get_session() as session:
            session.add(
                BacktestRun(
                    run_id=run.run_id,
                    strategy_name=run.strategy_name,
                    instrument=run.instrument,
                    timeframe=None,
                    start_date=ohlcv["timestamp"].min() if len(ohlcv) else None,
                    end_date=ohlcv["timestamp"].max() if len(ohlcv) else None,
                    split=run.split,
                    parameters=strategy.parameters,
                    metrics=run.metrics,
                )
            )

            for t in run.trades:
                commissions, fees, slippage, net_pnl = _apply_costs(t, t.setup.entry_price, quantity, market)
                session.add(
                    BacktestTrade(
                        run_id=run.run_id,
                        entry_timestamp=t.setup.timestamp,
                        exit_timestamp=t.exit_timestamp,
                        direction=t.setup.direction,
                        entry_price=t.setup.entry_price,
                        stop_price=t.setup.stop_price,
                        target_price=t.setup.target_price,
                        exit_price=t.exit_price,
                        r_multiple=t.r_multiple,
                        net_pnl=net_pnl,
                        outcome=t.outcome,
                    )
                )

                feat_row = feat_by_ts.loc[t.setup.timestamp] if t.setup.timestamp in feat_by_ts.index else None
                market_state_features = _row_to_clean_dict(feat_row) if feat_row is not None else {}
                regime_label = None
                if feat_row is not None:
                    from quant_intelligence.regimes.regime_engine import classify_regime

                    regime_label, _ = classify_regime(market_state_features)

                session.add(
                    StrategyObservation(
                        strategy_name=strategy.name,
                        instrument=run.instrument,
                        entry_timestamp=t.setup.timestamp,
                        exit_timestamp=t.exit_timestamp,
                        market_state_features=market_state_features,
                        regime_label=regime_label,
                        direction=t.setup.direction,
                        entry_price=t.setup.entry_price,
                        stop_price=t.setup.stop_price,
                        target_price=t.setup.target_price,
                        exit_price=t.exit_price,
                        r_multiple=t.r_multiple,
                        holding_period_bars=t.holding_period_bars,
                        mfe=t.mfe,
                        mae=t.mae,
                        gross_pnl=t.gross_pnl * quantity,
                        commissions=commissions,
                        fees=fees,
                        slippage=slippage,
                        net_pnl=net_pnl,
                        outcome="WIN" if t.r_multiple > 0 else ("LOSS" if t.r_multiple < 0 else "SCRATCH"),
                        source="backtest",
                        run_id=run.run_id,
                    )
                )
    except Exception:
        # Persistence must never crash a research run; caller still has run.trades/metrics in memory.
        pass


def _row_to_clean_dict(row: pd.Series) -> dict:
    d = row.to_dict()
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def walk_forward(
    strategy_factory,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    instrument: str,
    n_folds: int = 4,
    quantity: int = 50,
) -> list[BacktestResult]:
    """Rolling walk-forward: split the data into n_folds contiguous chronological
    windows; each fold's data is treated as a fresh out-of-sample test window.
    strategy_factory() must return a new strategy instance (parameters fixed in
    advance, i.e. NOT re-optimized per fold, to avoid look-ahead parameter fitting).
    """
    n = len(ohlcv)
    fold_size = n // n_folds
    results = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = n if fold == n_folds - 1 else (fold + 1) * fold_size
        fold_ohlcv = ohlcv.iloc[start:end].reset_index(drop=True)
        fold_features = features.iloc[start:end].reset_index(drop=True)
        if len(fold_ohlcv) < 50:
            continue
        strategy = strategy_factory()
        result = run_backtest(
            strategy, fold_ohlcv, fold_features, instrument, split=f"WALK_FORWARD_FOLD_{fold+1}", quantity=quantity
        )
        results.append(result)
    return results
