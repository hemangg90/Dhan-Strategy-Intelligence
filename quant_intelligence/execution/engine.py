"""Process-wide paper-trading engine.

Each market (NSE, MCX) has its own TradingEngine: its own PaperBroker (cash), daily risk counters, drawdown
peak, kill switch and risk limits, and runs the auto-trading scan in a background thread. Because it lives at module level (not in a
Streamlit session or page), switching pages - or closing the browser tab - does not stop it;
only `stop()` (or the server restarting / the cloud app sleeping) does.

`ENGINE` is the NSE account. `MCX_ENGINE` (also exported as `COMMODITY_RUNNER`) is the separate MCX account: it scans
commodities on its own thread and session hours and shares nothing with NSE. A trade is always checked and sized against
the account and limits of its own market.

Streamlit-free on purpose: a background thread has no ScriptRunContext, so nothing here
may call `st.*`.
"""
from __future__ import annotations

import datetime as dt
import json
import threading

from quant_intelligence.brokers.dhan_api_client import DhanApiClient
from quant_intelligence.brokers.paper_broker import PaperBroker
from quant_intelligence.config.settings import DATA_CACHE_DIR, SETTINGS
from quant_intelligence.execution.mover_selection import MoverConfig, Selection, select as select_movers
from quant_intelligence.execution.multi_cycle import run_multi_instrument_cycle
from quant_intelligence.execution.square_off import in_close_window, square_off_positions
from quant_intelligence.risk.risk_engine import AccountState
from quant_intelligence.utils.logging_utils import log_event
from quant_intelligence.utils.market_profile import MCX, NSE, MarketProfile, profile_for
from quant_intelligence.utils.timeutil import is_market_open, now_ist, today_ist


class ScanRunner:
    """A background scan loop over one watchlist in one market session.

    The NSE and MCX runners run at once, each a TradingEngine that owns its broker, cash, daily risk counters and kill
    switch. A runner holds only its own `scan_lock` for the whole scan; the owner's short `lock` is taken just around
    broker mutations (see multi_cycle), so a long NSE scan never blocks the commodity runner.
    """

    def __init__(self, name: str, profile: MarketProfile, owner: "TradingEngine | None" = None) -> None:
        self.name = name
        self.profile = profile
        self.owner = owner if owner is not None else self  # TradingEngine owns its own state
        self.scan_lock = threading.RLock()

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.running = False
        self.started_at = None
        self.cycles = 0
        self.last_cycle_at = None
        self.last_rows: list[dict] = []
        self.last_status = "Not started"
        self.last_error: str | None = None
        self.market_hours_only = True

        # Heatmap mover selection (see execution/mover_selection.py). None = scan everything, as always.
        self._mover_lock = threading.Lock()
        self._mover_config: MoverConfig | None = MoverConfig.from_settings() if SETTINGS.mover_selection_enabled else None
        self._mover_frozen: Selection | None = None  # the selection made when applied with auto-refresh off
        self.mover_selection: Selection | None = None  # what the last scan used (for the pages)

    # ---- heatmap mover selection -----------------------------------------------
    @property
    def mover_config(self) -> MoverConfig | None:
        return self._mover_config

    def set_mover_config(self, config: MoverConfig, frozen: Selection | None = None) -> None:
        """Scan only the heatmap's movers from the next cycle. With `config.auto_refresh` the selection is recomputed
        every cycle; otherwise `frozen` (made when the user pressed Apply) is kept until cleared."""
        with self._mover_lock:
            self._mover_config = config
            self._mover_frozen = None if config.auto_refresh else frozen
            self.mover_selection = frozen
        log_event("engine", f"Heatmap mover selection ON ({self.name}): top {config.n} per side, min move {config.min_abs_pct:g}%, "
                            f"{'refreshed every cycle' if config.auto_refresh else 'fixed until cleared'}", level="INFO")

    def clear_mover_config(self) -> None:
        with self._mover_lock:
            self._mover_config = None
            self._mover_frozen = None
            self.mover_selection = None
        log_event("engine", f"Heatmap mover selection OFF ({self.name}): scanning every instrument", level="INFO")

    def _resolve_selection(self, index_symbols, stock_symbols) -> Selection | None:
        with self._mover_lock:
            config, frozen = self._mover_config, self._mover_frozen
        if config is None:
            return None
        if frozen is not None:
            return frozen
        sel = select_movers(index_symbols, stock_symbols, "Commodity" if self.profile.name == "MCX" else "Stock", config)
        if not config.auto_refresh:
            with self._mover_lock:
                self._mover_frozen = sel
        return sel

    # ---- scanning ------------------------------------------------------------
    def run_cycle(self, index_symbols, stock_symbols, timeframe: str, lookback_days: int) -> list[dict]:
        """One full scan. Safe to call from the UI thread ('Run one cycle now') or the worker."""
        owner = self.owner
        with self.scan_lock:
            selection = self._resolve_selection(index_symbols, stock_symbols)
            self.mover_selection = selection
            indices, stocks = list(index_symbols), list(stock_symbols)
            if selection is not None:  # nothing selected -> nothing new is scanned; open positions are still monitored
                indices = selection.only(indices)
                # NSE's live heatmap can pick F&O stocks that are not on the watchlist: scan those too.
                stocks = selection.in_group("Stock") if selection.source == "nse_live" else selection.only(stocks)
            # Fresh client each cycle so a token updated in the sidebar/secrets applies immediately.
            rows, pnl_delta = run_multi_instrument_cycle(
                indices, stocks, timeframe, lookback_days,
                owner.broker, DhanApiClient(), owner.account_state,
                broker_lock=owner.lock, on_fill=owner.add_trade, selection=selection,
            )
            if selection is not None and not selection.usable:
                rows.insert(0, {"symbol": "-", "type": "selection", "strategy": "-", "status": "NO_SELECTION", "detail": selection.reason})
            owner.add_pnl(pnl_delta)
            self.last_rows = rows
            self.last_cycle_at = now_ist()
            self.cycles += 1
            return rows

    def square_off_now(self) -> list[dict]:
        """Close this market's open positions at their latest price (end of session, or on demand).
        Safe to call from the UI thread; prices are fetched outside the broker lock."""
        owner = self.owner
        return square_off_positions(owner.broker, DhanApiClient(), self.profile, owner.lock, owner.add_pnl)

    def start(self, index_symbols, stock_symbols, timeframe: str, lookback_days: int,
              interval_seconds: int | None = None, market_hours_only: bool = True) -> bool:
        from quant_intelligence.data.data_keeper import DATA_KEEPER

        DATA_KEEPER.start()  # candles stay current even between this runner's own scans
        with self.owner.lock:
            if self.running:
                return False
            self.market_hours_only = market_hours_only
            self._stop = threading.Event()  # fresh event per run: an old worker can never be revived
            self.running = True
            self.started_at = now_ist()
            self.last_error = None
            self.last_status = "Starting"
            interval = interval_seconds or SETTINGS.auto_trade_refresh_seconds
            self._thread = threading.Thread(
                target=self._loop,
                args=(self._stop, list(index_symbols), list(stock_symbols), timeframe, lookback_days, interval),
                name=f"auto-paper-trader-{self.name}",
                daemon=True,
            )
            self._thread.start()
        log_event("engine", f"Auto paper trading started ({self.name})", level="INFO")
        return True

    def stop(self) -> None:
        # No lock: must return immediately even while a long scan cycle holds it.
        self._stop.set()
        self.running = False
        self.last_status = "Stopped"
        log_event("engine", f"Auto paper trading stopped ({self.name})", level="INFO")

    def _loop(self, stop, index_symbols, stock_symbols, timeframe, lookback_days, interval) -> None:
        while not stop.is_set():
            try:
                market_open = is_market_open(profile=self.profile)
                closing = SETTINGS.eod_square_off and in_close_window(
                    now_ist(), self.profile, SETTINGS.eod_square_off_minutes
                )
                if self.market_hours_only and SETTINGS.eod_square_off and (closing or not market_open):
                    # Session ending or over: flatten this market's positions (also catches ones left
                    # from before a restart) instead of leaving them open with a frozen price.
                    self.square_off_now()
                if self.market_hours_only and not market_open:
                    self.last_status = f"Market closed - waiting for {self.profile.open:%H:%M} IST"
                elif self.market_hours_only and closing:
                    # No new trades in the last minutes: they would be squared off again immediately.
                    self.last_status = "Closing window - positions squared off, no new trades"
                elif self.owner.kill_switch:
                    self.last_status = "Kill switch engaged - not trading"
                else:
                    self.last_status = "Scanning"
                    self.run_cycle(index_symbols, stock_symbols, timeframe, lookback_days)
                    self.last_status = "Idle until next cycle"
                    self.last_error = None
            except Exception as e:  # a bad cycle must never kill the worker
                self.last_error = f"{type(e).__name__}: {e}"
                self.last_status = "Last cycle failed - will retry"
                log_event("engine", f"Auto cycle failed ({self.name}): {e}", level="ERROR")
            stop.wait(interval)

    def status(self) -> dict:
        alive = bool(self._thread and self._thread.is_alive())
        return {
            "running": self.running and alive,
            "started_at": self.started_at,
            "cycles": self.cycles,
            "last_cycle_at": self.last_cycle_at,
            "last_status": self.last_status,
            "last_error": self.last_error,
        }


class TradingEngine(ScanRunner):
    """One market's runner, which also owns that market's broker, risk counters and kill switch."""

    def __init__(self, name: str = "nse", profile: MarketProfile = NSE) -> None:
        super().__init__(name, profile)
        capital = SETTINGS.paper_starting_capital_mcx if profile.name == "MCX" else SETTINGS.paper_starting_capital
        self.broker = PaperBroker(starting_capital=capital)
        self.lock = threading.RLock()
        self.kill_switch = False
        self.trades_today = 0
        self.daily_pnl = 0.0  # realised today
        self.day_peak_pnl = 0.0  # highest marked-to-market daily P&L today (profit giveback lock)
        self.peak_equity = capital  # carried across days (drawdown is multi-day)
        self._day = today_ist()
        self._restored = False

    # ---- restart recovery ------------------------------------------------------
    def restore_from_db(self) -> None:
        """Rebuild today's book from the database after a process restart (idempotent).

        The broker lives in memory, so a restart used to empty it while the database kept the rows
        as OPEN - positions nobody monitored, and daily loss / trade-count limits that silently reset.
        Today's positions are reloaded and the daily counters restored. Cash is starting capital + ALL
        realised P&L to date, and the drawdown peak is the high-water mark of that realised equity curve
        (since the last manual peak reset), so a losing week still counts toward MAX_DRAWDOWN_PCT after a
        restart. OPEN rows from an earlier day can never be priced or closed properly, so they are
        marked STALE (kept, never deleted) rather than left looking live.
        """
        with self.lock:
            if self._restored:
                return
            self._restored = True
            try:
                from quant_intelligence.database.db import get_session, init_db
                from quant_intelligence.database.models import Position

                init_db()  # a page can be the first thing loaded; make sure new columns exist
                from quant_intelligence.risk.limit_overrides import load_persisted

                load_persisted()  # limits changed from the Risk Control pages survive a restart (every market)
                day_start = dt.datetime.combine(today_ist(), dt.time.min)
                cols = [c.name for c in Position.__table__.columns if c.name != "id"]
                with get_session() as session:
                    stale = [
                        r for r in session.query(Position).filter(
                            Position.status == "OPEN",
                            (Position.opened_at < day_start) | (Position.opened_at.is_(None)),
                        ).all()
                        if self._mine(r.underlying or r.instrument)
                    ]
                    for row in stale:
                        row.status = "STALE"
                        row.exit_reason = "STALE_ON_RESTART"
                    today_rows = [
                        {c: getattr(r, c) for c in cols}
                        for r in session.query(Position).filter(Position.opened_at >= day_start).all()
                        if self._mine(r.underlying or r.instrument)
                    ]
                    closed_history = [
                        (r.closed_at, r.net_pnl)
                        for r in session.query(Position)
                        .filter(Position.status == "CLOSED", Position.net_pnl.isnot(None))
                        .order_by(Position.closed_at)
                        .all()
                        if self._mine(r.underlying or r.instrument)
                    ]
                realised_today = 0.0
                for d in today_rows:
                    self.broker.positions[d["position_id"]] = d
                    if d["status"] == "CLOSED" and d.get("net_pnl") is not None:
                        realised_today += d["net_pnl"]
                self.trades_today = len(today_rows)
                self.daily_pnl = realised_today
                self.broker.cash, self.peak_equity = _equity_and_peak(
                    self.broker.capital, closed_history, _load_peak_reset(self.profile.name))
                log_event(
                    "engine",
                    f"Restored {sum(1 for d in today_rows if d['status'] == 'OPEN')} open position(s) and "
                    f"{len(today_rows)} trade(s) from today; marked {len(stale)} older OPEN row(s) STALE",
                    level="INFO",
                )
            except Exception as e:  # never block startup on recovery
                log_event("engine", f"Could not restore paper book from the database: {e}", level="ERROR")

    def _mine(self, symbol: str | None) -> bool:
        """Whether an instrument belongs to this engine's market."""
        return profile_for(symbol).name == self.profile.name

    # ---- this market's risk counters -----------------------------------------
    def _roll_day(self) -> None:
        today = today_ist()
        if today != self._day:
            self._day = today
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.day_peak_pnl = 0.0
            # peak_equity is NOT reset: drawdown is measured across days.

    def reset_drawdown_peak(self) -> float:
        """Start a fresh drawdown peak at the current marked-to-market equity (manual, from Risk Control).
        Persisted so a restart keeps it. Returns the new peak."""
        with self.lock:
            equity = self.broker.cash + _unrealised(self.broker.get_open_positions())
            self.peak_equity = equity
            _save_peak_reset(now_ist(), equity, self.profile.name)
            log_event("engine", f"{self.profile.name} drawdown peak reset to {equity:,.0f}", level="WARNING")
            return equity

    def add_trade(self, n: int = 1) -> None:
        with self.lock:
            self._roll_day()
            self.trades_today += n

    def add_pnl(self, amount: float) -> None:
        with self.lock:
            self._roll_day()
            self.daily_pnl += amount

    def account_state(self) -> AccountState:
        with self.lock:
            self._roll_day()
            open_positions = self.broker.get_open_positions()
            exposure_by_strategy: dict[str, float] = {}
            total_exposure = 0.0
            open_summary = []
            for p in open_positions:
                risk_amt = abs(p["entry_price"] - (p["stop_price"] or p["entry_price"])) * p["quantity"]
                exposure_by_strategy[p["strategy_name"]] = exposure_by_strategy.get(p["strategy_name"], 0.0) + risk_amt
                total_exposure += risk_amt
                open_summary.append({"instrument": p.get("underlying") or p.get("instrument"),
                                     "direction": p.get("direction"), "risk": risk_amt})

            # Mark to market: open losses count toward the daily-loss and drawdown limits before they close.
            unrealised = _unrealised(open_positions)
            equity = self.broker.cash + unrealised
            daily_pnl = self.daily_pnl + unrealised
            self.peak_equity = max(self.peak_equity, equity)
            self.day_peak_pnl = max(self.day_peak_pnl, daily_pnl)
            today = today_ist()
            closed_today = sorted(
                (
                    {"strategy": p.get("strategy_name"), "net_pnl": p.get("net_pnl"), "closed_at": p.get("closed_at")}
                    for p in self.broker.positions.values()
                    if p.get("status") == "CLOSED" and p.get("closed_at") is not None and p["closed_at"].date() == today
                ),
                key=lambda t: t["closed_at"],
            )
            return AccountState(
                equity=equity,
                peak_equity=self.peak_equity,
                daily_pnl=daily_pnl,
                open_positions_count=len(open_positions),
                trades_today=self.trades_today,
                exposure_by_strategy=exposure_by_strategy,
                total_exposure=total_exposure,
                broker_connected=self.broker.is_connected(),
                kill_switch_engaged=self.kill_switch,
                unrealised_pnl=unrealised,
                open_positions=open_summary,
                closed_today=closed_today,
                day_peak_pnl=self.day_peak_pnl,
                market=self.profile.name,
            )


# ---- drawdown bookkeeping helpers --------------------------------------------------------------
_PEAK_RESET_FILE = DATA_CACHE_DIR / "risk_peak_reset.json"


def _unrealised(open_positions: list[dict]) -> float:
    """Unrealised P&L from each position's last marked price (set by the position monitor). A position
    with no price yet counts as 0. Sign follows the option transaction (BUY gains when premium rises)."""
    total = 0.0
    for p in open_positions:
        last = p.get("last_price")
        if last is None:
            continue
        sign = 1 if (p.get("transaction") or p.get("direction")) in ("LONG", "BUY") else -1
        total += sign * (float(last) - p["entry_price"]) * p["quantity"]
    return total


def _equity_and_peak(capital: float, closed_history: list, peak_reset: dict | None) -> tuple[float, float]:
    """Realised equity (capital + every closed trade's P&L) and its high-water mark. With a manual peak
    reset, the mark starts from the equity at the reset and only later closes can raise it."""
    equity = capital
    reset_at = peak_reset["at"] if peak_reset else None
    peak = peak_reset["equity"] if peak_reset else capital
    for closed_at, pnl in closed_history:
        equity += pnl or 0.0
        if reset_at is None or (closed_at is not None and closed_at > reset_at):
            peak = max(peak, equity)
    return equity, max(peak, equity)


def _peak_reset_file(market: str = "NSE"):
    """One file per market. NSE keeps the original name, so a reset made before the split still applies to NSE."""
    if market.upper() == "MCX":
        return _PEAK_RESET_FILE.with_name(f"{_PEAK_RESET_FILE.stem}_mcx{_PEAK_RESET_FILE.suffix}")
    return _PEAK_RESET_FILE


def _load_peak_reset(market: str = "NSE") -> dict | None:
    try:
        data = json.loads(_peak_reset_file(market).read_text(encoding="utf-8"))
        return {"at": dt.datetime.fromisoformat(data["at"]), "equity": float(data["equity"])}
    except Exception:
        return None


def _save_peak_reset(at: dt.datetime, equity: float, market: str = "NSE") -> None:
    try:
        _peak_reset_file(market).write_text(json.dumps({"at": at.isoformat(), "equity": equity}), encoding="utf-8")
    except Exception as e:
        log_event("engine", f"Could not save the drawdown peak reset: {e}", level="WARNING")


ENGINE = TradingEngine("nse", NSE)  # the NSE paper account
MCX_ENGINE = TradingEngine("commodities", MCX)  # the separate MCX paper account
COMMODITY_RUNNER = MCX_ENGINE  # kept for the pages and tests that use this name


def engine_for_market(market: str | None) -> TradingEngine:
    """The engine (account) of a market name: "MCX" -> MCX_ENGINE, anything else -> ENGINE."""
    return MCX_ENGINE if (market or "").upper() == "MCX" else ENGINE


def engine_for(symbol: str | None) -> TradingEngine:
    """The engine whose account an instrument trades in (commodity underlyings -> MCX, everything else -> NSE)."""
    return engine_for_market(profile_for(symbol).name)
