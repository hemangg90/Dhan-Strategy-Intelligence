"""Read-only rows for the Settings & Data Status page. Only operating limits and switches: no credential, token or key is
ever read into these rows (credentials are shown as configured / not configured by the sidebar, never here)."""
from __future__ import annotations

from quant_intelligence.config.settings import SETTINGS


def mode_rows() -> list[dict]:
    s = SETTINGS
    return [
        {"Setting": "Trading mode", "Value": s.trading_mode, "Meaning": "PAPER simulates every order. LIVE also needs the confirmation phrase."},
        {"Setting": "Demo lockdown", "Value": "ON" if s.demo_lockdown else "off", "Meaning": "ON forces paper mode and hides credentials."},
        {"Setting": "Live orders possible", "Value": "YES" if s.live_mode_fully_authorized else "no",
         "Meaning": "Requires LIVE mode, the confirmation phrase and no lockdown."},
        {"Setting": "Tie-break trading (paper / live)", "Value": f"{'on' if s.tie_break_paper else 'off'} / {'on' if s.tie_break_live else 'off'}",
         "Meaning": "Trade when the top strategies are statistically tied, tagged and sized down."},
        {"Setting": "Analogue method", "Value": s.analogue_method, "Meaning": "legacy (default) or robust; see docs/REGIME_AND_ANALOGUES.md."},
        {"Setting": "Paper starting capital", "Value": f"{s.paper_starting_capital:,.0f}", "Meaning": "Simulated account size."},
    ]


def risk_rows() -> list[dict]:
    r = SETTINGS.risk
    rows = [
        ("Max risk per trade", f"{r.max_risk_per_trade_pct:g}% of equity", "What one stopped trade may cost; sizing divides this by the stop distance."),
        ("Max capital per trade", f"{r.max_capital_per_trade_pct:g}% of equity", "Cap on premium x quantity (0 disables)."),
        ("Max daily loss", f"{r.max_daily_loss_pct:g}%", "New entries stop at this loss."),
        ("Max drawdown", f"{r.max_drawdown_pct:g}%", "New entries stop at this fall from peak equity."),
        ("Strategy exposure", f"{r.max_strategy_exposure_pct:g}%", "Open risk in one strategy."),
        ("Portfolio exposure", f"{r.max_portfolio_exposure_pct:g}%", "Open risk in total."),
        ("Max trades per day", str(r.max_trades_per_day), ""),
        ("Max open positions", str(r.max_open_positions), "0 disables."),
        ("Correlated group", f"{r.max_group_positions} positions / {r.max_group_risk_pct:g}% risk", "Same-direction positions and risk per group."),
        ("Loss-streak pause", f"{r.loss_streak_pause} losses, {r.loss_streak_pause_min} min", "No new entries after consecutive losers."),
        ("Strategy daily loss stop", f"{r.strategy_max_losses_per_day} losses", "A strategy that lost this many today is done."),
        ("Profit lock", f"at {r.profit_lock_trigger_pct:g}%, give back {r.profit_lock_giveback:.0%}", "Stops trading if a good day is being given back."),
    ]
    return [{"Limit": a, "Value": b, "Meaning": c} for a, b, c in rows]


def housekeeping_rows() -> list[dict]:
    s = SETTINGS
    return [
        {"Setting": "Decision log retention", "Value": f"{s.decision_retention_days} days", "Meaning": "Older scan decisions are pruned."},
        {"Setting": "Auto-trade scan interval", "Value": f"{s.auto_trade_refresh_seconds} s", "Meaning": ""},
        {"Setting": "Quote refresh", "Value": f"{s.ltp_refresh_seconds} s", "Meaning": ""},
    ]
