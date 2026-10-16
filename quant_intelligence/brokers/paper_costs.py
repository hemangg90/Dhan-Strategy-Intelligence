"""Transaction costs for a simulated (paper) position, from the ACTUAL entry and exit prices.

Uses the same configured rates as the backtest cost model (`SETTINGS.costs`), so paper and backtest P&L are
comparable. Modelled, not a contract note: verify the rates against your broker's published charges.

Option positions (premium prices): brokerage on both legs, STT on the sell leg, exchange transaction charge and SEBI fee on
turnover, stamp duty on the buy leg, GST on brokerage + exchange charge + SEBI fee, plus one tick of exit slippage.
Non-option positions: brokerage + GST + exit slippage only. Statutory charges for equity / futures are NOT modelled
(the configuration has no rates for them), so that P&L is still somewhat optimistic; `costs_model` records which applied.
"""
from __future__ import annotations

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.market_profile import profile_for


def round_trip_costs(pos: dict, exit_price: float) -> tuple[float, str]:
    """(total costs in rupees, model name) for closing `pos` at `exit_price`. Entry slippage is already in
    `pos["entry_price"]`, so only the exit-side slippage is added here."""
    c = SETTINGS.costs
    qty = int(pos["quantity"])
    entry = float(pos["entry_price"])
    brokerage = c.brokerage_per_order_inr * 2
    exit_slippage = c.slippage_ticks * c.tick_size * qty

    if pos.get("option_type"):
        mcx = profile_for(pos.get("underlying") or pos.get("instrument")).name == "MCX"
        bought_first = (pos.get("transaction") or ("BUY" if pos["direction"] == "LONG" else "SELL")) == "BUY"
        buy_turnover = (entry if bought_first else exit_price) * qty
        sell_turnover = (exit_price if bought_first else entry) * qty
        turnover = buy_turnover + sell_turnover
        stt = sell_turnover * (c.opt_stt_sell_rate_mcx if mcx else c.opt_stt_sell_rate_nse)
        txn = turnover * (c.opt_txn_rate_mcx if mcx else c.opt_txn_rate_nse)
        sebi = turnover * c.sebi_rate
        stamp = buy_turnover * c.stamp_buy_rate
        gst = c.gst_rate * (brokerage + txn + sebi)
        return brokerage + stt + txn + sebi + stamp + gst + exit_slippage, "OPTION_PREMIUM"

    return brokerage + c.gst_rate * brokerage + exit_slippage, "BROKERAGE_ONLY"
