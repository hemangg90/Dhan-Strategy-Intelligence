"""Itemised charges for a paper option trade, at Dhan's brokerage and the statutory option rates.

The single definition of option charges: used by the paper broker when it closes a position
(brokers/paper_costs.py), to estimate charges on legacy rows and open positions, and by the order ticket. It uses the same
rates as the backtest cost model (config.settings.CostAssumptions), but on the ACTUAL entry/exit premiums, so the UI shows
gross P&L, charges and net P&L the way a broker contract note does.
"""
from __future__ import annotations

from dataclasses import dataclass

from quant_intelligence.config.settings import SETTINGS


@dataclass
class Charges:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst

    def as_rows(self) -> list[tuple[str, float]]:
        return [
            ("Brokerage", self.brokerage),
            ("STT", self.stt),
            ("Exchange charges", self.exchange),
            ("SEBI fee", self.sebi),
            ("Stamp duty", self.stamp),
            ("GST", self.gst),
        ]


def option_trade_charges(
    entry_price: float | None,
    exit_price: float | None,
    quantity: int | None,
    transaction: str = "BUY",
    market: str = "NSE",
) -> Charges:
    """Charges for opening and closing `quantity` units of an option (premium prices per unit).

    BUY: bought at entry, sold at exit. SELL: sold at entry, bought back at exit. STT applies to the
    sell leg's premium, stamp duty to the buy leg's; exchange and SEBI fees to both legs; GST to
    brokerage + exchange + SEBI.
    """
    c = SETTINGS.costs
    if not entry_price or not quantity:
        return Charges(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    exit_price = entry_price if exit_price is None else exit_price
    if transaction == "SELL":
        sell_turnover, buy_turnover = entry_price * quantity, exit_price * quantity
    else:
        buy_turnover, sell_turnover = entry_price * quantity, exit_price * quantity
    turnover = buy_turnover + sell_turnover

    mcx = market == "MCX"
    brokerage = c.brokerage_per_order_inr * 2
    stt = sell_turnover * (c.opt_stt_sell_rate_mcx if mcx else c.opt_stt_sell_rate_nse)
    exchange = turnover * (c.opt_txn_rate_mcx if mcx else c.opt_txn_rate_nse)
    sebi = turnover * c.sebi_rate
    stamp = buy_turnover * c.stamp_buy_rate
    gst = c.gst_rate * (brokerage + exchange + sebi)
    return Charges(brokerage, stt, exchange, sebi, stamp, gst)
