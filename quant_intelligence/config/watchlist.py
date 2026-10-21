"""Fixed stock watchlist traded together with the indices (SETTINGS.option_underlyings).

Symbols are Dhan/NSE trading symbols (verified against the Dhan scrip master):
note HDFC Bank is HDFCBANK and LIC is LICI on NSE.
"""
from __future__ import annotations

WATCHLIST_STOCKS: list[dict[str, str]] = [
    {"symbol": "RELIANCE", "name": "Reliance Industries", "sector": "Conglomerate (Energy/Telecom/Retail)"},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "sector": "Telecommunications"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank", "sector": "Financial Services (Banking)"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank", "sector": "Financial Services (Banking)"},
    {"symbol": "SBIN", "name": "State Bank of India", "sector": "Financial Services (Banking)"},
    {"symbol": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance", "sector": "Financial Services (NBFC)"},
    {"symbol": "LT", "name": "Larsen & Toubro", "sector": "Engineering & Infrastructure"},
    {"symbol": "LICI", "name": "Life Insurance Corporation", "sector": "Financial Services (Insurance)"},
    {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical Industries", "sector": "Healthcare"},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever", "sector": "Fast-Moving Consumer Goods (FMCG)"},
    {"symbol": "TITAN", "name": "Titan Company", "sector": "Consumer Cyclical"},
    {"symbol": "ADANIPORTS", "name": "Adani Ports & SEZ", "sector": "Infrastructure & Logistics"},
    {"symbol": "INFY", "name": "Infosys", "sector": "Information Technology"},
    {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank", "sector": "Financial Services (Banking)"},
]


# MCX commodities traded as options on the front-month future. `lot_size` is the exchange
# contract multiplier in PRICE units (P&L per 1.0 move in the quoted premium/price), because
# Dhan's scrip master reports SEM_LOT_UNITS=1 for every MCX contract. These come from MCX's
# published contract specs, NOT from Dhan - verify against the exchange before trusting P&L:
#   CRUDEOIL 100 bbl | CRUDEOILM 10 bbl | NATURALGAS 1250 mmBtu
# Only the energy contracts are kept. Gold, silver and copper (GOLD 1 kg quoted per 10 g -> x100, GOLDM 100 g -> x10,
# SILVER 30 kg, SILVERM 5 kg, COPPER 2500 kg) were removed from the watchlist; add a row here to bring one back.
WATCHLIST_COMMODITIES: list[dict] = [
    {"symbol": "CRUDEOIL", "name": "Crude Oil", "sector": "Energy", "lot_size": 100},
    {"symbol": "CRUDEOILM", "name": "Crude Oil Mini", "sector": "Energy", "lot_size": 10},
    {"symbol": "NATURALGAS", "name": "Natural Gas", "sector": "Energy", "lot_size": 1250},
]
COMMODITY_SYMBOLS: frozenset[str] = frozenset(c["symbol"] for c in WATCHLIST_COMMODITIES)
COMMODITY_LOT_SIZES: dict[str, int] = {c["symbol"]: c["lot_size"] for c in WATCHLIST_COMMODITIES}
