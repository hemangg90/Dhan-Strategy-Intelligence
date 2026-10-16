import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.brokers.base_broker import OrderRequest
from quant_intelligence.brokers.dhan_broker import DhanBroker
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.execution.setup_detector import SETUP_TRIGGERED, WAITING_FOR_SETUP, detect_chain_setup, detect_setup
from quant_intelligence.data_adapters.dhan_instrument_master import list_fno_stock_symbols
from quant_intelligence.options.option_selector import (
    OptionSelectionError,
    fetch_chain,
    get_underlying_info,
    option_segment_for,
    select_contract,
)
from quant_intelligence.options.premium_model import PremiumSizingError, translate_setup
from quant_intelligence.ranking.ranking_engine import TIE_BREAK_TAG, tradable
from quant_intelligence.risk.risk_engine import ProposedTrade, evaluate_trade
from quant_intelligence.strategies.registry import CHAIN_AWARE_STRATEGY_NAMES, get_strategy
from quant_intelligence.ui.credentials_panel import render_credentials_panel
from quant_intelligence.execution.engine import ENGINE
from quant_intelligence.ui.state import get_live_account_state, init_session_state, run_pipeline_cached
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
st.title("Live Options Trading")

st.error(
    "THIS PAGE PLACES REAL ORDERS WITH REAL MONEY through your Dhan account when fully "
    "authorized below. It is separate from Paper Trading on purpose - nothing here can be "
    "triggered accidentally from that page. Validate every strategy extensively in Paper "
    "Trading first."
)

if not SETTINGS.live_mode_fully_authorized:
    st.warning(
        "LIVE trading is NOT authorized. Set both TRADING_MODE=LIVE and "
        "TRADING_LIVE_CONFIRM=YES_I_UNDERSTAND_THE_RISK in .env to enable this page. "
        "Until then, every order below will be rejected by DhanBroker itself, independent "
        "of anything on this page."
    )

client = st.session_state["dhan_api_client"]
if "dhan_broker" not in st.session_state:
    st.session_state["dhan_broker"] = DhanBroker()
broker = st.session_state["dhan_broker"]

if not broker.is_connected():
    st.error("DhanBroker is not connected - enter your Dhan Client ID and Access Token below.")
    render_credentials_panel()
    st.stop()

output = run_pipeline_cached()
if output is None:
    st.error(st.session_state.get("pipeline_error", "Pipeline not available."))
    st.stop()

st.subheader("Option chain")
c1, c2 = st.columns([2, 1])
underlying_choices = list(SETTINGS.option_underlyings) + list_fno_stock_symbols()
underlying = c1.selectbox("Underlying", underlying_choices, key="option_underlying")
if c2.button("Fetch live option chain"):
    try:
        st.session_state["option_chain"] = fetch_chain(client, underlying, SETTINGS.option_expiry_preference, max_age=0)
    except Exception as e:
        st.session_state["option_chain"] = None
        st.error(f"Could not fetch option chain: {e}")

chain = st.session_state.get("option_chain")
if chain is not None and chain.underlying == underlying:
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Spot", round(chain.spot_price, 2))
    cc2.metric("ATM strike", chain.atm_strike)
    cc3.metric("Total PCR", round(chain.total_pcr, 2) if chain.total_pcr else "-")
    cc4.metric("ATM IV", round(chain.atm_iv, 2) if chain.atm_iv else "-")

st.divider()

ok_live, why_not = tradable(output.ranking, "LIVE")
if not ok_live:
    st.warning(f"NO TRADE: {why_not}")
    st.stop()

strategy_name = output.ranking.selected_strategy
st.success(f"Selected strategy: {strategy_name}")
if output.ranking.tie_break:
    st.error(
        f"TIE-BREAK SELECTION - REAL MONEY. {output.ranking.reason} Tie-break picks are the ones with NO statistically "
        f"proven edge (in research they average slightly negative after costs). The order will be tagged TIE-BREAK; "
        f"consider a smaller lot than usual (the paper auto-trader uses x{SETTINGS.tie_break_size_factor:g}). "
        "Nothing is sent without your confirmation, the risk engine and the live double gate."
    )
strategy = get_strategy(strategy_name)
recent_ohlcv = output.ohlcv.tail(5)
is_chain_strategy = strategy_name in CHAIN_AWARE_STRATEGY_NAMES

if is_chain_strategy and chain is None:
    st.warning(f"{strategy_name} needs a live option chain - fetch one above first.")
    st.stop()
setup_status = (
    detect_chain_setup(strategy, output.market_state, recent_ohlcv, chain)
    if is_chain_strategy
    else detect_setup(strategy, output.market_state, recent_ohlcv)
)

if setup_status.status == WAITING_FOR_SETUP:
    st.info(f"Status: WAITING FOR SETUP ({strategy_name} entry conditions not yet met on the latest bar)")
    st.stop()
if setup_status.status != SETUP_TRIGGERED:
    st.stop()

setup = setup_status.setup
transaction = setup.meta.get("transaction", "BUY")
st.success(f"Status: SETUP TRIGGERED - {setup.direction}, transaction {transaction}")

if chain is None or chain.underlying != underlying:
    st.warning("Fetch a live option chain for this underlying to resolve a tradeable contract.")
    st.stop()

moneyness_offset = st.slider("Moneyness offset (0 = ATM, + = further OTM)", 0, 5, SETTINGS.option_moneyness_offset)
try:
    lot_size = get_underlying_info(underlying)["lot_size"]
    contract = select_contract(chain, setup.direction, transaction, moneyness_offset, lot_size)
    premium_setup = translate_setup(setup, contract)
except (OptionSelectionError, PremiumSizingError) as e:
    st.error(f"Could not resolve a tradeable option: {e}")
    st.stop()

st.success(f"Resolved contract: {contract.trading_symbol}")
oc1, oc2, oc3, oc4 = st.columns(4)
oc1.metric("Premium (entry)", round(premium_setup.entry_price, 2))
oc2.metric("Premium stop", round(premium_setup.stop_price, 2))
oc3.metric("Premium target", round(premium_setup.target_price, 2))
oc4.metric("Delta", round(contract.delta, 3) if contract.delta is not None else "-")

lots = st.number_input("Lots", min_value=1, value=1, step=1)
quantity = int(lots) * contract.lot_size
st.caption(f"Order quantity: {lots} lot(s) x {contract.lot_size} = {quantity}")

amount_needed = premium_setup.entry_price * quantity
max_loss = abs(premium_setup.entry_price - premium_setup.stop_price) * quantity
try:
    available = float(broker.get_fund_limit().get("availabelBalance", 0.0))
except Exception:
    available = None
ac1, ac2, ac3 = st.columns(3)
ac1.metric("Amount needed" if transaction == "BUY" else "Premium value (margin extra)", f"₹{amount_needed:,.0f}")
ac2.metric("Risk at stop", f"₹{max_loss:,.0f}")
ac3.metric("Available funds", f"₹{available:,.0f}" if available is not None else "-")
if transaction == "BUY" and available is not None and amount_needed > available:
    st.warning("Amount needed exceeds available funds.")

confirm = st.checkbox("I have validated this strategy in Paper Trading and want to place a REAL order.")

if st.button("Submit to Risk Engine -> DhanBroker (LIVE)", type="primary", disabled=not confirm):
    account = get_live_account_state(broker)
    proposed = ProposedTrade(
        strategy_name=strategy_name,
        direction=setup.direction,
        entry_price=premium_setup.entry_price,
        stop_price=premium_setup.stop_price,
        target_price=premium_setup.target_price,
        quantity=quantity,
        relative_volume=output.market_state.get("relative_volume"),
        data_quality_status=output.data_quality_status, quote_time=getattr(output, "timestamp", None), long_premium=True,
        instrument=underlying,
    )
    decision = evaluate_trade(account, proposed)

    if not decision.approved:
        st.error(f"RISK ENGINE VETO ({decision.decision_id}): {decision.reason}")
    else:
        st.success(f"Risk approved ({decision.decision_id})")
        order = OrderRequest(
            strategy_name=strategy_name,
            instrument=underlying,
            direction=setup.direction,
            quantity=quantity,
            order_type="MARKET",
            price=premium_setup.entry_price,
            stop_price=premium_setup.stop_price,
            target_price=premium_setup.target_price,
            decision_id=decision.decision_id,
            security_id=contract.security_id,
            exchange_segment=option_segment_for(underlying),
            product_type=SETTINGS.option_product_type,
            transaction_type=contract.transaction,
            option_type=contract.option_type,
            strike=contract.strike,
            expiry=contract.expiry,
            lot_size=contract.lot_size,
            tag=TIE_BREAK_TAG if output.ranking.tie_break else None,
        )
        ack = broker.place_order(order)
        ENGINE.add_trade()
        if ack.status in ("FILLED", "TRADED", "PENDING", "TRANSIT"):
            st.success(f"LIVE order {ack.status}: {contract.transaction} {order.quantity} {contract.trading_symbol} (order {ack.order_id})")
        else:
            st.error(f"Order {ack.status}: {ack.reject_reason}")

st.divider()
st.subheader("Live positions (from Dhan)")
positions = broker.get_positions()
if positions:
    st.dataframe(positions, width="stretch", hide_index=True)
else:
    st.caption("No open live positions (or unable to fetch).")
