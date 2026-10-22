"""Option Chain: rank every strike of the live chain and show which one the best-strike selector would trade.

Read-only: it fetches the chain (one Dhan call, shared with the traders through the chain cache) and scores it
with options/strike_ranker.py. Nothing here places an order. The trading pages use the top strike only when
OPTION_SELECTION=BEST; the default remains ATM +/- OPTION_MONEYNESS_OFFSET.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.data_adapters.dhan_instrument_master import list_fno_stock_symbols
from quant_intelligence.options.option_selector import OptionSelectionError, fetch_chain, get_underlying_info, select_contract
from quant_intelligence.options.strike_ranker import BUY_PROFILE, SELL_PROFILE, rank_strikes
from quant_intelligence.ui.chain_views import ranked_frame
from quant_intelligence.ui.components import chip, empty_state, page_header
from quant_intelligence.ui.credentials_panel import render_credentials_panel
from quant_intelligence.ui.state import init_session_state
from quant_intelligence.ui.theme import apply_theme

apply_theme()
init_session_state()
page_header(
    "Option Chain",
    "Every strike of the live chain scored on delta, bid-ask spread, liquidity and time decay. A heuristic ranking, "
    "not a prediction: it has not been shown to beat the plain ATM pick.",
)
mode_chip = chip(f"Trading pages use: {SETTINGS.option_selection}", "info" if SETTINGS.option_selection == "BEST" else "muted", icon="")
st.markdown(mode_chip, unsafe_allow_html=True)
if SETTINGS.option_selection != "BEST":
    st.caption("Trading pages still pick ATM +/- the offset. Set OPTION_SELECTION=BEST in .env to let them use the top strike below.")

client = st.session_state["dhan_api_client"]
if not client.is_configured():
    st.error("Dhan credentials are not set - the option chain needs them.")
    render_credentials_panel()
    st.stop()

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
underlying = c1.selectbox("Underlying", list(SETTINGS.option_underlyings) + list_fno_stock_symbols(), key="oc_underlying")
side = c2.selectbox("Side", ["Calls (LONG view)", "Puts (SHORT view)"], key="oc_side")
transaction = c3.selectbox("Action", ["BUY", "SELL"], key="oc_transaction")
if c4.button("Fetch live chain", key="oc_fetch", width="stretch"):
    try:
        st.session_state["option_chain"] = fetch_chain(client, underlying, SETTINGS.option_expiry_preference, max_age=0)
    except Exception as e:
        st.session_state["option_chain"] = None
        st.error(f"Could not fetch the option chain: {e}")

chain = st.session_state.get("option_chain")
if chain is None or chain.underlying != underlying:
    empty_state("No chain loaded", " Pick an underlying and press Fetch live chain.")
    st.stop()

direction = "LONG" if side.startswith("Calls") else "SHORT"
try:
    lot_size = get_underlying_info(underlying)["lot_size"]
except OptionSelectionError:
    lot_size = 1
max_outlay = None
if transaction == "BUY":
    cap = st.number_input("Most one lot may cost (Rs, 0 = no cap)", min_value=0, value=0, step=1000, key="oc_cap")
    max_outlay = float(cap) if cap > 0 else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Spot", f"{chain.spot_price:,.2f}")
m2.metric("Expiry", chain.expiry)
m3.metric("ATM strike", f"{chain.atm_strike:g}" if chain.atm_strike is not None else "-")
m4.metric("Lot size", lot_size)

ranked = rank_strikes(chain, direction, transaction, lot_size, max_outlay)
eligible = [r for r in ranked if r.eligible]
profile = SELL_PROFILE if transaction == "SELL" else BUY_PROFILE
if not eligible:
    st.warning("No strike passes the filters (see the notes column). The trading pages would report no tradeable option.")
else:
    best = eligible[0]
    st.success(
        f"Best: **{chain.underlying} {chain.expiry} {best.strike:g} {best.option_type}** - score {best.score:.0f}/100, "
        f"premium {best.premium:,.2f}, delta {best.delta if best.delta is not None else '-'}, "
        f"one lot Rs {best.premium * lot_size:,.0f}"
    )
    try:
        atm_pick = select_contract(chain, direction, transaction, 0, lot_size)
        atm_rank = next((r for r in ranked if r.strike == atm_pick.strike), None)
        if atm_pick.strike != best.strike and atm_rank is not None:
            st.caption(
                f"The ATM pick would be {atm_pick.strike:g} (score "
                f"{atm_rank.score:.0f}{'' if atm_rank.eligible else ', filtered out: ' + '; '.join(atm_rank.notes)})."
            )
    except OptionSelectionError:
        pass

st.caption(
    f"Filters: |delta| {profile.delta_band[0]:.2f}-{profile.delta_band[1]:.2f}, spread under {profile.max_spread_pct:g}%, "
    "open interest and volume above zero. Score = 35% delta fit, 25% spread, 25% liquidity, 15% decay "
    f"(target |delta| {profile.delta_target:.2f}; {'a written' if profile.sell else 'a bought'} option "
    f"{'wants more' if profile.sell else 'should lose little'} decay). Missing inputs score a neutral 50 and are noted."
)
st.dataframe(ranked_frame(ranked), hide_index=True, width="stretch")
