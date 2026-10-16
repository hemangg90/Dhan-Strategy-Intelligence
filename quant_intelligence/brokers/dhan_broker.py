"""
DhanBroker adapter - LIVE order execution.

SAFETY: this class refuses to send any order unless BOTH:
  1. SETTINGS.trading_mode == "LIVE"
  2. SETTINGS.trading_live_confirm == "YES_I_UNDERSTAND_THE_RISK"
are true (see config/settings.py: SETTINGS.live_mode_fully_authorized).
This is checked again here, independently of the caller, so a bug elsewhere
in the app cannot silently enable live trading. Nothing in this class can
weaken or bypass that gate.

Orders are placed through `quant_intelligence.brokers.dhan_api_client.DhanApiClient`
against the Dhan v2 Orders API (see dhanhq.co/docs/v2/orders). This has been
tested against the documented request/response shape but NOT against a real
funded account - run extensive paper trading (PaperBroker, or this class's
`is_connected`/`get_positions` in LIVE-but-unauthorized inspection mode)
before enabling live order flow, and start with the smallest possible size.
"""
from __future__ import annotations

import requests

from quant_intelligence.brokers.base_broker import BaseBroker, OrderAck, OrderRequest
from quant_intelligence.brokers.dhan_api_client import DhanApiClient, DhanApiError
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.risk.risk_engine import consume_approval
from quant_intelligence.utils.logging_utils import log_event


class DhanBroker(BaseBroker):
    mode = "LIVE"
    name = "dhan"

    def __init__(self):
        # Credentials are read live: a broker built before a token was saved/renewed must not keep the old ones.
        self.client = DhanApiClient()
        # Set when an order request failed in transit (timeout / dropped connection): it may have reached Dhan, so
        # nothing further is sent until a human has checked Dhan's order book. Never cleared automatically.
        self.unreconciled_order: str | None = None

    @property
    def client_id(self) -> str:
        return SETTINGS.dhan_client_id

    @property
    def access_token(self) -> str:
        return SETTINGS.dhan_access_token

    def is_connected(self) -> bool:
        return bool(self.client_id and self.access_token)

    def place_order(self, order: OrderRequest) -> OrderAck:
        if not SETTINGS.live_mode_fully_authorized:
            reason = (
                "LIVE trading is not authorized. Set TRADING_MODE=LIVE and "
                "TRADING_LIVE_CONFIRM=YES_I_UNDERSTAND_THE_RISK in .env to enable."
            )
            log_event("dhan_broker", f"BLOCKED live order attempt: {reason}", level="ERROR")
            return OrderAck(order_id="BLOCKED", status="REJECTED", reject_reason=reason)

        if self.unreconciled_order:
            reason = (
                f"A previous live order ({self.unreconciled_order}) has an UNKNOWN outcome: the request may have reached Dhan. "
                "Check Dhan's order book, then restart the app to clear this block. No new order will be sent."
            )
            log_event("dhan_broker", f"BLOCKED live order attempt: {reason}", level="ERROR")
            return OrderAck(order_id="BLOCKED", status="REJECTED", reject_reason=reason)

        if not order.security_id:
            reason = "OrderRequest.security_id is required for live Dhan orders (resolve a contract first)"
            log_event("dhan_broker", f"BLOCKED live order attempt: {reason}", level="ERROR")
            return OrderAck(order_id="BLOCKED", status="REJECTED", reject_reason=reason)

        # Defence in depth: the order must carry a risk-engine approval issued moments ago, usable once. Every caller
        # already runs the risk engine; this makes skipping it (or replaying an approval) impossible by construction.
        if not consume_approval(order.decision_id):
            reason = "No valid, unused risk-engine approval for this order (missing, expired or already used)"
            log_event("dhan_broker", f"BLOCKED live order attempt: {reason}", level="ERROR", decision_id=order.decision_id)
            return OrderAck(order_id="BLOCKED", status="REJECTED", reject_reason=reason)

        transaction_type = order.transaction_type or ("BUY" if order.direction == "LONG" else "SELL")
        payload = {
            "dhanClientId": self.client_id,
            "transactionType": transaction_type,
            "exchangeSegment": order.exchange_segment,
            "productType": order.product_type,
            "orderType": order.order_type if order.order_type in ("LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_MARKET") else "MARKET",
            "validity": "DAY",
            "securityId": order.security_id,
            "quantity": order.quantity,
            "price": order.price or 0,
            # A tag (e.g. TIE-BREAK) travels to Dhan's own order book through correlationId.
            "correlationId": (f"TB-{order.decision_id or ''}" if order.tag else (order.decision_id or ""))[:30],
        }

        try:
            response = self.client.place_order(payload)
        except DhanApiError as e:
            log_event("dhan_broker", f"Live order REJECTED by Dhan API: {e}", level="ERROR", decision_id=order.decision_id)
            return OrderAck(order_id="ERROR", status="REJECTED", reject_reason=str(e))
        except requests.RequestException as e:
            # Timeout / connection reset AFTER the request was sent: Dhan may or may not have the order. Do not report a
            # rejection and do not retry; block further orders until the outcome is checked by a human.
            self.unreconciled_order = order.decision_id or "unknown"
            reason = (
                f"Order outcome UNKNOWN ({type(e).__name__}): the request may have reached Dhan. "
                "Check Dhan's order book before doing anything else; no automatic retry will be made."
            )
            log_event("dhan_broker", reason, level="ERROR", decision_id=order.decision_id)
            return OrderAck(order_id="UNKNOWN", status="UNKNOWN", reject_reason=reason)

        order_id = response.get("orderId", "UNKNOWN")
        status = response.get("orderStatus", "PENDING")
        log_event(
            "dhan_broker",
            f"LIVE order placed: {transaction_type} {order.quantity} {order.instrument} "
            f"(security_id={order.security_id}) -> {status}" + (f" [{order.tag}]" if order.tag else ""),
            level="INFO",
            order_id=order_id,
            decision_id=order.decision_id,
        )
        return OrderAck(order_id=order_id, status=status)

    def get_positions(self) -> list[dict]:
        try:
            return self.client.get_positions()
        except DhanApiError as e:
            log_event("dhan_broker", f"Failed to fetch live positions: {e}", level="ERROR")
            return []

    def get_fund_limit(self) -> dict:
        try:
            return self.client.get_fund_limit()
        except DhanApiError as e:
            log_event("dhan_broker", f"Failed to fetch fund limit: {e}", level="ERROR")
            return {}
