from __future__ import annotations

from decimal import Decimal

import requests
import structlog

from agent.execution.types import Fill
from agent.risk.types import PortfolioState
from agent.strategies.types import Action

logger = structlog.get_logger()

_TELEGRAM_SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlerter:
    """Telegram Bot alerter. Soft-fails silently when credentials are not set.

    Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)

    def send(self, text: str) -> None:
        """Send a plain-text message. Silently swallows all errors."""
        if not self._enabled:
            return
        try:
            url = _TELEGRAM_SEND_URL.format(token=self._token)
            requests.post(
                url,
                json={"chat_id": self._chat_id, "text": text},
                timeout=5,
            ).raise_for_status()
        except Exception:
            logger.warning("telegram_alerter.send_failed", exc_info=True)

    def send_fill_alert(self, fill: Fill) -> None:
        direction = "ENTER" if fill.action == Action.ENTER_LONG else "EXIT"
        self.send(
            f"[PAPER {direction}] {fill.symbol} x{fill.quantity} @ ₹{fill.fill_price}"
            f"\nStrategy: {fill.strategy_name}"
        )

    def send_rejection_alert(self, symbol: str, reason: str) -> None:
        self.send(f"[REJECTED] {symbol}: {reason}")

    def send_daily_summary(self, state: PortfolioState, *, session_count: int) -> None:
        pnl_sign = "+" if state.daily_pnl >= 0 else ""
        self.send(
            f"[DAY END] Session #{session_count}\n"
            f"NAV: ₹{state.nav:.2f}  P&L: {pnl_sign}₹{state.daily_pnl:.2f}\n"
            f"Orders: {state.orders_today}  Peak: ₹{state.peak_nav:.2f}"
        )
