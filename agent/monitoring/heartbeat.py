from __future__ import annotations

from datetime import datetime

import structlog

from agent.monitoring.alerter import TelegramAlerter
from agent.risk.types import PortfolioState

logger = structlog.get_logger()


class Heartbeat:
    """Periodic health-check logger. Optionally sends Telegram messages.

    alert_every_n_beats: send a Telegram alert every N beats (default 4 = every hour
    on a 15-minute bar loop). Set to 0 to disable Telegram alerts.
    """

    def __init__(
        self,
        alerter: TelegramAlerter | None = None,
        alert_every_n_beats: int = 4,
    ) -> None:
        self._alerter = alerter
        self._every_n = alert_every_n_beats
        self._count = 0

    @property
    def beat_count(self) -> int:
        return self._count

    def beat(self, state: PortfolioState, *, ts: datetime) -> None:
        self._count += 1
        logger.info(
            "heartbeat",
            beat=self._count,
            ts=ts.isoformat(),
            nav=str(state.nav),
            cash=str(state.cash),
            positions=list(state.positions.keys()),
            orders_today=state.orders_today,
        )
        if self._alerter and self._every_n > 0 and self._count % self._every_n == 0:
            self._alerter.send(
                f"[HEARTBEAT #{self._count}] {ts.strftime('%H:%M IST')}"
                f" | NAV ₹{state.nav:,.0f} | Orders: {state.orders_today}"
            )
