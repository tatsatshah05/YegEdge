# tests/monitoring/test_heartbeat.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from agent.monitoring.alerter import TelegramAlerter
from agent.monitoring.heartbeat import Heartbeat
from agent.risk.types import PortfolioState

IST = ZoneInfo("Asia/Kolkata")

T0 = datetime(2024, 1, 2, 10, 0, tzinfo=IST)

_STATE = PortfolioState(
    nav=Decimal("100000"), cash=Decimal("100000"),
    positions={}, daily_pnl=Decimal("0"),
    weekly_pnl=Decimal("0"), peak_nav=Decimal("100000"),
    orders_today=0, last_order_time={}, kill_switch_active=False,
    evaluation_time=T0,
)


def test_beat_logs_without_alerter() -> None:
    hb = Heartbeat(alerter=None)
    hb.beat(_STATE, ts=T0)  # must not raise


def test_beat_sends_telegram_when_alerter_configured() -> None:
    alerter = MagicMock(spec=TelegramAlerter)
    hb = Heartbeat(alerter=alerter, alert_every_n_beats=1)
    hb.beat(_STATE, ts=T0)
    alerter.send.assert_called_once()


def test_beat_respects_alert_every_n_beats() -> None:
    alerter = MagicMock(spec=TelegramAlerter)
    hb = Heartbeat(alerter=alerter, alert_every_n_beats=3)
    for _ in range(2):
        hb.beat(_STATE, ts=T0)
    alerter.send.assert_not_called()
    hb.beat(_STATE, ts=T0)  # 3rd beat — should send
    alerter.send.assert_called_once()


def test_beat_counter_increments() -> None:
    hb = Heartbeat(alerter=None)
    assert hb.beat_count == 0
    hb.beat(_STATE, ts=T0)
    assert hb.beat_count == 1
