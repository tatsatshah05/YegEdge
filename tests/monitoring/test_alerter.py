# tests/monitoring/test_alerter.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from agent.monitoring.alerter import TelegramAlerter

IST = ZoneInfo("Asia/Kolkata")


def _alerter(configured: bool = True) -> TelegramAlerter:
    return TelegramAlerter(
        bot_token="fake-token" if configured else "",
        chat_id="123456" if configured else "",
    )


def test_send_posts_to_telegram_when_configured() -> None:
    alerter = _alerter()
    with patch("agent.monitoring.alerter.requests.post") as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        alerter.send("Hello from YegEdge")
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert "Hello from YegEdge" in str(kwargs.get("json") or mock_post.call_args)


def test_send_does_nothing_when_unconfigured() -> None:
    alerter = _alerter(configured=False)
    with patch("agent.monitoring.alerter.requests.post") as mock_post:
        alerter.send("Should not send")
    mock_post.assert_not_called()


def test_send_soft_fails_on_http_error() -> None:
    alerter = _alerter()
    with patch("agent.monitoring.alerter.requests.post") as mock_post:
        mock_post.side_effect = Exception("network error")
        alerter.send("Test")  # must not raise


def test_send_fill_alert_includes_symbol_and_price() -> None:
    from agent.execution.types import ExecutionMode, Fill
    from agent.strategies.types import Action

    fill = Fill(
        order_id="paper-test",
        symbol="HDFCBANK",
        action=Action.ENTER_LONG,
        quantity=10,
        fill_price=Decimal("1710.00"),
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
        signal_id="HDFCBANK:enter_long:2024-01-02T09:15:00+05:30",
        strategy_name="trend_following_v1",
        execution_mode=ExecutionMode.PAPER,
    )
    alerter = _alerter()
    with patch("agent.monitoring.alerter.requests.post") as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        alerter.send_fill_alert(fill)
    call_json = mock_post.call_args.kwargs.get("json", {})
    assert "HDFCBANK" in str(call_json)
    assert "1710" in str(call_json)


def test_send_daily_summary_includes_nav() -> None:
    from agent.risk.types import PortfolioState

    state = PortfolioState(
        nav=Decimal("101500"),
        cash=Decimal("85000"),
        positions={},
        daily_pnl=Decimal("1500"),
        weekly_pnl=Decimal("1500"),
        peak_nav=Decimal("101500"),
        orders_today=2,
        last_order_time={},
        kill_switch_active=False,
        evaluation_time=datetime(2024, 1, 2, 15, 30, tzinfo=IST),
    )
    alerter = _alerter()
    with patch("agent.monitoring.alerter.requests.post") as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        alerter.send_daily_summary(state, session_count=3)
    call_json = mock_post.call_args.kwargs.get("json", {})
    assert "101500" in str(call_json)


def test_send_rejection_alert_does_not_raise() -> None:
    alerter = _alerter()
    with patch("agent.monitoring.alerter.requests.post") as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        alerter.send_rejection_alert("HDFCBANK", reason="max_open_positions")
