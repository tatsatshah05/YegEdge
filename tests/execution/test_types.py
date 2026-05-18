from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.execution.types import ExecutionMode, Fill
from agent.strategies.types import Action

IST = ZoneInfo("Asia/Kolkata")


def _make_fill(
    symbol: str = "HDFCBANK",
    action: Action = Action.ENTER_LONG,
    quantity: int = 10,
    fill_price: Decimal = Decimal("1710.00"),
) -> Fill:
    return Fill(
        order_id="paper-HDFCBANK-20240102091500",
        symbol=symbol,
        action=action,
        quantity=quantity,
        fill_price=fill_price,
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
        signal_id="HDFCBANK:enter_long:2024-01-02T09:15:00+05:30",
        strategy_name="trend_following_v1",
        execution_mode=ExecutionMode.PAPER,
    )


def test_execution_mode_values() -> None:
    assert ExecutionMode.PAPER == "paper"
    assert ExecutionMode.LIVE == "live"


def test_fill_construction() -> None:
    fill = _make_fill()
    assert fill.symbol == "HDFCBANK"
    assert fill.quantity == 10
    assert fill.execution_mode == ExecutionMode.PAPER


def test_fill_is_frozen() -> None:
    fill = _make_fill()
    with pytest.raises(AttributeError):
        fill.quantity = 20  # type: ignore[misc]


def test_fill_exit_long_negative_quantity_convention() -> None:
    fill = _make_fill(action=Action.EXIT_LONG, quantity=10)
    assert fill.quantity == 10
    assert fill.action == Action.EXIT_LONG


def test_fill_order_id_is_string() -> None:
    fill = _make_fill()
    assert isinstance(fill.order_id, str)
    assert len(fill.order_id) > 0
