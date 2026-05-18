from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.execution.types import ExecutionMode, Fill
from agent.portfolio.tracker import PortfolioTracker
from agent.strategies.types import Action

IST = ZoneInfo("Asia/Kolkata")

T0 = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
T1 = datetime(2024, 1, 2, 10, 15, tzinfo=IST)
T2 = datetime(2024, 1, 2, 11, 15, tzinfo=IST)

INITIAL_NAV = Decimal("100000")


def _make_fill(
    symbol: str = "HDFCBANK",
    action: Action = Action.ENTER_LONG,
    quantity: int = 10,
    price: Decimal = Decimal("1700.00"),
    ts: datetime = T0,
) -> Fill:
    return Fill(
        order_id=f"paper-{symbol}-{ts.strftime('%H%M%S')}",
        symbol=symbol,
        action=action,
        quantity=quantity,
        fill_price=price,
        timestamp=ts,
        signal_id=f"{symbol}:enter_long:{ts.isoformat()}",
        strategy_name="trend_following_v1",
        execution_mode=ExecutionMode.PAPER,
    )


def _tracker() -> PortfolioTracker:
    return PortfolioTracker(
        initial_nav=INITIAL_NAV,
        initial_cash=INITIAL_NAV,
        start_time=T0,
    )


def test_initial_state_has_no_positions() -> None:
    tracker = _tracker()
    state = tracker.state
    assert len(state.positions) == 0
    assert state.nav == INITIAL_NAV
    assert state.cash == INITIAL_NAV


def test_apply_enter_long_reduces_cash() -> None:
    tracker = _tracker()
    fill = _make_fill(quantity=10, price=Decimal("1700"))
    state = tracker.apply_fill(fill, evaluation_time=T0)
    expected_cash = INITIAL_NAV - (10 * Decimal("1700"))
    assert state.cash == expected_cash


def test_apply_enter_long_creates_position() -> None:
    tracker = _tracker()
    fill = _make_fill(symbol="HDFCBANK", quantity=10, price=Decimal("1700"))
    state = tracker.apply_fill(fill, evaluation_time=T0)
    assert "HDFCBANK" in state.positions
    assert state.positions["HDFCBANK"].quantity == 10
    assert state.positions["HDFCBANK"].average_price == Decimal("1700")


def test_apply_exit_long_removes_position_and_adds_cash() -> None:
    tracker = _tracker()
    tracker.apply_fill(_make_fill(quantity=10, price=Decimal("1700")), evaluation_time=T0)
    exit_fill = _make_fill(action=Action.EXIT_LONG, quantity=10, price=Decimal("1750"), ts=T1)
    state = tracker.apply_fill(exit_fill, evaluation_time=T1)
    assert "HDFCBANK" not in state.positions
    expected_cash = INITIAL_NAV - 10 * Decimal("1700") + 10 * Decimal("1750")
    assert state.cash == expected_cash


def test_daily_pnl_positive_on_profitable_exit() -> None:
    tracker = _tracker()
    tracker.apply_fill(_make_fill(quantity=10, price=Decimal("1700")), evaluation_time=T0)
    exit_fill = _make_fill(action=Action.EXIT_LONG, quantity=10, price=Decimal("1750"), ts=T1)
    state = tracker.apply_fill(exit_fill, evaluation_time=T1)
    assert state.daily_pnl == Decimal("500")  # 10 * (1750 - 1700)


def test_mark_to_market_updates_nav() -> None:
    tracker = _tracker()
    tracker.apply_fill(_make_fill(quantity=10, price=Decimal("1700")), evaluation_time=T0)
    state = tracker.mark_to_market({"HDFCBANK": Decimal("1800")}, evaluation_time=T1)
    expected_nav = (INITIAL_NAV - 10 * Decimal("1700")) + 10 * Decimal("1800")
    assert state.nav == expected_nav


def test_orders_today_increments_per_fill() -> None:
    tracker = _tracker()
    tracker.apply_fill(_make_fill(symbol="HDFCBANK"), evaluation_time=T0)
    state = tracker.apply_fill(_make_fill(symbol="TCS"), evaluation_time=T0)
    assert state.orders_today == 2


def test_peak_nav_tracks_high_water_mark() -> None:
    tracker = _tracker()
    tracker.apply_fill(_make_fill(quantity=10, price=Decimal("1700")), evaluation_time=T0)
    state = tracker.mark_to_market({"HDFCBANK": Decimal("1900")}, evaluation_time=T1)
    peak_after_gain = state.peak_nav
    state2 = tracker.mark_to_market({"HDFCBANK": Decimal("1600")}, evaluation_time=T2)
    assert state2.peak_nav == peak_after_gain


def test_last_order_time_recorded() -> None:
    tracker = _tracker()
    fill = _make_fill(ts=T0)
    state = tracker.apply_fill(fill, evaluation_time=T0)
    assert "HDFCBANK" in state.last_order_time
    assert state.last_order_time["HDFCBANK"] == T0


def test_kill_switch_defaults_false() -> None:
    tracker = _tracker()
    assert tracker.state.kill_switch_active is False


def test_multiple_symbols_tracked_independently() -> None:
    tracker = _tracker()
    tracker.apply_fill(
        _make_fill(symbol="HDFCBANK", quantity=10, price=Decimal("1700")), evaluation_time=T0
    )
    state = tracker.apply_fill(
        _make_fill(symbol="TCS", quantity=5, price=Decimal("3000")), evaluation_time=T0
    )
    assert "HDFCBANK" in state.positions
    assert "TCS" in state.positions
    assert state.positions["HDFCBANK"].quantity == 10
    assert state.positions["TCS"].quantity == 5


def test_apply_enter_long_addon_uses_weighted_average() -> None:
    """Add-on entry must compute weighted average price, not simple average."""
    tracker = _tracker()
    # First entry: 10 shares at 1700
    tracker.apply_fill(_make_fill(quantity=10, price=Decimal("1700")), evaluation_time=T0)
    # Add-on: 10 more shares at 1800 → weighted avg = (10*1700 + 10*1800) / 20 = 1750
    fill2 = _make_fill(quantity=10, price=Decimal("1800"), ts=T1)
    state = tracker.apply_fill(fill2, evaluation_time=T1)
    assert state.positions["HDFCBANK"].quantity == 20
    assert state.positions["HDFCBANK"].average_price == Decimal("1750.00")


def test_partial_exit_leaves_correct_residual() -> None:
    """Partial exit reduces quantity but keeps position with original average_price."""
    tracker = _tracker()
    tracker.apply_fill(_make_fill(quantity=10, price=Decimal("1700")), evaluation_time=T0)
    # Sell only 4 of 10 shares
    exit_fill = _make_fill(action=Action.EXIT_LONG, quantity=4, price=Decimal("1750"), ts=T1)
    state = tracker.apply_fill(exit_fill, evaluation_time=T1)
    assert "HDFCBANK" in state.positions
    assert state.positions["HDFCBANK"].quantity == 6
    assert state.positions["HDFCBANK"].average_price == Decimal("1700")
    # P&L = 4 * (1750 - 1700) = 200
    assert state.daily_pnl == Decimal("200")


def test_exit_long_with_no_position_is_no_op() -> None:
    """EXIT_LONG on a symbol with no open position must not affect orders_today or cash."""
    tracker = _tracker()
    initial_state = tracker.state
    exit_fill = _make_fill(action=Action.EXIT_LONG, quantity=10, price=Decimal("1750"), ts=T0)
    state = tracker.apply_fill(exit_fill, evaluation_time=T0)
    # No orders counted, no cash change
    assert state.orders_today == 0
    assert state.cash == INITIAL_NAV
    assert len(state.positions) == 0
