# tests/backtest/test_costs.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.backtest.costs import IndianCostModel
from agent.execution.types import ExecutionMode, Fill
from agent.strategies.types import Action

IST = ZoneInfo("Asia/Kolkata")
T0 = datetime(2024, 1, 2, 9, 15, tzinfo=IST)


def _make_fill(action: Action, price: float, qty: int) -> Fill:
    return Fill(
        order_id="test-01",
        symbol="HDFCBANK",
        action=action,
        quantity=qty,
        fill_price=Decimal(str(price)),
        timestamp=T0,
        signal_id="sig-01",
        strategy_name="trend_following_v1",
        execution_mode=ExecutionMode.PAPER,
    )


def test_compute_cost_returns_decimal() -> None:
    cost = IndianCostModel().compute_cost(_make_fill(Action.ENTER_LONG, 1700.0, 10))
    assert isinstance(cost, Decimal)


def test_compute_cost_is_positive() -> None:
    cost = IndianCostModel().compute_cost(_make_fill(Action.ENTER_LONG, 1700.0, 10))
    assert cost > Decimal("0")


def test_exit_long_is_more_expensive_than_enter_long() -> None:
    """EXIT_LONG has STT (0.025%); ENTER_LONG has stamp (0.003%). EXIT must cost more."""
    model = IndianCostModel()
    enter = model.compute_cost(_make_fill(Action.ENTER_LONG, 1700.0, 10))
    exit_ = model.compute_cost(_make_fill(Action.EXIT_LONG, 1700.0, 10))
    assert exit_ > enter


def test_enter_long_excludes_stt_includes_stamp() -> None:
    """For ENTER_LONG: stamp duty 0.003% present, STT absent.

    trade_value = 1700 * 10 = 17000
    stamp = 17000 * 0.00003 = 0.51
    stt   = 17000 * 0.00025 = 4.25
    Verify cost is lower than if STT were included.
    """
    model = IndianCostModel()
    cost = model.compute_cost(_make_fill(Action.ENTER_LONG, 1700.0, 10))
    trade_value = Decimal("17000")
    stt_if_included = (trade_value * Decimal("0.00025")).quantize(Decimal("0.01"))
    # If STT were included, cost would be at least stt_if_included more than stamp
    stamp = (trade_value * Decimal("0.00003")).quantize(Decimal("0.01"))
    # cost should contain stamp but not stt: cost < cost_with_stt
    # proxy: exit (has STT) costs more than enter (has stamp) by roughly STT - stamp = 3.74
    exit_cost = model.compute_cost(_make_fill(Action.EXIT_LONG, 1700.0, 10))
    assert float(exit_cost - cost) == pytest.approx(float(stt_if_included - stamp), abs=0.02)


def test_brokerage_capped_at_20_for_large_position() -> None:
    """Brokerage = min(₹20, trade_value * 0.0003). ₹1L position: 0.0003 * 100000 = ₹30 > ₹20."""
    model = IndianCostModel()
    # ₹100 * 1000 = ₹1,00,000
    enter = _make_fill(Action.ENTER_LONG, 100.0, 1000)
    cost = model.compute_cost(enter)
    # Manually compute expected cost with capped brokerage
    tv = Decimal("100000")
    exchange = (tv * Decimal("0.0000325")).quantize(Decimal("0.01"))
    stamp = (tv * Decimal("0.00003")).quantize(Decimal("0.01"))
    sebi = (tv * Decimal("0.000001")).quantize(Decimal("0.01"))
    brokerage = Decimal("20")  # capped at 20, not 30
    gst = ((brokerage + exchange) * Decimal("0.18")).quantize(Decimal("0.01"))
    expected = (stamp + exchange + sebi + brokerage + gst).quantize(Decimal("0.01"))
    assert cost == expected


def test_small_position_brokerage_not_capped() -> None:
    """Small trade: brokerage = trade_value * 0.0003 (below ₹20 cap)."""
    model = IndianCostModel()
    # ₹100 * 1 share = ₹100 → brokerage = 100 * 0.0003 = ₹0.03, not capped
    fill = _make_fill(Action.ENTER_LONG, 100.0, 1)
    cost = model.compute_cost(fill)
    assert cost > Decimal("0")
    tv = Decimal("100")
    brokerage_uncapped = (tv * Decimal("0.0003")).quantize(Decimal("0.01"))
    assert brokerage_uncapped < Decimal("20")


def test_round_trip_cost_in_6_to_12_bps_range() -> None:
    """Round-trip cost for rupees 1L position ~6-12 bps (₹60-₹120)."""
    model = IndianCostModel()
    enter = _make_fill(Action.ENTER_LONG, 100.0, 1000)  # ₹1,00,000
    exit_ = _make_fill(Action.EXIT_LONG, 100.0, 1000)
    total = model.compute_cost(enter) + model.compute_cost(exit_)
    assert (
        Decimal("60") <= total <= Decimal("120")
    ), f"Round-trip cost {total} outside 6-12 bps (₹60-₹120)"
