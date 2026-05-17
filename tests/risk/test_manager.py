from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.data.types import DataQuality, Position
from agent.risk.manager import RiskManager
from agent.risk.rules import load_risk_rules
from agent.risk.types import (
    PortfolioState,
    RejectionReason,
    RiskVerdict,
)
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")

_NAV = Decimal("83000.00")


def _signal(
    *,
    action: Action = Action.ENTER_LONG,
    symbol: str = "HDFCBANK",
    confidence: float = 0.75,
    suggested_stop: Decimal = Decimal("1660.00"),
    suggested_target: Decimal = Decimal("1780.00"),
    data_quality: DataQuality = DataQuality.OK,
) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        suggested_stop=suggested_stop,
        suggested_target=suggested_target,
        invalidation_condition="Close below EMA21",
        expected_r=2.0,
        time_horizon_hours=4,
        regime_fit=0.9,
        data_quality=data_quality,
        strategy_name="trend_following_v1",
        explanation="Test signal",
        timestamp=datetime(2024, 1, 2, 9, 45, tzinfo=IST),
    )


def _portfolio(
    *,
    nav: Decimal = _NAV,
    cash: Decimal = Decimal("50000.00"),
    positions: dict | None = None,
    daily_pnl: Decimal = Decimal("0.00"),
    weekly_pnl: Decimal = Decimal("0.00"),
    peak_nav: Decimal | None = None,
    orders_today: int = 0,
    last_order_time: dict | None = None,
    kill_switch_active: bool = False,
    evaluation_time: datetime | None = None,
) -> PortfolioState:
    return PortfolioState(
        nav=nav,
        cash=cash,
        positions=positions or {},
        daily_pnl=daily_pnl,
        weekly_pnl=weekly_pnl,
        peak_nav=peak_nav if peak_nav is not None else nav,
        orders_today=orders_today,
        last_order_time=last_order_time or {},
        kill_switch_active=kill_switch_active,
        evaluation_time=evaluation_time or datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(load_risk_rules())


def test_approved_signal_returns_approved(rm: RiskManager) -> None:
    decision = rm.evaluate(_signal(), _portfolio(), Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED
    assert decision.quantity > 0
    assert decision.rejection_reason == RejectionReason.NONE


def test_approved_decision_has_correct_prices(rm: RiskManager) -> None:
    decision = rm.evaluate(_signal(), _portfolio(), Decimal("1700.00"))
    assert decision.entry_price == Decimal("1700.00")
    assert decision.stop_price == Decimal("1660.00")
    assert decision.target_price == Decimal("1780.00")
    assert decision.risk_per_share == Decimal("40.00")


def test_quantity_capped_by_position_fraction(rm: RiskManager) -> None:
    # NAV=83000, max_risk=0.5% → 415, risk_per_share=40 → qty_by_risk=10
    # max_position=10% → 8300, at 1700/share → qty_by_size=4
    # Final = min(10, 4) = 4
    decision = rm.evaluate(_signal(), _portfolio(), Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED
    assert decision.quantity == 4


def test_position_value_equals_quantity_times_entry(rm: RiskManager) -> None:
    decision = rm.evaluate(_signal(), _portfolio(), Decimal("1700.00"))
    expected_value = Decimal(str(decision.quantity)) * Decimal("1700.00")
    assert decision.position_value == expected_value


def test_zero_quantity_rejects(rm: RiskManager) -> None:
    # NAV=100, max_risk=0.5% → 0.50, risk_per_share=40 → qty=0 → ZERO_QUANTITY
    port = _portfolio(nav=Decimal("100.00"), cash=Decimal("50.00"), peak_nav=Decimal("100.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.ZERO_QUANTITY
    assert decision.quantity == 0


def test_kill_switch_active_rejects_enter_long(rm: RiskManager) -> None:
    decision = rm.evaluate(_signal(), _portfolio(kill_switch_active=True), Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.KILL_SWITCH_ACTIVE
    assert decision.quantity == 0


def test_kill_switch_active_rejects_exit_long(rm: RiskManager) -> None:
    port = _portfolio(
        kill_switch_active=True,
        positions={
            "HDFCBANK": Position(
                symbol="HDFCBANK", quantity=5, average_price=Decimal("1700"), product="MIS"
            )
        },
    )
    decision = rm.evaluate(_signal(action=Action.EXIT_LONG), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.KILL_SWITCH_ACTIVE


def test_suspect_data_quality_rejects(rm: RiskManager) -> None:
    decision = rm.evaluate(
        _signal(data_quality=DataQuality.SUSPECT), _portfolio(), Decimal("1700.00")
    )
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.SUSPECT_DATA_QUALITY


def test_missing_data_quality_rejects(rm: RiskManager) -> None:
    decision = rm.evaluate(
        _signal(data_quality=DataQuality.MISSING), _portfolio(), Decimal("1700.00")
    )
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.SUSPECT_DATA_QUALITY


def test_partial_data_quality_is_allowed(rm: RiskManager) -> None:
    decision = rm.evaluate(
        _signal(data_quality=DataQuality.PARTIAL), _portfolio(), Decimal("1700.00")
    )
    assert decision.verdict == RiskVerdict.APPROVED


def test_insufficient_reward_risk_rejects(rm: RiskManager) -> None:
    # entry=1700, stop=1685, target=1710 → R/R = 10/15 = 0.67 < 1.5
    sig = _signal(suggested_stop=Decimal("1685.00"), suggested_target=Decimal("1710.00"))
    decision = rm.evaluate(sig, _portfolio(), Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.INSUFFICIENT_REWARD_RISK


def test_reward_risk_exactly_at_minimum_is_allowed(rm: RiskManager) -> None:
    # R/R = 1.5 exactly: entry=1700, stop=1660 (risk=40), target=1760 (reward=60)
    sig = _signal(suggested_stop=Decimal("1660.00"), suggested_target=Decimal("1760.00"))
    decision = rm.evaluate(sig, _portfolio(), Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_before_trading_window_rejects(rm: RiskManager) -> None:
    port = _portfolio(evaluation_time=datetime(2024, 1, 2, 9, 30, tzinfo=IST))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.OUTSIDE_TRADING_WINDOW


def test_after_trading_window_rejects(rm: RiskManager) -> None:
    port = _portfolio(evaluation_time=datetime(2024, 1, 2, 15, 0, tzinfo=IST))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.OUTSIDE_TRADING_WINDOW


def test_exactly_at_window_open_is_allowed(rm: RiskManager) -> None:
    port = _portfolio(evaluation_time=datetime(2024, 1, 2, 9, 45, tzinfo=IST))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_exactly_at_window_close_is_allowed(rm: RiskManager) -> None:
    port = _portfolio(evaluation_time=datetime(2024, 1, 2, 14, 45, tzinfo=IST))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_max_concurrent_positions_rejects(rm: RiskManager) -> None:
    positions = {
        f"SYM{i}": Position(
            symbol=f"SYM{i}", quantity=10, average_price=Decimal("100"), product="MIS"
        )
        for i in range(6)
    }
    port = _portfolio(positions=positions)
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.MAX_POSITIONS_REACHED


def test_five_positions_allows_sixth(rm: RiskManager) -> None:
    positions = {
        f"SYM{i}": Position(
            symbol=f"SYM{i}", quantity=10, average_price=Decimal("100"), product="MIS"
        )
        for i in range(5)
    }
    port = _portfolio(positions=positions)
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_cash_below_minimum_buffer_rejects(rm: RiskManager) -> None:
    # NAV=83000, min_cash=10% = 8300. cash=8000 < 8300 → reject
    port = _portfolio(nav=Decimal("83000.00"), cash=Decimal("8000.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.INSUFFICIENT_CASH


def test_cash_exactly_at_minimum_buffer_is_allowed(rm: RiskManager) -> None:
    # cash=8300 == 10% of 83000 → allowed (rule is strict <, not <=)
    port = _portfolio(nav=Decimal("83000.00"), cash=Decimal("8300.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_daily_loss_cap_rejects(rm: RiskManager) -> None:
    # NAV=83000, max_daily_loss=2% = 1660. daily_pnl=-1700 < -1660 → reject
    port = _portfolio(nav=Decimal("83000.00"), daily_pnl=Decimal("-1700.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.DAILY_LOSS_CAP


def test_daily_loss_exactly_at_cap_rejects(rm: RiskManager) -> None:
    port = _portfolio(nav=Decimal("83000.00"), daily_pnl=Decimal("-1660.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.DAILY_LOSS_CAP


def test_daily_loss_below_cap_is_allowed(rm: RiskManager) -> None:
    port = _portfolio(nav=Decimal("83000.00"), daily_pnl=Decimal("-1000.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_weekly_loss_cap_rejects(rm: RiskManager) -> None:
    # NAV=83000, max_weekly_loss=5% = 4150. weekly_pnl=-4200 → reject
    port = _portfolio(nav=Decimal("83000.00"), weekly_pnl=Decimal("-4200.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.WEEKLY_LOSS_CAP


def test_weekly_loss_exactly_at_cap_rejects(rm: RiskManager) -> None:
    port = _portfolio(nav=Decimal("83000.00"), weekly_pnl=Decimal("-4150.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.WEEKLY_LOSS_CAP


def test_drawdown_breach_rejects(rm: RiskManager) -> None:
    # peak=100000, nav=91500 → drawdown=8.5% > 8% → reject
    port = _portfolio(
        nav=Decimal("91500.00"),
        cash=Decimal("30000.00"),
        peak_nav=Decimal("100000.00"),
    )
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.DRAWDOWN_BREACH


def test_drawdown_exactly_at_threshold_rejects(rm: RiskManager) -> None:
    # drawdown = 8.0% exactly → reject (>= threshold)
    port = _portfolio(
        nav=Decimal("92000.00"),
        cash=Decimal("30000.00"),
        peak_nav=Decimal("100000.00"),
    )
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.DRAWDOWN_BREACH


def test_drawdown_below_threshold_is_allowed(rm: RiskManager) -> None:
    # peak=100000, nav=95000 → drawdown=5% < 8% → allowed
    port = _portfolio(
        nav=Decimal("95000.00"),
        cash=Decimal("30000.00"),
        peak_nav=Decimal("100000.00"),
    )
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_max_orders_today_rejects(rm: RiskManager) -> None:
    port = _portfolio(orders_today=4)
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.MAX_ORDERS_TODAY


def test_three_orders_today_allows_fourth(rm: RiskManager) -> None:
    port = _portfolio(orders_today=3)
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_symbol_in_cooldown_rejects(rm: RiskManager) -> None:
    eval_time = datetime(2024, 1, 2, 10, 0, tzinfo=IST)
    last_time = eval_time - timedelta(minutes=15)
    port = _portfolio(
        last_order_time={"HDFCBANK": last_time},
        evaluation_time=eval_time,
    )
    decision = rm.evaluate(_signal(symbol="HDFCBANK"), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.SYMBOL_COOLDOWN


def test_symbol_cooldown_expired_allows_order(rm: RiskManager) -> None:
    eval_time = datetime(2024, 1, 2, 10, 0, tzinfo=IST)
    last_time = eval_time - timedelta(minutes=31)
    port = _portfolio(
        last_order_time={"HDFCBANK": last_time},
        evaluation_time=eval_time,
    )
    decision = rm.evaluate(_signal(symbol="HDFCBANK"), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_different_symbol_not_in_cooldown(rm: RiskManager) -> None:
    eval_time = datetime(2024, 1, 2, 10, 0, tzinfo=IST)
    port = _portfolio(
        last_order_time={"INFY": eval_time - timedelta(minutes=5)},
        evaluation_time=eval_time,
    )
    decision = rm.evaluate(_signal(symbol="HDFCBANK"), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_exit_long_bypasses_entry_checks(rm: RiskManager) -> None:
    eval_time = datetime(2024, 1, 2, 15, 30, tzinfo=IST)
    port = _portfolio(
        orders_today=10,
        evaluation_time=eval_time,
        positions={
            "HDFCBANK": Position(
                symbol="HDFCBANK",
                quantity=5,
                average_price=Decimal("1700.00"),
                product="MIS",
            )
        },
    )
    decision = rm.evaluate(_signal(action=Action.EXIT_LONG), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_exit_long_quantity_matches_existing_position(rm: RiskManager) -> None:
    port = _portfolio(
        positions={
            "HDFCBANK": Position(
                symbol="HDFCBANK",
                quantity=7,
                average_price=Decimal("1700.00"),
                product="MIS",
            )
        }
    )
    decision = rm.evaluate(_signal(action=Action.EXIT_LONG), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED
    assert decision.quantity == 7


def test_exit_long_with_no_position_returns_zero_quantity(rm: RiskManager) -> None:
    port = _portfolio(positions={})
    decision = rm.evaluate(_signal(action=Action.EXIT_LONG), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED
    assert decision.quantity == 0
