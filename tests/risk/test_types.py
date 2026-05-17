from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.data.types import DataQuality, Position
from agent.risk.rules import RiskRules, load_risk_rules
from agent.risk.types import (
    PortfolioState,
    RejectionReason,
    RiskDecision,
    RiskVerdict,
)
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")
_TS = datetime(2024, 1, 2, 9, 45, tzinfo=IST)


def _signal() -> Signal:
    return Signal(
        symbol="HDFCBANK",
        action=Action.ENTER_LONG,
        confidence=0.75,
        suggested_stop=Decimal("1660.00"),
        suggested_target=Decimal("1780.00"),
        invalidation_condition="Close below EMA21",
        expected_r=2.0,
        time_horizon_hours=4,
        regime_fit=0.9,
        data_quality=DataQuality.OK,
        strategy_name="trend_following_v1",
        explanation="Test signal",
        timestamp=_TS,
    )


def test_risk_verdict_values() -> None:
    assert RiskVerdict.APPROVED == "approved"
    assert RiskVerdict.REJECTED == "rejected"


def test_rejection_reason_values() -> None:
    assert RejectionReason.NONE == "none"
    assert RejectionReason.KILL_SWITCH_ACTIVE == "kill_switch_active"
    assert RejectionReason.SUSPECT_DATA_QUALITY == "suspect_data_quality"
    assert RejectionReason.INSUFFICIENT_REWARD_RISK == "insufficient_reward_risk"
    assert RejectionReason.OUTSIDE_TRADING_WINDOW == "outside_trading_window"
    assert RejectionReason.MAX_POSITIONS_REACHED == "max_positions_reached"
    assert RejectionReason.INSUFFICIENT_CASH == "insufficient_cash"
    assert RejectionReason.DAILY_LOSS_CAP == "daily_loss_cap"
    assert RejectionReason.WEEKLY_LOSS_CAP == "weekly_loss_cap"
    assert RejectionReason.DRAWDOWN_BREACH == "drawdown_breach"
    assert RejectionReason.MAX_ORDERS_TODAY == "max_orders_today"
    assert RejectionReason.SYMBOL_COOLDOWN == "symbol_cooldown"
    assert RejectionReason.ZERO_QUANTITY == "zero_quantity"


def test_risk_decision_approved_construction() -> None:
    sig = _signal()
    decision = RiskDecision(
        verdict=RiskVerdict.APPROVED,
        quantity=4,
        entry_price=Decimal("1700.00"),
        stop_price=Decimal("1660.00"),
        target_price=Decimal("1780.00"),
        risk_per_share=Decimal("40.00"),
        position_value=Decimal("6800.00"),
        rejection_reason=RejectionReason.NONE,
        rejection_detail="",
        signal=sig,
    )
    assert decision.verdict == RiskVerdict.APPROVED
    assert decision.quantity == 4
    assert decision.rejection_reason == RejectionReason.NONE


def test_risk_decision_rejected_construction() -> None:
    sig = _signal()
    decision = RiskDecision(
        verdict=RiskVerdict.REJECTED,
        quantity=0,
        entry_price=Decimal("1700.00"),
        stop_price=Decimal("1660.00"),
        target_price=Decimal("1780.00"),
        risk_per_share=Decimal("0.00"),
        position_value=Decimal("0.00"),
        rejection_reason=RejectionReason.KILL_SWITCH_ACTIVE,
        rejection_detail="Kill switch is active — manual restart required",
        signal=sig,
    )
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.quantity == 0


def test_risk_decision_is_frozen() -> None:
    sig = _signal()
    decision = RiskDecision(
        verdict=RiskVerdict.APPROVED,
        quantity=4,
        entry_price=Decimal("1700.00"),
        stop_price=Decimal("1660.00"),
        target_price=Decimal("1780.00"),
        risk_per_share=Decimal("40.00"),
        position_value=Decimal("6800.00"),
        rejection_reason=RejectionReason.NONE,
        rejection_detail="",
        signal=sig,
    )
    with pytest.raises((AttributeError, TypeError)):
        decision.quantity = 99  # type: ignore[misc]


def test_portfolio_state_construction() -> None:
    port = PortfolioState(
        nav=Decimal("83000.00"),
        cash=Decimal("50000.00"),
        positions={},
        daily_pnl=Decimal("0.00"),
        weekly_pnl=Decimal("0.00"),
        peak_nav=Decimal("83000.00"),
        orders_today=0,
        last_order_time={},
        kill_switch_active=False,
        evaluation_time=datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    assert port.nav == Decimal("83000.00")
    assert port.kill_switch_active is False
    assert len(port.positions) == 0


def test_portfolio_state_with_positions() -> None:
    pos = Position(
        symbol="HDFCBANK",
        quantity=5,
        average_price=Decimal("1700.00"),
        product="MIS",
    )
    port = PortfolioState(
        nav=Decimal("83000.00"),
        cash=Decimal("41500.00"),
        positions={"HDFCBANK": pos},
        daily_pnl=Decimal("250.00"),
        weekly_pnl=Decimal("1200.00"),
        peak_nav=Decimal("84000.00"),
        orders_today=1,
        last_order_time={"HDFCBANK": datetime(2024, 1, 2, 9, 45, tzinfo=IST)},
        kill_switch_active=False,
        evaluation_time=datetime(2024, 1, 2, 10, 30, tzinfo=IST),
    )
    assert "HDFCBANK" in port.positions
    assert port.orders_today == 1


def test_load_risk_rules_parses_yaml() -> None:
    rules = load_risk_rules()
    assert rules.per_trade.max_risk_fraction == 0.005
    assert rules.per_trade.max_position_fraction == 0.10
    assert rules.per_trade.min_reward_risk == 1.5
    assert rules.portfolio.max_concurrent_positions == 6
    assert rules.portfolio.min_cash_fraction == 0.10
    assert rules.loss_caps.max_daily_loss_fraction == 0.02
    assert rules.loss_caps.max_weekly_loss_fraction == 0.05
    assert rules.loss_caps.max_drawdown_fraction == 0.08
    assert rules.frequency.max_new_orders_per_day == 4
    assert rules.frequency.symbol_cooldown_minutes == 30
    assert rules.windows.trade_start_ist == "09:45"
    assert rules.windows.trade_end_ist == "14:45"


def test_load_risk_rules_returns_risk_rules_instance() -> None:
    rules = load_risk_rules()
    assert isinstance(rules, RiskRules)
