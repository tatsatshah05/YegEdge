from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from agent.data.types import Position
from agent.strategies.types import Signal


class RiskVerdict(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class RejectionReason(StrEnum):
    NONE = "none"
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    SUSPECT_DATA_QUALITY = "suspect_data_quality"
    INSUFFICIENT_REWARD_RISK = "insufficient_reward_risk"
    OUTSIDE_TRADING_WINDOW = "outside_trading_window"
    MAX_POSITIONS_REACHED = "max_positions_reached"
    INSUFFICIENT_CASH = "insufficient_cash"
    DAILY_LOSS_CAP = "daily_loss_cap"
    WEEKLY_LOSS_CAP = "weekly_loss_cap"
    DRAWDOWN_BREACH = "drawdown_breach"
    MAX_ORDERS_TODAY = "max_orders_today"
    SYMBOL_COOLDOWN = "symbol_cooldown"
    ZERO_QUANTITY = "zero_quantity"


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Output of RiskManager.evaluate() — either an approved trade or a rejection.

    All monetary values are Decimal. quantity=0 and verdict=REJECTED always go together.
    """

    verdict: RiskVerdict
    quantity: int
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal
    risk_per_share: Decimal
    position_value: Decimal
    rejection_reason: RejectionReason
    rejection_detail: str
    signal: Signal


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Immutable snapshot of portfolio state passed into RiskManager.evaluate().

    dict fields (positions, last_order_time) make instances unhashable — never
    use PortfolioState as a dict key or in a set.
    """

    nav: Decimal
    cash: Decimal
    positions: dict[str, Position]
    daily_pnl: Decimal
    weekly_pnl: Decimal
    peak_nav: Decimal
    orders_today: int
    last_order_time: dict[str, datetime]
    kill_switch_active: bool
    evaluation_time: datetime
