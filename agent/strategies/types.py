from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from agent.data.types import DataQuality


class Action(StrEnum):
    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class Signal:
    """Structured output from a strategy.

    Consumed by the decision engine, AI layer, and risk manager.
    All prices (suggested_stop, suggested_target) are Decimal to prevent float drift.
    Polars DataFrames use Float64 for indicators; Decimal only appears in Signal fields.
    """

    symbol: str
    action: Action
    confidence: float  # [0.0, 1.0]
    suggested_stop: Decimal  # ATR-based stop price
    suggested_target: Decimal  # R-multiple target price
    invalidation_condition: str  # human-readable description
    expected_r: float  # (target - entry) / (entry - stop)
    time_horizon_hours: int
    regime_fit: float  # [0.0, 1.0]
    data_quality: DataQuality
    strategy_name: str
    explanation: str  # <= 120 chars, structured
    timestamp: datetime  # bar-open time this signal was generated from

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not 0.0 <= self.regime_fit <= 1.0:
            raise ValueError(f"regime_fit must be in [0.0, 1.0], got {self.regime_fit}")
        if self.suggested_stop >= self.suggested_target:
            raise ValueError(
                f"suggested_stop ({self.suggested_stop}) must be < "
                f"suggested_target ({self.suggested_target})"
            )
        if self.timestamp.tzinfo is None:
            raise ValueError("Signal.timestamp must be timezone-aware (IST)")
