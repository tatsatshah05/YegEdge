from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from agent.strategies.types import Action


class ExecutionMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class Fill:
    """Record of a simulated (paper) or real (live) order fill.

    order_id: deterministic string, stable across retries.
    quantity: always positive — action field carries direction.
    execution_mode: always PAPER in Phase 7; LIVE gated by LIVE_TRADING_ENABLED.
    """

    order_id: str
    symbol: str
    action: Action
    quantity: int
    fill_price: Decimal
    timestamp: datetime
    signal_id: str
    strategy_name: str
    execution_mode: ExecutionMode
