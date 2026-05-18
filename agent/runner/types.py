from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from agent.execution.types import Fill


@dataclass(frozen=True, slots=True)
class DailySessionResult:
    """Summary of one completed paper trading session."""

    session_date: date
    bars_processed: int
    signals_generated: int
    decisions_made: int
    fills: tuple[Fill, ...]
    rejections: int
    ai_cache_hits: int
    final_nav: Decimal
    daily_pnl: Decimal
    peak_nav: Decimal
