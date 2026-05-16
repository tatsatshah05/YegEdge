from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

Timeframe = Literal["15m", "60m", "1d"]


class DataQuality(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    SUSPECT = "suspect"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    timeframe: Timeframe
    timestamp: datetime  # bar-open, must be IST-aware
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    value: Decimal  # turnover in INR
    data_quality: DataQuality

    def __post_init__(self) -> None:
        import datetime as _dt

        if self.timestamp.tzinfo is None:
            raise ValueError("Bar.timestamp must be timezone-aware (IST)")
        # IST is UTC+05:30 (no DST). Any timezone not equal to +05:30 offset is wrong.
        ist_offset = _dt.timedelta(hours=5, minutes=30)
        actual_offset = self.timestamp.utcoffset()
        if actual_offset != ist_offset:
            raise ValueError(
                f"Bar.timestamp must be IST (UTC+05:30), got UTC offset {actual_offset!r}"
            )


@dataclass(frozen=True, slots=True)
class Order:
    symbol: str
    action: Literal["BUY", "SELL"]
    quantity: int
    order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"]
    price: Decimal | None
    trigger_price: Decimal | None
    product: Literal["MIS", "CNC"]
    client_order_id: str  # idempotency key


@dataclass(frozen=True, slots=True)
class OrderAck:
    order_id: str
    client_order_id: str
    status: Literal["OPEN", "COMPLETE", "REJECTED", "CANCELLED"]
    message: str


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    quantity: int  # negative = short
    average_price: Decimal
    product: Literal["MIS", "CNC"]


@dataclass(frozen=True, slots=True)
class Discrepancy:
    symbol: str
    local_quantity: int
    broker_quantity: int
    reason: str
