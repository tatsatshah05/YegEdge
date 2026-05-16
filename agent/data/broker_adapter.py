from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime

import polars as pl

from agent.data.types import Discrepancy, Order, OrderAck, Position


class BrokerAdapter(ABC):
    """Abstract broker adapter. The rest of the system only knows this interface.

    Concrete implementations: UpstoxAdapter, DhanAdapter, KiteAdapter.
    Swap broker by changing which concrete class is instantiated — nothing else changes.
    """

    @abstractmethod
    def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Return OHLCV bars as a Polars DataFrame.

        Columns required: symbol (Utf8), timeframe (Utf8), timestamp (Datetime[us,
        Asia/Kolkata]), open (Float64), high (Float64), low (Float64), close
        (Float64), volume (Int64), value (Float64).
        All timestamps must be IST-aware.
        """

    @abstractmethod
    async def stream_live(
        self,
        symbols: list[str],
        callback: Callable[[pl.DataFrame], None],
    ) -> None:
        """Subscribe to live tick stream.

        Callback receives one-row DataFrames per tick. Runs until cancelled via
        asyncio task cancellation.
        """

    @abstractmethod
    def place_order(self, order: Order) -> OrderAck:
        """Submit an order. Idempotency key is order.client_order_id."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel a live order by broker-assigned order_id."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return all open positions from the broker."""

    @abstractmethod
    def reconcile(self, expected: list[Position]) -> list[Discrepancy]:
        """Compare local position state against broker state.

        Return any mismatches.
        """
