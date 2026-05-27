from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True, slots=True)
class ClosedBar:
    symbol: str
    timeframe: str
    bar_open: datetime  # timezone-aware bar-start timestamp
    open: float
    high: float
    low: float
    close: float
    tick_count: int  # volume proxy — LTPC mode has no per-tick volume

    def to_dataframe(self) -> pl.DataFrame:
        tz_key = str(self.bar_open.tzinfo) if self.bar_open.tzinfo else "Asia/Kolkata"
        return pl.DataFrame(
            {
                "symbol": pl.Series([self.symbol], dtype=pl.Utf8),
                "timeframe": pl.Series([self.timeframe], dtype=pl.Utf8),
                "timestamp": pl.Series(
                    [self.bar_open],
                    dtype=pl.Datetime("us", tz_key),
                ),
                "open": pl.Series([self.open], dtype=pl.Float64),
                "high": pl.Series([self.high], dtype=pl.Float64),
                "low": pl.Series([self.low], dtype=pl.Float64),
                "close": pl.Series([self.close], dtype=pl.Float64),
                "volume": pl.Series([self.tick_count], dtype=pl.Int64),
                "value": pl.Series([self.close * self.tick_count], dtype=pl.Float64),
                "data_quality": pl.Series(["ok"], dtype=pl.Utf8),
            }
        )


class BarBuilder:
    """Aggregates LTP ticks into OHLCV bars aligned to a configurable market open.

    LTPC WebSocket mode delivers no per-tick volume; tick_count is used as a
    volume proxy. This is acceptable because TrendFollowingStrategy only uses
    price-based indicators (EMA, ATR, ADX).

    NSE defaults: tz=Asia/Kolkata, market_open_hour=9, market_open_minute=15
    NYSE defaults: tz=America/New_York, market_open_hour=9, market_open_minute=30
    """

    _TIMEFRAME_MINUTES: Final[dict[str, int]] = {
        "5m": 5,
        "15m": 15,
        "60m": 60,
        "1d": 375,  # 9:15 → 15:30 = 375 minutes
    }

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        tz: ZoneInfo = IST,
        market_open_hour: int = 9,
        market_open_minute: int = 15,
    ) -> None:
        self._symbol = symbol
        self._timeframe = timeframe
        self._bar_minutes = self._TIMEFRAME_MINUTES[timeframe]
        self._tz = tz
        self._market_open_hour = market_open_hour
        self._market_open_minute = market_open_minute
        self._current_slot: datetime | None = None
        self._open: float | None = None
        self._high: float | None = None
        self._low: float | None = None
        self._close: float | None = None
        self._tick_count: int = 0

    def on_tick(self, ltp: float, ts: datetime) -> ClosedBar | None:
        """Process one LTP tick. Returns a ClosedBar when a bar boundary is crossed."""
        slot = self._bar_start_for(ts)

        if self._current_slot is None:
            self._current_slot = slot
            self._start_bar(ltp)
            return None

        if slot == self._current_slot:
            self._update_bar(ltp)
            return None

        # New slot — close the current bar, start a new one
        closed = self._close_current()
        self._current_slot = slot
        self._start_bar(ltp)
        return closed

    def force_close(self) -> ClosedBar | None:
        """Close the current in-progress bar (call at session end: 15:30 IST)."""
        if self._current_slot is None or self._tick_count == 0:
            return None
        return self._close_current()

    def _bar_start_for(self, ts: datetime) -> datetime:
        local = ts.astimezone(self._tz)
        market_open = local.replace(
            hour=self._market_open_hour,
            minute=self._market_open_minute,
            second=0,
            microsecond=0,
        )
        if local <= market_open:
            return market_open
        elapsed = int((local - market_open).total_seconds()) // 60
        slot = elapsed // self._bar_minutes
        return market_open + timedelta(minutes=slot * self._bar_minutes)

    def _start_bar(self, ltp: float) -> None:
        self._open = ltp
        self._high = ltp
        self._low = ltp
        self._close = ltp
        self._tick_count = 1

    def _update_bar(self, ltp: float) -> None:
        assert self._high is not None and self._low is not None
        if ltp > self._high:
            self._high = ltp
        if ltp < self._low:
            self._low = ltp
        self._close = ltp
        self._tick_count += 1

    def _close_current(self) -> ClosedBar:
        assert self._current_slot is not None
        assert self._open is not None
        assert self._high is not None
        assert self._low is not None
        assert self._close is not None
        bar = ClosedBar(
            symbol=self._symbol,
            timeframe=self._timeframe,
            bar_open=self._current_slot,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            tick_count=self._tick_count,
        )
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self._tick_count = 0
        self._current_slot = None
        return bar
