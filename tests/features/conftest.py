from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

IST = ZoneInfo("Asia/Kolkata")


def make_ohlcv_df(
    closes: list[float],
    symbol: str = "TEST",
    timeframe: str = "60m",
    start: datetime | None = None,
    interval_minutes: int = 60,
) -> pl.DataFrame:
    """Build a valid OHLCV DataFrame from a list of close prices."""
    if start is None:
        start = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
    n = len(closes)
    timestamps = [start + timedelta(minutes=interval_minutes * i) for i in range(n)]
    highs = [c * 1.005 for c in closes]
    lows = [c * 0.995 for c in closes]
    opens = [c * 0.999 for c in closes]
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "timeframe": [timeframe] * n,
            "timestamp": pl.Series(timestamps, dtype=pl.Datetime("us", "Asia/Kolkata")),
            "open": pl.Series(opens, dtype=pl.Float64),
            "high": pl.Series(highs, dtype=pl.Float64),
            "low": pl.Series(lows, dtype=pl.Float64),
            "close": pl.Series(closes, dtype=pl.Float64),
            "volume": pl.Series([100_000] * n, dtype=pl.Int64),
            "value": pl.Series([c * 100_000 for c in closes], dtype=pl.Float64),
        }
    )


@pytest.fixture
def trend_up_df() -> pl.DataFrame:
    """100 bars of steadily rising price: 1000 → 1990 (+10/bar)."""
    return make_ohlcv_df([1000.0 + i * 10 for i in range(100)])


@pytest.fixture
def trend_down_df() -> pl.DataFrame:
    """100 bars of steadily falling price: 2000 → 1010 (-10/bar)."""
    return make_ohlcv_df([2000.0 - i * 10 for i in range(100)])


@pytest.fixture
def constant_df() -> pl.DataFrame:
    """100 bars of constant price: 1500."""
    return make_ohlcv_df([1500.0] * 100)


@pytest.fixture
def ranging_df() -> pl.DataFrame:
    """100 bars oscillating between 1480 and 1520 (sine wave pattern)."""
    import math

    closes = [1500.0 + 20.0 * math.sin(i * 0.3) for i in range(100)]
    return make_ohlcv_df(closes)


@pytest.fixture
def two_session_df() -> pl.DataFrame:
    """20 bars across two trading sessions (dates) for VWAP reset testing."""
    session1_start = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
    session2_start = datetime(2024, 1, 3, 9, 15, tzinfo=IST)
    closes1 = [1000.0 + i for i in range(10)]
    closes2 = [2000.0 + i for i in range(10)]
    df1 = make_ohlcv_df(closes1, start=session1_start)
    df2 = make_ohlcv_df(closes2, start=session2_start)
    return pl.concat([df1, df2]).sort("timestamp")
