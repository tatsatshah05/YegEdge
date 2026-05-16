from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from agent.features.indicators import add_ema, add_rsi

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _make_df(closes: list[float]) -> pl.DataFrame:
    """Minimal OHLCV DataFrame from a list of close prices (60m bars from 2024-01-02)."""
    n = len(closes)
    start = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
    timestamps = [start + timedelta(minutes=60 * i) for i in range(n)]
    highs = [c * 1.005 for c in closes]
    lows = [c * 0.995 for c in closes]
    opens = [c * 0.999 for c in closes]
    return pl.DataFrame(
        {
            "symbol": ["TEST"] * n,
            "timeframe": ["60m"] * n,
            "timestamp": pl.Series(timestamps, dtype=pl.Datetime("us", "Asia/Kolkata")),
            "open": pl.Series(opens, dtype=pl.Float64),
            "high": pl.Series(highs, dtype=pl.Float64),
            "low": pl.Series(lows, dtype=pl.Float64),
            "close": pl.Series(closes, dtype=pl.Float64),
            "volume": pl.Series([100_000] * n, dtype=pl.Int64),
            "value": pl.Series([c * 100_000 for c in closes], dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------


def test_add_ema_returns_column_named_ema_period() -> None:
    df = _make_df([100.0] * 30)
    result = add_ema(df, period=9)
    assert "ema_9" in result.columns


def test_add_ema_preserves_original_columns() -> None:
    df = _make_df([100.0] * 30)
    result = add_ema(df, period=9)
    for col in df.columns:
        assert col in result.columns


def test_add_ema_constant_price_converges_to_that_price() -> None:
    """EMA of a constant price series should equal that price after warmup."""
    df = _make_df([1500.0] * 50)
    result = add_ema(df, period=9)
    tail = result["ema_9"].tail(20).to_list()
    for v in tail:
        assert v is not None
        assert abs(v - 1500.0) < 0.01, f"EMA {v} should be ≈ 1500"


def test_add_ema_rising_prices_ema_lags_below_close() -> None:
    """On a rising series, EMA lags: it should be < current close for most bars."""
    closes = [1000.0 + i * 10 for i in range(50)]
    df = _make_df(closes)
    result = add_ema(df, period=9)
    tail_ema = result["ema_9"].tail(30).to_list()
    tail_close = result["close"].tail(30).to_list()
    lagging = [e < c for e, c in zip(tail_ema, tail_close, strict=False) if e is not None]
    assert all(lagging), "EMA should lag below close on a rising series"


def test_add_ema_different_periods_produce_different_columns() -> None:
    df = _make_df([100.0 + i for i in range(60)])
    result = add_ema(add_ema(df, period=9), period=21)
    assert "ema_9" in result.columns
    assert "ema_21" in result.columns
    ema9 = result["ema_9"][-1]
    ema21 = result["ema_21"][-1]
    close = result["close"][-1]
    assert abs(ema9 - close) < abs(ema21 - close), "Short EMA tracks closer to close"


def test_add_ema_custom_column() -> None:
    """add_ema can run on a column other than 'close'."""
    df = _make_df([100.0] * 30)
    result = add_ema(df, period=5, column="open")
    assert "ema_5" in result.columns


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------


def test_add_rsi_returns_column_named_rsi_period() -> None:
    df = _make_df([100.0 + i * 0.5 for i in range(40)])
    result = add_rsi(df, period=14)
    assert "rsi_14" in result.columns


def test_add_rsi_values_in_zero_to_hundred() -> None:
    """All non-null RSI values must be in [0, 100]."""
    closes = [100.0 + i * 2 for i in range(50)]
    df = _make_df(closes)
    result = add_rsi(df, period=14)
    values = result["rsi_14"].drop_nulls().to_list()
    assert len(values) > 0
    for v in values:
        assert 0.0 <= v <= 100.0, f"RSI value {v} outside [0, 100]"


def test_add_rsi_rising_series_above_50() -> None:
    """A steadily rising price series should produce RSI > 50 after warmup."""
    closes = [1000.0 + i * 5 for i in range(60)]
    df = _make_df(closes)
    result = add_rsi(df, period=14)
    tail = result["rsi_14"].tail(20).drop_nulls().to_list()
    assert all(v > 50 for v in tail), f"Rising series RSI should be > 50: {tail}"


def test_add_rsi_falling_series_below_50() -> None:
    """A steadily falling price series should produce RSI < 50 after warmup."""
    closes = [1000.0 - i * 5 for i in range(60)]
    df = _make_df(closes)
    result = add_rsi(df, period=14)
    tail = result["rsi_14"].tail(20).drop_nulls().to_list()
    assert all(v < 50 for v in tail), f"Falling series RSI should be < 50: {tail}"


def test_add_rsi_all_gains_gives_100() -> None:
    """When every bar is up, avg_loss = 0, RSI should be 100."""
    closes = [float(100 + i) for i in range(30)]
    df = _make_df(closes)
    result = add_rsi(df, period=14)
    tail = result["rsi_14"].tail(10).drop_nulls().to_list()
    assert all(v == 100.0 for v in tail), f"All-up series RSI should be 100: {tail}"
