from __future__ import annotations

import pytest

from agent.features.indicators import add_ema, add_rsi
from tests.features.conftest import make_ohlcv_df

# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------


def test_add_ema_returns_column_named_ema_period() -> None:
    df = make_ohlcv_df([100.0] * 30)
    result = add_ema(df, period=9)
    assert "ema_9" in result.columns


def test_add_ema_preserves_original_columns() -> None:
    df = make_ohlcv_df([100.0] * 30)
    result = add_ema(df, period=9)
    for col in df.columns:
        assert col in result.columns


def test_add_ema_constant_price_converges_to_that_price() -> None:
    """EMA of a constant price series should equal that price after warmup."""
    df = make_ohlcv_df([1500.0] * 50)
    result = add_ema(df, period=9)
    tail = result["ema_9"].tail(20).to_list()
    for v in tail:
        assert v is not None
        assert abs(v - 1500.0) < 0.01, f"EMA {v} should be ≈ 1500"


def test_add_ema_rising_prices_ema_lags_below_close() -> None:
    """On a rising series, EMA lags: it should be < current close for most bars."""
    closes = [1000.0 + i * 10 for i in range(50)]
    df = make_ohlcv_df(closes)
    result = add_ema(df, period=9)
    tail_ema = result["ema_9"].tail(30).to_list()
    tail_close = result["close"].tail(30).to_list()
    lagging = [e < c for e, c in zip(tail_ema, tail_close, strict=False) if e is not None]
    assert all(lagging), "EMA should lag below close on a rising series"


def test_add_ema_different_periods_produce_different_columns() -> None:
    df = make_ohlcv_df([100.0 + i for i in range(60)])
    result = add_ema(add_ema(df, period=9), period=21)
    assert "ema_9" in result.columns
    assert "ema_21" in result.columns
    ema9 = result["ema_9"][-1]
    ema21 = result["ema_21"][-1]
    close = result["close"][-1]
    assert abs(ema9 - close) < abs(ema21 - close), "Short EMA tracks closer to close"


def test_add_ema_custom_column() -> None:
    """add_ema can run on a column other than 'close'."""
    df = make_ohlcv_df([100.0] * 30)
    result = add_ema(df, period=5, column="open")
    assert "ema_5" in result.columns


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------


def test_add_rsi_returns_column_named_rsi_period() -> None:
    df = make_ohlcv_df([100.0 + i * 0.5 for i in range(40)])
    result = add_rsi(df, period=14)
    assert "rsi_14" in result.columns


def test_add_rsi_values_in_zero_to_hundred() -> None:
    """All non-null RSI values must be in [0, 100]."""
    closes = [100.0 + i * 2 for i in range(50)]
    df = make_ohlcv_df(closes)
    result = add_rsi(df, period=14)
    values = result["rsi_14"].drop_nulls().to_list()
    assert len(values) > 0
    for v in values:
        assert 0.0 <= v <= 100.0, f"RSI value {v} outside [0, 100]"


def test_add_rsi_rising_series_above_50() -> None:
    """A steadily rising price series should produce RSI > 50 after warmup."""
    closes = [1000.0 + i * 5 for i in range(60)]
    df = make_ohlcv_df(closes)
    result = add_rsi(df, period=14)
    tail = result["rsi_14"].tail(20).drop_nulls().to_list()
    assert all(v > 50 for v in tail), f"Rising series RSI should be > 50: {tail}"


def test_add_rsi_falling_series_below_50() -> None:
    """A steadily falling price series should produce RSI < 50 after warmup."""
    closes = [1000.0 - i * 5 for i in range(60)]
    df = make_ohlcv_df(closes)
    result = add_rsi(df, period=14)
    tail = result["rsi_14"].tail(20).drop_nulls().to_list()
    assert all(v < 50 for v in tail), f"Falling series RSI should be < 50: {tail}"


def test_add_rsi_all_gains_gives_100() -> None:
    """When every bar is up, avg_loss = 0, RSI should be 100."""
    closes = [float(100 + i) for i in range(30)]
    df = make_ohlcv_df(closes)
    result = add_rsi(df, period=14)
    tail = result["rsi_14"].tail(10).drop_nulls().to_list()
    assert all(v == 100.0 for v in tail), f"All-up series RSI should be 100: {tail}"


def test_add_rsi_all_losses_gives_0() -> None:
    """When every bar is down, avg_gain = 0, RSI should be 0."""
    closes = [float(100 - i) for i in range(30)]
    df = make_ohlcv_df(closes)
    result = add_rsi(df, period=14)
    tail = result["rsi_14"].tail(10).drop_nulls().to_list()
    assert all(v == 0.0 for v in tail), f"All-down series RSI should be 0: {tail}"


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------


def test_add_ema_invalid_period_raises() -> None:
    df = make_ohlcv_df([100.0] * 10)
    with pytest.raises(ValueError, match="period"):
        add_ema(df, period=0)


def test_add_rsi_invalid_period_raises() -> None:
    df = make_ohlcv_df([100.0] * 10)
    with pytest.raises(ValueError, match="period"):
        add_rsi(df, period=1)
