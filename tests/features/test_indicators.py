from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.features.indicators import add_adx, add_atr, add_ema, add_rsi, add_vwap
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


# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------


def test_add_atr_returns_column_named_atr_period() -> None:
    df = make_ohlcv_df([1000.0 + i for i in range(30)])
    result = add_atr(df, period=14)
    assert "atr_14" in result.columns


def test_add_atr_is_non_negative() -> None:
    """ATR must be >= 0 for all non-null values."""
    closes = [1000.0 + i * 3 for i in range(50)]
    df = make_ohlcv_df(closes)
    result = add_atr(df, period=14)
    values = result["atr_14"].drop_nulls().to_list()
    assert all(v >= 0.0 for v in values), "ATR must be non-negative"


def test_add_atr_constant_series_is_near_zero_after_warmup() -> None:
    """On a perfectly flat series (no gaps, constant OHLCV), ATR should approach zero."""
    IST = ZoneInfo("Asia/Kolkata")
    n = 60
    timestamps = [
        datetime(2024, 1, 2, 9, 15, tzinfo=IST) + timedelta(minutes=60 * i) for i in range(n)
    ]

    df = pl.DataFrame(
        {
            "symbol": ["TEST"] * n,
            "timeframe": ["60m"] * n,
            "timestamp": pl.Series(timestamps, dtype=pl.Datetime("us", "Asia/Kolkata")),
            "open": pl.Series([1500.0] * n, dtype=pl.Float64),
            "high": pl.Series([1500.0] * n, dtype=pl.Float64),
            "low": pl.Series([1500.0] * n, dtype=pl.Float64),
            "close": pl.Series([1500.0] * n, dtype=pl.Float64),
            "volume": pl.Series([100_000] * n, dtype=pl.Int64),
            "value": pl.Series([150_000_000.0] * n, dtype=pl.Float64),
        }
    )
    result = add_atr(df, period=14)
    tail = result["atr_14"].tail(20).drop_nulls().to_list()
    assert all(v < 0.001 for v in tail), f"Flat series ATR should be ~0: {tail}"


def test_add_atr_high_volatility_is_positive() -> None:
    """On a series with significant high-low range, ATR should be clearly positive."""
    import math

    closes = [1500.0 + 50 * math.sin(i * 0.5) for i in range(60)]
    df = make_ohlcv_df(closes)
    result = add_atr(df, period=14)
    tail = result["atr_14"].tail(20).drop_nulls().to_list()
    assert all(v > 5.0 for v in tail), f"Volatile series ATR should be > 5: {tail}"


def test_add_atr_preserves_original_columns() -> None:
    df = make_ohlcv_df([1000.0 + i for i in range(30)])
    result = add_atr(df, period=14)
    for col in df.columns:
        assert col in result.columns


def test_add_atr_invalid_period_raises() -> None:
    df = make_ohlcv_df([1000.0] * 10)
    with pytest.raises(ValueError, match="period"):
        add_atr(df, period=0)


# ---------------------------------------------------------------------------
# ADX tests
# ---------------------------------------------------------------------------


def test_add_adx_circuit_breaker_bars_produce_no_nan() -> None:
    """Bars with H=L=C (NSE circuit-breaker / halt) must not produce NaN in any ADX column."""
    IST = ZoneInfo("Asia/Kolkata")
    n = 60
    timestamps = [
        datetime(2024, 1, 2, 9, 15, tzinfo=IST) + timedelta(minutes=60 * i) for i in range(n)
    ]
    df = pl.DataFrame(
        {
            "symbol": ["TEST"] * n,
            "timeframe": ["60m"] * n,
            "timestamp": pl.Series(timestamps, dtype=pl.Datetime("us", "Asia/Kolkata")),
            "open": pl.Series([1500.0] * n, dtype=pl.Float64),
            "high": pl.Series([1500.0] * n, dtype=pl.Float64),
            "low": pl.Series([1500.0] * n, dtype=pl.Float64),
            "close": pl.Series([1500.0] * n, dtype=pl.Float64),
            "volume": pl.Series([100_000] * n, dtype=pl.Int64),
            "value": pl.Series([150_000_000.0] * n, dtype=pl.Float64),
        }
    )
    result = add_adx(df, period=14)
    for col in ("adx_14", "plus_di_14", "minus_di_14"):
        vals = result[col].to_list()
        nan_count = sum(1 for v in vals if isinstance(v, float) and math.isnan(v))
        assert nan_count == 0, f"{col} contains {nan_count} NaN values on circuit-breaker bars"


def test_add_adx_returns_expected_columns() -> None:
    df = make_ohlcv_df([1000.0 + i * 5 for i in range(60)])
    result = add_adx(df, period=14)
    assert "adx_14" in result.columns
    assert "plus_di_14" in result.columns
    assert "minus_di_14" in result.columns


def test_add_adx_values_in_zero_to_hundred() -> None:
    """ADX, +DI, -DI must all be in [0, 100] for non-null values."""
    closes = [1000.0 + i * 3 for i in range(80)]
    df = make_ohlcv_df(closes)
    result = add_adx(df, period=14)
    for col in ("adx_14", "plus_di_14", "minus_di_14"):
        values = result[col].drop_nulls().to_list()
        assert all(0.0 <= v <= 100.0 for v in values), f"{col} out of [0,100]"


def test_add_adx_uptrend_plus_di_dominates() -> None:
    """Strong uptrend → +DI should be > -DI after warmup."""
    closes = [1000.0 + i * 15 for i in range(80)]
    df = make_ohlcv_df(closes)
    result = add_adx(df, period=14)
    tail = result.tail(30)
    plus = tail["plus_di_14"].drop_nulls().to_list()
    minus = tail["minus_di_14"].drop_nulls().to_list()
    assert all(p > m for p, m in zip(plus, minus, strict=False)), "+DI should exceed -DI in uptrend"


def test_add_adx_downtrend_minus_di_dominates() -> None:
    """Strong downtrend → -DI should be > +DI after warmup."""
    closes = [2000.0 - i * 15 for i in range(80)]
    df = make_ohlcv_df(closes)
    result = add_adx(df, period=14)
    tail = result.tail(30)
    plus = tail["plus_di_14"].drop_nulls().to_list()
    minus = tail["minus_di_14"].drop_nulls().to_list()
    assert all(
        m > p for p, m in zip(plus, minus, strict=False)
    ), "-DI should exceed +DI in downtrend"


def test_add_adx_no_temp_columns_leaked() -> None:
    """No columns starting with '_' should appear in the result."""
    df = make_ohlcv_df([1000.0 + i * 5 for i in range(60)])
    result = add_adx(df, period=14)
    leaked = [c for c in result.columns if c.startswith("_")]
    assert leaked == [], f"Leaked temp columns: {leaked}"


def test_add_adx_invalid_period_raises() -> None:
    df = make_ohlcv_df([1000.0] * 10)
    with pytest.raises(ValueError, match="period"):
        add_adx(df, period=1)


# ---------------------------------------------------------------------------
# VWAP tests
# ---------------------------------------------------------------------------


def test_add_vwap_returns_vwap_column() -> None:
    df = make_ohlcv_df([1000.0 + i for i in range(20)])
    result = add_vwap(df)
    assert "vwap" in result.columns


def test_add_vwap_within_session_high_low_bounds() -> None:
    """VWAP on a constant-price session lies between that session's low and high."""
    # Use constant price so VWAP = typical_price = constant, within [low, high]
    df = make_ohlcv_df([1500.0] * 30)
    result = add_vwap(df)
    vwap_vals = result["vwap"].to_list()
    lows = result["low"].to_list()
    highs = result["high"].to_list()
    for v, lo, hi in zip(vwap_vals, lows, highs, strict=False):
        if v is not None:
            assert lo <= v <= hi, f"VWAP {v} outside [{lo}, {hi}]"


def test_add_vwap_resets_between_sessions(two_session_df: pl.DataFrame) -> None:
    """VWAP must reset to the first bar's typical price at each session boundary."""
    result = add_vwap(two_session_df)
    result = result.with_columns(pl.col("timestamp").dt.date().alias("_date"))
    for date_val in result["_date"].unique().sort().to_list():
        session = result.filter(pl.col("_date") == date_val)
        first_row = session.row(0, named=True)
        tp0 = (first_row["high"] + first_row["low"] + first_row["close"]) / 3.0
        vwap0 = first_row["vwap"]
        assert (
            abs(vwap0 - tp0) < 0.001
        ), f"Session {date_val}: VWAP[0]={vwap0} should equal typical_price[0]={tp0}"


def test_add_vwap_no_temp_columns_leaked() -> None:
    df = make_ohlcv_df([1000.0 + i for i in range(20)])
    result = add_vwap(df)
    leaked = [c for c in result.columns if c.startswith("_")]
    assert leaked == [], f"Leaked temp columns: {leaked}"


def test_add_vwap_preserves_original_columns() -> None:
    df = make_ohlcv_df([1000.0 + i for i in range(20)])
    result = add_vwap(df)
    for col in df.columns:
        assert col in result.columns
