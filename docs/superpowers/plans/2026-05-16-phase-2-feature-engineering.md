# Phase 2 — Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the feature engineering layer: pure Polars indicator functions (EMA, RSI, ATR, ADX, VWAP), a KMeans-based regime detector, and a FeaturePipeline orchestrator under `agent/features/`.

**Architecture:** All indicators are pure functions `(pl.DataFrame, **params) → pl.DataFrame` that add named columns. They chain cleanly via the `FeaturePipeline` orchestrator. Regime detection uses scikit-learn KMeans on three features (ADX, ATR%, 20-bar momentum) — the one interpretable ML exception permitted in V1. No pandas, no I/O, no side effects anywhere in this module.

**Tech Stack:** Python 3.11+, Polars ≥ 1.9 (native indicator math, `.ewm_mean`, `.over`), NumPy ≥ 1.26, scikit-learn ≥ 1.4 (KMeans), structlog, pytest.

**Critical conventions (carry these through every task):**
- `from __future__ import annotations` at the top of every `.py` file
- `logger = structlog.get_logger()` (not `log`)
- Black-formatted, Ruff-clean, 100-char line limit
- No `print()` — use `structlog`
- All prices in Polars are `Float64`; never `Decimal` in DataFrames
- Temp columns in Polars are prefixed with `_` and dropped before returning
- `frozen=True, slots=True` on dataclasses

---

## File Map

```
agent/features/
    __init__.py             # empty package marker
    indicators.py           # Pure functions: add_ema, add_rsi, add_atr, add_adx, add_vwap
    regime.py               # Regime(StrEnum) + RegimeDetector(KMeans wrapper)
    pipeline.py             # FeaturePipeline: chains indicators + regime

tests/features/
    __init__.py             # empty package marker
    conftest.py             # shared fixtures: make_ohlcv_df, trend/range/volatile helpers
    test_indicators.py      # one test class per indicator function
    test_regime.py          # RegimeDetector fit/predict/label assignment
    test_pipeline.py        # FeaturePipeline end-to-end
```

**Modified files:**
- `requirements.txt` — add `scikit-learn>=1.4`

**Review priority:** indicators.py → regime.py → pipeline.py. Indicators are the foundation; regime depends on adx_14 and atr_14 columns being present.

---

## Task 1: Package skeleton + EMA + RSI

**Files:**
- Create: `agent/features/__init__.py`
- Create: `tests/features/__init__.py`
- Create: `tests/features/conftest.py`
- Create: `agent/features/indicators.py` (EMA + RSI only; ATR, ADX, VWAP added in Tasks 2–3)
- Create: `tests/features/test_indicators.py` (EMA + RSI tests only)
- Modify: `requirements.txt`

- [ ] **Step 1: Add scikit-learn to requirements.txt**

Open `requirements.txt` and add this line after the `# --- Indicators ---` section:

```
scikit-learn>=1.4              # KMeans for regime detection
```

- [ ] **Step 2: Create package `__init__.py` files**

Create `agent/features/__init__.py` — leave it empty (just a package marker):

```python
```

Create `tests/features/__init__.py` — also empty:

```python
```

- [ ] **Step 3: Write the failing tests for EMA and RSI**

Create `tests/features/test_indicators.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

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
    assert f"ema_9" in result.columns


def test_add_ema_preserves_original_columns() -> None:
    df = _make_df([100.0] * 30)
    result = add_ema(df, period=9)
    for col in df.columns:
        assert col in result.columns


def test_add_ema_constant_price_converges_to_that_price() -> None:
    """EMA of a constant price series should equal that price after warmup."""
    df = _make_df([1500.0] * 50)
    result = add_ema(df, period=9)
    # After 30 bars the EMA should be very close to 1500
    tail = result["ema_9"].tail(20).to_list()
    for v in tail:
        assert v is not None
        assert abs(v - 1500.0) < 0.01, f"EMA {v} should be ≈ 1500"


def test_add_ema_rising_prices_ema_lags_below_close() -> None:
    """On a rising series, EMA lags: it should be < current close for most bars."""
    closes = [1000.0 + i * 10 for i in range(50)]
    df = _make_df(closes)
    result = add_ema(df, period=9)
    # After warmup (period bars), EMA should be below the current close
    tail_ema = result["ema_9"].tail(30).to_list()
    tail_close = result["close"].tail(30).to_list()
    lagging = [e < c for e, c in zip(tail_ema, tail_close) if e is not None]
    assert all(lagging), "EMA should lag below close on a rising series"


def test_add_ema_different_periods_produce_different_columns() -> None:
    df = _make_df([100.0 + i for i in range(60)])
    result = add_ema(add_ema(df, period=9), period=21)
    assert "ema_9" in result.columns
    assert "ema_21" in result.columns
    # Short EMA should track closer to close than long EMA
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
    # Check the last 20 bars (well past warmup)
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
```

- [ ] **Step 4: Run tests to confirm they fail**

```bash
cd /Users/tatsatshah/Desktop/yegedge
source .venv/bin/activate
python -m pytest tests/features/test_indicators.py -v --no-cov 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'agent.features'`

- [ ] **Step 5: Create `tests/features/conftest.py`**

```python
from __future__ import annotations

from datetime import date, datetime, timedelta
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
    """Build a valid OHLCV DataFrame from a list of close prices.

    Highs = close * 1.005, lows = close * 0.995, opens = close * 0.999.
    Timestamps are 60m apart from `start` (default: 2024-01-02 09:15 IST).
    """
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
```

- [ ] **Step 6: Implement `agent/features/indicators.py` (EMA + RSI only)**

```python
from __future__ import annotations

import polars as pl


def add_ema(df: pl.DataFrame, period: int, column: str = "close") -> pl.DataFrame:
    """Add an exponential moving average column ``ema_{period}`` to *df*.

    Uses standard EMA alpha = 2 / (period + 1), computed via Polars' ``ewm_mean``
    with ``span=period``.  The first bar's EMA equals the first close value;
    subsequent bars use the recursive EMA formula.

    Parameters
    ----------
    df:     DataFrame with at least a column named *column*.
    period: EMA lookback period (number of bars).
    column: Source column name (default ``"close"``).
    """
    if period < 1:
        raise ValueError(f"EMA period must be >= 1, got {period}")
    return df.with_columns(
        pl.col(column).ewm_mean(span=period, adjust=False).alias(f"ema_{period}")
    )


def add_rsi(df: pl.DataFrame, period: int = 14, column: str = "close") -> pl.DataFrame:
    """Add an RSI column ``rsi_{period}`` to *df*.

    Uses Wilder's smoothing (alpha = 1/period, i.e. ``com = period - 1``) to
    match the traditional RSI definition.  Edge cases:

    - When ``avg_loss == 0`` (all gains) → RSI = 100.
    - When ``avg_gain == 0`` (all losses) → RSI = 0.
    - The first bar always produces ``null`` (no prior close to diff against).

    Parameters
    ----------
    df:     DataFrame with at least a column named *column*.
    period: RSI lookback period (default 14).
    column: Source column name (default ``"close"``).
    """
    if period < 2:
        raise ValueError(f"RSI period must be >= 2, got {period}")
    return (
        df.with_columns(
            [
                pl.col(column)
                .diff()
                .clip(lower_bound=0.0)
                .ewm_mean(com=period - 1, adjust=False)
                .alias("_avg_gain"),
                (-pl.col(column))
                .diff()
                .clip(lower_bound=0.0)
                .ewm_mean(com=period - 1, adjust=False)
                .alias("_avg_loss"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_avg_loss") == 0.0)
            .then(pl.lit(100.0))
            .when(pl.col("_avg_gain") == 0.0)
            .then(pl.lit(0.0))
            .otherwise(
                100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss"))
            )
            .alias(f"rsi_{period}")
        )
        .drop(["_avg_gain", "_avg_loss"])
    )
```

- [ ] **Step 7: Run tests — all should pass**

```bash
python -m pytest tests/features/test_indicators.py -v --no-cov -k "ema or rsi"
```

Expected: `10 passed`

- [ ] **Step 8: Commit**

```bash
git add agent/features/__init__.py tests/features/__init__.py \
        tests/features/conftest.py tests/features/test_indicators.py \
        agent/features/indicators.py requirements.txt
git commit -m "feat(features): add package skeleton, add_ema and add_rsi indicators"
```

---

## Task 2: ATR + ADX indicators

**Files:**
- Modify: `agent/features/indicators.py` — add `add_atr` and `add_adx`
- Modify: `tests/features/test_indicators.py` — add ATR and ADX test cases

- [ ] **Step 1: Write the failing tests for ATR and ADX**

Append these to `tests/features/test_indicators.py` (after the existing RSI tests):

```python
from agent.features.indicators import add_atr, add_adx


# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------


def test_add_atr_returns_column_named_atr_period() -> None:
    df = _make_df([1000.0 + i for i in range(30)])
    result = add_atr(df, period=14)
    assert "atr_14" in result.columns


def test_add_atr_is_non_negative() -> None:
    """ATR must be >= 0 for all non-null values."""
    closes = [1000.0 + i * 3 for i in range(50)]
    df = _make_df(closes)
    result = add_atr(df, period=14)
    values = result["atr_14"].drop_nulls().to_list()
    assert all(v >= 0.0 for v in values), "ATR must be non-negative"


def test_add_atr_constant_series_is_near_zero_after_warmup() -> None:
    """On a perfectly flat series (no gaps, constant OHLCV), ATR should approach zero."""
    n = 60
    closes = [1500.0] * n
    # Build with zero spread: open=high=low=close
    timestamps = [
        datetime(2024, 1, 2, 9, 15, tzinfo=IST) + timedelta(minutes=60 * i)
        for i in range(n)
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
    # After warmup, ATR should be very close to 0 (no range, no gap)
    tail = result["atr_14"].tail(20).drop_nulls().to_list()
    assert all(v < 0.001 for v in tail), f"Flat series ATR should be ~0: {tail}"


def test_add_atr_high_volatility_is_positive() -> None:
    """On a series with significant high-low range, ATR should be clearly positive."""
    import math

    closes = [1500.0 + 50 * math.sin(i * 0.5) for i in range(60)]
    df = _make_df(closes)
    # _make_df sets high = close * 1.005, low = close * 0.995 → range ~1% of price
    result = add_atr(df, period=14)
    tail = result["atr_14"].tail(20).drop_nulls().to_list()
    assert all(v > 5.0 for v in tail), f"Volatile series ATR should be > 5: {tail}"


def test_add_atr_preserves_original_columns() -> None:
    df = _make_df([1000.0 + i for i in range(30)])
    result = add_atr(df, period=14)
    for col in df.columns:
        assert col in result.columns


# ---------------------------------------------------------------------------
# ADX tests
# ---------------------------------------------------------------------------


def test_add_adx_returns_expected_columns() -> None:
    df = _make_df([1000.0 + i * 5 for i in range(60)])
    result = add_adx(df, period=14)
    assert "adx_14" in result.columns
    assert "plus_di_14" in result.columns
    assert "minus_di_14" in result.columns


def test_add_adx_values_in_zero_to_hundred() -> None:
    """ADX, +DI, -DI must all be in [0, 100] for non-null values."""
    closes = [1000.0 + i * 3 for i in range(80)]
    df = _make_df(closes)
    result = add_adx(df, period=14)
    for col in ("adx_14", "plus_di_14", "minus_di_14"):
        values = result[col].drop_nulls().to_list()
        assert all(0.0 <= v <= 100.0 for v in values), f"{col} out of [0,100]"


def test_add_adx_uptrend_plus_di_dominates() -> None:
    """Strong uptrend → +DI should be > -DI after warmup."""
    closes = [1000.0 + i * 15 for i in range(80)]
    df = _make_df(closes)
    result = add_adx(df, period=14)
    tail = result.tail(30)
    plus = tail["plus_di_14"].drop_nulls().to_list()
    minus = tail["minus_di_14"].drop_nulls().to_list()
    assert all(p > m for p, m in zip(plus, minus)), "+DI should exceed -DI in uptrend"


def test_add_adx_downtrend_minus_di_dominates() -> None:
    """Strong downtrend → -DI should be > +DI after warmup."""
    closes = [2000.0 - i * 15 for i in range(80)]
    df = _make_df(closes)
    result = add_adx(df, period=14)
    tail = result.tail(30)
    plus = tail["plus_di_14"].drop_nulls().to_list()
    minus = tail["minus_di_14"].drop_nulls().to_list()
    assert all(m > p for p, m in zip(plus, minus)), "-DI should exceed +DI in downtrend"


def test_add_adx_no_temp_columns_leaked() -> None:
    """No columns starting with '_' should appear in the result."""
    df = _make_df([1000.0 + i * 5 for i in range(60)])
    result = add_adx(df, period=14)
    leaked = [c for c in result.columns if c.startswith("_")]
    assert leaked == [], f"Leaked temp columns: {leaked}"
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/features/test_indicators.py -v --no-cov -k "atr or adx"
```

Expected: `ImportError: cannot import name 'add_atr'`

- [ ] **Step 3: Add `add_atr` and `add_adx` to `agent/features/indicators.py`**

Append after the `add_rsi` function:

```python
def add_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Add Average True Range column ``atr_{period}`` to *df*.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|).
    Smoothed with Wilder's EWM (``com = period - 1``).  The first bar's TR is
    ``high - low`` (no prev_close available).

    Parameters
    ----------
    df:     DataFrame with ``high``, ``low``, ``close`` columns.
    period: ATR lookback period (default 14).
    """
    if period < 1:
        raise ValueError(f"ATR period must be >= 1, got {period}")
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return df.with_columns(
        tr.ewm_mean(com=period - 1, adjust=False).alias(f"atr_{period}")
    )


def add_adx(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Add ADX, +DI, and -DI columns to *df*.

    Columns added: ``adx_{period}``, ``plus_di_{period}``, ``minus_di_{period}``.

    Computation uses Wilder's smoothing (``com = period - 1``) throughout.
    Division-by-zero in DX is guarded: when (+DI + -DI) == 0, DX = 0.

    Parameters
    ----------
    df:     DataFrame with ``high``, ``low``, ``close`` columns.
    period: ADX lookback period (default 14).
    """
    if period < 2:
        raise ValueError(f"ADX period must be >= 2, got {period}")

    # Unique temp column names to avoid collisions with existing columns
    _tr = f"_adx_tr{period}"
    _pdm = f"_adx_pdm{period}"
    _mdm = f"_adx_mdm{period}"
    _dx = f"_adx_dx{period}"

    prev_high = pl.col("high").shift(1)
    prev_low = pl.col("low").shift(1)
    prev_close = pl.col("close").shift(1)

    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    up_move = pl.col("high") - prev_high
    down_move = prev_low - pl.col("low")
    plus_dm = (
        pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(pl.lit(0.0))
    )
    minus_dm = (
        pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(pl.lit(0.0))
    )

    return (
        df.with_columns(
            [
                tr.ewm_mean(com=period - 1, adjust=False).alias(_tr),
                plus_dm.ewm_mean(com=period - 1, adjust=False).alias(_pdm),
                minus_dm.ewm_mean(com=period - 1, adjust=False).alias(_mdm),
            ]
        )
        .with_columns(
            [
                (100.0 * pl.col(_pdm) / pl.col(_tr)).alias(f"plus_di_{period}"),
                (100.0 * pl.col(_mdm) / pl.col(_tr)).alias(f"minus_di_{period}"),
            ]
        )
        .with_columns(
            pl.when(
                (pl.col(f"plus_di_{period}") + pl.col(f"minus_di_{period}")) == 0.0
            )
            .then(pl.lit(0.0))
            .otherwise(
                100.0
                * (pl.col(f"plus_di_{period}") - pl.col(f"minus_di_{period}")).abs()
                / (pl.col(f"plus_di_{period}") + pl.col(f"minus_di_{period}"))
            )
            .alias(_dx)
        )
        .with_columns(
            pl.col(_dx).ewm_mean(com=period - 1, adjust=False).alias(f"adx_{period}")
        )
        .drop([_tr, _pdm, _mdm, _dx])
    )
```

- [ ] **Step 4: Run all indicator tests**

```bash
python -m pytest tests/features/test_indicators.py -v --no-cov
```

Expected: all EMA + RSI + ATR + ADX tests pass (`~19 passed`)

- [ ] **Step 5: Commit**

```bash
git add agent/features/indicators.py tests/features/test_indicators.py
git commit -m "feat(features): add add_atr and add_adx indicators (Wilder smoothing)"
```

---

## Task 3: VWAP indicator

**Files:**
- Modify: `agent/features/indicators.py` — add `add_vwap`
- Modify: `tests/features/test_indicators.py` — add VWAP tests

- [ ] **Step 1: Write the failing VWAP tests**

Append to `tests/features/test_indicators.py`:

```python
from agent.features.indicators import add_vwap


# ---------------------------------------------------------------------------
# VWAP tests
# ---------------------------------------------------------------------------


def test_add_vwap_returns_vwap_column() -> None:
    df = _make_df([1000.0 + i for i in range(20)])
    result = add_vwap(df)
    assert "vwap" in result.columns


def test_add_vwap_within_session_high_low_bounds() -> None:
    """VWAP must lie between the session's lowest low and highest high."""
    closes = [1000.0 + i for i in range(30)]
    df = _make_df(closes)
    result = add_vwap(df)
    vwap_vals = result["vwap"].to_list()
    lows = result["low"].to_list()
    highs = result["high"].to_list()
    for v, lo, hi in zip(vwap_vals, lows, highs):
        if v is not None:
            assert lo <= v <= hi, f"VWAP {v} outside [{lo}, {hi}]"


def test_add_vwap_resets_between_sessions(two_session_df: pl.DataFrame) -> None:
    """VWAP must reset to each session's first bar's typical price."""
    result = add_vwap(two_session_df)
    # Group by date and verify VWAP[0] of each session == typical_price[0]
    result = result.with_columns(pl.col("timestamp").dt.date().alias("_date"))
    for date_val in result["_date"].unique().sort().to_list():
        session = result.filter(pl.col("_date") == date_val)
        first_row = session.row(0, named=True)
        tp0 = (first_row["high"] + first_row["low"] + first_row["close"]) / 3.0
        vwap0 = first_row["vwap"]
        assert abs(vwap0 - tp0) < 0.001, (
            f"Session {date_val}: VWAP[0]={vwap0} should equal typical_price[0]={tp0}"
        )


def test_add_vwap_no_temp_columns_leaked() -> None:
    df = _make_df([1000.0 + i for i in range(20)])
    result = add_vwap(df)
    leaked = [c for c in result.columns if c.startswith("_")]
    assert leaked == [], f"Leaked temp columns: {leaked}"


def test_add_vwap_preserves_original_columns() -> None:
    df = _make_df([1000.0 + i for i in range(20)])
    result = add_vwap(df)
    for col in df.columns:
        assert col in result.columns
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/features/test_indicators.py -v --no-cov -k "vwap"
```

Expected: `ImportError: cannot import name 'add_vwap'`

- [ ] **Step 3: Add `add_vwap` to `agent/features/indicators.py`**

Append after `add_adx`:

```python
def add_vwap(df: pl.DataFrame) -> pl.DataFrame:
    """Add a session-VWAP column ``vwap`` to *df*.

    VWAP = cumulative(typical_price × volume) / cumulative(volume), reset each
    calendar date.  The typical price is (high + low + close) / 3.

    Only meaningful for intraday timeframes (15m, 60m).  On daily bars, VWAP
    equals the typical price of the single bar (degenerate case).

    Parameters
    ----------
    df:     DataFrame with ``high``, ``low``, ``close``, ``volume``, ``timestamp``
            columns.  ``timestamp`` must be timezone-aware (IST).
    """
    session = pl.col("timestamp").dt.date()
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    return (
        df.with_columns(tp.alias("_tp"))
        .with_columns((pl.col("_tp") * pl.col("volume")).alias("_tp_vol"))
        .with_columns(
            (
                pl.col("_tp_vol").cum_sum().over(session)
                / pl.col("volume").cast(pl.Float64).cum_sum().over(session)
            ).alias("vwap")
        )
        .drop(["_tp", "_tp_vol"])
    )
```

- [ ] **Step 4: Run all indicator tests**

```bash
python -m pytest tests/features/test_indicators.py -v --no-cov
```

Expected: all tests pass (`~24 passed`)

- [ ] **Step 5: Commit**

```bash
git add agent/features/indicators.py tests/features/test_indicators.py
git commit -m "feat(features): add add_vwap indicator with per-session reset"
```

---

## Task 4: Regime Detector

**Files:**
- Create: `agent/features/regime.py`
- Create: `tests/features/test_regime.py`

**Dependency note:** This task requires `adx_14` and `atr_14` columns produced by Task 2's `add_adx` and `add_atr`. Tests will call those functions to build properly enriched DataFrames.

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_regime.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.features.indicators import add_adx, add_atr
from agent.features.regime import Regime, RegimeDetector
from tests.features.conftest import make_ohlcv_df

IST = ZoneInfo("Asia/Kolkata")


def _enriched(closes: list[float]) -> pl.DataFrame:
    """Build a feature-enriched DataFrame with adx_14 and atr_14 columns."""
    df = make_ohlcv_df(closes)
    df = add_atr(df, period=14)
    df = add_adx(df, period=14)
    return df


# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------


def test_regime_values_are_strings() -> None:
    assert Regime.TRENDING == "trending"
    assert Regime.RANGING == "ranging"
    assert Regime.VOLATILE == "volatile"
    assert Regime.UNKNOWN == "unknown"


# ---------------------------------------------------------------------------
# RegimeDetector — before fit
# ---------------------------------------------------------------------------


def test_regime_detector_not_fit_by_default() -> None:
    rd = RegimeDetector()
    assert rd.is_fit is False


def test_predict_before_fit_returns_unknown() -> None:
    """predict() before fit() must tag every row as UNKNOWN."""
    rd = RegimeDetector()
    df = _enriched([1000.0 + i * 5 for i in range(40)])
    result = rd.predict(df)
    assert "regime" in result.columns
    assert all(v == Regime.UNKNOWN.value for v in result["regime"].to_list())


# ---------------------------------------------------------------------------
# RegimeDetector — fit
# ---------------------------------------------------------------------------


def test_fit_with_enough_data_marks_is_fit() -> None:
    rd = RegimeDetector()
    df = _enriched([1000.0 + i * 5 for i in range(100)])
    rd.fit(df)
    assert rd.is_fit is True


def test_fit_with_insufficient_data_leaves_unfit() -> None:
    """fit() with fewer than 60 non-null rows must leave the detector unfit."""
    rd = RegimeDetector()
    # Only 30 bars — after ADX warmup (~28 null rows), <60 non-null remain
    df = _enriched([1000.0 + i for i in range(30)])
    rd.fit(df)
    assert rd.is_fit is False


def test_fit_invalid_n_regimes_raises() -> None:
    with pytest.raises(ValueError, match="n_regimes"):
        RegimeDetector(n_regimes=1)
    with pytest.raises(ValueError, match="n_regimes"):
        RegimeDetector(n_regimes=5)


# ---------------------------------------------------------------------------
# RegimeDetector — predict after fit
# ---------------------------------------------------------------------------


def test_predict_after_fit_returns_valid_regime_values() -> None:
    """After fit(), every row must have a valid (non-UNKNOWN) Regime value."""
    closes = [1000.0 + i * 10 for i in range(120)]
    df = _enriched(closes)
    rd = RegimeDetector()
    rd.fit(df)
    result = rd.predict(df)
    valid_values = {r.value for r in Regime}
    for v in result["regime"].to_list():
        assert v in valid_values, f"Unexpected regime value: {v!r}"


def test_predict_adds_regime_column_without_modifying_input() -> None:
    """predict() must not modify the input DataFrame."""
    df = _enriched([1000.0 + i * 5 for i in range(100)])
    rd = RegimeDetector()
    rd.fit(df)
    original_cols = set(df.columns)
    result = rd.predict(df)
    assert set(df.columns) == original_cols, "Input DataFrame was modified"
    assert "regime" in result.columns
    assert "regime" not in df.columns


def test_predict_trending_label_on_strong_uptrend() -> None:
    """A strong, sustained uptrend should be classified as TRENDING."""
    # 200 bars of steady uptrend gives ADX plenty of time to rise above 25
    closes = [1000.0 + i * 20 for i in range(200)]
    df = _enriched(closes)
    rd = RegimeDetector()
    rd.fit(df)
    result = rd.predict(df)
    # Check the final 50 bars (well past warmup)
    tail_regimes = result["regime"].tail(50).to_list()
    trending_count = sum(1 for r in tail_regimes if r == Regime.TRENDING.value)
    assert trending_count > 30, (
        f"Expected majority TRENDING on uptrend, got {tail_regimes[-10:]}"
    )


def test_regime_column_dtype_is_string() -> None:
    df = _enriched([1000.0 + i * 5 for i in range(100)])
    rd = RegimeDetector()
    rd.fit(df)
    result = rd.predict(df)
    assert result["regime"].dtype == pl.Utf8, f"Expected Utf8, got {result['regime'].dtype}"
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/features/test_regime.py -v --no-cov 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'agent.features.regime'`

- [ ] **Step 3: Create `agent/features/regime.py`**

```python
from __future__ import annotations

from enum import StrEnum

import numpy as np
import polars as pl
import structlog
from sklearn.cluster import KMeans

logger = structlog.get_logger()

_MIN_FIT_ROWS: int = 60


class Regime(StrEnum):
    """Market regime label assigned by :class:`RegimeDetector`."""

    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class RegimeDetector:
    """Classify market regime using KMeans clustering.

    Features used (all from the output of ``add_atr`` + ``add_adx``):

    - ``adx_14``: trend strength (0–100)
    - ``atr_14 / close * 100``: normalised volatility (ATR as % of price)
    - ``close.pct_change(20) * 100``: 20-bar momentum

    Cluster → Regime mapping (post-hoc, based on centroid values):

    - Highest ADX centroid → ``TRENDING``
    - Of the remainder, highest ATR% centroid → ``VOLATILE``
    - Remaining cluster(s) → ``RANGING``

    Call :meth:`fit` on historical data before calling :meth:`predict`.
    ``predict`` returns ``UNKNOWN`` for every row when called before ``fit``.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42) -> None:
        if not (2 <= n_regimes <= 4):
            raise ValueError(f"n_regimes must be between 2 and 4, got {n_regimes}")
        self._n = n_regimes
        self._random_state = random_state
        self._model: KMeans | None = None
        self._label_map: dict[int, Regime] = {}

    @property
    def is_fit(self) -> bool:
        return self._model is not None

    def fit(self, df: pl.DataFrame) -> None:
        """Fit KMeans on *df*.  Requires ``adx_14``, ``atr_14``, ``close`` columns.

        Drops rows with null values in any feature column before fitting.
        Does nothing (leaves detector unfit) when fewer than
        ``_MIN_FIT_ROWS`` clean rows are available.
        """
        required = ["adx_14", "atr_14", "close"]
        clean = df.drop_nulls(subset=required)
        if len(clean) < _MIN_FIT_ROWS:
            logger.warning(
                "regime_detector.fit.insufficient_data",
                rows=len(clean),
                minimum=_MIN_FIT_ROWS,
            )
            return
        x = self._build_features(clean)
        model = KMeans(n_clusters=self._n, random_state=self._random_state, n_init=10)
        model.fit(x)
        self._label_map = self._assign_labels(model.cluster_centers_)
        self._model = model
        logger.info("regime_detector.fit.done", n_regimes=self._n, rows=len(clean))

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add a ``regime`` column to *df*.

        Returns ``UNKNOWN`` for every row when the detector has not been fit.
        Null-feature rows are filled with zeros before prediction (they will be
        assigned to the nearest centroid; the strategy layer should discard
        early-bar rows that have null indicators anyway).
        """
        if self._model is None:
            return df.with_columns(pl.lit(Regime.UNKNOWN.value).alias("regime"))
        x = self._build_features(df)
        clusters = self._model.predict(x)
        regime_values = [self._label_map[int(c)].value for c in clusters]
        return df.with_columns(
            pl.Series("regime", regime_values, dtype=pl.Utf8)
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_features(self, df: pl.DataFrame) -> np.ndarray:
        """Return a (n_rows, 3) float64 array: [adx, atr_pct, momentum_20]."""
        adx = df["adx_14"].fill_null(0.0).to_numpy().astype(np.float64)
        atr_pct = (
            (df["atr_14"] / df["close"] * 100.0).fill_null(0.0).to_numpy().astype(np.float64)
        )
        momentum = (
            df["close"]
            .pct_change(20)
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float64)
            * 100.0
        )
        return np.column_stack([adx, atr_pct, momentum])

    def _assign_labels(self, centers: np.ndarray) -> dict[int, Regime]:
        """Map cluster indices to Regime enum values based on centroid characteristics.

        centers shape: (n_clusters, 3) = [adx, atr_pct, momentum]
        """
        adx_col = centers[:, 0]
        atr_col = centers[:, 1]
        n = len(centers)

        trending_idx = int(np.argmax(adx_col))
        remaining = [i for i in range(n) if i != trending_idx]

        if len(remaining) == 1:
            # n_regimes == 2: only trending + ranging
            return {trending_idx: Regime.TRENDING, remaining[0]: Regime.RANGING}

        volatile_idx = remaining[int(np.argmax(atr_col[remaining]))]
        ranging_indices = [i for i in remaining if i != volatile_idx]

        label_map: dict[int, Regime] = {
            trending_idx: Regime.TRENDING,
            volatile_idx: Regime.VOLATILE,
        }
        for idx in ranging_indices:
            label_map[idx] = Regime.RANGING
        return label_map
```

- [ ] **Step 4: Run regime tests**

```bash
python -m pytest tests/features/test_regime.py -v --no-cov
```

Expected: all tests pass (`~11 passed`)

- [ ] **Step 5: Commit**

```bash
git add agent/features/regime.py tests/features/test_regime.py
git commit -m "feat(features): add RegimeDetector (KMeans, 3-class: trending/ranging/volatile)"
```

---

## Task 5: Feature Pipeline

**Files:**
- Create: `agent/features/pipeline.py`
- Create: `tests/features/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_pipeline.py`:

```python
from __future__ import annotations

import polars as pl
import pytest

from agent.features.pipeline import FeaturePipeline
from agent.features.regime import Regime, RegimeDetector
from tests.features.conftest import make_ohlcv_df

_EXPECTED_INDICATOR_COLS = {
    "ema_9",
    "ema_21",
    "ema_50",
    "rsi_14",
    "atr_14",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "vwap",
}


def _large_df(n: int = 150) -> pl.DataFrame:
    """150 bars of rising price — enough for all indicators to stabilize."""
    return make_ohlcv_df([1000.0 + i * 5 for i in range(n)])


# ---------------------------------------------------------------------------
# FeaturePipeline — without regime
# ---------------------------------------------------------------------------


def test_pipeline_adds_all_indicator_columns() -> None:
    df = _large_df()
    result = FeaturePipeline().run(df)
    for col in _EXPECTED_INDICATOR_COLS:
        assert col in result.columns, f"Missing column: {col}"


def test_pipeline_does_not_modify_input() -> None:
    df = _large_df()
    original_cols = set(df.columns)
    _ = FeaturePipeline().run(df)
    assert set(df.columns) == original_cols


def test_pipeline_no_temp_columns_leaked() -> None:
    df = _large_df()
    result = FeaturePipeline().run(df)
    leaked = [c for c in result.columns if c.startswith("_")]
    assert leaked == [], f"Leaked temp columns: {leaked}"


def test_pipeline_preserves_row_count() -> None:
    df = _large_df(n=80)
    result = FeaturePipeline().run(df)
    assert len(result) == len(df)


def test_pipeline_without_regime_has_no_regime_column() -> None:
    df = _large_df()
    result = FeaturePipeline().run(df)
    assert "regime" not in result.columns


# ---------------------------------------------------------------------------
# FeaturePipeline — with regime
# ---------------------------------------------------------------------------


def test_pipeline_with_regime_detector_adds_regime_column() -> None:
    df = _large_df(n=150)
    rd = RegimeDetector()
    rd.fit(df)  # pre-fit on a subset — pipeline.run will call predict only
    pipeline = FeaturePipeline(regime_detector=rd)
    result = pipeline.run(df)
    assert "regime" in result.columns


def test_pipeline_regime_values_are_valid() -> None:
    df = _large_df(n=150)
    rd = RegimeDetector()
    # Fit on enriched data (pipeline.run produces features, then predict)
    # We pre-fit on the raw df — the pipeline will add indicators first,
    # then call rd.predict on the enriched df.
    # To pre-fit properly, run the pipeline once to get features, then fit.
    pipeline = FeaturePipeline(regime_detector=rd)
    enriched = FeaturePipeline().run(df)  # get features without regime
    rd.fit(enriched)
    result = pipeline.run(df)
    valid_values = {r.value for r in Regime}
    for v in result["regime"].to_list():
        assert v in valid_values, f"Invalid regime: {v!r}"


def test_pipeline_with_unfit_regime_detector_returns_unknown() -> None:
    """An unfit RegimeDetector attached to the pipeline should produce UNKNOWN for all rows."""
    df = _large_df()
    rd = RegimeDetector()  # not fit
    pipeline = FeaturePipeline(regime_detector=rd)
    result = pipeline.run(df)
    assert "regime" in result.columns
    assert all(v == Regime.UNKNOWN.value for v in result["regime"].to_list())


# ---------------------------------------------------------------------------
# FeaturePipeline — edge cases
# ---------------------------------------------------------------------------


def test_pipeline_on_minimum_viable_df() -> None:
    """Pipeline should not raise on a very small DataFrame (just produce many nulls)."""
    df = make_ohlcv_df([1000.0, 1001.0, 1002.0])
    result = FeaturePipeline().run(df)
    # All indicator columns should exist, even if filled with nulls
    for col in _EXPECTED_INDICATOR_COLS:
        assert col in result.columns
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/features/test_pipeline.py -v --no-cov 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'agent.features.pipeline'`

- [ ] **Step 3: Create `agent/features/pipeline.py`**

```python
from __future__ import annotations

import polars as pl
import structlog

from agent.features.indicators import add_adx, add_atr, add_ema, add_rsi, add_vwap
from agent.features.regime import RegimeDetector

logger = structlog.get_logger()


class FeaturePipeline:
    """Compute all technical indicators and optional regime label for an OHLCV DataFrame.

    Usage::

        pipeline = FeaturePipeline()
        enriched_df = pipeline.run(raw_ohlcv_df)

    With regime detection::

        detector = RegimeDetector()
        detector.fit(historical_enriched_df)
        pipeline = FeaturePipeline(regime_detector=detector)
        enriched_df = pipeline.run(raw_ohlcv_df)

    The pipeline adds these columns (in order):

    - ``ema_9``, ``ema_21``, ``ema_50`` — exponential moving averages
    - ``rsi_14`` — RSI (Wilder smoothing, period 14)
    - ``atr_14`` — Average True Range (period 14)
    - ``adx_14``, ``plus_di_14``, ``minus_di_14`` — ADX directional system
    - ``vwap`` — session VWAP (resets each calendar date)
    - ``regime`` — if a fit :class:`RegimeDetector` is provided
    """

    def __init__(self, regime_detector: RegimeDetector | None = None) -> None:
        self._regime = regime_detector

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all indicators to *df* and return the enriched DataFrame.

        The input DataFrame is not modified.  Indicator columns are appended.

        Parameters
        ----------
        df:
            OHLCV DataFrame as produced by the Phase 1 data pipeline.
            Must have columns: symbol, timeframe, timestamp, open, high,
            low, close, volume, value.
        """
        result = df
        result = add_ema(result, period=9)
        result = add_ema(result, period=21)
        result = add_ema(result, period=50)
        result = add_rsi(result, period=14)
        result = add_atr(result, period=14)
        result = add_adx(result, period=14)
        result = add_vwap(result)
        if self._regime is not None:
            result = self._regime.predict(result)
        logger.debug(
            "feature_pipeline.run",
            rows=len(result),
            columns=len(result.columns),
            has_regime=self._regime is not None,
        )
        return result
```

- [ ] **Step 4: Run pipeline tests**

```bash
python -m pytest tests/features/test_pipeline.py -v --no-cov
```

Expected: all tests pass (`~9 passed`)

- [ ] **Step 5: Commit**

```bash
git add agent/features/pipeline.py tests/features/test_pipeline.py
git commit -m "feat(features): add FeaturePipeline orchestrating all indicators + regime"
```

---

## Task 6: Full test suite + coverage gate

**Files:**
- Run the full suite; add targeted tests if coverage falls below 70%

- [ ] **Step 1: Run the complete test suite**

```bash
cd /Users/tatsatshah/Desktop/yegedge
source .venv/bin/activate
python -m pytest tests/ -v --cov=agent --cov-report=term-missing
```

Expected: all tests pass (Phase 1 + Phase 2), coverage ≥ 70%.

If coverage for `agent/features/` is below 70%, identify uncovered branches via the `term-missing` report and add tests to cover them.

- [ ] **Step 2: Run linters**

```bash
python -m ruff check agent/features/ tests/features/
python -m black --check agent/features/ tests/features/
```

Expected: no issues.  Fix any before committing:

```bash
python -m black agent/features/ tests/features/
python -m ruff check --fix agent/features/ tests/features/
```

- [ ] **Step 3: Spot-check one full pipeline run manually**

```bash
python - <<'EOF'
from datetime import datetime
from zoneinfo import ZoneInfo
from tests.features.conftest import make_ohlcv_df
from agent.features.pipeline import FeaturePipeline
from agent.features.regime import RegimeDetector

IST = ZoneInfo("Asia/Kolkata")
df = make_ohlcv_df([1000.0 + i * 8 for i in range(200)])
rd = RegimeDetector()
pipeline = FeaturePipeline()
enriched = pipeline.run(df)
rd.fit(enriched)
pipeline_with_regime = FeaturePipeline(regime_detector=rd)
final = pipeline_with_regime.run(df)
print(final.select(["timestamp","close","ema_9","rsi_14","adx_14","vwap","regime"]).tail(5))
EOF
```

Expected: a 5-row table with all columns populated and regime labels visible.

- [ ] **Step 4: Commit**

```bash
git add agent/features/ tests/features/ requirements.txt
git commit -m "feat(features): Phase 2 complete — EMA, RSI, ATR, ADX, VWAP, RegimeDetector, FeaturePipeline"
```

---

## Verification

End-to-end smoke test (reads from Phase 1 Parquet cache if available, otherwise uses synthetic data):

```bash
python - <<'EOF'
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from agent.data.cache import ParquetCache
from agent.features.pipeline import FeaturePipeline
from agent.features.regime import RegimeDetector

IST = ZoneInfo("Asia/Kolkata")
cache = ParquetCache(Path("./data/cache"))
df = cache.read(
    symbol="HDFCBANK",
    timeframe="60m",
    start=datetime(2024, 1, 1, tzinfo=IST),
    end=datetime(2024, 12, 31, tzinfo=IST),
)
if len(df) == 0:
    print("Cache empty — run `python -m agent refresh --symbol HDFCBANK --timeframe 60m` first")
else:
    pipeline = FeaturePipeline()
    enriched = pipeline.run(df)
    rd = RegimeDetector()
    rd.fit(enriched)
    pipeline_r = FeaturePipeline(regime_detector=rd)
    final = pipeline_r.run(df)
    print(final.select(["timestamp","close","ema_21","rsi_14","adx_14","vwap","regime"]).tail(10))
    print(f"\nRegime distribution:\n{final['regime'].value_counts()}")
EOF
```

---

## Self-Review Checklist

- [x] **EMA** — `add_ema(df, period, column)` → `ema_{period}` column ✓
- [x] **RSI** — `add_rsi(df, period, column)` → `rsi_{period}`, handles avg_loss=0 and avg_gain=0 ✓
- [x] **ATR** — `add_atr(df, period)` → `atr_{period}`, Wilder smoothing ✓
- [x] **ADX** — `add_adx(df, period)` → `adx_{period}`, `plus_di_{period}`, `minus_di_{period}` ✓
- [x] **VWAP** — `add_vwap(df)` → `vwap`, resets per calendar date ✓
- [x] **Regime** — `RegimeDetector.fit/predict`, `Regime(StrEnum)`, 3 classes ✓
- [x] **Pipeline** — `FeaturePipeline.run(df)`, optional regime, no mutation of input ✓
- [x] **scikit-learn added to requirements.txt** ✓
- [x] **`from __future__ import annotations`** in every module ✓
- [x] **`logger = structlog.get_logger()`** (not `log`) ✓
- [x] **No temp columns leaked** in any indicator function ✓
- [x] **No pandas** anywhere in `agent/features/` ✓
- [x] **Period validation** raises `ValueError` in `add_ema`, `add_rsi`, `add_atr`, `add_adx` ✓
