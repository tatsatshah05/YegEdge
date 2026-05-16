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
            .otherwise(100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss")))
            .alias(f"rsi_{period}")
        )
        .drop(["_avg_gain", "_avg_loss"])
    )


def add_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Add Average True Range column ``atr_{period}`` to *df*.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|).
    Smoothed with Wilder's EWM (``com = period - 1``).  The first bar's TR is
    ``high - low`` (no prev_close available; Polars' max_horizontal ignores nulls).

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
    return df.with_columns(tr.ewm_mean(com=period - 1, adjust=False).alias(f"atr_{period}"))


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

    # Unique temp column names to avoid collisions with any existing columns
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
    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(pl.lit(0.0))
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
                pl.when(pl.col(_tr) == 0.0)
                .then(pl.lit(0.0))
                .otherwise(100.0 * pl.col(_pdm) / pl.col(_tr))
                .alias(f"plus_di_{period}"),
                pl.when(pl.col(_tr) == 0.0)
                .then(pl.lit(0.0))
                .otherwise(100.0 * pl.col(_mdm) / pl.col(_tr))
                .alias(f"minus_di_{period}"),
            ]
        )
        .with_columns(
            pl.when((pl.col(f"plus_di_{period}") + pl.col(f"minus_di_{period}")) == 0.0)
            .then(pl.lit(0.0))
            .otherwise(
                100.0
                * (pl.col(f"plus_di_{period}") - pl.col(f"minus_di_{period}")).abs()
                / (pl.col(f"plus_di_{period}") + pl.col(f"minus_di_{period}"))
            )
            .alias(_dx)
        )
        .with_columns(pl.col(_dx).ewm_mean(com=period - 1, adjust=False).alias(f"adx_{period}"))
        .drop([_tr, _pdm, _mdm, _dx])
    )


def add_vwap(df: pl.DataFrame) -> pl.DataFrame:
    """Add a session-VWAP column ``vwap`` to *df*.

    VWAP = cumulative(typical_price * volume) / cumulative(volume), reset each
    calendar date.  The typical price is (high + low + close) / 3.

    Only meaningful for intraday timeframes (15m, 60m).  On daily bars, VWAP
    equals the bar's typical price (degenerate single-bar case).

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
