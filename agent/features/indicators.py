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
