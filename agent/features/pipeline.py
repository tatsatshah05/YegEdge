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

    Columns added (in order):

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
