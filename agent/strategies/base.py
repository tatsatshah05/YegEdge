from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

from agent.strategies.types import Signal


class BaseStrategy(ABC):
    """Abstract base for all strategy implementations.

    Strategies are pure functions: they receive an enriched DataFrame (output
    of FeaturePipeline.run()) and return a list of Signals.  No I/O, no broker
    calls, no side effects.

    The DataFrame is expected to have at minimum:
    symbol (Utf8), timestamp (Datetime[us, Asia/Kolkata]), open, high, low,
    close, volume (all Float64 / Int64), data_quality (Utf8), and any
    indicator columns the concrete strategy declares as required.

    Column requirements are validated at the start of generate() — if any
    required column is missing, a ValueError is raised immediately.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier — written into every Signal.strategy_name."""

    @abstractmethod
    def generate(self, df: pl.DataFrame) -> list[Signal]:
        """Generate signals from an enriched OHLCV+indicator DataFrame.

        Parameters
        ----------
        df:
            Enriched DataFrame from FeaturePipeline.run().

        Returns
        -------
        list[Signal]
            One Signal per bar where an actionable condition is met.
            Returns an empty list when no conditions are triggered.
            Never raises on normal market data (only on malformed input).
        """
