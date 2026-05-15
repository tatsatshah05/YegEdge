from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def sample_bar_df() -> pl.DataFrame:
    """A minimal valid OHLCV DataFrame for validator and cache tests.

    Prices stored as Float64 in Polars (Polars can't construct typed columns from
    Python Decimal objects directly). The validator converts via Decimal(str(value))
    internally, so Float64 in storage is correct for the cache layer.
    """
    return pl.DataFrame(
        {
            "symbol": ["HDFCBANK"] * 3,
            "timeframe": ["60m"] * 3,
            "timestamp": [
                datetime(2024, 1, 2, 9, 15, tzinfo=IST),
                datetime(2024, 1, 2, 10, 15, tzinfo=IST),
                datetime(2024, 1, 2, 11, 15, tzinfo=IST),
            ],
            "open": [1700.00, 1710.00, 1705.00],
            "high": [1720.00, 1725.00, 1715.00],
            "low": [1695.00, 1700.00, 1698.00],
            "close": [1710.00, 1705.00, 1712.00],
            "volume": [100_000, 95_000, 110_000],
            "value": [171_000_000.0, 162_225_000.0, 188_320_000.0],
        }
    )


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Temp directory for Parquet cache tests."""
    d = tmp_path / "cache"
    d.mkdir()
    return d
