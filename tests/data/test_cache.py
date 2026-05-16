from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.data.cache import ParquetCache

IST = ZoneInfo("Asia/Kolkata")


def _make_bars(symbol: str, timeframe: str, year: int, n: int = 3) -> pl.DataFrame:
    from agent.data.types import DataQuality

    timestamps = [datetime(year, 1, i + 1, 9, 15, tzinfo=IST) for i in range(n)]
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "timeframe": [timeframe] * n,
            "timestamp": timestamps,
            "open": [100.00] * n,
            "high": [105.00] * n,
            "low": [99.00] * n,
            "close": [103.00] * n,
            "volume": [50_000] * n,
            "value": [5_000_000.0] * n,
            "data_quality": [DataQuality.OK.value] * n,
        }
    )


def test_write_and_read_roundtrip(cache_dir: Path) -> None:
    cache = ParquetCache(root=cache_dir)
    df = _make_bars("TCS", "60m", 2024)
    cache.write(df, symbol="TCS", timeframe="60m")

    result = cache.read(
        symbol="TCS",
        timeframe="60m",
        start=datetime(2024, 1, 1, tzinfo=IST),
        end=datetime(2024, 12, 31, tzinfo=IST),
    )
    assert len(result) == 3
    assert result["symbol"][0] == "TCS"


def test_write_creates_year_partitioned_file(cache_dir: Path) -> None:
    cache = ParquetCache(root=cache_dir)
    cache.write(_make_bars("INFY", "60m", 2024), symbol="INFY", timeframe="60m")
    expected = cache_dir / "60m" / "2024" / "INFY.parquet"
    assert expected.exists()


def test_last_timestamp_returns_max(cache_dir: Path) -> None:
    cache = ParquetCache(root=cache_dir)
    cache.write(_make_bars("WIPRO", "60m", 2024, n=5), symbol="WIPRO", timeframe="60m")
    ts = cache.last_timestamp(symbol="WIPRO", timeframe="60m")
    assert ts is not None
    assert ts == datetime(2024, 1, 5, 9, 15, tzinfo=IST)


def test_last_timestamp_none_when_no_data(cache_dir: Path) -> None:
    cache = ParquetCache(root=cache_dir)
    assert cache.last_timestamp(symbol="MISSING", timeframe="60m") is None


def test_write_appends_across_years(cache_dir: Path) -> None:
    cache = ParquetCache(root=cache_dir)
    df_2023 = _make_bars("AXISBANK", "60m", 2023)
    df_2024 = _make_bars("AXISBANK", "60m", 2024)
    cache.write(df_2023, symbol="AXISBANK", timeframe="60m")
    cache.write(df_2024, symbol="AXISBANK", timeframe="60m")

    result = cache.read(
        symbol="AXISBANK",
        timeframe="60m",
        start=datetime(2023, 1, 1, tzinfo=IST),
        end=datetime(2024, 12, 31, tzinfo=IST),
    )
    assert len(result) == 6


def test_coverage_report_structure(cache_dir: Path) -> None:
    cache = ParquetCache(root=cache_dir)
    cache.write(_make_bars("SBIN", "60m", 2024), symbol="SBIN", timeframe="60m")
    report = cache.coverage_report()
    assert "SBIN" in report
    assert "60m" in report["SBIN"]
    start, end = report["SBIN"]["60m"]
    assert start <= end
