from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

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


def test_incremental_write_deduplicates_and_latest_wins(cache_dir: Path) -> None:
    cache = ParquetCache(root=cache_dir)

    # First write: 3 bars at Jan 1, 2, 3 with close=103
    df_initial = _make_bars("HDFCBANK", "60m", 2024, n=3)
    cache.write(df_initial, symbol="HDFCBANK", timeframe="60m")

    # Second write: Jan 3 (overlap) with updated close=999, plus new Jan 4
    overlap_ts = datetime(2024, 1, 3, 9, 15, tzinfo=IST)
    new_ts = datetime(2024, 1, 4, 9, 15, tzinfo=IST)
    df_update = pl.DataFrame(
        {
            "symbol": ["HDFCBANK", "HDFCBANK"],
            "timeframe": ["60m", "60m"],
            "timestamp": [overlap_ts, new_ts],
            "open": [100.00, 100.00],
            "high": [105.00, 105.00],
            "low": [99.00, 99.00],
            "close": [999.00, 103.00],  # 999 should win over original 103 on overlap
            "volume": [50_000, 50_000],
            "value": [5_000_000.0, 5_000_000.0],
            "data_quality": ["ok", "ok"],
        }
    )
    cache.write(df_update, symbol="HDFCBANK", timeframe="60m")

    result = cache.read(
        symbol="HDFCBANK",
        timeframe="60m",
        start=datetime(2024, 1, 1, tzinfo=IST),
        end=datetime(2024, 12, 31, tzinfo=IST),
    )
    assert len(result) == 4  # Jan 1, 2, 3, 4 — no duplicate for Jan 3
    jan3 = result.filter(pl.col("timestamp") == overlap_ts)
    assert jan3["close"][0] == 999.00  # latest write wins


def test_coverage_report_structure(cache_dir: Path) -> None:
    cache = ParquetCache(root=cache_dir)
    cache.write(_make_bars("SBIN", "60m", 2024), symbol="SBIN", timeframe="60m")
    report = cache.coverage_report()
    assert "SBIN" in report
    assert "60m" in report["SBIN"]
    start, end = report["SBIN"]["60m"]
    assert start <= end


def test_coverage_report_aggregates_across_years(cache_dir: Path) -> None:
    cache = ParquetCache(root=cache_dir)
    cache.write(_make_bars("KOTAKBANK", "60m", 2023, n=3), symbol="KOTAKBANK", timeframe="60m")
    cache.write(_make_bars("KOTAKBANK", "60m", 2024, n=3), symbol="KOTAKBANK", timeframe="60m")
    report = cache.coverage_report()
    start, end = report["KOTAKBANK"]["60m"]
    # min must be in 2023, max must be in 2024 — not both from the same year
    assert start.year == 2023
    assert end.year == 2024
