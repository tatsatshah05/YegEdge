from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.data.types import DataQuality
from agent.data.validator import DataValidator

IST = ZoneInfo("Asia/Kolkata")


def _make_df(**overrides: object) -> pl.DataFrame:
    """Build a single-row bar DataFrame with sane defaults, overrideable per-column.

    Prices are floats — Polars doesn't infer pl.Decimal from Python Decimal objects.
    The validator converts via Decimal(str(value)) internally.
    """
    row: dict[str, object] = {
        "symbol": "HDFCBANK",
        "timeframe": "60m",
        "timestamp": datetime(2024, 1, 2, 9, 15, tzinfo=IST),
        "open": 1700.00,
        "high": 1720.00,
        "low": 1695.00,
        "close": 1710.00,
        "volume": 100_000,
        "value": 171_000_000.00,
    }
    row.update(overrides)
    return pl.DataFrame([row])


def test_valid_bar_gets_ok_quality() -> None:
    df = DataValidator().validate(_make_df())
    assert df["data_quality"][0] == DataQuality.OK.value


def test_high_below_open_gives_suspect() -> None:
    df = DataValidator().validate(
        _make_df(open=1700.0, high=1680.0, low=1660.0, close=1670.0)
    )
    assert df["data_quality"][0] == DataQuality.SUSPECT.value


def test_high_below_close_gives_suspect() -> None:
    df = DataValidator().validate(
        _make_df(high=1700.0, close=1750.0)
    )
    assert df["data_quality"][0] == DataQuality.SUSPECT.value


def test_low_above_open_gives_suspect() -> None:
    df = DataValidator().validate(
        _make_df(low=1750.0, open=1700.0)
    )
    assert df["data_quality"][0] == DataQuality.SUSPECT.value


def test_zero_volume_gives_partial() -> None:
    df = DataValidator().validate(_make_df(volume=0))
    assert df["data_quality"][0] == DataQuality.PARTIAL.value


def test_zero_price_gives_suspect() -> None:
    df = DataValidator().validate(_make_df(open=0.0))
    assert df["data_quality"][0] == DataQuality.SUSPECT.value


def test_outlier_price_jump_gives_suspect() -> None:
    # 60% single-bar price jump is suspicious
    df = _make_df(open=1700.0, close=2720.0)
    result = DataValidator().validate(df)
    assert result["data_quality"][0] == DataQuality.SUSPECT.value


def test_validate_preserves_ok_bars_in_multi_row_df(sample_bar_df: pl.DataFrame) -> None:
    result = DataValidator().validate(sample_bar_df)
    assert "data_quality" in result.columns
    assert all(q == DataQuality.OK.value for q in result["data_quality"].to_list())
