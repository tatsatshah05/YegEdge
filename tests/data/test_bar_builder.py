from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.data.bar_builder import BarBuilder, ClosedBar

IST = ZoneInfo("Asia/Kolkata")


def ts(h: int, m: int, s: int = 0) -> datetime:
    return datetime(2024, 1, 2, h, m, s, tzinfo=IST)


def test_no_bar_returned_on_first_tick() -> None:
    bb = BarBuilder("HDFCBANK", "60m")
    result = bb.on_tick(1710.0, ts(9, 15))
    assert result is None


def test_bar_closes_when_next_slot_tick_arrives() -> None:
    bb = BarBuilder("HDFCBANK", "60m")
    bb.on_tick(1700.0, ts(9, 15))
    bb.on_tick(1720.0, ts(9, 45))
    closed = bb.on_tick(1715.0, ts(10, 15))  # new slot → closes 9:15 bar
    assert closed is not None
    assert closed.symbol == "HDFCBANK"
    assert closed.open == 1700.0
    assert closed.high == 1720.0
    assert closed.low == 1700.0
    assert closed.close == 1720.0
    assert closed.tick_count == 2


def test_bar_start_aligned_to_market_open_not_tick_time() -> None:
    bb = BarBuilder("HDFCBANK", "60m")
    bb.on_tick(1700.0, ts(9, 20))  # arrives after 9:15, but slot is 9:15
    closed = bb.on_tick(1705.0, ts(10, 16))  # new slot
    assert closed is not None
    assert closed.bar_open == ts(9, 15)


def test_15m_bar_closes_at_15_minute_boundary() -> None:
    bb = BarBuilder("TCS", "15m")
    bb.on_tick(3500.0, ts(9, 15))
    bb.on_tick(3510.0, ts(9, 25))
    closed = bb.on_tick(3520.0, ts(9, 30))  # new 15m slot
    assert closed is not None
    assert closed.bar_open == ts(9, 15)
    assert closed.tick_count == 2


def test_force_close_returns_current_bar() -> None:
    bb = BarBuilder("INFY", "60m")
    bb.on_tick(1600.0, ts(9, 15))
    bb.on_tick(1620.0, ts(9, 55))
    closed = bb.force_close()
    assert closed is not None
    assert closed.open == 1600.0
    assert closed.high == 1620.0
    assert closed.tick_count == 2


def test_force_close_returns_none_when_no_ticks() -> None:
    bb = BarBuilder("WIPRO", "60m")
    assert bb.force_close() is None


def test_closed_bar_to_dataframe_has_required_columns() -> None:
    bb = BarBuilder("HDFCBANK", "60m")
    bb.on_tick(1700.0, ts(9, 15))
    closed = bb.on_tick(1710.0, ts(10, 15))
    assert closed is not None
    df = closed.to_dataframe()
    for col in (
        "symbol",
        "timeframe",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
        "data_quality",
    ):
        assert col in df.columns, f"Missing column: {col}"
    assert df["volume"][0] == 1  # 1 tick before close
    assert df["data_quality"][0] == "ok"


def test_tick_before_market_open_snaps_to_market_open_slot() -> None:
    bb = BarBuilder("SBIN", "60m")
    # Tick arrives at 9:10 — before 9:15 market open; should snap to 9:15 slot
    result = bb.on_tick(500.0, ts(9, 10))
    assert result is None  # no closed bar yet
    closed = bb.on_tick(505.0, ts(10, 16))
    assert closed is not None
    assert closed.bar_open == ts(9, 15)
