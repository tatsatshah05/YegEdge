from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.data.bar_builder import ClosedBar
from agent.portfolio.tracker import PortfolioTracker
from agent.runner.live_session import LiveSession

IST = ZoneInfo("Asia/Kolkata")


def _make_closed_bar(symbol: str = "HDFCBANK", bar_open: datetime | None = None) -> ClosedBar:
    if bar_open is None:
        bar_open = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
    return ClosedBar(
        symbol=symbol,
        timeframe="60m",
        bar_open=bar_open,
        open=1700.0,
        high=1720.0,
        low=1695.0,
        close=1710.0,
        tick_count=42,
    )


def _make_session(symbols: list[str] | None = None) -> LiveSession:
    if symbols is None:
        symbols = ["HDFCBANK"]
    portfolio = PortfolioTracker(
        initial_nav=Decimal("83000"),
        initial_cash=Decimal("83000"),
        start_time=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )
    return LiveSession(
        symbols=symbols,
        timeframe="60m",
        portfolio=portfolio,
        warmup_df=pl.DataFrame(),
    )


def test_live_session_creates_bar_builder_per_symbol() -> None:
    session = _make_session(["HDFCBANK", "TCS"])
    assert len(session._builders) == 2
    assert "HDFCBANK" in session._builders
    assert "TCS" in session._builders


def test_on_tick_returns_none_before_bar_closes() -> None:
    session = _make_session()
    ts = datetime(2024, 1, 2, 9, 20, tzinfo=IST)
    result = session._on_tick("HDFCBANK", 1710.0, ts)
    assert result is None


def test_on_tick_returns_closed_bar_at_slot_boundary() -> None:
    session = _make_session()
    session._on_tick("HDFCBANK", 1700.0, datetime(2024, 1, 2, 9, 15, tzinfo=IST))
    session._on_tick("HDFCBANK", 1720.0, datetime(2024, 1, 2, 9, 45, tzinfo=IST))
    result = session._on_tick("HDFCBANK", 1715.0, datetime(2024, 1, 2, 10, 15, tzinfo=IST))
    assert result is not None
    assert isinstance(result, ClosedBar)
    assert result.symbol == "HDFCBANK"


def test_on_tick_unknown_symbol_is_ignored() -> None:
    session = _make_session(["HDFCBANK"])
    result = session._on_tick("UNKNOWN", 100.0, datetime(2024, 1, 2, 9, 15, tzinfo=IST))
    assert result is None


def test_on_bar_closed_calls_process_bar() -> None:
    session = _make_session()
    called_with: list[pl.DataFrame] = []

    def fake_process_bar(df: pl.DataFrame, *, evaluation_time: datetime) -> list:
        called_with.append(df)
        return []

    # Need at least 2 rows for DailyLoop to run (warmup_df is empty here, so we
    # pre-populate _live_bars with one bar first, then call _on_bar_closed with a second)
    first_bar = _make_closed_bar(bar_open=datetime(2024, 1, 2, 9, 15, tzinfo=IST))
    session._live_bars["HDFCBANK"] = first_bar.to_dataframe()

    with patch.object(session._loop, "process_bar", side_effect=fake_process_bar):
        bar = _make_closed_bar(bar_open=datetime(2024, 1, 2, 10, 15, tzinfo=IST))
        session._on_bar_closed(bar)

    assert len(called_with) == 1


def test_is_within_session_returns_false_after_1530() -> None:
    session = _make_session()
    late = datetime(2024, 1, 2, 15, 31, tzinfo=IST)
    assert session._is_within_session(late) is False


def test_is_within_session_returns_true_at_1415() -> None:
    session = _make_session()
    mid = datetime(2024, 1, 2, 14, 15, tzinfo=IST)
    assert session._is_within_session(mid) is True
