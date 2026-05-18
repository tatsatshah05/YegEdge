# tests/backtest/test_runner.py
from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.backtest.costs import IndianCostModel
from agent.backtest.runner import BacktestRunner
from agent.data.cache import ParquetCache
from agent.execution.types import ExecutionMode, Fill
from agent.runner.types import DailySessionResult
from agent.strategies.types import Action

IST = ZoneInfo("Asia/Kolkata")


def _make_bars(session_date: date, n: int = 6) -> pl.DataFrame:
    """n bars starting at 09:15 IST on session_date, hourly."""
    base = datetime(session_date.year, session_date.month, session_date.day, 9, 15, tzinfo=IST)
    return pl.DataFrame(
        {
            "symbol": ["HDFCBANK"] * n,
            "timeframe": ["60m"] * n,
            "timestamp": [base + timedelta(hours=i) for i in range(n)],
            "open": [1700.0] * n,
            "high": [1720.0] * n,
            "low": [1695.0] * n,
            "close": [1710.0] * n,
            "volume": [100_000] * n,
            "value": [171_000_000.0] * n,
            "data_quality": ["ok"] * n,
        }
    )


def _fake_fill() -> Fill:
    return Fill(
        order_id="paper-HDFCBANK-20240102091500-sig-0001",
        symbol="HDFCBANK",
        action=Action.ENTER_LONG,
        quantity=10,
        fill_price=Decimal("1710.00"),
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
        signal_id="sig-0001",
        strategy_name="trend_following_v1",
        execution_mode=ExecutionMode.PAPER,
    )


def _fake_result(session_date: date, pnl: float = 0.0, fills: tuple = ()) -> DailySessionResult:
    return DailySessionResult(
        session_date=session_date,
        bars_processed=6,
        signals_generated=1,
        decisions_made=1,
        fills=fills,
        rejections=0,
        ai_cache_hits=0,
        final_nav=Decimal(str(83000.0 + pnl)),
        daily_pnl=Decimal(str(pnl)),
        peak_nav=Decimal(str(max(83000.0, 83000.0 + pnl))),
    )


def _make_runner(cache: ParquetCache) -> BacktestRunner:
    return BacktestRunner(
        strategy=MagicMock(),
        risk_manager=MagicMock(),
        cache=cache,
        initial_nav=Decimal("83000"),
    )


def test_runner_returns_empty_report_when_no_cache_data(tmp_path: Path) -> None:
    runner = _make_runner(ParquetCache(root=tmp_path / "cache"))
    report = runner.run(
        symbol="HDFCBANK",
        timeframe="60m",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 5),
    )
    assert len(report.sessions) == 0
    assert report.metrics.total_sessions == 0
    assert report.metrics.initial_nav == Decimal("83000")


def test_runner_creates_one_session_per_trading_day(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path / "cache")
    # Jan 2 (Tue) + Jan 3 (Wed) are both trading days
    bars = pl.concat([_make_bars(date(2024, 1, 2)), _make_bars(date(2024, 1, 3))])
    cache.write(bars, symbol="HDFCBANK", timeframe="60m")

    runner = _make_runner(cache)
    with patch("agent.backtest.runner.DailyLoop") as MockLoop:
        MockLoop.return_value.run.return_value = _fake_result(date(2024, 1, 2))
        report = runner.run(
            symbol="HDFCBANK",
            timeframe="60m",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
        )
    assert len(report.sessions) == 2


def test_runner_deducts_costs_from_gross_pnl(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path / "cache")
    cache.write(_make_bars(date(2024, 1, 2)), symbol="HDFCBANK", timeframe="60m")

    runner = _make_runner(cache)
    fill = _fake_fill()
    with patch("agent.backtest.runner.DailyLoop") as MockLoop:
        MockLoop.return_value.run.return_value = _fake_result(
            date(2024, 1, 2), pnl=1000.0, fills=(fill,)
        )
        report = runner.run(
            symbol="HDFCBANK",
            timeframe="60m",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
        )
    s = report.sessions[0]
    assert s.fills == 1
    assert s.gross_pnl == Decimal("1000.0")
    assert s.costs > Decimal("0")
    assert s.net_pnl == s.gross_pnl - s.costs
    assert s.net_pnl < s.gross_pnl


def test_runner_accumulates_nav_across_sessions(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path / "cache")
    bars = pl.concat([_make_bars(date(2024, 1, 2)), _make_bars(date(2024, 1, 3))])
    cache.write(bars, symbol="HDFCBANK", timeframe="60m")

    runner = _make_runner(cache)
    results_queue = [
        _fake_result(date(2024, 1, 2), pnl=500.0),
        _fake_result(date(2024, 1, 3), pnl=300.0),
    ]
    call_idx = 0

    def pop_result(*args, **kwargs):
        nonlocal call_idx
        r = results_queue[call_idx]
        call_idx += 1
        return r

    with patch("agent.backtest.runner.DailyLoop") as MockLoop:
        MockLoop.return_value.run.side_effect = pop_result
        report = runner.run(
            symbol="HDFCBANK",
            timeframe="60m",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
        )
    # Final NAV >= 83000 + 500 + 300 - costs (costs are small but non-zero)
    assert report.sessions[-1].final_nav > Decimal("83700")
    assert report.sessions[-1].final_nav <= Decimal("83800")


def test_runner_skips_days_with_no_bars_in_cache(tmp_path: Path) -> None:
    """A trading day with no cached bars is silently skipped (no DailyLoop created)."""
    cache = ParquetCache(root=tmp_path / "cache")
    # Only write Jan 2 — Jan 3 has no data
    cache.write(_make_bars(date(2024, 1, 2)), symbol="HDFCBANK", timeframe="60m")

    runner = _make_runner(cache)
    with patch("agent.backtest.runner.DailyLoop") as MockLoop:
        MockLoop.return_value.run.return_value = _fake_result(date(2024, 1, 2))
        report = runner.run(
            symbol="HDFCBANK",
            timeframe="60m",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
        )
    assert len(report.sessions) == 1
    assert MockLoop.return_value.run.call_count == 1
