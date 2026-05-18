# tests/backtest/test_cli_backtest.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent.cli import cli


def test_backtest_command_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["backtest", "--help"])
    assert result.exit_code == 0
    assert "--symbol" in result.output
    assert "--start" in result.output
    assert "--end" in result.output


def test_backtest_exits_when_cache_empty(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch("agent.cli.AppSettings") as MockSettings:
        s = MagicMock()
        s.parquet_cache_dir = tmp_path / "cache"
        s.paper_starting_capital = 83000.0
        MockSettings.return_value = s
        result = runner.invoke(
            cli,
            ["backtest", "--symbol", "HDFCBANK", "--start", "2024-01-02", "--end", "2024-01-05"],
        )
    assert result.exit_code == 1
    assert "No cached data" in result.output or "cache" in result.output.lower()


def test_backtest_prints_report_on_success(tmp_path: Path) -> None:
    """When BacktestRunner returns a non-empty report, backtest command prints metrics."""
    from datetime import date
    from decimal import Decimal

    from agent.backtest.metrics import BacktestMetrics, BacktestReport, SessionResult

    fake_session = SessionResult(
        session_date=date(2024, 1, 2),
        bars_processed=6,
        fills=2,
        gross_pnl=Decimal("1000"),
        costs=Decimal("50"),
        net_pnl=Decimal("950"),
        final_nav=Decimal("83950"),
    )
    fake_metrics = BacktestMetrics(
        total_sessions=1,
        winning_sessions=1,
        win_rate=1.0,
        total_gross_pnl=Decimal("1000"),
        total_costs=Decimal("50"),
        total_net_pnl=Decimal("950"),
        sharpe_ratio=1.5,
        max_drawdown=0.0,
        cagr=0.12,
        initial_nav=Decimal("83000"),
        final_nav=Decimal("83950"),
    )
    fake_report = BacktestReport(sessions=[fake_session], metrics=fake_metrics)

    runner = CliRunner()
    with (
        patch("agent.cli.AppSettings") as MockSettings,
        patch("agent.cli.BacktestRunner") as MockRunner,
    ):
        s = MagicMock()
        s.parquet_cache_dir = tmp_path / "cache"
        s.paper_starting_capital = 83000.0
        MockSettings.return_value = s

        MockRunner.return_value.run.return_value = fake_report

        result = runner.invoke(
            cli,
            ["backtest", "--symbol", "HDFCBANK", "--start", "2024-01-02", "--end", "2024-01-02"],
        )

    assert result.exit_code == 0
    assert "1" in result.output  # total_sessions
    assert "950" in result.output  # net_pnl somewhere in output
