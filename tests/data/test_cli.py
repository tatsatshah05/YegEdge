from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from agent.cli import cli


def test_cli_help_exits_cleanly() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "refresh" in result.output
    assert "verify" in result.output


def test_refresh_command_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["refresh", "--help"])
    assert result.exit_code == 0


def test_verify_command_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["verify", "--help"])
    assert result.exit_code == 0


def test_refresh_exits_1_when_no_access_token(tmp_path: object) -> None:
    mock_settings = MagicMock()
    mock_settings.upstox_access_token = ""
    with patch("agent.cli.AppSettings", return_value=mock_settings):
        result = CliRunner().invoke(cli, ["refresh"])
    assert result.exit_code == 1


def test_verify_reports_no_cached_data(tmp_path: object) -> None:
    from pathlib import Path

    mock_settings = MagicMock()
    # Point to a directory that doesn't exist so coverage_report returns {}
    if isinstance(tmp_path, Path):
        mock_settings.parquet_cache_dir = tmp_path / "nonexistent_cache"
    else:
        mock_settings.parquet_cache_dir = Path("/tmp/nonexistent_yegedge_cache_xyz")

    with patch("agent.cli.AppSettings", return_value=mock_settings):
        result = CliRunner().invoke(cli, ["verify"])

    assert result.exit_code == 1
    assert "No cached data" in result.output or result.exit_code == 1
