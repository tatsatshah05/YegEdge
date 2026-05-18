from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent.cli import cli


def test_run_paper_command_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-paper", "--help"])
    assert result.exit_code == 0
    assert "--date" in result.output


def test_run_paper_exits_when_no_cache(tmp_path: Path) -> None:
    """run-paper should exit cleanly with a message when cache has no data."""
    runner = CliRunner()
    with patch("agent.cli.AppSettings") as MockSettings:
        s = MagicMock()
        s.parquet_cache_dir = tmp_path / "cache"
        s.journal_db_path = tmp_path / "journal.db"
        s.upstox_access_token = ""
        MockSettings.return_value = s
        result = runner.invoke(cli, ["run-paper"])
    assert result.exit_code == 1
    assert "No cached data" in result.output or "cached" in result.output.lower()


def test_runner_module_has_main() -> None:
    from pathlib import Path
    import importlib.util
    path = Path(__file__).parent.parent.parent / "agent" / "runner" / "__main__.py"
    spec = importlib.util.spec_from_file_location("agent.runner.__main__", path)
    assert spec is not None
