from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent.cli import cli


def test_live_paper_command_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["live-paper", "--help"])
    assert result.exit_code == 0
    assert "--timeframe" in result.output


def test_live_paper_exits_when_no_access_token(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch("agent.cli.AppSettings") as MockSettings:
        s = MagicMock()
        s.parquet_cache_dir = tmp_path / "cache"
        s.journal_db_path = tmp_path / "journal.db"
        s.upstox_access_token = ""
        s.telegram_bot_token = ""
        s.telegram_chat_id = ""
        MockSettings.return_value = s
        result = runner.invoke(cli, ["live-paper"])
    assert result.exit_code == 1
    assert "token" in result.output.lower() or "access" in result.output.lower()


def test_live_paper_exits_when_no_cache(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch("agent.cli.AppSettings") as MockSettings:
        s = MagicMock()
        s.parquet_cache_dir = tmp_path / "cache"
        s.journal_db_path = tmp_path / "journal.db"
        s.upstox_access_token = "fake-token"
        s.telegram_bot_token = ""
        s.telegram_chat_id = ""
        MockSettings.return_value = s
        result = runner.invoke(cli, ["live-paper"])
    assert result.exit_code == 1
