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
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent.parent.parent / "agent" / "runner" / "__main__.py"
    spec = importlib.util.spec_from_file_location("agent.runner.__main__", path)
    assert spec is not None


def test_feature_pipeline_produces_required_strategy_columns() -> None:
    """Regression guard: FeaturePipeline.run() must add the columns TrendFollowingStrategy needs."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from agent.features.pipeline import FeaturePipeline

    IST = ZoneInfo("Asia/Kolkata")
    base_ts = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
    n = 60  # enough bars for EMA(50) + ADX(14) to be meaningful
    df = pl.DataFrame(
        {
            "symbol": ["HDFCBANK"] * n,
            "timeframe": ["60m"] * n,
            "timestamp": [base_ts + timedelta(hours=i) for i in range(n)],
            "open": [1700.0] * n,
            "high": [1720.0] * n,
            "low": [1695.0] * n,
            "close": [1710.0] * n,
            "volume": [100_000] * n,
            "value": [171_000_000.0] * n,
            "data_quality": ["ok"] * n,
        }
    )
    enriched = FeaturePipeline().run(df)
    for col in ("ema_21", "ema_50", "adx_14", "atr_14"):
        assert col in enriched.columns, f"Missing required column: {col}"
