from __future__ import annotations

from pathlib import Path

import pytest

from agent.monitoring.kill_switch import KillSwitch


def test_not_triggered_when_no_flag_file(tmp_path: Path) -> None:
    ks = KillSwitch(flag_path=tmp_path / ".kill_switch")
    assert ks.is_active() is False


def test_activate_creates_flag_file(tmp_path: Path) -> None:
    ks = KillSwitch(flag_path=tmp_path / ".kill_switch")
    ks.activate(reason="test activation")
    assert (tmp_path / ".kill_switch").exists()


def test_is_active_returns_true_when_flag_exists(tmp_path: Path) -> None:
    flag = tmp_path / ".kill_switch"
    flag.write_text("manual stop")
    ks = KillSwitch(flag_path=flag)
    assert ks.is_active() is True


def test_deactivate_removes_flag_file(tmp_path: Path) -> None:
    flag = tmp_path / ".kill_switch"
    flag.write_text("manual stop")
    ks = KillSwitch(flag_path=flag)
    ks.deactivate()
    assert not flag.exists()


def test_deactivate_is_idempotent_when_no_flag(tmp_path: Path) -> None:
    ks = KillSwitch(flag_path=tmp_path / ".kill_switch")
    ks.deactivate()  # should not raise
    assert ks.is_active() is False


def test_activate_reason_written_to_file(tmp_path: Path) -> None:
    ks = KillSwitch(flag_path=tmp_path / ".kill_switch")
    ks.activate(reason="drawdown limit hit")
    content = (tmp_path / ".kill_switch").read_text()
    assert "drawdown limit hit" in content


def test_portfolio_tracker_activate_kill_switch() -> None:
    from datetime import datetime
    from decimal import Decimal
    from zoneinfo import ZoneInfo
    from agent.portfolio.tracker import PortfolioTracker

    IST = ZoneInfo("Asia/Kolkata")
    tracker = PortfolioTracker(
        initial_nav=Decimal("100000"),
        initial_cash=Decimal("100000"),
        start_time=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )
    state = tracker.activate_kill_switch()
    assert state.kill_switch_active is True
