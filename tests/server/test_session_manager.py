from __future__ import annotations

from server.events import EventBus
from server.session_manager import SessionManager


def test_is_running_false_initially() -> None:
    bus = EventBus()
    manager = SessionManager(bus)
    assert manager.is_running is False


def test_portfolio_state_none_when_no_session() -> None:
    bus = EventBus()
    manager = SessionManager(bus)
    assert manager.portfolio_state is None


def test_last_bars_empty_initially() -> None:
    bus = EventBus()
    manager = SessionManager(bus)
    assert manager.last_bars == {}


def test_status_dict_shape() -> None:
    bus = EventBus()
    manager = SessionManager(bus)
    s = manager.status()
    assert s["running"] is False
    assert "timeframe" in s
    assert "symbols_count" in s
