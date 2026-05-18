# tests/runner/test_session_counter.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.runner.session_counter import PaperSessionCounter

LIVE_THRESHOLD = 60


def test_initial_count_is_zero(tmp_path: Path) -> None:
    counter = PaperSessionCounter(path=tmp_path / "sessions.json")
    assert counter.count() == 0


def test_increment_returns_new_count(tmp_path: Path) -> None:
    counter = PaperSessionCounter(path=tmp_path / "sessions.json")
    assert counter.increment() == 1
    assert counter.increment() == 2


def test_count_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    counter1 = PaperSessionCounter(path=path)
    counter1.increment()
    counter1.increment()

    counter2 = PaperSessionCounter(path=path)
    assert counter2.count() == 2


def test_not_ready_for_live_below_threshold(tmp_path: Path) -> None:
    counter = PaperSessionCounter(path=tmp_path / "sessions.json")
    for _ in range(59):
        counter.increment()
    assert counter.is_ready_for_live() is False


def test_ready_for_live_at_threshold(tmp_path: Path) -> None:
    counter = PaperSessionCounter(path=tmp_path / "sessions.json")
    for _ in range(60):
        counter.increment()
    assert counter.is_ready_for_live() is True


def test_count_file_contains_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    counter = PaperSessionCounter(path=path)
    counter.increment()
    data = json.loads(path.read_text())
    assert data["sessions_completed"] == 1
