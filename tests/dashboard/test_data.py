# tests/dashboard/test_data.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.journal.store import JournalStore
from agent.journal.types import JournalEntry, JournalEntryType
from dashboard.data import (
    compute_equity_stats,
    load_fills,
    load_pnl_history,
    load_rejections,
    load_session_count,
)

IST = ZoneInfo("Asia/Kolkata")
T0 = datetime(2024, 1, 2, 15, 30, tzinfo=IST)


def _seed_pnl(db_path: Path, session_date: str, nav: float, pnl: float, orders: int = 2) -> None:
    store = JournalStore(db_path=db_path)
    store.log(
        JournalEntry(
            entry_id=f"pnl-{session_date}",
            timestamp=T0,
            entry_type=JournalEntryType.PNL,
            symbol=None,
            payload=json.dumps(
                {
                    "session_date": session_date,
                    "final_nav": nav,
                    "daily_pnl": pnl,
                    "orders_today": orders,
                }
            ),
        )
    )


def _seed_fill(db_path: Path, entry_id: str, symbol: str, qty: int, price: float) -> None:
    store = JournalStore(db_path=db_path)
    store.log(
        JournalEntry(
            entry_id=entry_id,
            timestamp=T0,
            entry_type=JournalEntryType.FILL,
            symbol=symbol,
            payload=json.dumps(
                {"action": "enter_long", "quantity": qty, "price": price, "signal_id": "test"}
            ),
        )
    )


def _seed_rejection(db_path: Path, entry_id: str, symbol: str, reason: str) -> None:
    store = JournalStore(db_path=db_path)
    store.log(
        JournalEntry(
            entry_id=entry_id,
            timestamp=T0,
            entry_type=JournalEntryType.REJECTION,
            symbol=symbol,
            payload=json.dumps({"reason": reason, "detail": "test", "signal_id": "test"}),
        )
    )


def test_load_pnl_history_returns_correct_rows(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    _seed_pnl(db, "2024-01-02", nav=101500.0, pnl=1500.0)
    _seed_pnl(db, "2024-01-03", nav=102000.0, pnl=500.0)

    df = load_pnl_history(db)
    assert len(df) == 2
    assert "final_nav" in df.columns
    assert "daily_pnl" in df.columns
    assert "session_date" in df.columns


def test_load_pnl_history_empty_returns_schema(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    JournalStore(db_path=db)  # initialise empty DB
    df = load_pnl_history(db)
    assert len(df) == 0
    assert "final_nav" in df.columns


def test_load_fills_returns_correct_symbols(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    _seed_fill(db, "fill-1", "HDFCBANK", 10, 1710.0)
    _seed_fill(db, "fill-2", "TCS", 5, 3500.0)

    df = load_fills(db)
    assert len(df) == 2
    assert set(df["symbol"].to_list()) == {"HDFCBANK", "TCS"}
    assert "price" in df.columns
    assert "quantity" in df.columns


def test_load_rejections_returns_correct_reasons(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    _seed_rejection(db, "rej-1", "INFY", "max_positions_reached")
    _seed_rejection(db, "rej-2", "WIPRO", "daily_loss_cap")

    df = load_rejections(db)
    assert len(df) == 2
    assert "reason" in df.columns
    assert "max_positions_reached" in df["reason"].to_list()


def test_load_session_count_zero_when_no_file(tmp_path: Path) -> None:
    assert load_session_count(tmp_path / "sessions.json") == 0


def test_load_session_count_reflects_increments(tmp_path: Path) -> None:
    from agent.runner.session_counter import PaperSessionCounter

    path = tmp_path / "sessions.json"
    c = PaperSessionCounter(path=path)
    c.increment()
    c.increment()
    c.increment()
    assert load_session_count(path) == 3


def test_compute_equity_stats_empty_dataframe() -> None:
    empty_df = pl.DataFrame(
        schema={
            "session_date": pl.Utf8,
            "final_nav": pl.Float64,
            "daily_pnl": pl.Float64,
            "orders_today": pl.Int64,
        }
    )
    stats = compute_equity_stats(empty_df)
    assert stats["total_sessions"] == 0
    assert stats["win_rate"] == 0.0
    assert stats["current_nav"] == 0.0


def test_compute_equity_stats_with_data(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    _seed_pnl(db, "2024-01-02", nav=101500.0, pnl=1500.0)
    _seed_pnl(db, "2024-01-03", nav=101000.0, pnl=-500.0)
    _seed_pnl(db, "2024-01-04", nav=102000.0, pnl=1000.0)

    df = load_pnl_history(db)
    stats = compute_equity_stats(df)
    assert stats["total_sessions"] == 3
    assert stats["total_pnl"] == pytest.approx(2000.0)
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["current_nav"] == pytest.approx(102000.0)
