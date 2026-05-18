from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent.journal.store import JournalStore
from agent.journal.types import JournalEntry, JournalEntryType

IST = ZoneInfo("Asia/Kolkata")

T0 = datetime(2024, 1, 2, 9, 15, tzinfo=IST)


def _make_entry(
    entry_type: JournalEntryType = JournalEntryType.FILL,
    symbol: str = "HDFCBANK",
    payload: str = '{"action": "enter_long", "quantity": 10}',
) -> JournalEntry:
    return JournalEntry(
        entry_id="test-entry-001",
        timestamp=T0,
        entry_type=entry_type,
        symbol=symbol,
        payload=payload,
    )


def test_journal_entry_type_values() -> None:
    assert JournalEntryType.SIGNAL == "signal"
    assert JournalEntryType.DECISION == "decision"
    assert JournalEntryType.FILL == "fill"
    assert JournalEntryType.REJECTION == "rejection"
    assert JournalEntryType.PNL == "pnl"


def test_journal_entry_is_frozen() -> None:
    entry = _make_entry()
    with pytest.raises(AttributeError):
        entry.symbol = "TCS"  # type: ignore[misc]


def test_store_log_and_query_returns_entry(tmp_path: Path) -> None:
    store = JournalStore(db_path=tmp_path / "journal.db")
    entry = _make_entry()
    store.log(entry)
    results = store.query(limit=10)
    assert len(results) == 1
    assert results[0].symbol == "HDFCBANK"


def test_store_query_filter_by_entry_type(tmp_path: Path) -> None:
    store = JournalStore(db_path=tmp_path / "journal.db")
    store.log(_make_entry(entry_type=JournalEntryType.FILL))
    store.log(_make_entry(entry_type=JournalEntryType.SIGNAL, payload="{}"))
    fills = store.query(entry_type=JournalEntryType.FILL)
    assert len(fills) == 1
    assert fills[0].entry_type == JournalEntryType.FILL


def test_store_query_filter_by_symbol(tmp_path: Path) -> None:
    store = JournalStore(db_path=tmp_path / "journal.db")
    store.log(_make_entry(symbol="HDFCBANK"))
    store.log(_make_entry(symbol="TCS", payload="{}"))
    hdfc = store.query(symbol="HDFCBANK")
    assert len(hdfc) == 1
    assert hdfc[0].symbol == "HDFCBANK"


def test_store_is_append_only_and_ordered(tmp_path: Path) -> None:
    store = JournalStore(db_path=tmp_path / "journal.db")
    for i in range(5):
        store.log(
            JournalEntry(
                entry_id=f"entry-{i:03d}",
                timestamp=T0,
                entry_type=JournalEntryType.FILL,
                symbol="HDFCBANK",
                payload=f'{{"seq": {i}}}',
            )
        )
    results = store.query(limit=10)
    assert len(results) == 5
    ids = [r.entry_id for r in results]
    for i in range(5):
        assert f"entry-{i:03d}" in ids


def test_store_persists_across_open_close(tmp_path: Path) -> None:
    db_path = tmp_path / "journal.db"
    store1 = JournalStore(db_path=db_path)
    store1.log(_make_entry(payload='{"qty": 5}'))

    store2 = JournalStore(db_path=db_path)
    results = store2.query()
    assert len(results) == 1
    assert results[0].symbol == "HDFCBANK"


def test_store_query_limit_respected(tmp_path: Path) -> None:
    store = JournalStore(db_path=tmp_path / "journal.db")
    for i in range(20):
        store.log(
            JournalEntry(
                entry_id=f"entry-{i:03d}",
                timestamp=T0,
                entry_type=JournalEntryType.FILL,
                symbol="HDFCBANK",
                payload=f'{{"i": {i}}}',
            )
        )
    results = store.query(limit=5)
    assert len(results) == 5
