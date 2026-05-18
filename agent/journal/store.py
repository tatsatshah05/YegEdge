from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import structlog

from agent.journal.types import JournalEntry, JournalEntryType

logger = structlog.get_logger()

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS journal (
    row_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id    TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    entry_type  TEXT NOT NULL,
    symbol      TEXT,
    payload     TEXT NOT NULL
)
"""


class JournalStore:
    """Append-only SQLite journal.

    Uses Python's stdlib sqlite3 — no ORM. The table has no UPDATE or DELETE
    paths; every event is written once and read back exactly as stored.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def log(self, entry: JournalEntry) -> None:
        """Append a JournalEntry. Never raises on valid input."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO journal (entry_id, timestamp, entry_type, symbol, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    entry.entry_id,
                    entry.timestamp.isoformat(),
                    str(entry.entry_type),
                    entry.symbol,
                    entry.payload,
                ),
            )
        logger.debug(
            "journal.log",
            entry_id=entry.entry_id,
            entry_type=str(entry.entry_type),
            symbol=entry.symbol,
        )

    def query(
        self,
        *,
        entry_type: JournalEntryType | None = None,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[JournalEntry]:
        """Return recent entries, newest last, filtered optionally by type/symbol."""
        sql = "SELECT entry_id, timestamp, entry_type, symbol, payload FROM journal"
        conditions: list[str] = []
        params: list[object] = []

        if entry_type is not None:
            conditions.append("entry_type = ?")
            params.append(str(entry_type))
        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY row_id ASC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            JournalEntry(
                entry_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                entry_type=JournalEntryType(row[2]),
                symbol=row[3],
                payload=row[4],
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
