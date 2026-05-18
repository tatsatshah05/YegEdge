from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class JournalEntryType(StrEnum):
    SIGNAL = "signal"
    DECISION = "decision"
    FILL = "fill"
    REJECTION = "rejection"
    PNL = "pnl"


@dataclass(frozen=True, slots=True)
class JournalEntry:
    """Append-only audit record.

    payload is a JSON string containing the full event data. Using a string
    keeps the journal schema stable as event types evolve — no migrations needed
    to add fields inside payload.
    """

    entry_id: str  # unique, caller-assigned
    timestamp: datetime  # IST-aware
    entry_type: JournalEntryType
    symbol: str | None  # None for system-level events (e.g., PNL snapshots)
    payload: str  # JSON string
