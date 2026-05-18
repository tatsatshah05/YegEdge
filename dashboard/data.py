# dashboard/data.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from agent.data.cache import ParquetCache
from agent.journal.store import JournalStore
from agent.journal.types import JournalEntryType
from agent.runner.session_counter import PaperSessionCounter

_EMPTY_PNL_SCHEMA = {
    "session_date": pl.Utf8,
    "final_nav": pl.Float64,
    "daily_pnl": pl.Float64,
    "orders_today": pl.Int64,
}

_EMPTY_FILL_SCHEMA = {
    "timestamp": pl.Utf8,
    "symbol": pl.Utf8,
    "action": pl.Utf8,
    "quantity": pl.Int64,
    "price": pl.Float64,
    "signal_id": pl.Utf8,
}

_EMPTY_REJECTION_SCHEMA = {
    "timestamp": pl.Utf8,
    "symbol": pl.Utf8,
    "reason": pl.Utf8,
    "detail": pl.Utf8,
}


def load_pnl_history(db_path: Path, limit: int = 500) -> pl.DataFrame:
    """Load PNL journal entries as a DataFrame.

    Returns columns: session_date (str), final_nav (float), daily_pnl (float),
    orders_today (int). Returns an empty DataFrame with correct schema when no data.
    """
    store = JournalStore(db_path=db_path)
    entries = store.query(entry_type=JournalEntryType.PNL, limit=limit)
    rows = []
    for e in entries:
        try:
            d = json.loads(e.payload)
            rows.append(
                {
                    "session_date": str(d.get("session_date", "")),
                    "final_nav": float(d.get("final_nav", 0)),
                    "daily_pnl": float(d.get("daily_pnl", 0)),
                    "orders_today": int(d.get("orders_today", 0)),
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    if not rows:
        return pl.DataFrame(schema=_EMPTY_PNL_SCHEMA)
    return pl.DataFrame(rows)


def load_fills(db_path: Path, limit: int = 1000) -> pl.DataFrame:
    """Load FILL journal entries as a DataFrame.

    Returns columns: timestamp, symbol, action, quantity (int), price (float),
    signal_id (str). Returns empty DataFrame with correct schema when no data.
    """
    store = JournalStore(db_path=db_path)
    entries = store.query(entry_type=JournalEntryType.FILL, limit=limit)
    rows = []
    for e in entries:
        try:
            d = json.loads(e.payload)
            rows.append(
                {
                    "timestamp": e.timestamp.isoformat(),
                    "symbol": e.symbol or "",
                    "action": str(d.get("action", "")),
                    "quantity": int(d.get("quantity", 0)),
                    "price": float(d.get("price", 0)),
                    "signal_id": str(d.get("signal_id", "")),
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    if not rows:
        return pl.DataFrame(schema=_EMPTY_FILL_SCHEMA)
    return pl.DataFrame(rows)


def load_rejections(db_path: Path, limit: int = 1000) -> pl.DataFrame:
    """Load REJECTION journal entries as a DataFrame.

    Returns columns: timestamp, symbol, reason, detail.
    """
    store = JournalStore(db_path=db_path)
    entries = store.query(entry_type=JournalEntryType.REJECTION, limit=limit)
    rows = []
    for e in entries:
        try:
            d = json.loads(e.payload)
            rows.append(
                {
                    "timestamp": e.timestamp.isoformat(),
                    "symbol": e.symbol or "",
                    "reason": str(d.get("reason", "")),
                    "detail": str(d.get("detail", "")),
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    if not rows:
        return pl.DataFrame(schema=_EMPTY_REJECTION_SCHEMA)
    return pl.DataFrame(rows)


def load_session_count(json_path: Path) -> int:
    """Return the number of completed paper sessions from the counter JSON file."""
    return PaperSessionCounter(path=json_path).count()


def load_coverage_summary(
    cache_root: Path,
) -> dict[str, dict[str, tuple[datetime, datetime]]]:
    """Return the Parquet cache coverage report."""
    return ParquetCache(root=cache_root).coverage_report()


def compute_equity_stats(pnl_df: pl.DataFrame) -> dict[str, float]:
    """Compute headline stats from a PNL history DataFrame.

    Returns: total_sessions, total_pnl, win_rate (0-1), current_nav.
    All values are 0.0 / 0 when pnl_df is empty.
    """
    if len(pnl_df) == 0:
        return {"total_sessions": 0, "total_pnl": 0.0, "win_rate": 0.0, "current_nav": 0.0}
    total_sessions = len(pnl_df)
    total_pnl = float(pnl_df["daily_pnl"].sum())
    wins = int((pnl_df["daily_pnl"] > 0).sum())
    win_rate = wins / total_sessions
    current_nav = float(pnl_df["final_nav"][-1])
    return {
        "total_sessions": float(total_sessions),
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "current_nav": current_nav,
    }
