from __future__ import annotations

import dataclasses

from agent.decision.types import ResearchNote
from agent.strategies.types import Signal


def _regime_bucket(regime_fit: float) -> str:
    if regime_fit >= 0.8:
        return "strong"
    if regime_fit >= 0.5:
        return "moderate"
    return "weak"


def _rr_bucket(expected_r: float) -> str:
    if expected_r >= 3.0:
        return "high"
    if expected_r >= 2.0:
        return "medium"
    return "low"


def _cache_key(signal: Signal) -> str:
    return f"{signal.action}:{_regime_bucket(signal.regime_fit)}:{_rr_bucket(signal.expected_r)}"


class NoteCache:
    """In-memory cache for ResearchNote objects, keyed by (action, regime_bucket, rr_bucket).

    Symbol is intentionally excluded from the key: structurally identical signals on
    different symbols share a cache entry because the research note captures the pattern,
    not symbol-specific data.
    """

    def __init__(self) -> None:
        self._store: dict[str, ResearchNote] = {}

    def get(self, signal: Signal) -> ResearchNote | None:
        """Return a cached note (with cached=True) or None on miss."""
        key = _cache_key(signal)
        note = self._store.get(key)
        if note is None:
            return None
        return dataclasses.replace(note, cached=True)

    def put(self, signal: Signal, note: ResearchNote) -> None:
        """Store note under the cache key derived from signal."""
        key = _cache_key(signal)
        self._store[key] = note

    @property
    def size(self) -> int:
        return len(self._store)
