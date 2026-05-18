from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.ai.cache import NoteCache
from agent.data.types import DataQuality
from agent.decision.types import ResearchNote
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")


def _make_signal(
    symbol: str = "HDFCBANK",
    action: Action = Action.ENTER_LONG,
    regime_fit: float = 0.9,
    expected_r: float = 2.5,
) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        confidence=0.75,
        suggested_stop=Decimal("1680.00"),
        suggested_target=Decimal("1750.00"),
        invalidation_condition="Close below EMA21",
        expected_r=expected_r,
        time_horizon_hours=4,
        regime_fit=regime_fit,
        data_quality=DataQuality.OK,
        strategy_name="trend_following_v1",
        explanation="EMA cross",
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )


def _make_note(veto: bool = False) -> ResearchNote:
    return ResearchNote(
        signal_id="HDFCBANK:enter_long:2024-01-02T09:15:00+05:30",
        bullish_case="Strong trend.",
        bearish_case="Could reverse.",
        dominant_risk="Earnings surprise.",
        regime_fit_assessment="Trending regime suits strategy.",
        confidence_qualitative="HIGH",
        veto=veto,
        veto_reason=None,
        model_used="claude-haiku-4-5-20251001",
        tokens_used=150,
        cached=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_miss_returns_none() -> None:
    """get on empty cache returns None."""
    cache = NoteCache()
    signal = _make_signal()
    assert cache.get(signal) is None


@pytest.mark.parametrize(
    "regime_fit,expected_regime,expected_r,expected_rr",
    [
        (0.0, "weak", 1.99, "low"),
        (0.49, "weak", 1.99, "low"),
        (0.5, "moderate", 2.0, "medium"),
        (0.79, "moderate", 2.99, "medium"),
        (0.8, "strong", 3.0, "high"),
        (1.0, "strong", 4.0, "high"),
    ],
)
def test_bucket_boundaries(
    regime_fit: float,
    expected_regime: str,
    expected_r: float,
    expected_rr: str,
) -> None:
    from agent.ai.cache import _cache_key

    sig = _make_signal(regime_fit=regime_fit, expected_r=expected_r)
    key = _cache_key(sig)
    assert key == f"{sig.action}:{expected_regime}:{expected_rr}"


def test_hit_sets_cached_true() -> None:
    """After put, get returns the note with cached=True even if the original had cached=False."""
    cache = NoteCache()
    signal = _make_signal()
    note = _make_note()
    assert note.cached is False  # confirm the original is False
    cache.put(signal, note)
    result = cache.get(signal)
    assert result is not None
    assert result.cached is True


def test_same_pattern_hits_cache() -> None:
    """Two signals with same action/regime_fit/expected_r but different symbols share a cache entry."""
    cache = NoteCache()
    signal_hdfc = _make_signal(symbol="HDFCBANK")
    signal_infy = _make_signal(symbol="INFY")
    note = _make_note()

    cache.put(signal_hdfc, note)

    # Different symbol, same structural pattern — should hit
    result = cache.get(signal_infy)
    assert result is not None
    assert result.cached is True


def test_different_action_misses_cache() -> None:
    """ENTER_LONG and EXIT_LONG signals with same regime_fit/expected_r have separate cache entries."""
    cache = NoteCache()
    signal_enter = _make_signal(action=Action.ENTER_LONG)
    signal_exit = _make_signal(action=Action.EXIT_LONG)
    note = _make_note()

    cache.put(signal_enter, note)

    # EXIT_LONG was never stored — should miss
    result = cache.get(signal_exit)
    assert result is None


def test_size_property() -> None:
    """size increments correctly as distinct keys are inserted."""
    cache = NoteCache()
    assert cache.size == 0

    # First entry
    signal_enter = _make_signal(action=Action.ENTER_LONG, regime_fit=0.9, expected_r=2.5)
    cache.put(signal_enter, _make_note())
    assert cache.size == 1

    # Second entry: different action → different key
    signal_exit = _make_signal(action=Action.EXIT_LONG, regime_fit=0.9, expected_r=2.5)
    cache.put(signal_exit, _make_note())
    assert cache.size == 2

    # Overwriting an existing key must not increase size
    cache.put(signal_enter, _make_note())
    assert cache.size == 2
