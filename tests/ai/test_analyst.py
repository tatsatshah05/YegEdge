from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from agent.ai.analyst import AIAnalyst
from agent.ai.cache import NoteCache
from agent.data.types import DataQuality
from agent.decision.types import ResearchNote
from agent.strategies.types import Action, Signal
from config.settings import AppSettings

IST = ZoneInfo("Asia/Kolkata")


def _make_signal(
    confidence: float = 0.75,
    regime_fit: float = 0.9,
    expected_r: float = 2.5,
) -> Signal:
    return Signal(
        symbol="HDFCBANK",
        action=Action.ENTER_LONG,
        confidence=confidence,
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


def _make_settings(spend_cap: float = 1500.0) -> AppSettings:
    s = AppSettings()
    object.__setattr__(s, "anthropic_api_key", "test-key")
    object.__setattr__(s, "max_monthly_api_spend_inr", Decimal(str(spend_cap)))
    object.__setattr__(s, "claude_model_primary", "claude-sonnet-4-6")
    object.__setattr__(s, "claude_model_cheap", "claude-haiku-4-5-20251001")
    return s


def _fake_claude_response(veto: bool = False) -> MagicMock:
    """Build a fake anthropic response mimicking tool_use output."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {
        "bullish_case": "Strong trend.",
        "bearish_case": "Could reverse.",
        "dominant_risk": "Earnings surprise.",
        "regime_fit_assessment": "Trending regime suits strategy.",
        "confidence_qualitative": "HIGH",
        "veto": veto,
        "veto_reason": "Earnings tonight." if veto else None,
    }
    usage = MagicMock()
    usage.input_tokens = 200
    usage.output_tokens = 100
    response = MagicMock()
    response.content = [tool_block]
    response.usage = usage
    return response


def _make_analyst(
    settings: AppSettings | None = None,
    mock_response: MagicMock | None = None,
) -> tuple[AIAnalyst, MagicMock]:
    settings = settings or _make_settings()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response or _fake_claude_response()
    analyst = AIAnalyst(settings=settings, _client=mock_client)
    return analyst, mock_client


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


def test_analyse_returns_research_note() -> None:
    analyst, _ = _make_analyst()
    note = analyst.analyse(_make_signal())
    assert isinstance(note, ResearchNote)
    assert note.bullish_case == "Strong trend."
    assert note.veto is False


def test_analyse_uses_cheap_model_for_high_confidence() -> None:
    analyst, mock_client = _make_analyst()
    analyst.analyse(_make_signal(confidence=0.80))
    call_kwargs = mock_client.messages.create.call_args
    assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"


def test_analyse_uses_primary_model_for_low_confidence() -> None:
    analyst, mock_client = _make_analyst()
    analyst.analyse(_make_signal(confidence=0.79))
    call_kwargs = mock_client.messages.create.call_args
    assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"


def test_analyse_caches_result_and_hits_on_second_call() -> None:
    analyst, mock_client = _make_analyst()
    sig = _make_signal()
    analyst.analyse(sig)
    analyst.analyse(sig)  # second call should hit cache
    assert mock_client.messages.create.call_count == 1  # Claude called only once


def test_cache_hit_returns_note_with_cached_true() -> None:
    analyst, _ = _make_analyst()
    sig = _make_signal()
    analyst.analyse(sig)
    note2 = analyst.analyse(sig)
    assert note2.cached is True


def test_analyse_propagates_signal_id_from_cache_hit() -> None:
    """Cache hit must carry the current signal's signal_id, not the cached one."""
    analyst, _ = _make_analyst()
    sig = _make_signal()
    analyst.analyse(sig)
    note2 = analyst.analyse(sig)
    expected_id = f"{sig.symbol}:{sig.action}:{sig.timestamp.isoformat()}"
    assert note2.signal_id == expected_id


def test_analyse_veto_propagated() -> None:
    analyst, _ = _make_analyst(mock_response=_fake_claude_response(veto=True))
    note = analyst.analyse(_make_signal())
    assert note.veto is True
    assert note.veto_reason == "Earnings tonight."


def test_budget_exceeded_returns_degraded_note_without_api_call() -> None:
    settings = _make_settings(spend_cap=0.0)  # cap = 0 → always over budget
    analyst, mock_client = _make_analyst(settings=settings)
    note = analyst.analyse(_make_signal())
    assert mock_client.messages.create.call_count == 0
    assert note.veto is False
    assert "Budget cap" in note.bullish_case


def test_spend_tally_increases_after_api_call() -> None:
    analyst, _ = _make_analyst()
    assert analyst._spend_inr == Decimal("0")
    analyst.analyse(_make_signal())
    assert analyst._spend_inr > Decimal("0")


def test_missing_tool_use_block_raises_runtime_error() -> None:
    """If Claude returns no tool_use block, raise RuntimeError."""
    mock_response = MagicMock()
    mock_response.content = []  # no tool blocks
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
    analyst, _ = _make_analyst(mock_response=mock_response)
    with pytest.raises(RuntimeError, match="submit_research_note"):
        analyst.analyse(_make_signal())


def test_suspect_data_quality_returns_degraded_note_without_api_call() -> None:
    import dataclasses

    from agent.data.types import DataQuality

    sig = _make_signal()
    # Override data_quality to SUSPECT using dataclasses.replace since Signal is frozen
    sig_suspect = dataclasses.replace(sig, data_quality=DataQuality.SUSPECT)
    analyst, mock_client = _make_analyst()
    note = analyst.analyse(sig_suspect)
    assert mock_client.messages.create.call_count == 0
    assert note.veto is False
    assert "suspect" in note.dominant_risk.lower()


def test_budget_check_precedes_cache_lookup() -> None:
    """Even with a cache hit available, over-budget returns degraded note."""
    settings = _make_settings(spend_cap=1500.0)
    analyst, _ = _make_analyst(settings=settings)
    sig = _make_signal()
    # Populate cache with a real note via first call
    analyst.analyse(sig)
    assert analyst._spend_inr > Decimal("0")
    # Now exhaust the budget
    object.__setattr__(settings, "max_monthly_api_spend_inr", Decimal("0"))
    # Second call: cache has a hit, but budget is exceeded — must return degraded note
    note2 = analyst.analyse(sig)
    assert "Budget cap" in note2.bullish_case
