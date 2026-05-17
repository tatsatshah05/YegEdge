from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.data.types import DataQuality
from agent.decision.types import Decision, DecisionStatus, ResearchNote
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")


def _make_signal(
    symbol: str = "HDFCBANK",
    action: Action = Action.ENTER_LONG,
    confidence: float = 0.7,
) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        suggested_stop=Decimal("1680.00"),
        suggested_target=Decimal("1760.00"),
        invalidation_condition="Close below EMA21",
        expected_r=2.0,
        time_horizon_hours=4,
        regime_fit=0.8,
        data_quality=DataQuality.OK,
        strategy_name="trend_following_v1",
        explanation="EMA21 crossed above EMA50",
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )


def test_decision_status_values() -> None:
    assert DecisionStatus.PENDING == "pending"
    assert DecisionStatus.WAIT_FOR_CONFIRMATION == "wait_for_confirmation"
    assert DecisionStatus.SKIPPED == "skipped"


def test_research_note_construction() -> None:
    note = ResearchNote(
        signal_id="HDFCBANK:enter_long:2024-01-02T09:15:00+05:30",
        bullish_case="Strong trend, high ADX.",
        bearish_case="Overbought in short term.",
        dominant_risk="Reversal if FII selling picks up.",
        regime_fit_assessment="Trending regime suits strategy.",
        confidence_qualitative="HIGH",
        veto=False,
        veto_reason=None,
        model_used="claude-haiku-4-5-20251001",
        tokens_used=150,
        cached=False,
    )
    assert note.veto is False
    assert note.veto_reason is None


def test_research_note_is_frozen() -> None:
    note = ResearchNote(
        signal_id="id",
        bullish_case="b",
        bearish_case="br",
        dominant_risk="r",
        regime_fit_assessment="a",
        confidence_qualitative="LOW",
        veto=False,
        veto_reason=None,
        model_used="claude-haiku-4-5-20251001",
        tokens_used=100,
        cached=False,
    )
    with pytest.raises(AttributeError):
        note.veto = True  # type: ignore[misc]


def test_decision_pending_construction() -> None:
    sig = _make_signal()
    d = Decision(
        signal=sig,
        status=DecisionStatus.PENDING,
        signal_id="HDFCBANK:enter_long:2024-01-02T09:15:00+05:30",
        merged_from=("trend_following_v1",),
        research_note=None,
        skip_reason="",
        timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    assert d.status == DecisionStatus.PENDING
    assert d.skip_reason == ""


def test_decision_is_frozen() -> None:
    sig = _make_signal()
    d = Decision(
        signal=sig,
        status=DecisionStatus.PENDING,
        signal_id="id",
        merged_from=("trend_following_v1",),
        research_note=None,
        skip_reason="",
        timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    with pytest.raises(AttributeError):
        d.status = DecisionStatus.SKIPPED  # type: ignore[misc]


def test_decision_skipped_has_reason() -> None:
    sig = _make_signal()
    d = Decision(
        signal=sig,
        status=DecisionStatus.SKIPPED,
        signal_id="id",
        merged_from=("trend_following_v1",),
        research_note=None,
        skip_reason="Already holding position in HDFCBANK",
        timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    assert "HDFCBANK" in d.skip_reason


def test_decision_with_veto_research_note() -> None:
    sig = _make_signal()
    note = ResearchNote(
        signal_id="id",
        bullish_case="b",
        bearish_case="br",
        dominant_risk="r",
        regime_fit_assessment="a",
        confidence_qualitative="MEDIUM",
        veto=True,
        veto_reason="High risk given current macro environment",
        model_used="claude-haiku-4-5-20251001",
        tokens_used=200,
        cached=True,
    )
    d = Decision(
        signal=sig,
        status=DecisionStatus.WAIT_FOR_CONFIRMATION,
        signal_id="id",
        merged_from=("trend_following_v1",),
        research_note=note,
        skip_reason="AI veto: High risk given current macro environment",
        timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    assert d.research_note is not None
    assert d.research_note.veto is True
