# tests/ai/test_prompt.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from agent.ai.prompt import build_prompt
from agent.data.types import DataQuality
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")


def _make_signal(
    symbol: str = "HDFCBANK",
    action: Action = Action.ENTER_LONG,
    confidence: float = 0.75,
    regime_fit: float = 0.8,
    expected_r: float = 2.0,
) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        suggested_stop=Decimal("1680.00"),
        suggested_target=Decimal("1760.00"),
        invalidation_condition="Close below EMA21",
        expected_r=expected_r,
        time_horizon_hours=4,
        regime_fit=regime_fit,
        data_quality=DataQuality.OK,
        strategy_name="trend_following_v1",
        explanation="EMA21 crossed above EMA50 (ADX=28.5, vol_ratio=1.20)",
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )


def test_prompt_contains_symbol() -> None:
    prompt = build_prompt(_make_signal(symbol="TCS"))
    assert "TCS" in prompt


def test_prompt_contains_action() -> None:
    prompt = build_prompt(_make_signal(action=Action.ENTER_LONG))
    assert "ENTER_LONG" in prompt or "enter_long" in prompt or "long" in prompt.lower()


def test_prompt_contains_stop_and_target() -> None:
    prompt = build_prompt(_make_signal())
    assert "1680" in prompt
    assert "1760" in prompt


def test_prompt_contains_expected_r() -> None:
    prompt = build_prompt(_make_signal(expected_r=2.5))
    assert "2.5" in prompt


def test_prompt_contains_explanation() -> None:
    prompt = build_prompt(_make_signal())
    assert "EMA21" in prompt


def test_prompt_with_portfolio_summary_includes_it() -> None:
    summary = "Holding 2 positions: SBIN (10 shares), TCS (5 shares)"
    prompt = build_prompt(_make_signal(), portfolio_summary=summary)
    assert "SBIN" in prompt
