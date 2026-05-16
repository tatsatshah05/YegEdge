from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.data.types import DataQuality
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")

_TS = datetime(2024, 1, 2, 9, 15, tzinfo=IST)


def _valid_signal(**overrides: object) -> Signal:
    """Return a Signal with valid defaults; override individual fields via kwargs."""
    kwargs: dict[str, object] = {
        "symbol": "HDFCBANK",
        "action": Action.ENTER_LONG,
        "confidence": 0.7,
        "suggested_stop": Decimal("1680.00"),
        "suggested_target": Decimal("1760.00"),
        "invalidation_condition": "Close below EMA21",
        "expected_r": 2.0,
        "time_horizon_hours": 4,
        "regime_fit": 0.9,
        "data_quality": DataQuality.OK,
        "strategy_name": "trend_following_v1",
        "explanation": "EMA21 crossed above EMA50 with ADX=28",
        "timestamp": _TS,
    }
    kwargs.update(overrides)
    return Signal(**kwargs)  # type: ignore[arg-type]


def test_action_values() -> None:
    assert Action.ENTER_LONG == "enter_long"
    assert Action.EXIT_LONG == "exit_long"
    assert Action.HOLD == "hold"


def test_signal_valid_construction() -> None:
    sig = _valid_signal()
    assert sig.symbol == "HDFCBANK"
    assert sig.action == Action.ENTER_LONG
    assert sig.confidence == 0.7
    assert sig.suggested_stop == Decimal("1680.00")
    assert sig.suggested_target == Decimal("1760.00")
    assert sig.expected_r == 2.0
    assert sig.regime_fit == 0.9
    assert sig.data_quality == DataQuality.OK
    assert sig.strategy_name == "trend_following_v1"
    assert sig.timestamp == _TS


def test_signal_is_frozen() -> None:
    sig = _valid_signal()
    with pytest.raises((AttributeError, TypeError)):
        sig.confidence = 0.5  # type: ignore[misc]


def test_signal_confidence_above_1_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _valid_signal(confidence=1.1)


def test_signal_confidence_below_0_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _valid_signal(confidence=-0.1)


def test_signal_regime_fit_above_1_raises() -> None:
    with pytest.raises(ValueError, match="regime_fit"):
        _valid_signal(regime_fit=1.5)


def test_signal_regime_fit_below_0_raises() -> None:
    with pytest.raises(ValueError, match="regime_fit"):
        _valid_signal(regime_fit=-0.1)


def test_signal_stop_gte_target_raises() -> None:
    with pytest.raises(ValueError, match="suggested_stop"):
        _valid_signal(
            suggested_stop=Decimal("1760.00"),
            suggested_target=Decimal("1680.00"),
        )


def test_signal_stop_equal_target_raises() -> None:
    with pytest.raises(ValueError, match="suggested_stop"):
        _valid_signal(
            suggested_stop=Decimal("1700.00"),
            suggested_target=Decimal("1700.00"),
        )


def test_signal_naive_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _valid_signal(timestamp=datetime(2024, 1, 2, 9, 15))  # naive
