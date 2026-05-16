# tests/data/test_types.py
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.data.types import Bar, DataQuality, Order

IST = ZoneInfo("Asia/Kolkata")


def test_data_quality_values():
    assert DataQuality.OK == "ok"
    assert DataQuality.PARTIAL == "partial"
    assert DataQuality.SUSPECT == "suspect"
    assert DataQuality.MISSING == "missing"


def test_bar_is_frozen():
    bar = Bar(
        symbol="HDFCBANK",
        timeframe="60m",
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
        open=Decimal("1700.00"),
        high=Decimal("1720.00"),
        low=Decimal("1695.00"),
        close=Decimal("1710.00"),
        volume=100000,
        value=Decimal("171000000.00"),
        data_quality=DataQuality.OK,
    )
    with pytest.raises(FrozenInstanceError):
        bar.symbol = "INFY"  # type: ignore[misc]


def test_bar_requires_ist_timestamp():
    # Naive datetimes must be rejected — enforced by __post_init__
    with pytest.raises(ValueError, match="timezone-aware"):
        Bar(
            symbol="HDFCBANK",
            timeframe="60m",
            timestamp=datetime(2024, 1, 2, 9, 15),  # naive — no tzinfo
            open=Decimal("1700.00"),
            high=Decimal("1720.00"),
            low=Decimal("1695.00"),
            close=Decimal("1710.00"),
            volume=100000,
            value=Decimal("171000000.00"),
            data_quality=DataQuality.OK,
        )


def test_order_is_frozen():
    order = Order(
        symbol="TCS",
        action="BUY",
        quantity=10,
        order_type="MARKET",
        price=None,
        trigger_price=None,
        product="MIS",
        client_order_id="test-001",
    )
    with pytest.raises(FrozenInstanceError):
        order.quantity = 20  # type: ignore[misc]


def test_settings_live_trading_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    from config.settings import AppSettings

    settings = AppSettings()
    assert settings.live_trading_enabled is False


def test_bar_rejects_utc_timestamp() -> None:
    import datetime as _dt

    utc_ts = _dt.datetime(2024, 1, 2, 3, 45, tzinfo=_dt.UTC)  # 9:15 IST = 3:45 UTC
    with pytest.raises(ValueError, match="IST"):
        Bar(
            symbol="HDFCBANK",
            timeframe="60m",
            timestamp=utc_ts,
            open=Decimal("1700.00"),
            high=Decimal("1720.00"),
            low=Decimal("1695.00"),
            close=Decimal("1710.00"),
            volume=100000,
            value=Decimal("171000000.00"),
            data_quality=DataQuality.OK,
        )
