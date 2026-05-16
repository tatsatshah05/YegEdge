# tests/data/test_calendar.py
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from agent.data.calendar import NseTradingCalendar

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def cal() -> NseTradingCalendar:
    return NseTradingCalendar()


def test_weekday_is_trading_day(cal: NseTradingCalendar) -> None:
    # 2024-01-02 is a Tuesday — normal trading day
    assert cal.is_trading_day(date(2024, 1, 2)) is True


def test_saturday_is_not_trading_day(cal: NseTradingCalendar) -> None:
    assert cal.is_trading_day(date(2024, 1, 6)) is False


def test_sunday_is_not_trading_day(cal: NseTradingCalendar) -> None:
    assert cal.is_trading_day(date(2024, 1, 7)) is False


def test_nse_holiday_is_not_trading_day(cal: NseTradingCalendar) -> None:
    # Republic Day 2024 = 2024-01-26 (Friday) — NSE closed
    assert cal.is_trading_day(date(2024, 1, 26)) is False


def test_trading_sessions_returns_only_trading_days(cal: NseTradingCalendar) -> None:
    sessions = cal.trading_sessions(date(2024, 1, 1), date(2024, 1, 12))
    # Jan 1 (holiday), Jan 6-7 (weekend), Jan 8 (Monday, open)
    assert date(2024, 1, 1) not in sessions   # New Year
    assert date(2024, 1, 6) not in sessions   # Saturday
    assert date(2024, 1, 2) in sessions       # Tuesday
    assert all(cal.is_trading_day(d) for d in sessions)


def test_market_open_at_930(cal: NseTradingCalendar) -> None:
    # 9:30 IST on a trading day — market is open
    dt = datetime(2024, 1, 2, 9, 30, tzinfo=IST)
    assert cal.is_market_open(dt) is True


def test_market_closed_before_915(cal: NseTradingCalendar) -> None:
    dt = datetime(2024, 1, 2, 9, 10, tzinfo=IST)
    assert cal.is_market_open(dt) is False


def test_market_closed_after_1530(cal: NseTradingCalendar) -> None:
    dt = datetime(2024, 1, 2, 15, 31, tzinfo=IST)
    assert cal.is_market_open(dt) is False


def test_market_closed_on_weekend(cal: NseTradingCalendar) -> None:
    dt = datetime(2024, 1, 6, 10, 0, tzinfo=IST)
    assert cal.is_market_open(dt) is False
