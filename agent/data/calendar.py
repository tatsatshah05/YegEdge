from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import yaml

IST = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN = time(9, 15)
_MARKET_CLOSE = time(15, 30)

_CONFIG_DIR = Path(__file__).parents[2] / "config"
_NSE_HOLIDAYS_FILE = _CONFIG_DIR / "nse_holidays.yaml"


def _load_supplemental_holidays() -> frozenset[date]:
    """Load NSE-specific holidays from config/nse_holidays.yaml.

    These supplement the XBOM base calendar with dates that exchange-calendars
    does not capture (e.g. special closures announced by NSE during the year).
    """
    if not _NSE_HOLIDAYS_FILE.exists():
        return frozenset()
    with _NSE_HOLIDAYS_FILE.open() as fh:
        data: dict[int, list[str]] = yaml.safe_load(fh) or {}
    holidays: set[date] = set()
    for year_dates in data.values():
        if year_dates:
            for ds in year_dates:
                holidays.add(date.fromisoformat(ds))
    return frozenset(holidays)


class NseTradingCalendar:
    """NSE equity-segment trading calendar.

    Uses the XBOM (BSE) exchange-calendars base — BSE and NSE share the same
    national/exchange holiday schedule — supplemented by NSE-specific closures
    declared in ``config/nse_holidays.yaml``.
    """

    def __init__(self) -> None:
        self._cal = xcals.get_calendar("XBOM")
        self._supplemental: frozenset[date] = _load_supplemental_holidays()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_trading_day(self, d: date) -> bool:
        """Return True if *d* is an NSE equity trading session."""
        if d in self._supplemental:
            return False
        return bool(self._cal.is_session(d))

    def trading_sessions(self, start: date, end: date) -> list[date]:
        """Return all NSE trading session dates in [start, end] inclusive."""
        raw: list[date] = [s.date() for s in self._cal.sessions_in_range(start, end)]
        return [d for d in raw if d not in self._supplemental]

    def is_market_open(self, dt: datetime) -> bool:
        """Return True if *dt* falls within normal NSE market hours (9:15–15:30 IST).

        Args:
            dt: An IST-aware (or any-tz-aware) datetime. Naive datetimes are rejected.

        Raises:
            ValueError: If *dt* is tz-naive.
        """
        if dt.tzinfo is None:
            raise ValueError("datetime must be timezone-aware (pass IST-aware datetime)")
        ist_dt = dt.astimezone(IST)
        if not self.is_trading_day(ist_dt.date()):
            return False
        t = ist_dt.time().replace(second=0, microsecond=0)
        return _MARKET_OPEN <= t <= _MARKET_CLOSE

    def next_open(self, dt: datetime) -> datetime:
        """Return the next market open at or after *dt* (returns IST-aware datetime).

        If *dt* is before 9:15 on a trading day, returns 9:15 on that same day.
        Otherwise returns 9:15 on the next trading session.

        Args:
            dt: An IST-aware (or any-tz-aware) datetime.

        Raises:
            ValueError: If *dt* is tz-naive.
            RuntimeError: If no future trading session can be found within the
                calendar's range.
        """
        if dt.tzinfo is None:
            raise ValueError("datetime must be timezone-aware (pass IST-aware datetime)")
        ist_dt = dt.astimezone(IST)
        d = ist_dt.date()
        if self.is_trading_day(d) and ist_dt.time() < _MARKET_OPEN:
            return datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)
        # Scan forward up to one year for the next session
        lookahead_end = date(d.year + 1, d.month, d.day)
        sessions = self.trading_sessions(d, lookahead_end)
        future = [s for s in sessions if s > d]
        if not future:
            raise RuntimeError(
                f"No future NSE trading sessions found after {d} within calendar range"
            )
        nxt = future[0]
        return datetime(nxt.year, nxt.month, nxt.day, 9, 15, tzinfo=IST)
