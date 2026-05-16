from __future__ import annotations

from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.data.upstox_adapter import UPSTOX_TIMEFRAME_MAP, UpstoxAdapter

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Minimal instruments data for mocking _load_instruments
# ---------------------------------------------------------------------------
SAMPLE_INSTRUMENTS = [
    {
        "tradingsymbol": "HDFCBANK",
        "isin": "INE040A01034",
        "exchange": "NSE",
        "instrument_type": "EQ",
    },
    {
        "tradingsymbol": "TCS",
        "isin": "INE467B01029",
        "exchange": "NSE",
        "instrument_type": "EQ",
    },
]

# Fake candle response from Upstox v3 API
FAKE_CANDLE_RESPONSE = {
    "status": "success",
    "data": {
        "candles": [
            ["2024-01-02T09:15:00+05:30", 1700.0, 1720.0, 1695.0, 1710.0, 100000, 171000000.0],
            ["2024-01-02T10:15:00+05:30", 1710.0, 1725.0, 1700.0, 1705.0, 95000, 162225000.0],
        ]
    },
}

EMPTY_CANDLE_RESPONSE = {
    "status": "success",
    "data": {"candles": []},
}


# ---------------------------------------------------------------------------
# Fixture — patches _load_instruments so no real CDN calls happen
# ---------------------------------------------------------------------------
@pytest.fixture
def adapter() -> UpstoxAdapter:
    with patch.object(UpstoxAdapter, "_load_instruments") as mock_load:
        mock_load.return_value = pl.DataFrame(SAMPLE_INSTRUMENTS)
        inst = UpstoxAdapter(access_token="fake-token")
    # Assign the DataFrame directly so all subsequent method calls see it
    inst._instruments = pl.DataFrame(SAMPLE_INSTRUMENTS)
    return inst


# ---------------------------------------------------------------------------
# Helper: build a mock requests.Response for fetch_historical tests
# ---------------------------------------------------------------------------
def _make_mock_response(payload: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_symbol_to_instrument_key(adapter: UpstoxAdapter) -> None:
    """HDFCBANK should map to NSE_EQ|INE040A01034."""
    key = adapter._symbol_to_instrument_key("HDFCBANK")
    assert key == "NSE_EQ|INE040A01034"


def test_unknown_symbol_raises(adapter: UpstoxAdapter) -> None:
    """A symbol not in the instrument master must raise KeyError."""
    with pytest.raises(KeyError):
        adapter._symbol_to_instrument_key("UNKNOWN_XYZ")


def test_instrument_key_to_symbol(adapter: UpstoxAdapter) -> None:
    """Reverse map: instrument key -> tradingsymbol."""
    symbol = adapter._instrument_key_to_symbol("NSE_EQ|INE467B01029")
    assert symbol == "TCS"


def test_fetch_historical_returns_polars_dataframe(adapter: UpstoxAdapter) -> None:
    """fetch_historical returns a pl.DataFrame with expected columns and row count."""
    from datetime import datetime

    mock_resp = _make_mock_response(FAKE_CANDLE_RESPONSE)

    with patch("agent.data.upstox_adapter.requests.get", return_value=mock_resp):
        df = adapter.fetch_historical(
            symbol="HDFCBANK",
            timeframe="15m",
            start=datetime(2024, 1, 2, tzinfo=IST),
            end=datetime(2024, 1, 2, tzinfo=IST),
        )

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2
    assert "symbol" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns
    assert "value" in df.columns
    assert "timestamp" in df.columns
    assert "timeframe" in df.columns


def test_fetch_historical_timestamps_are_ist(adapter: UpstoxAdapter) -> None:
    """All timestamps in the returned DataFrame must carry IST timezone info."""
    from datetime import datetime

    mock_resp = _make_mock_response(FAKE_CANDLE_RESPONSE)

    with patch("agent.data.upstox_adapter.requests.get", return_value=mock_resp):
        df = adapter.fetch_historical(
            symbol="HDFCBANK",
            timeframe="60m",
            start=datetime(2024, 1, 2, tzinfo=IST),
            end=datetime(2024, 1, 2, tzinfo=IST),
        )

    # Polars stores Datetime with timezone; check the dtype string contains IST label
    ts_dtype = df["timestamp"].dtype
    # Polars Datetime dtype carries tz info as string — should be Asia/Kolkata
    assert "Asia/Kolkata" in str(ts_dtype), f"Expected IST timezone in dtype, got {ts_dtype}"

    # Also verify values round-trip correctly by converting to Python datetimes
    for ts in df["timestamp"].to_list():
        assert ts is not None
        # After Polars → Python, the datetime should be timezone-aware
        assert ts.tzinfo is not None


def test_fetch_historical_empty_candles_returns_empty_df(adapter: UpstoxAdapter) -> None:
    """When the API returns an empty candles list, return an empty DataFrame."""
    from datetime import datetime

    mock_resp = _make_mock_response(EMPTY_CANDLE_RESPONSE)

    with patch("agent.data.upstox_adapter.requests.get", return_value=mock_resp):
        df = adapter.fetch_historical(
            symbol="TCS",
            timeframe="1d",
            start=datetime(2024, 1, 1, tzinfo=IST),
            end=datetime(2024, 1, 31, tzinfo=IST),
        )

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 0


def test_upstox_timeframe_map_covers_required_timeframes() -> None:
    """The exported UPSTOX_TIMEFRAME_MAP must include all three required timeframes."""
    assert "15m" in UPSTOX_TIMEFRAME_MAP
    assert "60m" in UPSTOX_TIMEFRAME_MAP
    assert "1d" in UPSTOX_TIMEFRAME_MAP
    # Verify shape of values
    for tf, (unit, interval) in UPSTOX_TIMEFRAME_MAP.items():
        assert isinstance(unit, str), f"unit for {tf} must be str"
        assert isinstance(interval, int), f"interval for {tf} must be int"
