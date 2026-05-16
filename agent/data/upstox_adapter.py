from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import requests
import structlog

from agent.data.broker_adapter import BrokerAdapter
from agent.data.types import Discrepancy, Order, OrderAck, Position

log = structlog.get_logger()

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Public constant — maps our internal timeframe strings to Upstox v3 API params
# ---------------------------------------------------------------------------
UPSTOX_TIMEFRAME_MAP: dict[str, tuple[str, int]] = {
    "15m": ("minutes", 15),
    "60m": ("hours", 1),
    "1d": ("days", 1),
}

_INSTRUMENTS_CDN_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
)

_HISTORICAL_URL_TEMPLATE = (
    "https://api.upstox.com/v3/historical-candle"
    "/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"
)


class UpstoxAdapter(BrokerAdapter):
    """Concrete BrokerAdapter for the Upstox broker (REST + WebSocket).

    Uses the Upstox v3 REST API for historical data directly via ``requests``
    (bypassing the upstox_client SDK to avoid its auth complexity).

    WebSocket live streaming is implemented in Task 8; ``stream_live`` raises
    ``NotImplementedError`` here.
    """

    def __init__(
        self,
        access_token: str,
        instruments_cache_path: str | None = None,
    ) -> None:
        self._access_token = access_token
        self._instruments: pl.DataFrame = self._load_instruments(instruments_cache_path)

    # ------------------------------------------------------------------
    # Instrument master helpers
    # ------------------------------------------------------------------

    def _load_instruments(self, cache_path: str | None = None) -> pl.DataFrame:
        """Load the NSE instrument master into a Polars DataFrame.

        If *cache_path* is provided and the file exists, reads it from disk
        (JSON array).  Otherwise downloads the gzipped JSON from the Upstox
        CDN, decompresses it, and returns the result.
        """
        if cache_path is not None:
            p = Path(cache_path)
            if p.exists():
                log.info(
                    "upstox_adapter.instruments.cache_hit",
                    path=str(p),
                )
                with p.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return pl.DataFrame(data)

        log.info(
            "upstox_adapter.instruments.downloading",
            url=_INSTRUMENTS_CDN_URL,
        )
        resp = requests.get(_INSTRUMENTS_CDN_URL, timeout=30)
        resp.raise_for_status()
        data = json.loads(gzip.decompress(resp.content))
        log.info(
            "upstox_adapter.instruments.loaded",
            count=len(data),
        )
        return pl.DataFrame(data)

    def _symbol_to_instrument_key(self, symbol: str) -> str:
        """Return the Upstox instrument key for a given NSE trading symbol.

        Format: ``NSE_EQ|{ISIN}``

        Raises
        ------
        KeyError
            If *symbol* is not found in the NSE EQ instrument master.
        """
        matches = self._instruments.filter(
            (pl.col("tradingsymbol") == symbol) & (pl.col("exchange") == "NSE")
        )
        if len(matches) == 0:
            raise KeyError(
                f"Symbol '{symbol}' not found in NSE instrument master. "
                "Refresh the instruments cache or verify the symbol name."
            )
        isin: str = matches["isin"][0]
        return f"NSE_EQ|{isin}"

    def _instrument_key_to_symbol(self, instrument_key: str) -> str:
        """Return the NSE trading symbol for a given Upstox instrument key.

        Format of *instrument_key*: ``NSE_EQ|{ISIN}``

        Raises
        ------
        KeyError
            If the ISIN derived from *instrument_key* is not found.
        """
        _, isin = instrument_key.split("|", maxsplit=1)
        matches = self._instruments.filter(pl.col("isin") == isin)
        if len(matches) == 0:
            raise KeyError(
                f"ISIN '{isin}' (from key '{instrument_key}') not found in instrument master."
            )
        return str(matches["tradingsymbol"][0])

    # ------------------------------------------------------------------
    # BrokerAdapter: fetch_historical
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch OHLCV bars from the Upstox v3 historical-candle endpoint.

        Returns a Polars DataFrame with columns:
          symbol (Utf8), timeframe (Utf8), timestamp (Datetime[us, Asia/Kolkata]),
          open (Float64), high (Float64), low (Float64), close (Float64),
          volume (Int64), value (Float64).

        Returns an empty ``pl.DataFrame()`` when the API returns no candles.

        Parameters
        ----------
        symbol:
            NSE trading symbol, e.g. ``"HDFCBANK"``.
        timeframe:
            One of ``"15m"``, ``"60m"``, ``"1d"``.
        start, end:
            Closed date interval.  Must be IST-aware datetimes.
        """
        if timeframe not in UPSTOX_TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Valid options: {list(UPSTOX_TIMEFRAME_MAP)}"
            )

        unit, interval = UPSTOX_TIMEFRAME_MAP[timeframe]
        instrument_key = self._symbol_to_instrument_key(symbol)

        from_date = start.strftime("%Y-%m-%d")
        to_date = end.strftime("%Y-%m-%d")

        url = _HISTORICAL_URL_TEMPLATE.format(
            instrument_key=instrument_key,
            unit=unit,
            interval=interval,
            to_date=to_date,
            from_date=from_date,
        )

        log.info(
            "upstox_adapter.fetch_historical.request",
            symbol=symbol,
            timeframe=timeframe,
            from_date=from_date,
            to_date=to_date,
        )

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        payload = resp.json()
        candles: list[list] = payload.get("data", {}).get("candles", [])

        if not candles:
            log.info(
                "upstox_adapter.fetch_historical.empty",
                symbol=symbol,
                timeframe=timeframe,
            )
            return pl.DataFrame()

        # Each candle: [ts_str, open, high, low, close, volume, value_or_oi]
        timestamps: list[datetime] = []
        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        volumes: list[int] = []
        values: list[float] = []

        for candle in candles:
            ts_str, o, h, l, c, vol, val = candle
            ts = datetime.fromisoformat(ts_str).astimezone(IST)
            timestamps.append(ts)
            opens.append(float(o))
            highs.append(float(h))
            lows.append(float(l))
            closes.append(float(c))
            volumes.append(int(vol))
            values.append(float(val))

        df = pl.DataFrame(
            {
                "symbol": [symbol] * len(candles),
                "timeframe": [timeframe] * len(candles),
                "timestamp": pl.Series(timestamps).dt.convert_time_zone("Asia/Kolkata"),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "value": values,
            }
        ).sort("timestamp")

        log.info(
            "upstox_adapter.fetch_historical.done",
            symbol=symbol,
            timeframe=timeframe,
            rows=len(df),
        )
        return df

    # ------------------------------------------------------------------
    # BrokerAdapter: stubs — implemented in later phases
    # ------------------------------------------------------------------

    async def stream_live(
        self,
        symbols: list[str],
        callback: Callable[[pl.DataFrame], None],
    ) -> None:
        """WebSocket live tick stream — implemented in Task 8."""
        raise NotImplementedError("Implemented in Task 8")

    def place_order(self, order: Order) -> OrderAck:
        """Order placement — implemented in Phase 5."""
        raise NotImplementedError("Implemented in Phase 5")

    def cancel_order(self, order_id: str) -> None:
        """Order cancellation — implemented in Phase 5."""
        raise NotImplementedError("Implemented in Phase 5")

    def get_positions(self) -> list[Position]:
        """Position retrieval — implemented in Phase 5."""
        raise NotImplementedError("Implemented in Phase 5")

    def reconcile(self, expected: list[Position]) -> list[Discrepancy]:
        """Position reconciliation — implemented in Phase 5."""
        raise NotImplementedError("Implemented in Phase 5")
