from __future__ import annotations

import asyncio
import gzip
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Final
from urllib.parse import quote
from zoneinfo import ZoneInfo

import polars as pl
import requests
import structlog
import upstox_client
from upstox_client.feeder.market_data_streamer_v3 import MarketDataStreamerV3

from agent.data.broker_adapter import BrokerAdapter
from agent.data.types import Discrepancy, Order, OrderAck, Position

logger = structlog.get_logger()

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Public constant — maps our internal timeframe strings to Upstox v3 API params
# ---------------------------------------------------------------------------
UPSTOX_TIMEFRAME_MAP: Final[dict[str, tuple[str, int]]] = {
    "5m": ("minutes", 5),
    "15m": ("minutes", 15),
    "60m": ("hours", 1),
    "1d": ("days", 1),
}

_INSTRUMENTS_CDN_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

_OHLCV_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "timestamp": pl.Datetime("us", "Asia/Kolkata"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "value": pl.Float64,
}

_REQUIRED_INSTRUMENT_COLUMNS: Final[frozenset[str]] = frozenset(
    {"trading_symbol", "segment", "instrument_key"}
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
        CDN, decompresses it, and returns the result.  Raises `RuntimeError`
        on any I/O or parse failure with a structured log entry.
        """
        df: pl.DataFrame
        if cache_path is not None:
            p = Path(cache_path)
            if p.exists():
                logger.info("upstox_adapter.instruments.cache_hit", path=str(p))
                try:
                    with p.open("r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    df = pl.DataFrame(data)
                except Exception as exc:
                    logger.error(
                        "upstox_adapter.instruments.cache_read_error",
                        path=str(p),
                        error=str(exc),
                    )
                    raise RuntimeError(f"Failed to read instrument cache from {p}: {exc}") from exc
                self._validate_instrument_columns(df)
                return df
            logger.info(
                "upstox_adapter.instruments.cache_miss_falling_back_to_cdn",
                path=str(p),
            )

        logger.info("upstox_adapter.instruments.downloading", url=_INSTRUMENTS_CDN_URL)
        try:
            resp = requests.get(_INSTRUMENTS_CDN_URL, timeout=30)
            resp.raise_for_status()
            data = json.loads(gzip.decompress(resp.content))
        except Exception as exc:
            logger.error(
                "upstox_adapter.instruments.download_error",
                url=_INSTRUMENTS_CDN_URL,
                error=str(exc),
            )
            raise RuntimeError(
                f"Failed to download instrument master from Upstox CDN: {exc}"
            ) from exc
        df = pl.DataFrame(data)
        logger.info("upstox_adapter.instruments.loaded", count=len(df))
        self._validate_instrument_columns(df)
        return df

    @staticmethod
    def _validate_instrument_columns(df: pl.DataFrame) -> None:
        missing = _REQUIRED_INSTRUMENT_COLUMNS - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Instrument master is missing required columns: {missing}. "
                "The Upstox CDN data format may have changed — refresh the cache."
            )

    def _symbol_to_instrument_key(self, symbol: str) -> str:
        """Return the Upstox instrument key for a given NSE trading symbol.

        Format: ``NSE_EQ|{ISIN}``

        Raises
        ------
        KeyError
            If *symbol* is not found in the NSE EQ instrument master.
        """
        matches = self._instruments.filter(
            (pl.col("trading_symbol") == symbol) & (pl.col("segment") == "NSE_EQ")
        )
        if len(matches) == 0:
            raise KeyError(
                f"Symbol '{symbol}' not found in NSE instrument master. "
                "Refresh the instruments cache or verify the symbol name."
            )
        return str(matches["instrument_key"][0])

    def _instrument_key_to_symbol(self, instrument_key: str) -> str:
        """Return the NSE trading symbol for a given Upstox instrument key.

        Format of *instrument_key*: ``NSE_EQ|{ISIN}``

        Raises
        ------
        KeyError
            If the ISIN derived from *instrument_key* is not found.
        """
        matches = self._instruments.filter(pl.col("instrument_key") == instrument_key)
        if len(matches) == 0:
            raise KeyError(
                f"Instrument key '{instrument_key}' not found in instrument master."
            )
        return str(matches["trading_symbol"][0])

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
        encoded_key = quote(instrument_key, safe="")  # encode the pipe character

        from_date = start.strftime("%Y-%m-%d")
        to_date = end.strftime("%Y-%m-%d")

        url = (
            f"https://api.upstox.com/v3/historical-candle"
            f"/{encoded_key}/{unit}/{interval}/{to_date}/{from_date}"
        )

        logger.info(
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
            logger.info(
                "upstox_adapter.fetch_historical.empty",
                symbol=symbol,
                timeframe=timeframe,
            )
            return pl.DataFrame(schema=_OHLCV_SCHEMA)

        # Each candle: [ts_str, open, high, low, close, volume, value_or_oi]
        timestamps: list[datetime] = []
        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        volumes: list[int] = []
        values: list[float] = []

        for idx, candle in enumerate(candles):
            if len(candle) < 7:
                logger.warning(
                    "upstox_adapter.fetch_historical.malformed_candle",
                    symbol=symbol,
                    index=idx,
                    candle=candle,
                )
                continue
            ts_str = candle[0]
            o, h, lo, c, vol, val = candle[1], candle[2], candle[3], candle[4], candle[5], candle[6]
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                logger.error(
                    "upstox_adapter.fetch_historical.naive_timestamp",
                    symbol=symbol,
                    index=idx,
                    ts_str=ts_str,
                )
                raise ValueError(
                    f"Upstox returned a naive (no timezone) timestamp at candle {idx}: {ts_str!r}. "
                    "Expected an ISO 8601 string with UTC offset."
                )
            timestamps.append(ts.astimezone(IST))
            opens.append(float(o))
            highs.append(float(h))
            lows.append(float(lo))
            closes.append(float(c))
            volumes.append(round(float(vol)))
            values.append(float(val))

        if not timestamps:
            return pl.DataFrame(schema=_OHLCV_SCHEMA)

        df = pl.DataFrame(
            {
                "symbol": pl.Series([symbol] * len(timestamps), dtype=pl.Utf8),
                "timeframe": pl.Series([timeframe] * len(timestamps), dtype=pl.Utf8),
                "timestamp": pl.Series(timestamps, dtype=pl.Datetime("us", "Asia/Kolkata")),
                "open": pl.Series(opens, dtype=pl.Float64),
                "high": pl.Series(highs, dtype=pl.Float64),
                "low": pl.Series(lows, dtype=pl.Float64),
                "close": pl.Series(closes, dtype=pl.Float64),
                "volume": pl.Series(volumes, dtype=pl.Int64),
                "value": pl.Series(values, dtype=pl.Float64),
            }
        ).sort("timestamp")

        logger.info(
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
        """WebSocket live tick stream using Upstox MarketDataStreamerV3.

        Subscribes to LTPC (Last Trade Price + Change) ticks for the given
        symbols and invokes *callback* with a one-row ``pl.DataFrame`` for each
        tick received.  The coroutine stays alive until cancelled by the caller.

        Parameters
        ----------
        symbols:
            NSE trading symbols to subscribe to, e.g. ``["HDFCBANK", "TCS"]``.
        callback:
            Callable invoked for every valid tick.  Receives a one-row DataFrame
            with columns: symbol (Utf8), ltp (Float64), timestamp (Datetime[us, Asia/Kolkata]).
        """
        instrument_keys = [self._symbol_to_instrument_key(s) for s in symbols]

        config = upstox_client.Configuration()
        config.access_token = self._access_token
        api_client = upstox_client.ApiClient(configuration=config)

        streamer = MarketDataStreamerV3(
            api_client=api_client,
            instrumentKeys=instrument_keys,
            mode="ltpc",
        )

        streamer.on("message", lambda msg: self._handle_tick(msg, callback=callback))

        logger.info(
            "upstox_adapter.stream_live.connecting",
            symbols=symbols,
            instrument_keys=instrument_keys,
        )

        # streamer.connect() is synchronous and blocks until disconnected.
        # Wrap in run_in_executor so it doesn't block the asyncio event loop.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, streamer.connect)

        logger.info("upstox_adapter.stream_live.disconnected", symbols=symbols)

    def _handle_tick(
        self,
        msg: dict,  # type: ignore[type-arg]
        *,
        callback: Callable[[pl.DataFrame], None],
    ) -> None:
        """Parse a raw WebSocket protobuf-decoded message and fire *callback*.

        Each entry in ``msg["feeds"]`` corresponds to one instrument.  A
        one-row DataFrame is emitted per valid LTPC feed entry.

        Errors in individual feed entries are caught and logged so a single
        bad tick never crashes the stream.
        """
        for instrument_key, feed_data in msg.get("feeds", {}).items():
            try:
                try:
                    symbol = self._instrument_key_to_symbol(instrument_key)
                except KeyError:
                    logger.debug(
                        "upstox_adapter.handle_tick.unknown_instrument_key",
                        instrument_key=instrument_key,
                    )
                    continue

                ltp_data = feed_data.get("ltpc")
                if ltp_data is None:
                    logger.debug(
                        "upstox_adapter.handle_tick.no_ltpc",
                        instrument_key=instrument_key,
                    )
                    continue

                if "ltt" not in ltp_data or "ltp" not in ltp_data:
                    logger.debug(
                        "upstox_adapter.handle_tick.incomplete_ltpc",
                        symbol=symbol,
                        keys=list(ltp_data.keys()),
                    )
                    continue

                # ltt is epoch ms as a string (e.g. "1704168600000")
                ltp = float(ltp_data["ltp"])
                ts = datetime.fromtimestamp(int(ltp_data["ltt"]) / 1000, tz=IST)

                row_df = pl.DataFrame(
                    {
                        "symbol": [symbol],
                        "ltp": pl.Series([ltp], dtype=pl.Float64),
                        "timestamp": pl.Series([ts], dtype=pl.Datetime("us", "Asia/Kolkata")),
                    }
                )
                callback(row_df)

            except Exception as exc:
                logger.error(
                    "upstox_adapter.handle_tick.error",
                    instrument_key=instrument_key,
                    error=str(exc),
                )

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
