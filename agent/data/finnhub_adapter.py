from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import requests
import structlog

from agent.data.broker_adapter import BrokerAdapter
from agent.data.types import Discrepancy, Order, OrderAck, Position

logger = structlog.get_logger()

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

_FINNHUB_RESOLUTION: dict[str, str] = {
    "5m": "5",
    "15m": "15",
    "60m": "60",
    "1d": "D",
}

_FINNHUB_WS_URL = "wss://ws.finnhub.io"
_FINNHUB_REST_URL = "https://finnhub.io/api/v1"

# Free tier: 60 API calls/minute — space historical fetches to stay under limit
_RATE_LIMIT_DELAY = 1.1  # seconds between historical requests


class FinnhubAdapter(BrokerAdapter):
    """BrokerAdapter backed by Finnhub for NYSE real-time tick streaming.

    Free tier: up to 60 simultaneous WebSocket symbol subscriptions,
    60 REST calls/minute for historical data.

    Live ticks come from trade events (real-time, not delayed).
    Historical candles are fetched via REST and returned with ET timestamps.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Historical data (REST)
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        resolution = _FINNHUB_RESOLUTION.get(timeframe)
        if resolution is None:
            raise ValueError(f"Unsupported timeframe for Finnhub: {timeframe!r}")

        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
            "token": self._api_key,
        }
        try:
            resp = requests.get(f"{_FINNHUB_REST_URL}/stock/candle", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("finnhub_adapter.fetch_historical.error", symbol=symbol, error=str(exc))
            return pl.DataFrame()

        if data.get("s") != "ok" or not data.get("t"):
            logger.info("finnhub_adapter.fetch_historical.no_data", symbol=symbol, status=data.get("s"))
            return pl.DataFrame()

        timestamps = [datetime.fromtimestamp(t, tz=UTC).astimezone(ET) for t in data["t"]]
        opens = [float(v) for v in data["o"]]
        highs = [float(v) for v in data["h"]]
        lows = [float(v) for v in data["l"]]
        closes = [float(v) for v in data["c"]]
        volumes = [int(v) for v in data["v"]]
        n = len(timestamps)

        return pl.DataFrame(
            {
                "symbol": pl.Series([symbol] * n, dtype=pl.Utf8),
                "timeframe": pl.Series([timeframe] * n, dtype=pl.Utf8),
                "timestamp": pl.Series(timestamps, dtype=pl.Datetime("us", "America/New_York")),
                "open": pl.Series(opens, dtype=pl.Float64),
                "high": pl.Series(highs, dtype=pl.Float64),
                "low": pl.Series(lows, dtype=pl.Float64),
                "close": pl.Series(closes, dtype=pl.Float64),
                "volume": pl.Series(volumes, dtype=pl.Int64),
                "value": pl.Series(
                    [closes[i] * volumes[i] for i in range(n)], dtype=pl.Float64
                ),
                "data_quality": pl.Series(["ok"] * n, dtype=pl.Utf8),
            }
        ).sort("timestamp")

    # ------------------------------------------------------------------
    # Live tick stream (WebSocket)
    # ------------------------------------------------------------------

    async def stream_live(
        self,
        symbols: list[str],
        callback: Callable[[pl.DataFrame], None],
    ) -> None:
        """Subscribe to real-time trade ticks via Finnhub WebSocket.

        Free tier supports up to 60 simultaneous subscriptions.
        Runs until the asyncio task is cancelled.
        """
        try:
            import websockets  # type: ignore[import]
        except ImportError:
            raise RuntimeError("websockets package required: pip install websockets")

        uri = f"{_FINNHUB_WS_URL}?token={self._api_key}"
        # Respect free-tier 60-symbol limit
        active_symbols = symbols[:60]
        if len(symbols) > 60:
            logger.warning(
                "finnhub_adapter.stream_live.symbol_limit",
                total=len(symbols),
                subscribed=60,
                note="Free tier limited to 60 symbols",
            )

        logger.info("finnhub_adapter.stream_live.connecting", symbols=len(active_symbols))

        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                for sym in active_symbols:
                    await ws.send(json.dumps({"type": "subscribe", "symbol": sym}))

                logger.info("finnhub_adapter.stream_live.subscribed", symbols=len(active_symbols))

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("type") != "trade":
                        continue

                    for trade in msg.get("data", []):
                        try:
                            sym = str(trade["s"])
                            price = float(trade["p"])
                            ts_ms = int(trade["t"])
                            ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
                            df = pl.DataFrame(
                                {
                                    "symbol": pl.Series([sym], dtype=pl.Utf8),
                                    "ltp": pl.Series([price], dtype=pl.Float64),
                                    "timestamp": pl.Series(
                                        [ts], dtype=pl.Datetime("us", "UTC")
                                    ),
                                }
                            )
                            callback(df)
                        except Exception as exc:
                            logger.warning(
                                "finnhub_adapter.trade_parse_error", error=str(exc)
                            )

        except asyncio.CancelledError:
            logger.info("finnhub_adapter.stream_live.cancelled")
            raise
        except Exception as exc:
            logger.error("finnhub_adapter.stream_live.error", error=str(exc))
            raise

    # ------------------------------------------------------------------
    # Order management (not supported — paper execution handled in-code)
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> OrderAck:
        raise NotImplementedError("FinnhubAdapter is data-only. Use PaperExecution.")

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("FinnhubAdapter is data-only.")

    def get_positions(self) -> list[Position]:
        raise NotImplementedError("FinnhubAdapter is data-only.")

    def reconcile(self, expected: list[Position]) -> list[Discrepancy]:
        raise NotImplementedError("FinnhubAdapter is data-only.")
