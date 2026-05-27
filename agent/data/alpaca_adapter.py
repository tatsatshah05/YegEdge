from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import structlog

from agent.data.broker_adapter import BrokerAdapter
from agent.data.types import Discrepancy, Order, OrderAck, Position

logger = structlog.get_logger()

ET = ZoneInfo("America/New_York")

_TIMEFRAME_MAP: dict[str, object] = {}  # populated lazily on first use


def _build_timeframe(tf: str) -> object:
    """Return an alpaca-py TimeFrame for our timeframe string."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore[import]

    mapping = {
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "60m": TimeFrame.Hour,
        "1d": TimeFrame.Day,
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe for Alpaca: {tf!r}")
    return mapping[tf]


class AlpacaAdapter(BrokerAdapter):
    """Alpaca paper-trading adapter for NYSE stocks.

    Uses alpaca-py SDK (pip install alpaca-py).
    Historical data via StockHistoricalDataClient.
    Live ticks via StockDataStream subscribing to trades.

    Paper trading URL: https://paper-api.alpaca.markets
    All prices are USD; timestamps are UTC (converted to ET for BarBuilder).
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets") -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url
        self._stream: object | None = None

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch OHLCV bars from Alpaca. Returns Polars DataFrame with ET timestamps."""
        from alpaca.data import StockHistoricalDataClient  # type: ignore[import]
        from alpaca.data.requests import StockBarsRequest  # type: ignore[import]

        client = StockHistoricalDataClient(self._api_key, self._api_secret)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_build_timeframe(timeframe),
            start=start,
            end=end,
        )
        bars_response = client.get_stock_bars(request)
        bars = bars_response.get(symbol, [])
        if not bars:
            return pl.DataFrame()

        rows = {
            "symbol": [],
            "timeframe": [],
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "value": [],
            "data_quality": [],
        }
        for bar in bars:
            ts: datetime = bar.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=ZoneInfo("UTC"))
            ts_et = ts.astimezone(ET)
            rows["symbol"].append(symbol)
            rows["timeframe"].append(timeframe)
            rows["timestamp"].append(ts_et)
            rows["open"].append(float(bar.open))
            rows["high"].append(float(bar.high))
            rows["low"].append(float(bar.low))
            rows["close"].append(float(bar.close))
            rows["volume"].append(int(bar.volume))
            rows["value"].append(float(bar.close) * int(bar.volume))
            rows["data_quality"].append("ok")

        return pl.DataFrame(
            {
                "symbol": pl.Series(rows["symbol"], dtype=pl.Utf8),
                "timeframe": pl.Series(rows["timeframe"], dtype=pl.Utf8),
                "timestamp": pl.Series(rows["timestamp"], dtype=pl.Datetime("us", "America/New_York")),
                "open": pl.Series(rows["open"], dtype=pl.Float64),
                "high": pl.Series(rows["high"], dtype=pl.Float64),
                "low": pl.Series(rows["low"], dtype=pl.Float64),
                "close": pl.Series(rows["close"], dtype=pl.Float64),
                "volume": pl.Series(rows["volume"], dtype=pl.Int64),
                "value": pl.Series(rows["value"], dtype=pl.Float64),
                "data_quality": pl.Series(rows["data_quality"], dtype=pl.Utf8),
            }
        )

    # ------------------------------------------------------------------
    # Live tick stream
    # ------------------------------------------------------------------

    async def stream_live(
        self,
        symbols: list[str],
        callback: Callable[[pl.DataFrame], None],
    ) -> None:
        """Subscribe to live trade ticks for all symbols via Alpaca WebSocket.

        Runs until the asyncio task is cancelled. Each trade fires callback with
        a one-row DataFrame: symbol, ltp, timestamp.
        """
        from alpaca.data.live import StockDataStream  # type: ignore[import]

        wss = StockDataStream(self._api_key, self._api_secret, feed="iex")
        self._stream = wss

        async def on_trade(trade: object) -> None:
            try:
                ts: datetime = trade.timestamp  # type: ignore[attr-defined]
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=ZoneInfo("UTC"))
                df = pl.DataFrame(
                    {
                        "symbol": pl.Series([trade.symbol], dtype=pl.Utf8),  # type: ignore[attr-defined]
                        "ltp": pl.Series([float(trade.price)], dtype=pl.Float64),  # type: ignore[attr-defined]
                        "timestamp": pl.Series([ts], dtype=pl.Datetime("us", "UTC")),
                    }
                )
                callback(df)
            except Exception as exc:
                logger.warning("alpaca_adapter.trade_handler_error", error=str(exc))

        wss.subscribe_trades(on_trade, *symbols)
        logger.info("alpaca_adapter.stream_live.connecting", symbols=len(symbols))

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, wss.run)
        except asyncio.CancelledError:
            logger.info("alpaca_adapter.stream_live.cancelled")
            try:
                wss.stop()
            except Exception:
                pass
            raise
        except Exception as exc:
            logger.error("alpaca_adapter.stream_live.error", error=str(exc))
            raise

    # ------------------------------------------------------------------
    # Order management (paper trading — not yet wired to live execution)
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> OrderAck:
        raise NotImplementedError("Alpaca order placement not implemented for paper trading")

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("Alpaca cancel_order not implemented")

    def get_positions(self) -> list[Position]:
        raise NotImplementedError("Alpaca get_positions not implemented")

    def reconcile(self, expected: list[Position]) -> list[Discrepancy]:
        raise NotImplementedError("Alpaca reconcile not implemented")
