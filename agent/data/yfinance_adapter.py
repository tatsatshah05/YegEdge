from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl
import structlog
import yfinance as yf

from agent.data.broker_adapter import BrokerAdapter
from agent.data.types import Discrepancy, Order, OrderAck, Position

logger = structlog.get_logger()
IST = ZoneInfo("Asia/Kolkata")

YFINANCE_INTERVAL_MAP: Final[dict[str, str]] = {
    "5m": "5m",
    "15m": "15m",
    "60m": "1h",
    "1d": "1d",
}

# Data retention limits in yfinance (approximate)
# 5m → 60 days, 15m → 60 days, 60m → 730 days, 1d → full history
YFINANCE_HISTORY_LIMIT_DAYS: Final[dict[str, int]] = {
    "5m": 59,
    "15m": 59,
    "60m": 729,
    "1d": 3650,
}

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


def _to_yf_symbol(symbol: str) -> str:
    """Add .NS suffix for NSE equity symbols."""
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"


def _to_ist(ts_raw: object) -> datetime:
    """Convert a pandas Timestamp or datetime to IST-aware datetime."""
    if hasattr(ts_raw, "to_pydatetime"):
        ts_py: datetime = ts_raw.to_pydatetime()  # type: ignore[union-attr]
    else:
        ts_py = ts_raw  # type: ignore[assignment]
    if ts_py.tzinfo is not None:
        return ts_py.astimezone(IST)
    return ts_py.replace(tzinfo=IST)


class YFinanceAdapter(BrokerAdapter):
    """BrokerAdapter backed by Yahoo Finance (yfinance). No API key required.

    Provides ~15-minute delayed NSE data. Order placement raises
    NotImplementedError — pair with PaperExecution for paper trading.

    Data availability limits (yfinance restriction):
      15m → last 60 days only
      60m → last 730 days
      1d  → full history
    """

    def __init__(self, poll_interval_seconds: int = 60) -> None:
        self._poll_interval = poll_interval_seconds

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
        if timeframe not in YFINANCE_INTERVAL_MAP:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Valid options: {list(YFINANCE_INTERVAL_MAP)}"
            )

        limit_days = YFINANCE_HISTORY_LIMIT_DAYS[timeframe]
        earliest_allowed = datetime.now(tz=IST) - timedelta(days=limit_days)
        if start < earliest_allowed:
            logger.warning(
                "yfinance_adapter.fetch_historical.clamping_start",
                symbol=symbol,
                timeframe=timeframe,
                requested_start=str(start.date()),
                clamped_start=str(earliest_allowed.date()),
            )
            start = earliest_allowed

        yf_interval = YFINANCE_INTERVAL_MAP[timeframe]
        yf_symbol = _to_yf_symbol(symbol)

        logger.info(
            "yfinance_adapter.fetch_historical.request",
            symbol=symbol,
            timeframe=timeframe,
            start=str(start.date()),
            end=str(end.date()),
        )

        ticker = yf.Ticker(yf_symbol)
        df_raw = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=yf_interval,
            auto_adjust=True,
            prepost=False,
        )

        if df_raw.empty:
            logger.info(
                "yfinance_adapter.fetch_historical.empty",
                symbol=symbol,
                timeframe=timeframe,
            )
            return pl.DataFrame(schema=_OHLCV_SCHEMA)

        timestamps: list[datetime] = []
        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        volumes: list[int] = []

        for ts_raw, row in df_raw.iterrows():
            timestamps.append(_to_ist(ts_raw))
            opens.append(float(row["Open"]))
            highs.append(float(row["High"]))
            lows.append(float(row["Low"]))
            closes.append(float(row["Close"]))
            volumes.append(int(row["Volume"]))

        if not timestamps:
            return pl.DataFrame(schema=_OHLCV_SCHEMA)

        n = len(timestamps)
        # Approximate turnover as close × volume (yfinance doesn't provide INR turnover)
        values = [closes[i] * float(volumes[i]) for i in range(n)]

        df = pl.DataFrame(
            {
                "symbol": pl.Series([symbol] * n, dtype=pl.Utf8),
                "timeframe": pl.Series([timeframe] * n, dtype=pl.Utf8),
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
            "yfinance_adapter.fetch_historical.done",
            symbol=symbol,
            timeframe=timeframe,
            rows=len(df),
        )
        return df

    # ------------------------------------------------------------------
    # BrokerAdapter: stream_live (polling)
    # ------------------------------------------------------------------

    async def stream_live(
        self,
        symbols: list[str],
        callback: Callable[[pl.DataFrame], None],
    ) -> None:
        """Poll yfinance every poll_interval_seconds and emit new 1m closes as ticks.

        Tracks the last emitted timestamp per symbol to avoid duplicates.
        Data is ~15 minutes delayed for NSE via Yahoo Finance.
        """
        logger.info(
            "yfinance_adapter.stream_live.start",
            symbols=symbols,
            poll_interval=self._poll_interval,
        )

        last_ts: dict[str, datetime | None] = {s: None for s in symbols}

        while True:
            try:
                await asyncio.sleep(self._poll_interval)

                for symbol in symbols:
                    try:
                        yf_symbol = _to_yf_symbol(symbol)
                        ticker = yf.Ticker(yf_symbol)
                        df_raw = ticker.history(period="1d", interval="1m", prepost=False)

                        if df_raw.empty:
                            continue

                        for ts_raw, row in df_raw.iterrows():
                            ts_ist = _to_ist(ts_raw)
                            prev = last_ts[symbol]
                            if prev is not None and ts_ist <= prev:
                                continue

                            ltp = float(row["Close"])
                            tick_df = pl.DataFrame(
                                {
                                    "symbol": [symbol],
                                    "ltp": pl.Series([ltp], dtype=pl.Float64),
                                    "timestamp": pl.Series(
                                        [ts_ist], dtype=pl.Datetime("us", "Asia/Kolkata")
                                    ),
                                }
                            )
                            callback(tick_df)
                            last_ts[symbol] = ts_ist

                        logger.debug(
                            "yfinance_adapter.stream_live.polled",
                            symbol=symbol,
                            last_ts=str(last_ts[symbol]),
                        )

                    except Exception as exc:
                        logger.error(
                            "yfinance_adapter.stream_live.symbol_error",
                            symbol=symbol,
                            error=str(exc),
                        )

            except asyncio.CancelledError:
                logger.info("yfinance_adapter.stream_live.cancelled")
                raise
            except Exception as exc:
                logger.error("yfinance_adapter.stream_live.error", error=str(exc))

    # ------------------------------------------------------------------
    # BrokerAdapter: order operations (not supported — use PaperExecution)
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> OrderAck:
        raise NotImplementedError("YFinanceAdapter is read-only. Use PaperExecution.")

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("YFinanceAdapter is read-only.")

    def get_positions(self) -> list[Position]:
        raise NotImplementedError("YFinanceAdapter is read-only.")

    def reconcile(self, expected: list[Position]) -> list[Discrepancy]:
        raise NotImplementedError("YFinanceAdapter is read-only.")
