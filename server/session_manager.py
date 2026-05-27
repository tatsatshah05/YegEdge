from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import structlog

from agent.data.bar_builder import ClosedBar
from agent.data.cache import ParquetCache
from agent.data.universe import UniverseLoader
from agent.data.yfinance_adapter import YFinanceAdapter
from agent.execution.types import Fill
from agent.monitoring.alerter import TelegramAlerter
from agent.monitoring.kill_switch import KillSwitch
from agent.portfolio.tracker import PortfolioTracker
from agent.runner.live_session import LiveSession
from config.settings import AppSettings
from server.events import EventBus

logger = structlog.get_logger()
IST = ZoneInfo("Asia/Kolkata")
ET = ZoneInfo("America/New_York")


async def _fetch_yf_warmup(
    symbols: list[str],
    timeframe: str,
    warmup_bars: int,
    end: datetime,
) -> list[pl.DataFrame]:
    """Parallel-fetch recent bars via yfinance for NSE symbols (adds .NS suffix).

    Used as a fallback when the Parquet cache has no data for *timeframe*.
    """
    adapter = YFinanceAdapter()
    start = end - timedelta(days=5)
    loop = asyncio.get_running_loop()

    async def _one(sym: str) -> pl.DataFrame:
        try:
            df = await loop.run_in_executor(
                None, lambda s=sym: adapter.fetch_historical(s, timeframe, start, end)
            )
            return df.tail(warmup_bars) if len(df) > 0 else pl.DataFrame()
        except Exception as exc:
            logger.warning(
                "session_manager.warmup_yf_symbol_failed",
                symbol=sym,
                error=str(exc),
            )
            return pl.DataFrame()

    results = await asyncio.gather(*[_one(s) for s in symbols])
    return [df for df in results if len(df) > 0]


async def _fetch_alpaca_warmup(
    symbols: list[str],
    timeframe: str,
    warmup_bars: int,
    api_key: str,
    api_secret: str,
    base_url: str,
) -> list[pl.DataFrame]:
    """Parallel-fetch recent bars via Alpaca for NYSE symbols."""
    from agent.data.alpaca_adapter import AlpacaAdapter

    adapter = AlpacaAdapter(api_key, api_secret, base_url)
    end = datetime.now(tz=ET)
    start = end - timedelta(days=5)
    loop = asyncio.get_running_loop()

    async def _one(sym: str) -> pl.DataFrame:
        try:
            df = await loop.run_in_executor(
                None, lambda s=sym: adapter.fetch_historical(s, timeframe, start, end)
            )
            return df.tail(warmup_bars) if len(df) > 0 else pl.DataFrame()
        except Exception as exc:
            logger.warning("session_manager.warmup_alpaca_symbol_failed", symbol=sym, error=str(exc))
            return pl.DataFrame()

    results = await asyncio.gather(*[_one(s) for s in symbols])
    return [df for df in results if len(df) > 0]


async def _fetch_finnhub_warmup(
    symbols: list[str],
    timeframe: str,
    warmup_bars: int,
    api_key: str,
) -> list[pl.DataFrame]:
    """Sequential-fetch recent bars via Finnhub REST for NYSE symbols.

    Sequential (not parallel) to respect the 60 calls/minute free-tier limit.
    """
    from agent.data.finnhub_adapter import FinnhubAdapter

    adapter = FinnhubAdapter(api_key)
    end = datetime.now(tz=ET)
    start = end - timedelta(days=5)
    loop = asyncio.get_running_loop()
    frames: list[pl.DataFrame] = []

    for sym in symbols:
        try:
            df = await loop.run_in_executor(
                None, lambda s=sym: adapter.fetch_historical(s, timeframe, start, end)
            )
            if len(df) > 0:
                frames.append(df.tail(warmup_bars))
            # Small delay to stay under 60 calls/minute rate limit
            await asyncio.sleep(1.1)
        except Exception as exc:
            logger.warning("session_manager.warmup_finnhub_symbol_failed", symbol=sym, error=str(exc))

    return frames


class SessionManager:
    """Manages the LiveSession lifecycle and publishes events to the EventBus.

    Owns creation and teardown of LiveSession, wires bar-close callbacks into
    EventBus publishes, and exposes read-only state for FastAPI routes.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._session: LiveSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._portfolio: PortfolioTracker | None = None
        self._kill_switch: KillSwitch | None = None
        self._last_bars: dict[str, dict[str, Any]] = {}
        self._timeframe: str = "5m"
        self._symbols: list[str] = []
        self._exchange: str = "NSE"
        self._started_at: datetime | None = None

    # ------------------------------------------------------------------
    # Read-only properties (safe to call from FastAPI route handlers)
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True only while the background asyncio task is alive."""
        return self._task is not None and not self._task.done()

    @property
    def portfolio_state(self) -> dict[str, Any] | None:
        """Serialisable snapshot of the current PortfolioState, or None if no session."""
        if self._portfolio is None:
            return None
        s = self._portfolio.state
        return {
            "nav": float(s.nav),
            "cash": float(s.cash),
            "daily_pnl": float(s.daily_pnl),
            "peak_nav": float(s.peak_nav),
            "orders_today": s.orders_today,
            "positions": {
                sym: {
                    "quantity": pos.quantity,
                    "average_price": float(pos.average_price),
                    "product": pos.product,
                }
                for sym, pos in s.positions.items()
            },
        }

    @property
    def last_bars(self) -> dict[str, dict[str, Any]]:
        """Most-recent closed bar per symbol. Returns a shallow copy."""
        return dict(self._last_bars)

    def status(self) -> dict[str, Any]:
        """Summary dict consumed by the /api/status FastAPI route."""
        return {
            "running": self.is_running,
            "timeframe": self._timeframe,
            "symbols_count": len(self._symbols),
            "exchange": self._exchange,
            "started_at": self._started_at.isoformat() if self._started_at else None,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        timeframe: str = "5m",
        warmup_bars: int = 100,
        exchange: str = "NSE",
    ) -> None:
        """Build and start a LiveSession in the background.

        exchange: "NSE" (Upstox/yfinance) or "NYSE" (Alpaca paper trading).
        Raises RuntimeError if a session is already running.
        """
        if self.is_running:
            raise RuntimeError("Session already running")

        settings = AppSettings()
        self._exchange = exchange

        # --- Exchange-specific configuration ----------------------------
        if exchange == "NYSE":
            session_tz = ET
            market_open_hour = 9
            market_open_minute = 30
            session_end = time(16, 0)
            universe_path = Path("config/universe_nyse.yaml")
            now = datetime.now(tz=ET)
        else:
            session_tz = IST
            market_open_hour = 9
            market_open_minute = 15
            session_end = time(15, 30)
            universe_path = Path("config/universe.yaml")
            now = datetime.now(tz=IST)

        today = now.date()
        session_start = datetime(
            today.year, today.month, today.day,
            market_open_hour, market_open_minute,
            tzinfo=session_tz,
        )

        # --- Universe & cache warmup -----------------------------------
        cache = ParquetCache(root=settings.parquet_cache_dir)
        report = cache.coverage_report()
        universe = UniverseLoader(universe_path)
        symbols = universe.symbols()
        self._symbols = symbols
        self._timeframe = timeframe

        warmup_frames: list[pl.DataFrame] = []

        if exchange == "NYSE":
            # NYSE warmup: prefer Finnhub (sequential, rate-limited), fall back to Alpaca
            if settings.finnhub_api_key:
                logger.info(
                    "session_manager.warmup_finnhub_fetching",
                    timeframe=timeframe,
                    symbols=len(symbols),
                )
                warmup_frames = await _fetch_finnhub_warmup(
                    symbols, timeframe, warmup_bars, settings.finnhub_api_key
                )
                logger.info(
                    "session_manager.warmup_finnhub_done",
                    symbols_loaded=len(warmup_frames),
                )
            elif settings.alpaca_api_key:
                logger.info(
                    "session_manager.warmup_alpaca_fetching",
                    timeframe=timeframe,
                    symbols=len(symbols),
                )
                warmup_frames = await _fetch_alpaca_warmup(
                    symbols,
                    timeframe,
                    warmup_bars,
                    settings.alpaca_api_key,
                    settings.alpaca_api_secret,
                    settings.alpaca_base_url,
                )
                logger.info(
                    "session_manager.warmup_alpaca_done",
                    symbols_loaded=len(warmup_frames),
                )
            else:
                logger.warning(
                    "session_manager.warmup_nyse_skipped_no_keys",
                    note="indicators will be cold for first ~30 bars",
                )
        else:
            # NSE: warmup from Parquet cache, fall back to yfinance
            for sym in symbols:
                if sym not in report or timeframe not in report.get(sym, {}):
                    continue
                sym_earliest, _ = report[sym][timeframe]
                all_sym = cache.read(
                    symbol=sym, timeframe=timeframe, start=sym_earliest, end=session_start
                )
                if len(all_sym) == 0:
                    continue
                warmup_frames.append(all_sym.tail(warmup_bars))

            if not warmup_frames:
                logger.info(
                    "session_manager.warmup_cache_empty_fetching_yfinance",
                    timeframe=timeframe,
                    symbols=len(symbols),
                )
                warmup_frames = await _fetch_yf_warmup(
                    symbols, timeframe, warmup_bars, session_start
                )
                logger.info(
                    "session_manager.warmup_yfinance_done",
                    symbols_loaded=len(warmup_frames),
                )

        warmup_df = pl.concat(warmup_frames) if warmup_frames else pl.DataFrame()

        # --- Portfolio & kill-switch -----------------------------------
        self._portfolio = PortfolioTracker(
            initial_nav=Decimal(str(settings.paper_starting_capital)),
            initial_cash=Decimal(str(settings.paper_starting_capital)),
            start_time=session_start,
        )

        ks_path = Path("./data/.kill_switch")
        ks_path.parent.mkdir(parents=True, exist_ok=True)
        if ks_path.exists():
            ks_path.unlink()
        self._kill_switch = KillSwitch(flag_path=ks_path)

        alerter = TelegramAlerter(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )

        # Keep local references for the closure below
        bus = self._bus
        manager = self

        def on_bar_closed(bar: object, fills: list[Fill]) -> None:
            if not isinstance(bar, ClosedBar):
                raise TypeError(f"Expected ClosedBar, got {type(bar)}")

            # Update last-bar cache
            manager._last_bars[bar.symbol] = {
                "symbol": bar.symbol,
                "bar_open": bar.bar_open.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "tick_count": bar.tick_count,
            }

            # Publish bar_closed event
            t = asyncio.create_task(
                bus.publish({
                    "type": "bar_closed",
                    "ts": bar.bar_open.isoformat(),
                    "data": manager._last_bars[bar.symbol],
                })
            )
            manager._background_tasks.add(t)
            t.add_done_callback(manager._background_tasks.discard)

            # Publish fill events
            for fill in fills:
                t = asyncio.create_task(
                    bus.publish({
                        "type": "fill",
                        "ts": bar.bar_open.isoformat(),
                        "data": {
                            "symbol": fill.symbol,
                            "action": str(fill.action),
                            "quantity": fill.quantity,
                            "price": float(fill.fill_price),
                            "order_id": fill.order_id,
                        },
                    })
                )
                manager._background_tasks.add(t)
                t.add_done_callback(manager._background_tasks.discard)

            # Publish portfolio snapshot
            if manager._portfolio is not None:
                state = manager._portfolio.state
                t = asyncio.create_task(
                    bus.publish({
                        "type": "portfolio",
                        "ts": bar.bar_open.isoformat(),
                        "data": {
                            "nav": float(state.nav),
                            "daily_pnl": float(state.daily_pnl),
                            "cash": float(state.cash),
                            "orders_today": state.orders_today,
                        },
                    })
                )
                manager._background_tasks.add(t)
                t.add_done_callback(manager._background_tasks.discard)

        # --- Build session --------------------------------------------
        self._session = LiveSession(
            symbols=symbols,
            timeframe=timeframe,
            portfolio=self._portfolio,
            warmup_df=warmup_df,
            alerter=alerter,
            kill_switch=self._kill_switch,
            on_bar_closed=on_bar_closed,
            session_tz=session_tz,
            session_end=session_end,
            market_open_hour=market_open_hour,
            market_open_minute=market_open_minute,
        )

        # --- Adapter selection ----------------------------------------
        if exchange == "NYSE":
            if settings.finnhub_api_key:
                from agent.data.finnhub_adapter import FinnhubAdapter
                adapter = FinnhubAdapter(api_key=settings.finnhub_api_key)
                logger.info("session_manager.adapter", broker="finnhub", exchange="NYSE")
            elif settings.alpaca_api_key:
                from agent.data.alpaca_adapter import AlpacaAdapter
                adapter = AlpacaAdapter(
                    api_key=settings.alpaca_api_key,
                    api_secret=settings.alpaca_api_secret,
                    base_url=settings.alpaca_base_url,
                )
                logger.info("session_manager.adapter", broker="alpaca", exchange="NYSE")
            else:
                self._session = None
                raise RuntimeError(
                    "No NYSE data source configured. Set FINNHUB_API_KEY in .env."
                )
        elif settings.broker == "yfinance":
            adapter = YFinanceAdapter()
            logger.info("session_manager.adapter", broker="yfinance", exchange="NSE")
        else:
            if not settings.upstox_access_token:
                self._session = None
                raise RuntimeError(
                    "UPSTOX_ACCESS_TOKEN is not set. Run daily login or set BROKER=yfinance."
                )
            from agent.data.upstox_adapter import UpstoxAdapter
            adapter = UpstoxAdapter(access_token=settings.upstox_access_token)
            logger.info("session_manager.adapter", broker="upstox", exchange="NSE")

        # Tick timezone fallback — Alpaca ticks are UTC-aware, Upstox may be naive
        tick_tz = session_tz

        async def _run() -> None:
            def on_tick_df(df: pl.DataFrame) -> None:
                if len(df) == 0:
                    return
                sym = str(df["symbol"][0])
                ltp = float(df["ltp"][0])
                ts = df["timestamp"][0]
                if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=tick_tz)
                if manager._session is not None:
                    manager._session.put_tick(sym, ltp, ts)

            stream_task = asyncio.create_task(
                adapter.stream_live(symbols, callback=on_tick_df)
            )

            def _log_stream_error(t: asyncio.Task[None]) -> None:
                if not t.cancelled() and t.exception() is not None:
                    logger.error(
                        "session_manager.stream_task_failed",
                        error=str(t.exception()),
                        exc_info=t.exception(),
                    )

            stream_task.add_done_callback(_log_stream_error)
            try:
                if manager._session is not None:
                    await manager._session.run()
            finally:
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass

        self._task = asyncio.create_task(_run())
        self._started_at = datetime.now(tz=session_tz)

        await bus.publish({
            "type": "session_started",
            "ts": self._started_at.isoformat(),
            "data": {"timeframe": timeframe, "symbols": symbols, "exchange": exchange},
        })
        logger.info(
            "session_manager.started",
            timeframe=timeframe,
            symbols=len(symbols),
            exchange=exchange,
        )

    async def stop(self) -> None:
        """Gracefully stop the running session (10-second timeout then cancel)."""
        if not self.is_running:
            return

        if self._kill_switch is not None:
            self._kill_switch.activate(reason="Stopped via web terminal")

        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=10.0)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()

        self._session = None
        self._task = None
        self._started_at = None

        await self._bus.publish({
            "type": "session_stopped",
            "ts": datetime.now(tz=IST).isoformat(),
            "data": {},
        })
        logger.info("session_manager.stopped")
