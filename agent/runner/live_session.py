from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Callable
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import structlog

from agent.data.bar_builder import BarBuilder, ClosedBar
from agent.execution.paper import PaperExecution
from agent.features.pipeline import FeaturePipeline
from agent.journal.store import JournalStore
from agent.monitoring.alerter import TelegramAlerter
from agent.monitoring.heartbeat import Heartbeat
from agent.monitoring.kill_switch import KillSwitch
from agent.portfolio.tracker import PortfolioTracker
from agent.risk.manager import RiskManager
from agent.risk.rules import load_risk_rules
from agent.runner.daily_loop import DailyLoop
from agent.strategies.trend_following import TrendFollowingStrategy

logger = structlog.get_logger()

IST = ZoneInfo("Asia/Kolkata")
_SESSION_END = time(15, 30)


class LiveSession:
    """Real-time paper trading session.

    Receives LTP ticks via put_tick(), aggregates them into OHLCV bars with
    BarBuilder (one per symbol), enriches each closed bar with FeaturePipeline,
    and runs the full DailyLoop pipeline.

    Thread safety: put_tick() is safe to call from any thread (e.g., WebSocket
    callback thread). Ticks are queued and processed serially by the async loop.
    """

    def __init__(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        portfolio: PortfolioTracker,
        warmup_df: pl.DataFrame,
        kill_switch: KillSwitch | None = None,
        alerter: TelegramAlerter | None = None,
        heartbeat: Heartbeat | None = None,
        on_bar_closed: Callable[[object, list[object]], None] | None = None,
    ) -> None:
        self._symbols = symbols
        self._timeframe = timeframe
        self._portfolio = portfolio
        self._warmup_df = warmup_df
        self._pipeline = FeaturePipeline()
        self._builders: dict[str, BarBuilder] = {s: BarBuilder(s, timeframe) for s in symbols}
        # Accumulated live bars per symbol (grows as bars close during session)
        self._live_bars: dict[str, pl.DataFrame] = {s: pl.DataFrame() for s in symbols}
        self._queue: asyncio.Queue[tuple[str, float, datetime]] = asyncio.Queue()
        self._loop = self._make_daily_loop(portfolio, alerter, heartbeat, kill_switch)
        self._kill_switch = kill_switch
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._on_bar_closed_cb = on_bar_closed

    def put_tick(self, symbol: str, ltp: float, ts: datetime) -> None:
        """Thread-safe. Called from WebSocket callback thread."""
        if self._event_loop is None:
            raise RuntimeError("LiveSession.run() must be awaited before put_tick() is called")
        asyncio.run_coroutine_threadsafe(
            self._queue.put((symbol, ltp, ts)),
            self._event_loop,
        )

    async def run(self) -> None:
        """Main async loop. Processes ticks until 15:30 IST or kill switch."""
        self._event_loop = asyncio.get_running_loop()
        logger.info("live_session.start", symbols=self._symbols, timeframe=self._timeframe)
        while True:
            try:
                symbol, ltp, ts = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except TimeoutError:
                if not self._is_within_session(datetime.now(tz=IST)):
                    break
                continue

            if not self._is_within_session(ts):
                break

            if self._kill_switch and self._kill_switch.is_active():
                logger.warning("live_session.kill_switch_active")
                break

            closed = self._on_tick(symbol, ltp, ts)
            if closed is not None:
                self._on_bar_closed(closed)

        # Force-close all in-progress bars at session end
        for sym in self._symbols:
            closed = self._builders[sym].force_close()
            if closed is not None:
                self._on_bar_closed(closed)

        logger.info("live_session.end")

    # ------------------------------------------------------------------
    # Testable sync internals
    # ------------------------------------------------------------------

    def _on_tick(self, symbol: str, ltp: float, ts: datetime) -> ClosedBar | None:
        """Route one tick to the correct BarBuilder."""
        if symbol not in self._builders:
            return None
        return self._builders[symbol].on_tick(ltp, ts)

    def _on_bar_closed(self, bar: ClosedBar) -> None:
        """Append closed bar, re-enrich window, call DailyLoop.process_bar."""
        new_row = bar.to_dataframe()
        existing = self._live_bars[bar.symbol]
        self._live_bars[bar.symbol] = (
            pl.concat([existing, new_row]) if len(existing) > 0 else new_row
        )

        # Build full window: warmup + all live bars for this symbol
        sym_warmup = (
            self._warmup_df.filter(pl.col("symbol") == bar.symbol)
            if len(self._warmup_df) > 0
            else pl.DataFrame()
        )
        live_sym = self._live_bars[bar.symbol]
        raw_window = pl.concat([sym_warmup, live_sym]) if len(sym_warmup) > 0 else live_sym

        if len(raw_window) < 2:
            return

        enriched = self._pipeline.run(raw_window)
        fills = self._loop.process_bar(enriched, evaluation_time=bar.bar_open)
        if self._on_bar_closed_cb is not None:
            self._on_bar_closed_cb(bar, fills)

    def _is_within_session(self, ts: datetime) -> bool:
        ist = ts.astimezone(IST)
        return ist.time() <= _SESSION_END

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------

    def _make_daily_loop(
        self,
        portfolio: PortfolioTracker,
        alerter: TelegramAlerter | None,
        heartbeat: Heartbeat | None,
        kill_switch: KillSwitch | None,
    ) -> DailyLoop:
        _tmp = Path(tempfile.mkdtemp())
        strategy = TrendFollowingStrategy()
        risk_manager = RiskManager(rules=load_risk_rules())
        executor = PaperExecution()
        journal = JournalStore(db_path=_tmp / "live_journal.db")
        _alerter = alerter or TelegramAlerter("", "")
        _heartbeat = heartbeat or Heartbeat(alerter=None)
        _kill_switch = kill_switch or KillSwitch(flag_path=_tmp / ".kill_switch")

        return DailyLoop(
            strategy=strategy,
            risk_manager=risk_manager,
            executor=executor,
            portfolio=portfolio,
            journal=journal,
            analyst=None,
            kill_switch=_kill_switch,
            heartbeat=_heartbeat,
            alerter=_alerter,
        )
