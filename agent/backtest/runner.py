# agent/backtest/runner.py
from __future__ import annotations

import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import structlog

from agent.backtest.costs import IndianCostModel
from agent.backtest.metrics import BacktestReport, SessionResult, compute_metrics
from agent.data.cache import ParquetCache
from agent.data.calendar import NseTradingCalendar
from agent.execution.paper import PaperExecution
from agent.features.pipeline import FeaturePipeline
from agent.journal.store import JournalStore
from agent.monitoring.alerter import TelegramAlerter
from agent.monitoring.heartbeat import Heartbeat
from agent.monitoring.kill_switch import KillSwitch
from agent.portfolio.tracker import PortfolioTracker
from agent.risk.manager import RiskManager
from agent.runner.daily_loop import DailyLoop
from agent.strategies.trend_following import TrendFollowingStrategy

log = structlog.get_logger()
IST = ZoneInfo("Asia/Kolkata")


class BacktestRunner:
    """Replay historical OHLCV bars session-by-session through the full strategy pipeline.

    AI analyst is set to None — zero Anthropic API calls during backtest.
    A temporary SQLite journal is used per run — the production journal is never touched.
    KillSwitch is scoped to a temp directory — cannot trigger the production kill switch.
    Alerts are suppressed via TelegramAlerter("", "").

    Each session creates a fresh PortfolioTracker initialised from the prior session's
    ending NAV. IndianCostModel deducts realistic NSE transaction costs per fill, and
    NAV compounds across sessions.
    """

    def __init__(
        self,
        *,
        strategy: TrendFollowingStrategy,
        risk_manager: RiskManager,
        cache: ParquetCache,
        initial_nav: Decimal,
        cost_model: IndianCostModel | None = None,
    ) -> None:
        self._strategy = strategy
        self._risk_manager = risk_manager
        self._cache = cache
        self._initial_nav = initial_nav
        self._cost_model = cost_model or IndianCostModel()

    def run(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        warmup_bars: int = 100,
    ) -> BacktestReport:
        """Run the backtest for one symbol over [start_date, end_date].

        Returns a BacktestReport with per-session SessionResult entries and
        aggregate BacktestMetrics. Returns an empty report if no cached data
        exists for the symbol/timeframe or if there are no trading days in range.
        """
        calendar = NseTradingCalendar()
        trading_days = calendar.trading_sessions(start_date, end_date)

        if not trading_days:
            return BacktestReport(sessions=[], metrics=compute_metrics([], self._initial_nav))

        # Load full history — 1 year before start for indicator warm-up seeding
        load_from = datetime(start_date.year - 1, start_date.month, start_date.day, tzinfo=IST)
        load_to = datetime(end_date.year, end_date.month, end_date.day, 23, 59, tzinfo=IST)
        raw = self._cache.read(symbol=symbol, timeframe=timeframe, start=load_from, end=load_to)

        if len(raw) == 0:
            log.warning("backtest.no_cache_data", symbol=symbol, timeframe=timeframe)
            return BacktestReport(sessions=[], metrics=compute_metrics([], self._initial_nav))

        # Enrich once — FeaturePipeline is safe for single-symbol DataFrames
        enriched = FeaturePipeline().run(raw)

        running_nav = self._initial_nav
        session_results: list[SessionResult] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            journal = JournalStore(db_path=tmp / "backtest.db")

            for session_date in trading_days:
                session_start = datetime(
                    session_date.year,
                    session_date.month,
                    session_date.day,
                    9,
                    15,
                    tzinfo=IST,
                )
                session_end = datetime(
                    session_date.year,
                    session_date.month,
                    session_date.day,
                    15,
                    30,
                    tzinfo=IST,
                )

                session_df = enriched.filter(
                    (pl.col("timestamp") >= session_start) & (pl.col("timestamp") <= session_end)
                )
                if len(session_df) == 0:
                    log.debug(
                        "backtest.session_skipped_no_bars",
                        date=str(session_date),
                        symbol=symbol,
                    )
                    continue

                warmup_df = enriched.filter(pl.col("timestamp") < session_start).tail(warmup_bars)

                portfolio = PortfolioTracker(
                    initial_nav=running_nav,
                    initial_cash=running_nav,
                    start_time=session_start,
                )

                # KillSwitch scoped to temp dir — cannot trigger production kill switch.
                # Heartbeat has no alerter — no Telegram messages during backtest.
                # analyst=None — zero Anthropic API calls.
                # TelegramAlerter("", "") — alerts suppressed.
                loop = DailyLoop(
                    strategy=self._strategy,
                    risk_manager=self._risk_manager,
                    executor=PaperExecution(),
                    portfolio=portfolio,
                    journal=journal,
                    analyst=None,
                    kill_switch=KillSwitch(flag_path=tmp / ".kill_switch"),
                    heartbeat=Heartbeat(),
                    alerter=TelegramAlerter("", ""),
                )

                result = loop.run(
                    session_date=session_date,
                    warmup_df=warmup_df,
                    session_df=session_df,
                )

                costs = sum(
                    (self._cost_model.compute_cost(f) for f in result.fills),
                    Decimal("0"),
                )
                net_pnl = result.daily_pnl - costs
                running_nav = running_nav + net_pnl

                session_results.append(
                    SessionResult(
                        session_date=session_date,
                        bars_processed=result.bars_processed,
                        fills=len(result.fills),
                        gross_pnl=result.daily_pnl,
                        costs=costs,
                        net_pnl=net_pnl,
                        final_nav=running_nav,
                    )
                )

                log.info(
                    "backtest.session_done",
                    date=str(session_date),
                    symbol=symbol,
                    bars=result.bars_processed,
                    fills=len(result.fills),
                    net_pnl=str(net_pnl),
                    running_nav=str(running_nav),
                )

        return BacktestReport(
            sessions=session_results,
            metrics=compute_metrics(session_results, self._initial_nav),
        )
