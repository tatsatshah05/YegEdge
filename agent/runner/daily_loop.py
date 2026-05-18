from __future__ import annotations

import json
import time as _time
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import structlog

from agent.ai.analyst import AIAnalyst
from agent.decision.engine import DecisionEngine
from agent.decision.types import Decision, DecisionStatus
from agent.execution.paper import PaperExecution
from agent.execution.types import Fill
from agent.journal.store import JournalStore
from agent.journal.types import JournalEntry, JournalEntryType
from agent.monitoring.alerter import TelegramAlerter
from agent.monitoring.heartbeat import Heartbeat
from agent.monitoring.kill_switch import KillSwitch
from agent.portfolio.tracker import PortfolioTracker
from agent.risk.manager import RiskManager
from agent.risk.types import RiskVerdict
from agent.runner.types import DailySessionResult
from agent.strategies.trend_following import TrendFollowingStrategy

logger = structlog.get_logger()

IST = ZoneInfo("Asia/Kolkata")


def _ensure_ist(ts: object) -> datetime:
    """Convert a Polars timestamp (or any datetime-like) to IST-aware datetime."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=IST)
        return ts.astimezone(IST)
    # Polars may return integers (microseconds since epoch) in some configs — handle string
    return datetime.fromisoformat(str(ts)).replace(tzinfo=IST)


class DailyLoop:
    """Orchestrates one paper trading session.

    Takes pre-loaded DataFrames (warmup history + session bars) and runs the full
    pipeline bar-by-bar. All external dependencies are injected for testability.
    """

    def __init__(
        self,
        *,
        strategy: TrendFollowingStrategy,
        risk_manager: RiskManager,
        executor: PaperExecution,
        portfolio: PortfolioTracker,
        journal: JournalStore,
        analyst: AIAnalyst | None,
        kill_switch: KillSwitch,
        heartbeat: Heartbeat,
        alerter: TelegramAlerter,
    ) -> None:
        self._strategy = strategy
        self._risk_manager = risk_manager
        self._executor = executor
        self._portfolio = portfolio
        self._journal = journal
        self._analyst = analyst
        self._kill_switch = kill_switch
        self._heartbeat = heartbeat
        self._alerter = alerter
        self._decision_engine = DecisionEngine()
        self._last_signals = 0
        self._last_decisions = 0
        self._last_rejections = 0

    def process_bar(
        self,
        df: pl.DataFrame,
        *,
        evaluation_time: datetime,
    ) -> list[Fill]:
        """Run one enriched bar window through the full pipeline. Returns fills."""
        if len(df) < 2:
            return []

        signals = self._strategy.generate(df)
        if not signals:
            self._last_signals = 0
            self._last_decisions = 0
            self._last_rejections = 0
            return []

        decisions = self._decision_engine.evaluate(
            signals,
            self._portfolio.state,
            evaluation_time=evaluation_time,
        )

        fills: list[Fill] = []
        for decision in decisions:
            self._log_decision(decision, evaluation_time)

            if decision.status != DecisionStatus.PENDING:
                continue

            signal = decision.signal

            # Optional AI annotation — result is used for logging only here;
            # the DecisionEngine already consumed notes before this point.
            # When analyst is provided, we call it for its side-effects (cache warm-up).
            if self._analyst is not None:
                self._analyst.analyse(signal)

            risk_dec = self._risk_manager.evaluate(
                signal,
                self._portfolio.state,
                signal.suggested_target,
            )

            if risk_dec.verdict == RiskVerdict.APPROVED:
                fill = self._executor.submit(
                    decision, risk_dec, submitted_at=evaluation_time
                )
                self._portfolio.apply_fill(fill, evaluation_time=evaluation_time)
                self._journal.log(
                    JournalEntry(
                        entry_id=fill.order_id,
                        timestamp=_ensure_ist(fill.timestamp),
                        entry_type=JournalEntryType.FILL,
                        symbol=fill.symbol,
                        payload=json.dumps({
                            "action": str(fill.action),
                            "quantity": fill.quantity,
                            "price": str(fill.fill_price),
                            "signal_id": fill.signal_id,
                        }),
                    )
                )
                self._alerter.send_fill_alert(fill)
                fills.append(fill)
            else:
                # Use monotonic ns to guarantee uniqueness when the same signal_id
                # appears in multiple bar windows (e.g., same signal replays across
                # expanded windows in process_bar called from run()).
                rej_id = (
                    f"rej-{signal.symbol}"
                    f"-{evaluation_time.isoformat()}"
                    f"-{decision.signal_id[-12:]}"
                    f"-{_time.monotonic_ns()}"
                )
                self._journal.log(
                    JournalEntry(
                        entry_id=rej_id,
                        timestamp=evaluation_time,
                        entry_type=JournalEntryType.REJECTION,
                        symbol=signal.symbol,
                        payload=json.dumps({
                            "reason": str(risk_dec.rejection_reason),
                            "detail": risk_dec.rejection_detail,
                            "signal_id": decision.signal_id,
                        }),
                    )
                )
                self._alerter.send_rejection_alert(
                    signal.symbol, reason=str(risk_dec.rejection_reason)
                )

        self._last_signals = len(signals)
        self._last_decisions = len(decisions)
        self._last_rejections = len(decisions) - len(fills)
        return fills

    def run(
        self,
        *,
        session_date: date,
        warmup_df: pl.DataFrame,
        session_df: pl.DataFrame,
    ) -> DailySessionResult:
        """Run one full trading session.

        warmup_df: historical bars preceding session_date (for indicator warm-up).
        session_df: intraday bars for session_date.
        """
        if self._kill_switch.is_active():
            logger.warning("daily_loop.kill_switch_active_at_start", date=str(session_date))
            state = self._portfolio.state
            return DailySessionResult(
                session_date=session_date,
                bars_processed=0,
                signals_generated=0,
                decisions_made=0,
                fills=(),
                rejections=0,
                ai_cache_hits=0,
                final_nav=state.nav,
                daily_pnl=state.daily_pnl,
                peak_nav=state.peak_nav,
            )

        logger.info("daily_loop.session_start", date=str(session_date), bars=len(session_df))

        all_fills: list[Fill] = []
        bars_processed = 0
        signals_generated = 0
        decisions_made = 0
        rejections_count = 0

        combined = pl.concat([warmup_df, session_df]) if len(warmup_df) > 0 else session_df
        warmup_len = len(warmup_df)

        session_timestamps = session_df["timestamp"].to_list()

        for i, ts in enumerate(session_timestamps):
            if self._kill_switch.is_active():
                logger.warning("daily_loop.kill_switch_triggered", bar_index=i)
                self._portfolio.activate_kill_switch()
                break

            window = combined.slice(0, warmup_len + i + 1)
            evaluation_time = _ensure_ist(ts)

            fills = self.process_bar(window, evaluation_time=evaluation_time)
            bars_processed += 1
            signals_generated += self._last_signals
            decisions_made += self._last_decisions
            rejections_count += self._last_rejections
            all_fills.extend(fills)

            self._heartbeat.beat(self._portfolio.state, ts=evaluation_time)

        # End-of-session P&L journal entry — always written even if no bars processed
        # (kill switch path short-circuits before this block, so session_timestamps is
        # always populated when we reach here).
        final_state = self._portfolio.state
        if session_timestamps:
            last_ts = _ensure_ist(session_timestamps[-1])
        else:
            last_ts = datetime.now(tz=IST)

        self._journal.log(
            JournalEntry(
                entry_id=f"pnl-{session_date.isoformat()}",
                timestamp=last_ts,
                entry_type=JournalEntryType.PNL,
                symbol=None,
                payload=json.dumps({
                    "session_date": str(session_date),
                    "final_nav": str(final_state.nav),
                    "daily_pnl": str(final_state.daily_pnl),
                    "orders_today": final_state.orders_today,
                }),
            )
        )

        analyst_cache_hits = (
            self._analyst._cache.size if self._analyst is not None else 0
        )

        self._alerter.send_daily_summary(final_state, session_count=1)

        result = DailySessionResult(
            session_date=session_date,
            bars_processed=bars_processed,
            signals_generated=signals_generated,
            decisions_made=decisions_made,
            fills=tuple(all_fills),
            rejections=rejections_count,
            ai_cache_hits=analyst_cache_hits,
            final_nav=final_state.nav,
            daily_pnl=final_state.daily_pnl,
            peak_nav=final_state.peak_nav,
        )

        logger.info(
            "daily_loop.session_end",
            date=str(session_date),
            bars=bars_processed,
            fills=len(all_fills),
            final_nav=str(result.final_nav),
            daily_pnl=str(result.daily_pnl),
        )
        return result

    def _log_decision(self, decision: Decision, evaluation_time: datetime) -> None:
        """Journal every decision (PENDING, SKIPPED, WAIT_FOR_CONFIRMATION)."""
        try:
            self._journal.log(
                JournalEntry(
                    entry_id=(
                        f"dec-{decision.signal_id}"
                        f"-{evaluation_time.isoformat()}"
                        f"-{_time.monotonic_ns()}"
                    ),
                    timestamp=evaluation_time,
                    entry_type=JournalEntryType.DECISION,
                    symbol=decision.signal.symbol,
                    payload=json.dumps({
                        "signal_id": decision.signal_id,
                        "status": str(decision.status),
                        "skip_reason": decision.skip_reason,
                    }),
                )
            )
        except Exception:
            logger.warning(
                "daily_loop.journal_decision_failed",
                signal_id=decision.signal_id,
                exc_info=True,
            )
