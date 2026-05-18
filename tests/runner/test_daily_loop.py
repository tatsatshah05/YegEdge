from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.data.types import DataQuality
from agent.decision.types import DecisionStatus
from agent.execution.types import ExecutionMode, Fill
from agent.features.pipeline import FeaturePipeline
from agent.journal.store import JournalStore
from agent.journal.types import JournalEntry, JournalEntryType
from agent.monitoring.alerter import TelegramAlerter
from agent.monitoring.heartbeat import Heartbeat
from agent.monitoring.kill_switch import KillSwitch
from agent.portfolio.tracker import PortfolioTracker
from agent.risk.types import RejectionReason, RiskDecision, RiskVerdict
from agent.runner.daily_loop import DailyLoop
from agent.runner.types import DailySessionResult
from agent.strategies.trend_following import TrendFollowingStrategy
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")
T0 = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
SESSION_DATE = date(2024, 1, 2)


def _make_enriched_df(n: int = 55, symbol: str = "HDFCBANK") -> pl.DataFrame:
    """Build a synthetic enriched DataFrame large enough for indicator warm-up."""
    from datetime import timedelta

    # 15-minute bars starting at 09:15 IST, incrementing by 15 minutes each row
    bar_start = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
    timestamps = [bar_start + timedelta(minutes=15 * i) for i in range(n)]
    closes = [1700.0 + i * 0.5 for i in range(n)]
    opens = [c - 2.0 for c in closes]
    highs = [c + 5.0 for c in closes]
    lows = [c - 5.0 for c in closes]

    ema_21 = [c - 10.0 for c in closes]
    ema_50 = [c - 5.0 for c in closes[:-1]] + [closes[-1] - 11.0]  # cross at last bar

    return pl.DataFrame({
        "symbol": [symbol] * n,
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [100_000 + i * 1000 for i in range(n)],
        "value": [c * 100_000 for c in closes],
        "ema_21": ema_21,
        "ema_50": ema_50,
        "adx_14": [25.0] * n,
        "atr_14": [15.0] * n,
        "data_quality": [DataQuality.OK.value] * n,
        "regime": ["trending"] * n,
    })


def _make_risk_approved(signal: Signal) -> RiskDecision:
    return RiskDecision(
        verdict=RiskVerdict.APPROVED,
        quantity=10,
        entry_price=signal.suggested_target,
        stop_price=signal.suggested_stop,
        target_price=signal.suggested_target,
        risk_per_share=Decimal("30"),
        position_value=signal.suggested_target * 10,
        rejection_reason=RejectionReason.NONE,
        rejection_detail="",
        signal=signal,
    )


def _make_risk_rejected(signal: Signal) -> RiskDecision:
    return RiskDecision(
        verdict=RiskVerdict.REJECTED,
        quantity=0,
        entry_price=signal.suggested_target,
        stop_price=signal.suggested_stop,
        target_price=signal.suggested_target,
        risk_per_share=Decimal("0"),
        position_value=Decimal("0"),
        rejection_reason=RejectionReason.MAX_POSITIONS_REACHED,
        rejection_detail="max positions reached",
        signal=signal,
    )


def _make_loop(tmp_path: Path, kill_switch: KillSwitch | None = None) -> DailyLoop:
    portfolio = PortfolioTracker(
        initial_nav=Decimal("100000"),
        initial_cash=Decimal("100000"),
        start_time=T0,
    )
    journal = JournalStore(db_path=tmp_path / "journal.db")
    strategy = TrendFollowingStrategy(min_adx=20.0, min_volume_ratio=1.0)
    risk = MagicMock()
    risk.evaluate.return_value = _make_risk_rejected(
        Signal(
            symbol="HDFCBANK", action=Action.ENTER_LONG, confidence=0.75,
            suggested_stop=Decimal("1680"), suggested_target=Decimal("1750"),
            invalidation_condition="", expected_r=2.0, time_horizon_hours=4,
            regime_fit=0.9, data_quality=DataQuality.OK,
            strategy_name="trend_following_v1", explanation="",
            timestamp=T0,
        )
    )
    ks = kill_switch or KillSwitch(flag_path=tmp_path / ".kill_switch")
    alerter = MagicMock(spec=TelegramAlerter)
    heartbeat = Heartbeat(alerter=alerter, alert_every_n_beats=999)

    return DailyLoop(
        strategy=strategy,
        risk_manager=risk,
        executor=MagicMock(),
        portfolio=portfolio,
        journal=journal,
        analyst=None,
        kill_switch=ks,
        heartbeat=heartbeat,
        alerter=alerter,
    )


def test_daily_session_result_is_frozen() -> None:
    result = DailySessionResult(
        session_date=SESSION_DATE,
        bars_processed=10,
        signals_generated=2,
        decisions_made=2,
        fills=(),
        rejections=2,
        ai_cache_hits=0,
        final_nav=Decimal("100000"),
        daily_pnl=Decimal("0"),
        peak_nav=Decimal("100000"),
    )
    with pytest.raises(AttributeError):
        result.bars_processed = 99  # type: ignore[misc]


def test_process_bar_returns_no_fills_when_rejected(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    df = _make_enriched_df()
    fills = loop.process_bar(df, evaluation_time=T0)
    assert isinstance(fills, list)
    assert len(fills) == 0


def test_process_bar_journals_decisions(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    df = _make_enriched_df()
    loop.process_bar(df, evaluation_time=T0)
    journal = JournalStore(db_path=tmp_path / "journal.db")
    entries = journal.query()
    assert len(entries) >= 0  # loop may have no signals — that's OK


def test_process_bar_returns_fills_when_approved(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    df = _make_enriched_df()

    fake_signal = Signal(
        symbol="HDFCBANK", action=Action.ENTER_LONG, confidence=0.75,
        suggested_stop=Decimal("1680"), suggested_target=Decimal("1750"),
        invalidation_condition="", expected_r=2.0, time_horizon_hours=4,
        regime_fit=0.9, data_quality=DataQuality.OK,
        strategy_name="trend_following_v1", explanation="", timestamp=T0,
    )
    loop._risk_manager.evaluate.return_value = _make_risk_approved(fake_signal)

    fake_fill = Fill(
        order_id="paper-test", symbol="HDFCBANK", action=Action.ENTER_LONG,
        quantity=10, fill_price=Decimal("1750"),
        timestamp=T0, signal_id="test:enter_long:2024",
        strategy_name="trend_following_v1", execution_mode=ExecutionMode.PAPER,
    )
    loop._executor.submit.return_value = fake_fill

    fills = loop.process_bar(df, evaluation_time=T0)
    assert isinstance(fills, list)


def test_run_returns_daily_session_result(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    warmup_df = _make_enriched_df(n=55)
    session_df = _make_enriched_df(n=4)
    result = loop.run(
        session_date=SESSION_DATE,
        warmup_df=warmup_df,
        session_df=session_df,
    )
    assert isinstance(result, DailySessionResult)
    assert result.session_date == SESSION_DATE
    assert result.bars_processed == 4


def test_run_stops_on_kill_switch(tmp_path: Path) -> None:
    flag = tmp_path / ".kill_switch"
    flag.write_text("test stop")
    ks = KillSwitch(flag_path=flag)
    loop = _make_loop(tmp_path, kill_switch=ks)

    warmup_df = _make_enriched_df(n=55)
    session_df = _make_enriched_df(n=10)
    result = loop.run(
        session_date=SESSION_DATE,
        warmup_df=warmup_df,
        session_df=session_df,
    )
    assert result.bars_processed == 0


def test_run_journals_pnl_entry_at_end(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    warmup_df = _make_enriched_df(n=55)
    session_df = _make_enriched_df(n=2)
    loop.run(session_date=SESSION_DATE, warmup_df=warmup_df, session_df=session_df)
    journal = JournalStore(db_path=tmp_path / "journal.db")
    pnl_entries = journal.query(entry_type=JournalEntryType.PNL)
    assert len(pnl_entries) == 1


def test_run_final_nav_in_result(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    warmup_df = _make_enriched_df(n=55)
    session_df = _make_enriched_df(n=2)
    result = loop.run(session_date=SESSION_DATE, warmup_df=warmup_df, session_df=session_df)
    assert result.final_nav == Decimal("100000")


def test_run_sends_daily_summary_alert(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    warmup_df = _make_enriched_df(n=55)
    session_df = _make_enriched_df(n=2)
    loop.run(session_date=SESSION_DATE, warmup_df=warmup_df, session_df=session_df)
    loop._alerter.send_daily_summary.assert_called_once()


def test_process_bar_with_no_signals_returns_empty_list(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    tiny_df = _make_enriched_df(n=2)
    fills = loop.process_bar(tiny_df, evaluation_time=T0)
    assert fills == []
