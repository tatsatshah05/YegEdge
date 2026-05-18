from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.data.types import DataQuality
from agent.decision.types import Decision, DecisionStatus, ResearchNote
from agent.execution.paper import PaperExecution
from agent.execution.types import ExecutionMode, Fill
from agent.risk.types import PortfolioState, RejectionReason, RiskDecision, RiskVerdict
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")

EVAL_TIME = datetime(2024, 1, 2, 10, 0, tzinfo=IST)


def _make_signal(symbol: str = "HDFCBANK", action: Action = Action.ENTER_LONG) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        confidence=0.75,
        suggested_stop=Decimal("1680.00"),
        suggested_target=Decimal("1760.00"),
        invalidation_condition="Close below EMA21",
        expected_r=2.0,
        time_horizon_hours=4,
        regime_fit=0.8,
        data_quality=DataQuality.OK,
        strategy_name="trend_following_v1",
        explanation="EMA21 crossed above EMA50",
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )


def _make_decision(signal: Signal) -> Decision:
    return Decision(
        signal=signal,
        status=DecisionStatus.PENDING,
        signal_id=f"{signal.symbol}:{signal.action}:{signal.timestamp.isoformat()}",
        merged_from=("trend_following_v1",),
        research_note=None,
        skip_reason="",
        timestamp=EVAL_TIME,
    )


def _make_risk_decision(
    signal: Signal,
    quantity: int = 10,
    entry_price: Decimal = Decimal("1710.00"),
) -> RiskDecision:
    return RiskDecision(
        verdict=RiskVerdict.APPROVED,
        quantity=quantity,
        entry_price=entry_price,
        stop_price=Decimal("1680.00"),
        target_price=Decimal("1760.00"),
        risk_per_share=Decimal("30.00"),
        position_value=entry_price * quantity,
        rejection_reason=RejectionReason.NONE,
        rejection_detail="",
        signal=signal,
    )


def test_submit_returns_fill() -> None:
    engine = PaperExecution()
    sig = _make_signal()
    decision = _make_decision(sig)
    risk_dec = _make_risk_decision(sig)
    fill = engine.submit(decision, risk_dec, submitted_at=EVAL_TIME)
    assert isinstance(fill, Fill)


def test_submit_fill_price_matches_risk_entry() -> None:
    engine = PaperExecution()
    sig = _make_signal()
    risk_dec = _make_risk_decision(sig, entry_price=Decimal("1715.50"))
    fill = engine.submit(_make_decision(sig), risk_dec, submitted_at=EVAL_TIME)
    assert fill.fill_price == Decimal("1715.50")


def test_submit_fill_quantity_matches_risk_quantity() -> None:
    engine = PaperExecution()
    sig = _make_signal()
    risk_dec = _make_risk_decision(sig, quantity=25)
    fill = engine.submit(_make_decision(sig), risk_dec, submitted_at=EVAL_TIME)
    assert fill.quantity == 25


def test_submit_execution_mode_is_paper() -> None:
    engine = PaperExecution()
    sig = _make_signal()
    fill = engine.submit(_make_decision(sig), _make_risk_decision(sig), submitted_at=EVAL_TIME)
    assert fill.execution_mode == ExecutionMode.PAPER


def test_submit_order_id_is_deterministic() -> None:
    engine = PaperExecution()
    sig = _make_signal()
    d = _make_decision(sig)
    rd = _make_risk_decision(sig)
    fill1 = engine.submit(d, rd, submitted_at=EVAL_TIME)
    fill2 = engine.submit(d, rd, submitted_at=EVAL_TIME)
    assert fill1.order_id == fill2.order_id


def test_submit_action_propagated() -> None:
    engine = PaperExecution()
    sig = _make_signal(action=Action.EXIT_LONG)
    fill = engine.submit(_make_decision(sig), _make_risk_decision(sig), submitted_at=EVAL_TIME)
    assert fill.action == Action.EXIT_LONG


def test_submit_signal_id_in_fill() -> None:
    engine = PaperExecution()
    sig = _make_signal()
    d = _make_decision(sig)
    fill = engine.submit(d, _make_risk_decision(sig), submitted_at=EVAL_TIME)
    assert fill.signal_id == d.signal_id
