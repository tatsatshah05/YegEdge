from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.data.types import DataQuality, Position
from agent.decision.engine import DecisionEngine
from agent.decision.types import DecisionStatus, ResearchNote
from agent.risk.types import PortfolioState
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")

EVAL_TIME = datetime(2024, 1, 2, 10, 0, tzinfo=IST)


def _make_signal(
    symbol: str = "HDFCBANK",
    action: Action = Action.ENTER_LONG,
    confidence: float = 0.7,
    strategy_name: str = "trend_following_v1",
) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        suggested_stop=Decimal("1680.00"),
        suggested_target=Decimal("1760.00"),
        invalidation_condition="Close below EMA21",
        expected_r=2.0,
        time_horizon_hours=4,
        regime_fit=0.8,
        data_quality=DataQuality.OK,
        strategy_name=strategy_name,
        explanation="EMA21 crossed above EMA50",
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )


def _empty_portfolio() -> PortfolioState:
    return PortfolioState(
        nav=Decimal("100000"),
        cash=Decimal("90000"),
        positions={},
        daily_pnl=Decimal("0"),
        weekly_pnl=Decimal("0"),
        peak_nav=Decimal("100000"),
        orders_today=0,
        last_order_time={},
        kill_switch_active=False,
        evaluation_time=EVAL_TIME,
    )


def _portfolio_with_position(symbol: str) -> PortfolioState:
    pos = Position(symbol=symbol, quantity=10, average_price=Decimal("1700"), product="MIS")
    return PortfolioState(
        nav=Decimal("100000"),
        cash=Decimal("73000"),
        positions={symbol: pos},
        daily_pnl=Decimal("0"),
        weekly_pnl=Decimal("0"),
        peak_nav=Decimal("100000"),
        orders_today=1,
        last_order_time={},
        kill_switch_active=False,
        evaluation_time=EVAL_TIME,
    )


def _make_note(signal_id: str, veto: bool, reason: str | None = None) -> ResearchNote:
    return ResearchNote(
        signal_id=signal_id,
        bullish_case="Trending strongly.",
        bearish_case="Overbought short-term.",
        dominant_risk="FII selling.",
        regime_fit_assessment="Good fit.",
        confidence_qualitative="HIGH",
        veto=veto,
        veto_reason=reason,
        model_used="claude-haiku-4-5-20251001",
        tokens_used=150,
        cached=False,
    )


# --- Basic ---


def test_empty_signals_returns_empty() -> None:
    engine = DecisionEngine()
    result = engine.evaluate([], _empty_portfolio(), evaluation_time=EVAL_TIME)
    assert result == []


def test_hold_signals_are_filtered() -> None:
    engine = DecisionEngine()
    hold_sig = _make_signal(action=Action.HOLD)
    result = engine.evaluate([hold_sig], _empty_portfolio(), evaluation_time=EVAL_TIME)
    assert result == []


def test_single_enter_long_becomes_pending() -> None:
    engine = DecisionEngine()
    sig = _make_signal(action=Action.ENTER_LONG)
    result = engine.evaluate([sig], _empty_portfolio(), evaluation_time=EVAL_TIME)
    assert len(result) == 1
    assert result[0].status == DecisionStatus.PENDING
    assert result[0].signal is sig


def test_single_exit_long_becomes_pending() -> None:
    engine = DecisionEngine()
    sig = _make_signal(action=Action.EXIT_LONG)
    result = engine.evaluate([sig], _empty_portfolio(), evaluation_time=EVAL_TIME)
    assert len(result) == 1
    assert result[0].status == DecisionStatus.PENDING


def test_evaluation_time_naive_raises() -> None:
    engine = DecisionEngine()
    with pytest.raises(ValueError, match="IST-aware"):
        engine.evaluate([], _empty_portfolio(), evaluation_time=datetime(2024, 1, 2, 10, 0))


# --- Deduplication ---


def test_same_symbol_action_two_strategies_deduplicates() -> None:
    engine = DecisionEngine()
    sig_a = _make_signal(confidence=0.7, strategy_name="trend_following_v1")
    sig_b = _make_signal(confidence=0.9, strategy_name="mean_reversion_v1")
    result = engine.evaluate([sig_a, sig_b], _empty_portfolio(), evaluation_time=EVAL_TIME)
    assert len(result) == 1
    assert result[0].signal.confidence == 0.9
    assert "trend_following_v1" in result[0].merged_from
    assert "mean_reversion_v1" in result[0].merged_from


def test_different_symbols_produce_separate_decisions() -> None:
    engine = DecisionEngine()
    sig1 = _make_signal(symbol="HDFCBANK")
    sig2 = _make_signal(symbol="TCS")
    result = engine.evaluate([sig1, sig2], _empty_portfolio(), evaluation_time=EVAL_TIME)
    assert len(result) == 2
    symbols = {d.signal.symbol for d in result}
    assert symbols == {"HDFCBANK", "TCS"}


def test_enter_and_exit_same_symbol_are_separate_decisions() -> None:
    engine = DecisionEngine()
    enter = _make_signal(action=Action.ENTER_LONG)
    exit_ = _make_signal(action=Action.EXIT_LONG)
    result = engine.evaluate([enter, exit_], _empty_portfolio(), evaluation_time=EVAL_TIME)
    assert len(result) == 2
    actions = {d.signal.action for d in result}
    assert Action.ENTER_LONG in actions
    assert Action.EXIT_LONG in actions


# --- Portfolio context ---


def test_enter_long_skipped_when_already_holding() -> None:
    engine = DecisionEngine()
    sig = _make_signal(symbol="HDFCBANK", action=Action.ENTER_LONG)
    portfolio = _portfolio_with_position("HDFCBANK")
    result = engine.evaluate([sig], portfolio, evaluation_time=EVAL_TIME)
    assert len(result) == 1
    assert result[0].status == DecisionStatus.SKIPPED
    assert "HDFCBANK" in result[0].skip_reason


def test_exit_long_not_skipped_even_when_holding() -> None:
    engine = DecisionEngine()
    sig = _make_signal(symbol="HDFCBANK", action=Action.EXIT_LONG)
    portfolio = _portfolio_with_position("HDFCBANK")
    result = engine.evaluate([sig], portfolio, evaluation_time=EVAL_TIME)
    assert len(result) == 1
    assert result[0].status == DecisionStatus.PENDING


def test_enter_long_not_skipped_when_holding_different_symbol() -> None:
    engine = DecisionEngine()
    sig = _make_signal(symbol="TCS", action=Action.ENTER_LONG)
    portfolio = _portfolio_with_position("HDFCBANK")
    result = engine.evaluate([sig], portfolio, evaluation_time=EVAL_TIME)
    assert len(result) == 1
    assert result[0].status == DecisionStatus.PENDING


# --- Veto handling ---


def test_veto_true_produces_wait_for_confirmation() -> None:
    engine = DecisionEngine()
    sig = _make_signal()
    signal_id = f"{sig.symbol}:{sig.action}:{sig.timestamp.isoformat()}"
    note = _make_note(signal_id, veto=True, reason="Macro risk too high")
    result = engine.evaluate(
        [sig],
        _empty_portfolio(),
        research_notes={signal_id: note},
        evaluation_time=EVAL_TIME,
    )
    assert len(result) == 1
    assert result[0].status == DecisionStatus.WAIT_FOR_CONFIRMATION
    assert "Macro risk too high" in result[0].skip_reason


def test_veto_false_research_note_still_pending() -> None:
    engine = DecisionEngine()
    sig = _make_signal()
    signal_id = f"{sig.symbol}:{sig.action}:{sig.timestamp.isoformat()}"
    note = _make_note(signal_id, veto=False)
    result = engine.evaluate(
        [sig],
        _empty_portfolio(),
        research_notes={signal_id: note},
        evaluation_time=EVAL_TIME,
    )
    assert result[0].status == DecisionStatus.PENDING
    assert result[0].research_note is not None


def test_no_research_note_still_pending() -> None:
    engine = DecisionEngine()
    sig = _make_signal()
    result = engine.evaluate([sig], _empty_portfolio(), evaluation_time=EVAL_TIME)
    assert result[0].status == DecisionStatus.PENDING
    assert result[0].research_note is None


def test_veto_with_no_reason_uses_fallback_message() -> None:
    engine = DecisionEngine()
    sig = _make_signal()
    signal_id = f"{sig.symbol}:{sig.action}:{sig.timestamp.isoformat()}"
    note = _make_note(signal_id, veto=True, reason=None)
    result = engine.evaluate(
        [sig],
        _empty_portfolio(),
        research_notes={signal_id: note},
        evaluation_time=EVAL_TIME,
    )
    assert result[0].status == DecisionStatus.WAIT_FOR_CONFIRMATION
    assert result[0].skip_reason != ""


# --- signal_id, merged_from, timestamp ---


def test_signal_id_format() -> None:
    engine = DecisionEngine()
    sig = _make_signal(symbol="HDFCBANK", action=Action.ENTER_LONG)
    result = engine.evaluate([sig], _empty_portfolio(), evaluation_time=EVAL_TIME)
    expected_id = f"HDFCBANK:enter_long:{sig.timestamp.isoformat()}"
    assert result[0].signal_id == expected_id


def test_signal_id_is_deterministic() -> None:
    engine = DecisionEngine()
    sig = _make_signal()
    result1 = engine.evaluate([sig], _empty_portfolio(), evaluation_time=EVAL_TIME)
    result2 = engine.evaluate([sig], _empty_portfolio(), evaluation_time=EVAL_TIME)
    assert result1[0].signal_id == result2[0].signal_id


def test_merged_from_single_strategy() -> None:
    engine = DecisionEngine()
    sig = _make_signal(strategy_name="trend_following_v1")
    result = engine.evaluate([sig], _empty_portfolio(), evaluation_time=EVAL_TIME)
    assert result[0].merged_from == ("trend_following_v1",)


def test_decision_timestamp_matches_evaluation_time() -> None:
    engine = DecisionEngine()
    sig = _make_signal()
    result = engine.evaluate([sig], _empty_portfolio(), evaluation_time=EVAL_TIME)
    assert result[0].timestamp == EVAL_TIME


def test_merged_from_multiple_strategies_sorted() -> None:
    engine = DecisionEngine()
    sig_z = _make_signal(confidence=0.7, strategy_name="zebra_strategy")
    sig_a = _make_signal(confidence=0.8, strategy_name="alpha_strategy")
    sig_m = _make_signal(confidence=0.9, strategy_name="middle_strategy")
    result = engine.evaluate([sig_z, sig_a, sig_m], _empty_portfolio(), evaluation_time=EVAL_TIME)
    assert len(result) == 1
    assert result[0].merged_from == ("alpha_strategy", "middle_strategy", "zebra_strategy")
