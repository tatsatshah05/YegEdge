# Phase 5 — Decision Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `agent/decision/` — the pure aggregation layer that takes signals from one or more strategies, deduplicates by (symbol, action), applies portfolio context, handles AI research note vetoes, and emits a `Decision` per actionable (symbol, action) pair for the risk manager.

**Architecture:** The decision engine sits between strategies and the risk manager in the data flow: `list[Signal] → DecisionEngine → list[Decision] → RiskManager`. It is a pure function over its inputs (no I/O, no broker calls). Every decision — including skipped ones — is returned so the journal layer can log the full picture. The AI research layer (Phase 7) is not built yet; the engine accepts an optional `research_notes` dict that defaults to `None`, so it works without the AI layer.

**Tech Stack:** Python 3.11+, `dataclasses` (frozen+slots), `enum.StrEnum`, `structlog`, `pytest`. All types follow the pattern established in Phases 3 and 4.

---

## Context for subagent workers

You are implementing Phase 5 of a trading agent called YegEdge. The project lives at `/Users/tatsatshah/Desktop/yegedge`. There is a `.venv/` virtualenv — always activate it before running commands:

```bash
source /Users/tatsatshah/Desktop/yegedge/.venv/bin/activate
```

The current branch is `phase-2-feature-engineering`. Do not create a new branch.

**Already built (do not modify):**
- `agent/data/types.py` — `DataQuality`, `Position`, `Order`, `Bar`
- `agent/strategies/types.py` — `Action` (ENTER_LONG / EXIT_LONG / HOLD), `Signal`
- `agent/risk/types.py` — `RiskDecision`, `PortfolioState`, `RiskVerdict`, `RejectionReason`
- `agent/risk/manager.py` — `RiskManager.evaluate(signal, portfolio, entry_price) → RiskDecision`

**Conventions you must follow:**
- `from __future__ import annotations` at the top of every `.py` file
- `logger = structlog.get_logger()` (not `log`)
- `@dataclass(frozen=True, slots=True)` on all dataclasses
- No `print()` — use structlog
- All monetary values are `Decimal`; `float` is only used for ratios/weights
- IST-aware `datetime` everywhere (`ZoneInfo("Asia/Kolkata")`)
- No comments unless the WHY is non-obvious

**Key types you will reference (copy exactly — do not re-define):**

```python
# from agent.strategies.types
class Action(StrEnum):
    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"
    HOLD = "hold"

@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    action: Action
    confidence: float
    suggested_stop: Decimal
    suggested_target: Decimal
    invalidation_condition: str
    expected_r: float
    time_horizon_hours: int
    regime_fit: float
    data_quality: DataQuality
    strategy_name: str
    explanation: str
    timestamp: datetime  # IST-aware

# from agent.risk.types
@dataclass(frozen=True, slots=True)
class PortfolioState:
    nav: Decimal
    cash: Decimal
    positions: dict[str, Position]
    daily_pnl: Decimal
    weekly_pnl: Decimal
    peak_nav: Decimal
    orders_today: int
    last_order_time: dict[str, datetime]
    kill_switch_active: bool
    evaluation_time: datetime
```

---

## File Map

```
agent/decision/
    __init__.py       — empty package marker
    types.py          — DecisionStatus, ResearchNote, Decision
    engine.py         — DecisionEngine

tests/decision/
    __init__.py       — empty package marker
    test_types.py     — 7 tests for types
    test_engine.py    — 20 tests for DecisionEngine
```

---

## Task 1: Types — DecisionStatus, ResearchNote, Decision

**Files:**
- Create: `agent/decision/__init__.py`
- Create: `agent/decision/types.py`
- Create: `tests/decision/__init__.py`
- Test: `tests/decision/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/decision/test_types.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.decision.types import Decision, DecisionStatus, ResearchNote
from agent.data.types import DataQuality
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")


def _make_signal(
    symbol: str = "HDFCBANK",
    action: Action = Action.ENTER_LONG,
    confidence: float = 0.7,
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
        strategy_name="trend_following_v1",
        explanation="EMA21 crossed above EMA50",
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )


def test_decision_status_values() -> None:
    assert DecisionStatus.PENDING == "pending"
    assert DecisionStatus.WAIT_FOR_CONFIRMATION == "wait_for_confirmation"
    assert DecisionStatus.SKIPPED == "skipped"


def test_research_note_construction() -> None:
    note = ResearchNote(
        signal_id="HDFCBANK:enter_long:2024-01-02T09:15:00+05:30",
        bullish_case="Strong trend, high ADX.",
        bearish_case="Overbought in short term.",
        dominant_risk="Reversal if FII selling picks up.",
        regime_fit_assessment="Trending regime suits strategy.",
        confidence_qualitative="HIGH",
        veto=False,
        veto_reason=None,
        model_used="claude-haiku-4-5-20251001",
        tokens_used=150,
        cached=False,
    )
    assert note.veto is False
    assert note.veto_reason is None


def test_research_note_is_frozen() -> None:
    note = ResearchNote(
        signal_id="id",
        bullish_case="b",
        bearish_case="br",
        dominant_risk="r",
        regime_fit_assessment="a",
        confidence_qualitative="LOW",
        veto=False,
        veto_reason=None,
        model_used="claude-haiku-4-5-20251001",
        tokens_used=100,
        cached=False,
    )
    with pytest.raises(Exception):
        note.veto = True  # type: ignore[misc]


def test_decision_pending_construction() -> None:
    sig = _make_signal()
    d = Decision(
        signal=sig,
        status=DecisionStatus.PENDING,
        signal_id="HDFCBANK:enter_long:2024-01-02T09:15:00+05:30",
        merged_from=("trend_following_v1",),
        research_note=None,
        skip_reason="",
        timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    assert d.status == DecisionStatus.PENDING
    assert d.skip_reason == ""


def test_decision_is_frozen() -> None:
    sig = _make_signal()
    d = Decision(
        signal=sig,
        status=DecisionStatus.PENDING,
        signal_id="id",
        merged_from=("trend_following_v1",),
        research_note=None,
        skip_reason="",
        timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    with pytest.raises(Exception):
        d.status = DecisionStatus.SKIPPED  # type: ignore[misc]


def test_decision_skipped_has_reason() -> None:
    sig = _make_signal()
    d = Decision(
        signal=sig,
        status=DecisionStatus.SKIPPED,
        signal_id="id",
        merged_from=("trend_following_v1",),
        research_note=None,
        skip_reason="Already holding position in HDFCBANK",
        timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    assert "HDFCBANK" in d.skip_reason


def test_decision_with_veto_research_note() -> None:
    sig = _make_signal()
    note = ResearchNote(
        signal_id="id",
        bullish_case="b",
        bearish_case="br",
        dominant_risk="r",
        regime_fit_assessment="a",
        confidence_qualitative="MEDIUM",
        veto=True,
        veto_reason="High risk given current macro environment",
        model_used="claude-haiku-4-5-20251001",
        tokens_used=200,
        cached=True,
    )
    d = Decision(
        signal=sig,
        status=DecisionStatus.WAIT_FOR_CONFIRMATION,
        signal_id="id",
        merged_from=("trend_following_v1",),
        research_note=note,
        skip_reason="AI veto: High risk given current macro environment",
        timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    assert d.research_note is not None
    assert d.research_note.veto is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/tatsatshah/Desktop/yegedge
source .venv/bin/activate
python -m pytest tests/decision/test_types.py -v --no-cov 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'agent.decision'`

- [ ] **Step 3: Create package skeletons**

Create `agent/decision/__init__.py` with content `# intentionally empty`.
Create `tests/decision/__init__.py` with content `# intentionally empty`.

- [ ] **Step 4: Write `agent/decision/types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from agent.strategies.types import Signal


class DecisionStatus(StrEnum):
    PENDING = "pending"
    WAIT_FOR_CONFIRMATION = "wait_for_confirmation"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ResearchNote:
    """Schema-bounded AI output consumed by DecisionEngine.

    Defined here so the decision engine can consume research notes without
    importing from agent/ai/ (which is built in Phase 7). The AI layer will
    produce instances of this exact type via a Pydantic output schema.

    confidence_qualitative is one of: "LOW", "MEDIUM", "HIGH".
    veto_reason is None when veto is False.
    """

    signal_id: str
    bullish_case: str
    bearish_case: str
    dominant_risk: str
    regime_fit_assessment: str
    confidence_qualitative: str
    veto: bool
    veto_reason: str | None
    model_used: str
    tokens_used: int
    cached: bool


@dataclass(frozen=True, slots=True)
class Decision:
    """Output of DecisionEngine.evaluate(). Consumed by RiskManager.evaluate().

    status=PENDING: forward to risk manager.
    status=WAIT_FOR_CONFIRMATION: AI vetoed; reconsider next bar without skip.
    status=SKIPPED: suppressed by dedup or portfolio context; journal only.

    skip_reason is empty string when status=PENDING.
    merged_from lists all strategy names that contributed signals for this
    (symbol, action) pair before deduplication selected the best one.
    signal_id is "{symbol}:{action}:{best_signal.timestamp.isoformat()}".
    """

    signal: Signal
    status: DecisionStatus
    signal_id: str
    merged_from: tuple[str, ...]
    research_note: ResearchNote | None
    skip_reason: str
    timestamp: datetime
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/decision/test_types.py -v --no-cov
```

Expected: `7 passed`

- [ ] **Step 6: Commit**

```bash
git add agent/decision/__init__.py agent/decision/types.py \
        tests/decision/__init__.py tests/decision/test_types.py
git commit -m "feat(decision): add DecisionStatus, ResearchNote, and Decision types"
```

---

## Task 2: DecisionEngine

**Files:**
- Create: `agent/decision/engine.py`
- Test: `tests/decision/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/decision/test_engine.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.decision.engine import DecisionEngine
from agent.decision.types import DecisionStatus, ResearchNote
from agent.data.types import DataQuality, Position
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
        engine.evaluate(
            [], _empty_portfolio(), evaluation_time=datetime(2024, 1, 2, 10, 0)
        )


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/decision/test_engine.py -v --no-cov 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'DecisionEngine'`

- [ ] **Step 3: Write `agent/decision/engine.py`**

```python
from __future__ import annotations

from datetime import datetime

import structlog

from agent.decision.types import Decision, DecisionStatus, ResearchNote
from agent.risk.types import PortfolioState
from agent.strategies.types import Action, Signal

logger = structlog.get_logger()


class DecisionEngine:
    """Pure aggregation layer: list[Signal] → list[Decision].

    Takes signals from one or more strategies for a single evaluation cycle,
    deduplicates by (symbol, action), applies portfolio context, handles AI
    research note vetoes, and returns one Decision per unique (symbol, action)
    pair — including SKIPPED decisions so every signal appears in the journal.

    No I/O. No side effects. Pure function over its inputs.
    """

    def evaluate(
        self,
        signals: list[Signal],
        portfolio: PortfolioState,
        research_notes: dict[str, ResearchNote] | None = None,
        *,
        evaluation_time: datetime,
    ) -> list[Decision]:
        """Process one evaluation cycle.

        Parameters
        ----------
        signals:
            All signals from all strategies for this bar.
        portfolio:
            Current portfolio snapshot.
        research_notes:
            Optional mapping of signal_id → ResearchNote from the AI layer.
            Signals with no matching note proceed without AI input.
            signal_id format: "{symbol}:{action}:{signal.timestamp.isoformat()}"
        evaluation_time:
            IST-aware datetime for this evaluation cycle.

        Returns
        -------
        list[Decision]
            One Decision per unique (symbol, action) pair.
        """
        if evaluation_time.tzinfo is None:
            raise ValueError("evaluation_time must be IST-aware")

        notes = research_notes or {}
        decisions: list[Decision] = []

        # Step 1: Filter HOLD signals — carry no actionable intent.
        actionable = [s for s in signals if s.action != Action.HOLD]

        # Step 2: Deduplicate by (symbol, action): highest confidence wins.
        groups: dict[tuple[str, str], list[Signal]] = {}
        for sig in actionable:
            key = (sig.symbol, str(sig.action))
            groups.setdefault(key, []).append(sig)

        for (symbol, action_str), group in groups.items():
            best = max(group, key=lambda s: s.confidence)
            merged_from = tuple(sorted({s.strategy_name for s in group}))
            signal_id = f"{symbol}:{action_str}:{best.timestamp.isoformat()}"
            note = notes.get(signal_id)

            # Step 3: Portfolio context — suppress duplicate ENTER_LONG.
            if action_str == Action.ENTER_LONG and symbol in portfolio.positions:
                decisions.append(
                    Decision(
                        signal=best,
                        status=DecisionStatus.SKIPPED,
                        signal_id=signal_id,
                        merged_from=merged_from,
                        research_note=note,
                        skip_reason=f"Already holding position in {symbol}",
                        timestamp=evaluation_time,
                    )
                )
                continue

            # Step 4: Veto handling — downgrade for AI-flagged risk.
            if note is not None and note.veto:
                decisions.append(
                    Decision(
                        signal=best,
                        status=DecisionStatus.WAIT_FOR_CONFIRMATION,
                        signal_id=signal_id,
                        merged_from=merged_from,
                        research_note=note,
                        skip_reason=f"AI veto: {note.veto_reason or 'no reason given'}",
                        timestamp=evaluation_time,
                    )
                )
                continue

            # Step 5: Approve for risk manager.
            decisions.append(
                Decision(
                    signal=best,
                    status=DecisionStatus.PENDING,
                    signal_id=signal_id,
                    merged_from=merged_from,
                    research_note=note,
                    skip_reason="",
                    timestamp=evaluation_time,
                )
            )

        logger.debug(
            "decision_engine.evaluate.done",
            total_signals=len(signals),
            decisions=len(decisions),
            pending=sum(1 for d in decisions if d.status == DecisionStatus.PENDING),
            skipped=sum(1 for d in decisions if d.status == DecisionStatus.SKIPPED),
            vetoed=sum(
                1
                for d in decisions
                if d.status == DecisionStatus.WAIT_FOR_CONFIRMATION
            ),
        )
        return decisions
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/decision/test_engine.py -v --no-cov
```

Expected: `20 passed`

- [ ] **Step 5: Commit**

```bash
git add agent/decision/engine.py tests/decision/test_engine.py
git commit -m "feat(decision): add DecisionEngine with dedup, portfolio context, and veto handling"
```

---

## Task 3: Full Test Suite + Coverage Gate

**Files:**
- Verify: all existing test files pass together with coverage

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/tatsatshah/Desktop/yegedge
source .venv/bin/activate
python -m pytest tests/ -v --cov=agent --cov-report=term-missing
```

Expected: **224+ tests pass** (197 from Phases 1–4 + 27 new). Coverage ≥ 70% total; `agent/decision/` should be ≥ 90%.

- [ ] **Step 2: Run linters**

```bash
ruff check agent/decision/ tests/decision/
black --check agent/decision/ tests/decision/
```

Expected: no issues. If ruff or black flags anything, fix before continuing.

- [ ] **Step 3: Run end-to-end spot-check**

Run this inline script to verify the full pipeline → strategy → decision engine → risk manager chain:

```bash
python - <<'EOF'
from __future__ import annotations
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo
import polars as pl

from agent.features.pipeline import FeaturePipeline
from agent.strategies.trend_following import TrendFollowingStrategy
from agent.decision.engine import DecisionEngine
from agent.decision.types import DecisionStatus
from agent.risk.manager import RiskManager
from agent.risk.rules import load_risk_rules
from agent.risk.types import PortfolioState
from agent.data.types import DataQuality

IST = ZoneInfo("Asia/Kolkata")
EVAL_TIME = datetime(2024, 1, 2, 10, 15, tzinfo=IST)

# 80-bar V-shape to force a golden cross
n = 80
base = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
timestamps = [base + timedelta(hours=i) for i in range(n)]
closes = [120.0 - i * 0.8 for i in range(40)] + [88.0 + i * 1.2 for i in range(40)]

df = pl.DataFrame({
    "symbol": ["HDFCBANK"] * n,
    "timeframe": ["60m"] * n,
    "timestamp": timestamps,
    "open": [c - 0.5 for c in closes],
    "high": [c + 2.0 for c in closes],
    "low": [c - 2.0 for c in closes],
    "close": closes,
    "volume": [200_000] * n,
    "value": [c * 200_000 for c in closes],
    "data_quality": [DataQuality.OK.value] * n,
})

enriched = FeaturePipeline().run(df)
signals = TrendFollowingStrategy().generate(enriched)
print(f"Signals: {len(signals)}")

portfolio = PortfolioState(
    nav=Decimal("200000"), cash=Decimal("180000"), positions={},
    daily_pnl=Decimal("0"), weekly_pnl=Decimal("0"), peak_nav=Decimal("200000"),
    orders_today=0, last_order_time={}, kill_switch_active=False,
    evaluation_time=EVAL_TIME,
)

engine = DecisionEngine()
decisions = engine.evaluate(signals, portfolio, evaluation_time=EVAL_TIME)
print(f"Decisions: {len(decisions)}")
for d in decisions:
    print(f"  [{d.signal.action}] status={d.status} merged_from={d.merged_from}")

rm = RiskManager(load_risk_rules())
for d in decisions:
    if d.status == DecisionStatus.PENDING:
        entry = d.signal.suggested_stop + Decimal("5")
        risk_decision = rm.evaluate(d.signal, portfolio, entry)
        print(f"  RiskDecision: verdict={risk_decision.verdict} qty={risk_decision.quantity} reason={risk_decision.rejection_reason}")

print("END-TO-END SPOT-CHECK PASSED")
EOF
```

Expected: prints signal/decision counts, no exceptions, ends with `END-TO-END SPOT-CHECK PASSED`.

- [ ] **Step 4: Commit**

```bash
git add -p  # review any remaining unstaged changes
git commit -m "test(decision): Phase 5 full suite passes coverage gate (≥70%), linters clean"
```

---

## Review Priority

1. **Task 1 (types)** — foundation for Task 2; review `signal_id` format convention before Task 2 runs.
2. **Task 2 (engine)** — core logic; verify dedup selects `max(confidence)` not first-seen, and that EXIT_LONG bypasses the portfolio-context check.
3. **Task 3 (full suite)** — coverage confirmation only; no new code.

## Self-review checklist

- [ ] `signal_id` format `"{symbol}:{action}:{timestamp.isoformat()}"` is consistent between `types.py` docstring, `engine.py` computation, and all test fixtures.
- [ ] `merged_from` is `tuple[str, ...]` (immutable) — confirmed in `types.py` and populated with `tuple(sorted(...))` in `engine.py`.
- [ ] EXIT_LONG signals bypass the portfolio-position check (only ENTER_LONG is guarded).
- [ ] `evaluate()` raises `ValueError` on naive `evaluation_time` before doing any other work.
- [ ] `research_notes=None` is handled by `notes = research_notes or {}` — no `KeyError` when notes are absent.
- [ ] All dataclasses use `frozen=True, slots=True`.
- [ ] `from __future__ import annotations` is the first line of every `.py` file.
- [ ] `logger = structlog.get_logger()` (not `log`).
