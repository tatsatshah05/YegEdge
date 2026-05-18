# Phase 7 — Paper Execution + Portfolio + Journal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three tightly coupled modules: `agent/execution/` (paper trade fills), `agent/portfolio/` (position and P&L tracking), and `agent/journal/` (append-only SQLite audit log). Together they form the complete paper-trading loop.

**Architecture:** `PaperExecution.submit(decision, risk_decision) → Fill` simulates an immediate fill at entry_price. `PortfolioTracker.apply_fill(fill)` updates position state and returns an immutable `PortfolioState` snapshot. `JournalStore.log(entry)` appends every event to SQLite — signals, decisions, fills, rejections — so nothing is ever silently dropped. All three modules are pure over their inputs except for SQLite I/O.

**Tech Stack:** Python 3.11+, `dataclasses`, `enum.StrEnum`, `sqlite3` (stdlib — no ORM for the append-only log), `structlog`, `pytest`.

---

## Context for subagent workers

**Project:** `/Users/tatsatshah/Desktop/yegedge`
**Branch:** `phase-2-feature-engineering`
**Virtualenv:** `source /Users/tatsatshah/Desktop/yegedge/.venv/bin/activate`

**Conventions:**
- `from __future__ import annotations` first line every `.py`
- `logger = structlog.get_logger()` (not `log`)
- `@dataclass(frozen=True, slots=True)` on all dataclasses
- No `print()` — structlog only
- Monetary: `Decimal`. Ratios: `float`. IST-aware `datetime` everywhere.

**Key types already defined — do NOT redefine:**

```python
# agent/data/types.py
@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    quantity: int       # negative = short (not used in V1 — MIS long only)
    average_price: Decimal
    product: Literal["MIS", "CNC"]

# agent/risk/types.py
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

@dataclass(frozen=True, slots=True)
class RiskDecision:
    verdict: RiskVerdict
    quantity: int
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal
    risk_per_share: Decimal
    position_value: Decimal
    rejection_reason: RejectionReason
    rejection_detail: str
    signal: Signal

# agent/decision/types.py
@dataclass(frozen=True, slots=True)
class Decision:
    signal: Signal
    status: DecisionStatus
    signal_id: str
    merged_from: tuple[str, ...]
    research_note: ResearchNote | None
    skip_reason: str
    timestamp: datetime

# agent/strategies/types.py
class Action(StrEnum):
    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"
    HOLD = "hold"
```

---

## File Map

```
agent/execution/
    __init__.py        — empty
    types.py           — Fill dataclass, ExecutionMode StrEnum
    paper.py           — PaperExecution class

agent/portfolio/
    __init__.py        — empty
    tracker.py         — PortfolioTracker class

agent/journal/
    __init__.py        — empty
    types.py           — JournalEntry dataclass, JournalEntryType StrEnum
    store.py           — JournalStore (SQLite, append-only)

tests/execution/
    __init__.py        — empty
    test_types.py      — 5 tests
    test_paper.py      — 7 tests

tests/portfolio/
    __init__.py        — empty
    test_tracker.py    — 10 tests

tests/journal/
    __init__.py        — empty
    test_store.py      — 8 tests
```

---

## Task 1: Execution Types + PaperExecution

**Files:**
- Create: `agent/execution/__init__.py`
- Create: `agent/execution/types.py`
- Create: `agent/execution/paper.py`
- Create: `tests/execution/__init__.py`
- Test: `tests/execution/test_types.py`
- Test: `tests/execution/test_paper.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/execution/test_types.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.execution.types import ExecutionMode, Fill
from agent.strategies.types import Action

IST = ZoneInfo("Asia/Kolkata")


def _make_fill(
    symbol: str = "HDFCBANK",
    action: Action = Action.ENTER_LONG,
    quantity: int = 10,
    fill_price: Decimal = Decimal("1710.00"),
) -> Fill:
    return Fill(
        order_id="paper-HDFCBANK-20240102091500",
        symbol=symbol,
        action=action,
        quantity=quantity,
        fill_price=fill_price,
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
        signal_id="HDFCBANK:enter_long:2024-01-02T09:15:00+05:30",
        strategy_name="trend_following_v1",
        execution_mode=ExecutionMode.PAPER,
    )


def test_execution_mode_values() -> None:
    assert ExecutionMode.PAPER == "paper"
    assert ExecutionMode.LIVE == "live"


def test_fill_construction() -> None:
    fill = _make_fill()
    assert fill.symbol == "HDFCBANK"
    assert fill.quantity == 10
    assert fill.execution_mode == ExecutionMode.PAPER


def test_fill_is_frozen() -> None:
    fill = _make_fill()
    with pytest.raises(AttributeError):
        fill.quantity = 20  # type: ignore[misc]


def test_fill_exit_long_negative_quantity_convention() -> None:
    fill = _make_fill(action=Action.EXIT_LONG, quantity=10)
    # EXIT_LONG fills carry positive quantity — the caller decides the sign convention.
    assert fill.quantity == 10
    assert fill.action == Action.EXIT_LONG


def test_fill_order_id_is_string() -> None:
    fill = _make_fill()
    assert isinstance(fill.order_id, str)
    assert len(fill.order_id) > 0
```

```python
# tests/execution/test_paper.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.decision.types import Decision, DecisionStatus, ResearchNote
from agent.execution.paper import PaperExecution
from agent.execution.types import ExecutionMode, Fill
from agent.data.types import DataQuality
from agent.risk.types import RiskDecision, RiskVerdict, RejectionReason, PortfolioState
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")

EVAL_TIME = datetime(2024, 1, 2, 10, 0, tzinfo=IST)


def _make_signal(symbol: str = "HDFCBANK", action: Action = Action.ENTER_LONG) -> Signal:
    return Signal(
        symbol=symbol, action=action, confidence=0.75,
        suggested_stop=Decimal("1680.00"), suggested_target=Decimal("1760.00"),
        invalidation_condition="Close below EMA21", expected_r=2.0,
        time_horizon_hours=4, regime_fit=0.8, data_quality=DataQuality.OK,
        strategy_name="trend_following_v1",
        explanation="EMA21 crossed above EMA50",
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )


def _make_decision(signal: Signal) -> Decision:
    return Decision(
        signal=signal, status=DecisionStatus.PENDING,
        signal_id=f"{signal.symbol}:{signal.action}:{signal.timestamp.isoformat()}",
        merged_from=("trend_following_v1",), research_note=None, skip_reason="",
        timestamp=EVAL_TIME,
    )


def _make_risk_decision(
    signal: Signal,
    quantity: int = 10,
    entry_price: Decimal = Decimal("1710.00"),
) -> RiskDecision:
    return RiskDecision(
        verdict=RiskVerdict.APPROVED, quantity=quantity,
        entry_price=entry_price,
        stop_price=Decimal("1680.00"), target_price=Decimal("1760.00"),
        risk_per_share=Decimal("30.00"),
        position_value=entry_price * quantity,
        rejection_reason=RejectionReason.NONE, rejection_detail="",
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/tatsatshah/Desktop/yegedge && source .venv/bin/activate
python -m pytest tests/execution/ -v --no-cov 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'agent.execution'`

- [ ] **Step 3: Create package skeletons**

Create `agent/execution/__init__.py` and `tests/execution/__init__.py` — both `# intentionally empty`.

- [ ] **Step 4: Write `agent/execution/types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from agent.strategies.types import Action


class ExecutionMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class Fill:
    """Record of a simulated (paper) or real (live) order fill.

    order_id: deterministic string, stable across retries.
    quantity: always positive — action field carries direction.
    execution_mode: always PAPER in Phase 7; LIVE gated by LIVE_TRADING_ENABLED.
    """

    order_id: str
    symbol: str
    action: Action
    quantity: int
    fill_price: Decimal
    timestamp: datetime
    signal_id: str
    strategy_name: str
    execution_mode: ExecutionMode
```

- [ ] **Step 5: Write `agent/execution/paper.py`**

```python
from __future__ import annotations

from datetime import datetime

import structlog

from agent.decision.types import Decision
from agent.execution.types import ExecutionMode, Fill
from agent.risk.types import RiskDecision

logger = structlog.get_logger()


class PaperExecution:
    """Simulate paper trade fills. No broker calls; fills are instantaneous at entry_price.

    submit() is idempotent: the same (decision, risk_decision, submitted_at) always
    produces the same order_id, so retries are safe.
    """

    def submit(
        self,
        decision: Decision,
        risk_decision: RiskDecision,
        *,
        submitted_at: datetime,
    ) -> Fill:
        """Simulate an immediate fill at risk_decision.entry_price.

        Parameters
        ----------
        decision:
            The approved decision from DecisionEngine.
        risk_decision:
            The approved RiskDecision containing quantity and entry_price.
        submitted_at:
            IST-aware evaluation time (used in order_id and fill timestamp).
        """
        order_id = (
            f"paper-{decision.signal.symbol}"
            f"-{submitted_at.strftime('%Y%m%d%H%M%S')}"
            f"-{decision.signal_id[-8:]}"
        )

        fill = Fill(
            order_id=order_id,
            symbol=decision.signal.symbol,
            action=decision.signal.action,
            quantity=risk_decision.quantity,
            fill_price=risk_decision.entry_price,
            timestamp=submitted_at,
            signal_id=decision.signal_id,
            strategy_name=decision.signal.strategy_name,
            execution_mode=ExecutionMode.PAPER,
        )

        logger.info(
            "paper_execution.filled",
            order_id=order_id,
            symbol=fill.symbol,
            action=str(fill.action),
            quantity=fill.quantity,
            price=str(fill.fill_price),
        )
        return fill
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/execution/ -v --no-cov
```

Expected: `12 passed`

- [ ] **Step 7: Commit**

```bash
git add agent/execution/__init__.py agent/execution/types.py agent/execution/paper.py \
        tests/execution/__init__.py tests/execution/test_types.py tests/execution/test_paper.py
git commit -m "feat(execution): add Fill type, ExecutionMode, and PaperExecution"
```

---

## Task 2: PortfolioTracker

**Files:**
- Create: `agent/portfolio/__init__.py`
- Create: `agent/portfolio/tracker.py`
- Create: `tests/portfolio/__init__.py`
- Test: `tests/portfolio/test_tracker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/portfolio/test_tracker.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.execution.types import ExecutionMode, Fill
from agent.portfolio.tracker import PortfolioTracker
from agent.strategies.types import Action

IST = ZoneInfo("Asia/Kolkata")

T0 = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
T1 = datetime(2024, 1, 2, 10, 15, tzinfo=IST)
T2 = datetime(2024, 1, 2, 11, 15, tzinfo=IST)

INITIAL_NAV = Decimal("100000")


def _make_fill(
    symbol: str = "HDFCBANK",
    action: Action = Action.ENTER_LONG,
    quantity: int = 10,
    price: Decimal = Decimal("1700.00"),
    ts: datetime = T0,
) -> Fill:
    return Fill(
        order_id=f"paper-{symbol}-{ts.strftime('%H%M%S')}",
        symbol=symbol,
        action=action,
        quantity=quantity,
        fill_price=price,
        timestamp=ts,
        signal_id=f"{symbol}:enter_long:{ts.isoformat()}",
        strategy_name="trend_following_v1",
        execution_mode=ExecutionMode.PAPER,
    )


def _tracker() -> PortfolioTracker:
    return PortfolioTracker(
        initial_nav=INITIAL_NAV,
        initial_cash=INITIAL_NAV,
        start_time=T0,
    )


def test_initial_state_has_no_positions() -> None:
    tracker = _tracker()
    state = tracker.state
    assert len(state.positions) == 0
    assert state.nav == INITIAL_NAV
    assert state.cash == INITIAL_NAV


def test_apply_enter_long_reduces_cash() -> None:
    tracker = _tracker()
    fill = _make_fill(quantity=10, price=Decimal("1700"))
    state = tracker.apply_fill(fill, evaluation_time=T0)
    expected_cash = INITIAL_NAV - (10 * Decimal("1700"))
    assert state.cash == expected_cash


def test_apply_enter_long_creates_position() -> None:
    tracker = _tracker()
    fill = _make_fill(symbol="HDFCBANK", quantity=10, price=Decimal("1700"))
    state = tracker.apply_fill(fill, evaluation_time=T0)
    assert "HDFCBANK" in state.positions
    assert state.positions["HDFCBANK"].quantity == 10
    assert state.positions["HDFCBANK"].average_price == Decimal("1700")


def test_apply_exit_long_removes_position_and_adds_cash() -> None:
    tracker = _tracker()
    tracker.apply_fill(_make_fill(quantity=10, price=Decimal("1700")), evaluation_time=T0)
    exit_fill = _make_fill(action=Action.EXIT_LONG, quantity=10, price=Decimal("1750"), ts=T1)
    state = tracker.apply_fill(exit_fill, evaluation_time=T1)
    assert "HDFCBANK" not in state.positions
    # Cash should be roughly initial - buy + sell proceeds
    expected_cash = INITIAL_NAV - 10 * Decimal("1700") + 10 * Decimal("1750")
    assert state.cash == expected_cash


def test_daily_pnl_positive_on_profitable_exit() -> None:
    tracker = _tracker()
    tracker.apply_fill(_make_fill(quantity=10, price=Decimal("1700")), evaluation_time=T0)
    exit_fill = _make_fill(action=Action.EXIT_LONG, quantity=10, price=Decimal("1750"), ts=T1)
    state = tracker.apply_fill(exit_fill, evaluation_time=T1)
    assert state.daily_pnl == Decimal("500")  # 10 * (1750 - 1700)


def test_mark_to_market_updates_nav() -> None:
    tracker = _tracker()
    tracker.apply_fill(_make_fill(quantity=10, price=Decimal("1700")), evaluation_time=T0)
    state = tracker.mark_to_market({"HDFCBANK": Decimal("1800")}, evaluation_time=T1)
    expected_nav = (INITIAL_NAV - 10 * Decimal("1700")) + 10 * Decimal("1800")
    assert state.nav == expected_nav


def test_orders_today_increments_per_fill() -> None:
    tracker = _tracker()
    tracker.apply_fill(_make_fill(symbol="HDFCBANK"), evaluation_time=T0)
    state = tracker.apply_fill(_make_fill(symbol="TCS"), evaluation_time=T0)
    assert state.orders_today == 2


def test_peak_nav_tracks_high_water_mark() -> None:
    tracker = _tracker()
    tracker.apply_fill(_make_fill(quantity=10, price=Decimal("1700")), evaluation_time=T0)
    state = tracker.mark_to_market({"HDFCBANK": Decimal("1900")}, evaluation_time=T1)
    peak_after_gain = state.peak_nav
    state2 = tracker.mark_to_market({"HDFCBANK": Decimal("1600")}, evaluation_time=T2)
    # Peak should not decrease when NAV falls
    assert state2.peak_nav == peak_after_gain


def test_last_order_time_recorded() -> None:
    tracker = _tracker()
    fill = _make_fill(ts=T0)
    state = tracker.apply_fill(fill, evaluation_time=T0)
    assert "HDFCBANK" in state.last_order_time
    assert state.last_order_time["HDFCBANK"] == T0


def test_kill_switch_defaults_false() -> None:
    tracker = _tracker()
    assert tracker.state.kill_switch_active is False


def test_multiple_symbols_tracked_independently() -> None:
    tracker = _tracker()
    tracker.apply_fill(_make_fill(symbol="HDFCBANK", quantity=10, price=Decimal("1700")), evaluation_time=T0)
    state = tracker.apply_fill(_make_fill(symbol="TCS", quantity=5, price=Decimal("3000")), evaluation_time=T0)
    assert "HDFCBANK" in state.positions
    assert "TCS" in state.positions
    assert state.positions["HDFCBANK"].quantity == 10
    assert state.positions["TCS"].quantity == 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/portfolio/test_tracker.py -v --no-cov 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'agent.portfolio'`

- [ ] **Step 3: Create package skeleton**

Create `agent/portfolio/__init__.py` and `tests/portfolio/__init__.py` — both `# intentionally empty`.

- [ ] **Step 4: Write `agent/portfolio/tracker.py`**

```python
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import structlog

from agent.data.types import Position
from agent.execution.types import Fill
from agent.risk.types import PortfolioState
from agent.strategies.types import Action

logger = structlog.get_logger()


class PortfolioTracker:
    """Mutable tracker that produces immutable PortfolioState snapshots.

    apply_fill() updates internal state and returns a snapshot.
    mark_to_market() revalues open positions at current prices and returns a snapshot.
    state property returns the latest snapshot without modifying internal state.
    """

    def __init__(
        self,
        initial_nav: Decimal,
        initial_cash: Decimal,
        start_time: datetime,
    ) -> None:
        self._cash = initial_cash
        self._initial_nav = initial_nav
        self._peak_nav = initial_nav
        self._daily_pnl = Decimal("0")
        self._weekly_pnl = Decimal("0")
        self._positions: dict[str, Position] = {}
        self._orders_today: int = 0
        self._last_order_time: dict[str, datetime] = {}
        self._kill_switch_active: bool = False
        self._evaluation_time: datetime = start_time

    def apply_fill(self, fill: Fill, *, evaluation_time: datetime) -> PortfolioState:
        """Apply a fill to internal state and return a PortfolioState snapshot.

        ENTER_LONG: deduct cash, create/increase position at fill_price.
        EXIT_LONG:  add cash proceeds, realize P&L, remove/reduce position.
        """
        self._evaluation_time = evaluation_time
        self._orders_today += 1
        self._last_order_time[fill.symbol] = fill.timestamp

        if fill.action == Action.ENTER_LONG:
            cost = fill.fill_price * fill.quantity
            self._cash -= cost
            existing = self._positions.get(fill.symbol)
            if existing is None:
                self._positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    quantity=fill.quantity,
                    average_price=fill.fill_price,
                    product="MIS",
                )
            else:
                # Weighted average for add-on entries
                total_qty = existing.quantity + fill.quantity
                avg_price = (
                    existing.average_price * existing.quantity
                    + fill.fill_price * fill.quantity
                ) / Decimal(str(total_qty))
                self._positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    quantity=total_qty,
                    average_price=avg_price.quantize(Decimal("0.01")),
                    product="MIS",
                )

        elif fill.action == Action.EXIT_LONG:
            existing = self._positions.get(fill.symbol)
            if existing is not None:
                realized_pnl = (fill.fill_price - existing.average_price) * fill.quantity
                self._daily_pnl += realized_pnl
                self._weekly_pnl += realized_pnl
                proceeds = fill.fill_price * fill.quantity
                self._cash += proceeds
                remaining_qty = existing.quantity - fill.quantity
                if remaining_qty <= 0:
                    del self._positions[fill.symbol]
                else:
                    self._positions[fill.symbol] = Position(
                        symbol=fill.symbol,
                        quantity=remaining_qty,
                        average_price=existing.average_price,
                        product="MIS",
                    )

        nav = self._compute_nav()
        if nav > self._peak_nav:
            self._peak_nav = nav

        logger.debug(
            "portfolio_tracker.apply_fill",
            symbol=fill.symbol,
            action=str(fill.action),
            quantity=fill.quantity,
            nav=str(nav),
            cash=str(self._cash),
        )
        return self._snapshot(nav)

    def mark_to_market(
        self,
        prices: dict[str, Decimal],
        *,
        evaluation_time: datetime,
    ) -> PortfolioState:
        """Revalue open positions at current prices. Does not modify positions."""
        self._evaluation_time = evaluation_time
        nav = self._cash + sum(
            pos.average_price * pos.quantity
            if prices.get(pos.symbol) is None
            else prices[pos.symbol] * pos.quantity
            for pos in self._positions.values()
        )
        if nav > self._peak_nav:
            self._peak_nav = nav
        return self._snapshot(nav)

    @property
    def state(self) -> PortfolioState:
        return self._snapshot(self._compute_nav())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_nav(self) -> Decimal:
        return self._cash + sum(
            pos.average_price * pos.quantity for pos in self._positions.values()
        )

    def _snapshot(self, nav: Decimal) -> PortfolioState:
        return PortfolioState(
            nav=nav,
            cash=self._cash,
            positions=dict(self._positions),
            daily_pnl=self._daily_pnl,
            weekly_pnl=self._weekly_pnl,
            peak_nav=self._peak_nav,
            orders_today=self._orders_today,
            last_order_time=dict(self._last_order_time),
            kill_switch_active=self._kill_switch_active,
            evaluation_time=self._evaluation_time,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/portfolio/test_tracker.py -v --no-cov
```

Expected: `10 passed`

- [ ] **Step 6: Commit**

```bash
git add agent/portfolio/__init__.py agent/portfolio/tracker.py \
        tests/portfolio/__init__.py tests/portfolio/test_tracker.py
git commit -m "feat(portfolio): add PortfolioTracker with fill application and mark-to-market"
```

---

## Task 3: Journal Store

**Files:**
- Create: `agent/journal/__init__.py`
- Create: `agent/journal/types.py`
- Create: `agent/journal/store.py`
- Create: `tests/journal/__init__.py`
- Test: `tests/journal/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/journal/test_store.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent.journal.types import JournalEntry, JournalEntryType
from agent.journal.store import JournalStore

IST = ZoneInfo("Asia/Kolkata")

T0 = datetime(2024, 1, 2, 9, 15, tzinfo=IST)


def _make_entry(
    entry_type: JournalEntryType = JournalEntryType.FILL,
    symbol: str = "HDFCBANK",
    payload: str = '{"action": "enter_long", "quantity": 10}',
) -> JournalEntry:
    return JournalEntry(
        entry_id="test-entry-001",
        timestamp=T0,
        entry_type=entry_type,
        symbol=symbol,
        payload=payload,
    )


def test_journal_entry_type_values() -> None:
    assert JournalEntryType.SIGNAL == "signal"
    assert JournalEntryType.DECISION == "decision"
    assert JournalEntryType.FILL == "fill"
    assert JournalEntryType.REJECTION == "rejection"
    assert JournalEntryType.PNL == "pnl"


def test_journal_entry_is_frozen() -> None:
    entry = _make_entry()
    with pytest.raises(AttributeError):
        entry.symbol = "TCS"  # type: ignore[misc]


def test_store_log_and_query_returns_entry(tmp_path: Path) -> None:
    store = JournalStore(db_path=tmp_path / "journal.db")
    entry = _make_entry()
    store.log(entry)
    results = store.query(limit=10)
    assert len(results) == 1
    assert results[0].symbol == "HDFCBANK"


def test_store_query_filter_by_entry_type(tmp_path: Path) -> None:
    store = JournalStore(db_path=tmp_path / "journal.db")
    store.log(_make_entry(entry_type=JournalEntryType.FILL))
    store.log(_make_entry(entry_type=JournalEntryType.SIGNAL, payload="{}"))
    fills = store.query(entry_type=JournalEntryType.FILL)
    assert len(fills) == 1
    assert fills[0].entry_type == JournalEntryType.FILL


def test_store_query_filter_by_symbol(tmp_path: Path) -> None:
    store = JournalStore(db_path=tmp_path / "journal.db")
    store.log(_make_entry(symbol="HDFCBANK"))
    store.log(_make_entry(symbol="TCS", payload="{}"))
    hdfc = store.query(symbol="HDFCBANK")
    assert len(hdfc) == 1
    assert hdfc[0].symbol == "HDFCBANK"


def test_store_is_append_only_and_ordered(tmp_path: Path) -> None:
    store = JournalStore(db_path=tmp_path / "journal.db")
    for i in range(5):
        store.log(JournalEntry(
            entry_id=f"entry-{i:03d}",
            timestamp=T0,
            entry_type=JournalEntryType.FILL,
            symbol="HDFCBANK",
            payload=f'{{"seq": {i}}}',
        ))
    results = store.query(limit=10)
    assert len(results) == 5
    # All entries present — append-only, nothing deleted
    ids = [r.entry_id for r in results]
    for i in range(5):
        assert f"entry-{i:03d}" in ids


def test_store_persists_across_open_close(tmp_path: Path) -> None:
    db_path = tmp_path / "journal.db"
    store1 = JournalStore(db_path=db_path)
    store1.log(_make_entry(payload='{"qty": 5}'))

    store2 = JournalStore(db_path=db_path)
    results = store2.query()
    assert len(results) == 1
    assert results[0].symbol == "HDFCBANK"


def test_store_query_limit_respected(tmp_path: Path) -> None:
    store = JournalStore(db_path=tmp_path / "journal.db")
    for i in range(20):
        store.log(JournalEntry(
            entry_id=f"entry-{i:03d}", timestamp=T0,
            entry_type=JournalEntryType.FILL, symbol="HDFCBANK",
            payload=f'{{"i": {i}}}',
        ))
    results = store.query(limit=5)
    assert len(results) == 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/journal/test_store.py -v --no-cov 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'agent.journal'`

- [ ] **Step 3: Create package skeletons**

Create `agent/journal/__init__.py` and `tests/journal/__init__.py` — both `# intentionally empty`.

- [ ] **Step 4: Write `agent/journal/types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class JournalEntryType(StrEnum):
    SIGNAL = "signal"
    DECISION = "decision"
    FILL = "fill"
    REJECTION = "rejection"
    PNL = "pnl"


@dataclass(frozen=True, slots=True)
class JournalEntry:
    """Append-only audit record.

    payload is a JSON string containing the full event data. Using a string
    keeps the journal schema stable as event types evolve — no migrations needed
    to add fields inside payload.
    """

    entry_id: str       # unique, caller-assigned
    timestamp: datetime  # IST-aware
    entry_type: JournalEntryType
    symbol: str | None  # None for system-level events (e.g., PNL snapshots)
    payload: str        # JSON string
```

- [ ] **Step 5: Write `agent/journal/store.py`**

```python
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import structlog

from agent.journal.types import JournalEntry, JournalEntryType

logger = structlog.get_logger()

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS journal (
    row_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id    TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    entry_type  TEXT NOT NULL,
    symbol      TEXT,
    payload     TEXT NOT NULL
)
"""


class JournalStore:
    """Append-only SQLite journal.

    Uses Python's stdlib sqlite3 — no ORM. The table has no UPDATE or DELETE
    paths; every event is written once and read back exactly as stored.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def log(self, entry: JournalEntry) -> None:
        """Append a JournalEntry. Never raises on valid input."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO journal (entry_id, timestamp, entry_type, symbol, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    entry.entry_id,
                    entry.timestamp.isoformat(),
                    str(entry.entry_type),
                    entry.symbol,
                    entry.payload,
                ),
            )
        logger.debug(
            "journal.log",
            entry_id=entry.entry_id,
            entry_type=str(entry.entry_type),
            symbol=entry.symbol,
        )

    def query(
        self,
        *,
        entry_type: JournalEntryType | None = None,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[JournalEntry]:
        """Return recent entries, newest last, filtered optionally by type/symbol."""
        sql = "SELECT entry_id, timestamp, entry_type, symbol, payload FROM journal"
        conditions: list[str] = []
        params: list[object] = []

        if entry_type is not None:
            conditions.append("entry_type = ?")
            params.append(str(entry_type))
        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY row_id ASC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            JournalEntry(
                entry_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                entry_type=JournalEntryType(row[2]),
                symbol=row[3],
                payload=row[4],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/journal/test_store.py -v --no-cov
```

Expected: `8 passed`

- [ ] **Step 7: Commit**

```bash
git add agent/journal/__init__.py agent/journal/types.py agent/journal/store.py \
        tests/journal/__init__.py tests/journal/test_store.py
git commit -m "feat(journal): add append-only SQLite JournalStore and JournalEntry types"
```

---

## Task 4: Full Test Suite + Integration Test

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/tatsatshah/Desktop/yegedge && source .venv/bin/activate
python -m pytest tests/ -v --cov=agent --cov-report=term-missing 2>&1 | tail -35
```

Expected: **282+ tests pass** (246+ from Phases 1–6 + 35+ new). Coverage ≥ 70% total; `agent/execution/`, `agent/portfolio/`, `agent/journal/` all ≥ 85%.

- [ ] **Step 2: Run linters**

```bash
ruff check agent/execution/ agent/portfolio/ agent/journal/ \
           tests/execution/ tests/portfolio/ tests/journal/ \
  && black --check agent/execution/ agent/portfolio/ agent/journal/ \
           tests/execution/ tests/portfolio/ tests/journal/ \
  && echo CLEAN
```

Expected: no issues.

- [ ] **Step 3: Run integration spot-check**

```bash
python - <<'EOF'
from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo
import tempfile, json

from agent.decision.types import Decision, DecisionStatus
from agent.execution.paper import PaperExecution
from agent.execution.types import ExecutionMode
from agent.journal.store import JournalStore
from agent.journal.types import JournalEntry, JournalEntryType
from agent.portfolio.tracker import PortfolioTracker
from agent.risk.types import RiskDecision, RiskVerdict, RejectionReason, PortfolioState
from agent.data.types import DataQuality
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")
T0 = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
T1 = datetime(2024, 1, 2, 10, 15, tzinfo=IST)

# Build signal, decision, risk_decision
sig = Signal(
    symbol="HDFCBANK", action=Action.ENTER_LONG, confidence=0.75,
    suggested_stop=Decimal("1680"), suggested_target=Decimal("1760"),
    invalidation_condition="Close below EMA21", expected_r=2.0,
    time_horizon_hours=4, regime_fit=0.8, data_quality=DataQuality.OK,
    strategy_name="trend_following_v1",
    explanation="EMA21 crossed above EMA50",
    timestamp=T0,
)
decision = Decision(
    signal=sig, status=DecisionStatus.PENDING,
    signal_id=f"{sig.symbol}:{sig.action}:{sig.timestamp.isoformat()}",
    merged_from=("trend_following_v1",), research_note=None, skip_reason="",
    timestamp=T0,
)
risk_dec = RiskDecision(
    verdict=RiskVerdict.APPROVED, quantity=10,
    entry_price=Decimal("1710"), stop_price=Decimal("1680"),
    target_price=Decimal("1760"), risk_per_share=Decimal("30"),
    position_value=Decimal("17100"), rejection_reason=RejectionReason.NONE,
    rejection_detail="", signal=sig,
)

# Paper execution
executor = PaperExecution()
fill = executor.submit(decision, risk_dec, submitted_at=T0)
assert fill.execution_mode == ExecutionMode.PAPER
assert fill.fill_price == Decimal("1710")
print(f"Fill: {fill.symbol} x{fill.quantity} @ {fill.fill_price}")

# Portfolio tracker
tracker = PortfolioTracker(initial_nav=Decimal("100000"), initial_cash=Decimal("100000"), start_time=T0)
state = tracker.apply_fill(fill, evaluation_time=T0)
assert "HDFCBANK" in state.positions
print(f"Portfolio: nav={state.nav} cash={state.cash} positions={list(state.positions.keys())}")

state2 = tracker.mark_to_market({"HDFCBANK": Decimal("1750")}, evaluation_time=T1)
print(f"After MTM: nav={state2.nav}")
assert state2.nav > state.nav  # price went up

# Journal
with tempfile.TemporaryDirectory() as tmpdir:
    store = JournalStore(db_path=Path(tmpdir) / "journal.db")
    store.log(JournalEntry(
        entry_id=fill.order_id,
        timestamp=fill.timestamp,
        entry_type=JournalEntryType.FILL,
        symbol=fill.symbol,
        payload=json.dumps({"action": str(fill.action), "quantity": fill.quantity, "price": str(fill.fill_price)}),
    ))
    entries = store.query()
    assert len(entries) == 1
    assert entries[0].symbol == "HDFCBANK"
    print(f"Journal: {len(entries)} entry logged")

print("END-TO-END INTEGRATION PASSED")
EOF
```

Expected: prints fill/portfolio/journal lines and ends with `END-TO-END INTEGRATION PASSED`.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-05-17-phase-7-execution-portfolio-journal.md
git commit -m "test(execution/portfolio/journal): Phase 7 full suite passes coverage gate (≥70%)"
```

---

## Self-review checklist

- [ ] `Fill.quantity` is always positive — action carries direction
- [ ] `PortfolioTracker.apply_fill` EXIT_LONG: adds full proceeds to cash, removes position
- [ ] `PortfolioTracker._compute_nav` uses `average_price` for open positions (not mark-to-market) — mark_to_market is explicit
- [ ] `JournalStore` uses WAL mode for SQLite — prevents corruption on concurrent reads
- [ ] `JournalEntry.payload` is a JSON string — schema-stable as event structure evolves
- [ ] No `print()` anywhere — structlog only
- [ ] `from __future__ import annotations` first line in every `.py`
