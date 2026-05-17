# Phase 4 — Risk Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Risk Manager — the inviolable gatekeeper that approves or rejects every `Signal` before it reaches the execution layer, enforcing all rules from `config/risk_rules.yaml`.

**Architecture:** `RiskManager.evaluate(signal, portfolio_state, entry_price) → RiskDecision` is a pure function of its inputs. The caller (decision engine) is responsible for constructing `PortfolioState` from live portfolio/P&L data. No I/O, no broker calls, no time.now() inside the manager. Rules are loaded once from YAML into typed Pydantic models; the manager never reads YAML directly.

**Tech Stack:** Python 3.11+, Pydantic v2 (rule config), `Decimal` for all monetary math, `dataclasses` (frozen + slots), structlog, pytest.

**Critical conventions (same as prior phases):**
- `from __future__ import annotations` at the top of every `.py` file
- `logger = structlog.get_logger()` (not `log`)
- `frozen=True, slots=True` on dataclasses
- No `print()` — use `structlog`
- `Decimal` everywhere money is involved — never `float` for INR amounts

**Rules implemented in Phase 4** (from `config/risk_rules.yaml`):

| Rule | YAML key | Reject reason |
|------|----------|---------------|
| Kill switch active | `kill_switch.auto_reset` | `KILL_SWITCH_ACTIVE` |
| Suspect/missing data quality | implicit | `SUSPECT_DATA_QUALITY` |
| R/R below minimum | `per_trade.min_reward_risk` | `INSUFFICIENT_REWARD_RISK` |
| Outside trading window | `windows.trade_start_ist / trade_end_ist` | `OUTSIDE_TRADING_WINDOW` |
| Max concurrent positions | `portfolio.max_concurrent_positions` | `MAX_POSITIONS_REACHED` |
| Cash below minimum buffer | `portfolio.min_cash_fraction` | `INSUFFICIENT_CASH` |
| Daily loss cap | `loss_caps.max_daily_loss_fraction` | `DAILY_LOSS_CAP` |
| Weekly loss cap | `loss_caps.max_weekly_loss_fraction` | `WEEKLY_LOSS_CAP` |
| Max drawdown | `loss_caps.max_drawdown_fraction` | `DRAWDOWN_BREACH` |
| Max orders today | `frequency.max_new_orders_per_day` | `MAX_ORDERS_TODAY` |
| Symbol cooldown | `frequency.symbol_cooldown_minutes` | `SYMBOL_COOLDOWN` |
| Position sizing → 0 shares | derived | `ZERO_QUANTITY` |

**Deferred to later phases** (require external data not available yet):
- Sector/factor exposure limits (needs UniverseLoader sector map)
- Pairwise correlation check (needs 60-day rolling returns)
- Earnings/event blackouts (needs earnings calendar)
- Consecutive loser cooldown (needs trade history)
- Sharp drawdown size reduction (needs session history)
- Liquidity filters: min dollar volume, spread bps (needs tick data)

**EXIT_LONG policy:** EXIT_LONG signals bypass all entry checks (R/R, window, positions, cash, loss caps, orders, cooldown) — you must always be able to exit an open position. Only the kill switch blocks EXIT_LONG.

---

## File Map

```
agent/risk/
    __init__.py             # empty package marker
    types.py                # RiskVerdict, RejectionReason, RiskDecision, PortfolioState
    rules.py                # Pydantic models for risk_rules.yaml + load_risk_rules()
    manager.py              # RiskManager.evaluate()

tests/risk/
    __init__.py             # empty package marker
    test_types.py           # sanity checks: enum values, dataclass construction, load_risk_rules
    test_manager.py         # one test per rule that deliberately triggers it + happy path
```

**No modifications to existing files.** Cross-module imports:
- `agent/risk/types.py` imports `Position` from `agent.data.types` and `Signal` from `agent.strategies.types`
- `agent/risk/manager.py` imports `DataQuality` from `agent.data.types`, `Action` from `agent.strategies.types`

---

## Task 1: Types + Config Loader

**Files:**
- Create: `agent/risk/__init__.py`
- Create: `agent/risk/types.py`
- Create: `agent/risk/rules.py`
- Create: `tests/risk/__init__.py`
- Create: `tests/risk/test_types.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/risk/test_types.py`:

```python
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.data.types import DataQuality, Position
from agent.risk.rules import RiskRules, load_risk_rules
from agent.risk.types import (
    PortfolioState,
    RejectionReason,
    RiskDecision,
    RiskVerdict,
)
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")
_TS = datetime(2024, 1, 2, 9, 45, tzinfo=IST)


def _signal() -> Signal:
    return Signal(
        symbol="HDFCBANK",
        action=Action.ENTER_LONG,
        confidence=0.75,
        suggested_stop=Decimal("1660.00"),
        suggested_target=Decimal("1780.00"),
        invalidation_condition="Close below EMA21",
        expected_r=2.0,
        time_horizon_hours=4,
        regime_fit=0.9,
        data_quality=DataQuality.OK,
        strategy_name="trend_following_v1",
        explanation="Test signal",
        timestamp=_TS,
    )


# ---------------------------------------------------------------------------
# Enum sanity
# ---------------------------------------------------------------------------


def test_risk_verdict_values() -> None:
    assert RiskVerdict.APPROVED == "approved"
    assert RiskVerdict.REJECTED == "rejected"


def test_rejection_reason_values() -> None:
    assert RejectionReason.NONE == "none"
    assert RejectionReason.KILL_SWITCH_ACTIVE == "kill_switch_active"
    assert RejectionReason.SUSPECT_DATA_QUALITY == "suspect_data_quality"
    assert RejectionReason.INSUFFICIENT_REWARD_RISK == "insufficient_reward_risk"
    assert RejectionReason.OUTSIDE_TRADING_WINDOW == "outside_trading_window"
    assert RejectionReason.MAX_POSITIONS_REACHED == "max_positions_reached"
    assert RejectionReason.INSUFFICIENT_CASH == "insufficient_cash"
    assert RejectionReason.DAILY_LOSS_CAP == "daily_loss_cap"
    assert RejectionReason.WEEKLY_LOSS_CAP == "weekly_loss_cap"
    assert RejectionReason.DRAWDOWN_BREACH == "drawdown_breach"
    assert RejectionReason.MAX_ORDERS_TODAY == "max_orders_today"
    assert RejectionReason.SYMBOL_COOLDOWN == "symbol_cooldown"
    assert RejectionReason.ZERO_QUANTITY == "zero_quantity"


# ---------------------------------------------------------------------------
# RiskDecision construction
# ---------------------------------------------------------------------------


def test_risk_decision_approved_construction() -> None:
    sig = _signal()
    decision = RiskDecision(
        verdict=RiskVerdict.APPROVED,
        quantity=4,
        entry_price=Decimal("1700.00"),
        stop_price=Decimal("1660.00"),
        target_price=Decimal("1780.00"),
        risk_per_share=Decimal("40.00"),
        position_value=Decimal("6800.00"),
        rejection_reason=RejectionReason.NONE,
        rejection_detail="",
        signal=sig,
    )
    assert decision.verdict == RiskVerdict.APPROVED
    assert decision.quantity == 4
    assert decision.rejection_reason == RejectionReason.NONE


def test_risk_decision_rejected_construction() -> None:
    sig = _signal()
    decision = RiskDecision(
        verdict=RiskVerdict.REJECTED,
        quantity=0,
        entry_price=Decimal("1700.00"),
        stop_price=Decimal("1660.00"),
        target_price=Decimal("1780.00"),
        risk_per_share=Decimal("0.00"),
        position_value=Decimal("0.00"),
        rejection_reason=RejectionReason.KILL_SWITCH_ACTIVE,
        rejection_detail="Kill switch is active — manual restart required",
        signal=sig,
    )
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.quantity == 0


def test_risk_decision_is_frozen() -> None:
    sig = _signal()
    decision = RiskDecision(
        verdict=RiskVerdict.APPROVED,
        quantity=4,
        entry_price=Decimal("1700.00"),
        stop_price=Decimal("1660.00"),
        target_price=Decimal("1780.00"),
        risk_per_share=Decimal("40.00"),
        position_value=Decimal("6800.00"),
        rejection_reason=RejectionReason.NONE,
        rejection_detail="",
        signal=sig,
    )
    with pytest.raises((AttributeError, TypeError)):
        decision.quantity = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PortfolioState construction
# ---------------------------------------------------------------------------


def test_portfolio_state_construction() -> None:
    port = PortfolioState(
        nav=Decimal("83000.00"),
        cash=Decimal("50000.00"),
        positions={},
        daily_pnl=Decimal("0.00"),
        weekly_pnl=Decimal("0.00"),
        peak_nav=Decimal("83000.00"),
        orders_today=0,
        last_order_time={},
        kill_switch_active=False,
        evaluation_time=datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    assert port.nav == Decimal("83000.00")
    assert port.kill_switch_active is False
    assert len(port.positions) == 0


def test_portfolio_state_with_positions() -> None:
    pos = Position(
        symbol="HDFCBANK",
        quantity=5,
        average_price=Decimal("1700.00"),
        product="MIS",
    )
    port = PortfolioState(
        nav=Decimal("83000.00"),
        cash=Decimal("41500.00"),
        positions={"HDFCBANK": pos},
        daily_pnl=Decimal("250.00"),
        weekly_pnl=Decimal("1200.00"),
        peak_nav=Decimal("84000.00"),
        orders_today=1,
        last_order_time={"HDFCBANK": datetime(2024, 1, 2, 9, 45, tzinfo=IST)},
        kill_switch_active=False,
        evaluation_time=datetime(2024, 1, 2, 10, 30, tzinfo=IST),
    )
    assert "HDFCBANK" in port.positions
    assert port.orders_today == 1


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def test_load_risk_rules_parses_yaml() -> None:
    rules = load_risk_rules()
    assert rules.per_trade.max_risk_fraction == 0.005
    assert rules.per_trade.max_position_fraction == 0.10
    assert rules.per_trade.min_reward_risk == 1.5
    assert rules.portfolio.max_concurrent_positions == 6
    assert rules.portfolio.min_cash_fraction == 0.10
    assert rules.loss_caps.max_daily_loss_fraction == 0.02
    assert rules.loss_caps.max_weekly_loss_fraction == 0.05
    assert rules.loss_caps.max_drawdown_fraction == 0.08
    assert rules.frequency.max_new_orders_per_day == 4
    assert rules.frequency.symbol_cooldown_minutes == 30
    assert rules.windows.trade_start_ist == "09:45"
    assert rules.windows.trade_end_ist == "14:45"


def test_load_risk_rules_returns_risk_rules_instance() -> None:
    rules = load_risk_rules()
    assert isinstance(rules, RiskRules)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/tatsatshah/Desktop/yegedge && source .venv/bin/activate && pytest tests/risk/test_types.py -v --no-cov 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'agent.risk'`

- [ ] **Step 3: Create package markers**

Create `agent/risk/__init__.py` — empty (`# intentionally empty`).
Create `tests/risk/__init__.py` — empty (`# intentionally empty`).

- [ ] **Step 4: Create `agent/risk/types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from agent.data.types import Position
from agent.strategies.types import Signal


class RiskVerdict(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class RejectionReason(StrEnum):
    NONE = "none"
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    SUSPECT_DATA_QUALITY = "suspect_data_quality"
    INSUFFICIENT_REWARD_RISK = "insufficient_reward_risk"
    OUTSIDE_TRADING_WINDOW = "outside_trading_window"
    MAX_POSITIONS_REACHED = "max_positions_reached"
    INSUFFICIENT_CASH = "insufficient_cash"
    DAILY_LOSS_CAP = "daily_loss_cap"
    WEEKLY_LOSS_CAP = "weekly_loss_cap"
    DRAWDOWN_BREACH = "drawdown_breach"
    MAX_ORDERS_TODAY = "max_orders_today"
    SYMBOL_COOLDOWN = "symbol_cooldown"
    ZERO_QUANTITY = "zero_quantity"


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Output of RiskManager.evaluate() — either an approved trade or a rejection.

    All monetary values are Decimal. quantity=0 and verdict=REJECTED always go together.
    """

    verdict: RiskVerdict
    quantity: int               # shares to trade; 0 when rejected
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal
    risk_per_share: Decimal     # entry_price - stop_price
    position_value: Decimal     # quantity * entry_price
    rejection_reason: RejectionReason
    rejection_detail: str       # human-readable; empty string when approved
    signal: Signal              # originating signal (for journaling)


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Immutable snapshot of portfolio state passed into RiskManager.evaluate().

    The caller (decision engine) constructs this from live portfolio + P&L data.
    dict fields (positions, last_order_time) make instances unhashable — never
    use PortfolioState as a dict key or in a set.
    """

    nav: Decimal                            # current net asset value
    cash: Decimal                           # available cash
    positions: dict[str, Position]         # symbol → open Position
    daily_pnl: Decimal                     # today's realized + unrealized P&L
    weekly_pnl: Decimal                    # this week's P&L
    peak_nav: Decimal                      # historical NAV peak (for drawdown)
    orders_today: int                       # count of new orders placed today
    last_order_time: dict[str, datetime]   # symbol → last order IST timestamp
    kill_switch_active: bool
    evaluation_time: datetime              # IST-aware "now" for window/cooldown checks
```

- [ ] **Step 5: Create `agent/risk/rules.py`**

```python
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class PerTradeRules(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_risk_fraction: float        # 0.005 — max loss per trade as fraction of NAV
    max_position_fraction: float    # 0.10  — max position size as fraction of NAV
    min_reward_risk: float          # 1.5   — minimum R/R required to enter


class PortfolioRules(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_concurrent_positions: int   # 6
    min_cash_fraction: float        # 0.10 — minimum cash buffer as fraction of NAV


class LossCaps(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_daily_loss_fraction: float   # 0.02
    max_weekly_loss_fraction: float  # 0.05
    max_drawdown_fraction: float     # 0.08 — triggers kill switch


class FrequencyRules(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_new_orders_per_day: int     # 4
    symbol_cooldown_minutes: int    # 30


class TradingWindowRules(BaseModel):
    model_config = ConfigDict(extra="ignore")

    trade_start_ist: str            # "09:45"
    trade_end_ist: str              # "14:45"


class RiskRules(BaseModel):
    per_trade: PerTradeRules
    portfolio: PortfolioRules
    loss_caps: LossCaps
    frequency: FrequencyRules
    windows: TradingWindowRules


def load_risk_rules(path: Path = Path("config/risk_rules.yaml")) -> RiskRules:
    """Load and parse risk_rules.yaml into typed Pydantic models.

    Extra YAML keys (stop_loss, take_profit, etc.) are ignored by each sub-model.
    """
    with path.open() as f:
        data = yaml.safe_load(f)
    return RiskRules(
        per_trade=PerTradeRules(**data["per_trade"]),
        portfolio=PortfolioRules(**data["portfolio"]),
        loss_caps=LossCaps(**data["loss_caps"]),
        frequency=FrequencyRules(**data["frequency"]),
        windows=TradingWindowRules(**data["windows"]),
    )
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
cd /Users/tatsatshah/Desktop/yegedge && source .venv/bin/activate && pytest tests/risk/test_types.py -v --no-cov
```

Expected: `12 passed`

- [ ] **Step 7: Lint and format**

```bash
ruff check agent/risk/types.py agent/risk/rules.py tests/risk/test_types.py
black agent/risk/types.py agent/risk/rules.py tests/risk/test_types.py
```

Fix any issues, re-run tests.

- [ ] **Step 8: Commit**

```bash
git add agent/risk/__init__.py agent/risk/types.py agent/risk/rules.py \
        tests/risk/__init__.py tests/risk/test_types.py
git commit -m "feat(risk): add RiskVerdict, RejectionReason, RiskDecision, PortfolioState and RiskRules config loader"
```

---

## Task 2: RiskManager

**Files:**
- Create: `agent/risk/manager.py`
- Create: `tests/risk/test_manager.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/risk/test_manager.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.data.types import DataQuality, Position
from agent.risk.manager import RiskManager
from agent.risk.rules import load_risk_rules
from agent.risk.types import (
    PortfolioState,
    RejectionReason,
    RiskVerdict,
)
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")

_NAV = Decimal("83000.00")


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


def _signal(
    *,
    action: Action = Action.ENTER_LONG,
    symbol: str = "HDFCBANK",
    confidence: float = 0.75,
    suggested_stop: Decimal = Decimal("1660.00"),
    suggested_target: Decimal = Decimal("1780.00"),
    data_quality: DataQuality = DataQuality.OK,
) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        suggested_stop=suggested_stop,
        suggested_target=suggested_target,
        invalidation_condition="Close below EMA21",
        expected_r=2.0,
        time_horizon_hours=4,
        regime_fit=0.9,
        data_quality=data_quality,
        strategy_name="trend_following_v1",
        explanation="Test signal",
        timestamp=datetime(2024, 1, 2, 9, 45, tzinfo=IST),
    )


def _portfolio(
    *,
    nav: Decimal = _NAV,
    cash: Decimal = Decimal("50000.00"),
    positions: dict | None = None,
    daily_pnl: Decimal = Decimal("0.00"),
    weekly_pnl: Decimal = Decimal("0.00"),
    peak_nav: Decimal | None = None,
    orders_today: int = 0,
    last_order_time: dict | None = None,
    kill_switch_active: bool = False,
    evaluation_time: datetime | None = None,
) -> PortfolioState:
    return PortfolioState(
        nav=nav,
        cash=cash,
        positions=positions or {},
        daily_pnl=daily_pnl,
        weekly_pnl=weekly_pnl,
        peak_nav=peak_nav if peak_nav is not None else nav,
        orders_today=orders_today,
        last_order_time=last_order_time or {},
        kill_switch_active=kill_switch_active,
        evaluation_time=evaluation_time or datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(load_risk_rules())


# ---------------------------------------------------------------------------
# Happy path — all rules pass
# ---------------------------------------------------------------------------


def test_approved_signal_returns_approved(rm: RiskManager) -> None:
    # entry=1700, stop=1660, target=1780
    # R/R = (1780-1700)/(1700-1660) = 80/40 = 2.0 >= 1.5 ✓
    decision = rm.evaluate(_signal(), _portfolio(), Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED
    assert decision.quantity > 0
    assert decision.rejection_reason == RejectionReason.NONE


def test_approved_decision_has_correct_prices(rm: RiskManager) -> None:
    decision = rm.evaluate(_signal(), _portfolio(), Decimal("1700.00"))
    assert decision.entry_price == Decimal("1700.00")
    assert decision.stop_price == Decimal("1660.00")
    assert decision.target_price == Decimal("1780.00")
    assert decision.risk_per_share == Decimal("40.00")


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def test_quantity_capped_by_position_fraction(rm: RiskManager) -> None:
    # NAV=83000, max_risk=0.5% → 415 INR, risk_per_share=40 → qty_by_risk=10
    # max_position=10% → 8300 INR, at 1700/share → qty_by_size=4
    # Final quantity = min(10, 4) = 4
    decision = rm.evaluate(_signal(), _portfolio(), Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED
    assert decision.quantity == 4


def test_position_value_equals_quantity_times_entry(rm: RiskManager) -> None:
    decision = rm.evaluate(_signal(), _portfolio(), Decimal("1700.00"))
    expected_value = Decimal(str(decision.quantity)) * Decimal("1700.00")
    assert decision.position_value == expected_value


def test_zero_quantity_rejects(rm: RiskManager) -> None:
    # NAV=100, max_risk=0.5% → 0.50 INR. risk_per_share=40. qty=floor(0.50/40)=0 → ZERO_QUANTITY
    port = _portfolio(nav=Decimal("100.00"), cash=Decimal("50.00"), peak_nav=Decimal("100.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.ZERO_QUANTITY
    assert decision.quantity == 0


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


def test_kill_switch_active_rejects_enter_long(rm: RiskManager) -> None:
    decision = rm.evaluate(
        _signal(), _portfolio(kill_switch_active=True), Decimal("1700.00")
    )
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.KILL_SWITCH_ACTIVE
    assert decision.quantity == 0


def test_kill_switch_active_rejects_exit_long(rm: RiskManager) -> None:
    """Kill switch blocks EXIT_LONG too — everything stops on kill switch."""
    port = _portfolio(
        kill_switch_active=True,
        positions={"HDFCBANK": Position(symbol="HDFCBANK", quantity=5, average_price=Decimal("1700"), product="MIS")},
    )
    decision = rm.evaluate(_signal(action=Action.EXIT_LONG), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.KILL_SWITCH_ACTIVE


# ---------------------------------------------------------------------------
# Data quality gate
# ---------------------------------------------------------------------------


def test_suspect_data_quality_rejects(rm: RiskManager) -> None:
    decision = rm.evaluate(
        _signal(data_quality=DataQuality.SUSPECT), _portfolio(), Decimal("1700.00")
    )
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.SUSPECT_DATA_QUALITY


def test_missing_data_quality_rejects(rm: RiskManager) -> None:
    decision = rm.evaluate(
        _signal(data_quality=DataQuality.MISSING), _portfolio(), Decimal("1700.00")
    )
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.SUSPECT_DATA_QUALITY


def test_partial_data_quality_is_allowed(rm: RiskManager) -> None:
    decision = rm.evaluate(
        _signal(data_quality=DataQuality.PARTIAL), _portfolio(), Decimal("1700.00")
    )
    assert decision.verdict == RiskVerdict.APPROVED


# ---------------------------------------------------------------------------
# Reward/risk ratio
# ---------------------------------------------------------------------------


def test_insufficient_reward_risk_rejects(rm: RiskManager) -> None:
    # entry=1700, stop=1685, target=1710 → R/R = 10/15 = 0.67 < 1.5
    sig = _signal(suggested_stop=Decimal("1685.00"), suggested_target=Decimal("1710.00"))
    decision = rm.evaluate(sig, _portfolio(), Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.INSUFFICIENT_REWARD_RISK


def test_reward_risk_exactly_at_minimum_is_allowed(rm: RiskManager) -> None:
    # R/R = 1.5 exactly: entry=1700, stop=1660 (risk=40), target=1760 (reward=60)
    sig = _signal(suggested_stop=Decimal("1660.00"), suggested_target=Decimal("1760.00"))
    decision = rm.evaluate(sig, _portfolio(), Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


# ---------------------------------------------------------------------------
# Trading window
# ---------------------------------------------------------------------------


def test_before_trading_window_rejects(rm: RiskManager) -> None:
    # 09:30 IST — before 09:45 start
    port = _portfolio(evaluation_time=datetime(2024, 1, 2, 9, 30, tzinfo=IST))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.OUTSIDE_TRADING_WINDOW


def test_after_trading_window_rejects(rm: RiskManager) -> None:
    # 15:00 IST — after 14:45 end
    port = _portfolio(evaluation_time=datetime(2024, 1, 2, 15, 0, tzinfo=IST))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.OUTSIDE_TRADING_WINDOW


def test_exactly_at_window_open_is_allowed(rm: RiskManager) -> None:
    port = _portfolio(evaluation_time=datetime(2024, 1, 2, 9, 45, tzinfo=IST))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_exactly_at_window_close_is_allowed(rm: RiskManager) -> None:
    port = _portfolio(evaluation_time=datetime(2024, 1, 2, 14, 45, tzinfo=IST))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


# ---------------------------------------------------------------------------
# Max concurrent positions
# ---------------------------------------------------------------------------


def test_max_concurrent_positions_rejects(rm: RiskManager) -> None:
    # 6 positions already open; max is 6 → reject before opening a 7th
    positions = {
        f"SYM{i}": Position(symbol=f"SYM{i}", quantity=10, average_price=Decimal("100"), product="MIS")
        for i in range(6)
    }
    port = _portfolio(positions=positions)
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.MAX_POSITIONS_REACHED


def test_five_positions_allows_sixth(rm: RiskManager) -> None:
    positions = {
        f"SYM{i}": Position(symbol=f"SYM{i}", quantity=10, average_price=Decimal("100"), product="MIS")
        for i in range(5)
    }
    port = _portfolio(positions=positions)
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


# ---------------------------------------------------------------------------
# Minimum cash buffer
# ---------------------------------------------------------------------------


def test_cash_below_minimum_buffer_rejects(rm: RiskManager) -> None:
    # NAV=83000, min_cash=10% = 8300. cash=8000 < 8300 → reject
    port = _portfolio(nav=Decimal("83000.00"), cash=Decimal("8000.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.INSUFFICIENT_CASH


def test_cash_exactly_at_minimum_buffer_is_allowed(rm: RiskManager) -> None:
    # cash=8300 == 10% of 83000 → allowed (rule is <, not <=)
    port = _portfolio(nav=Decimal("83000.00"), cash=Decimal("8300.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


# ---------------------------------------------------------------------------
# Daily loss cap
# ---------------------------------------------------------------------------


def test_daily_loss_cap_rejects(rm: RiskManager) -> None:
    # NAV=83000, max_daily_loss=2% = 1660. daily_pnl=-1700 < -1660 → reject
    port = _portfolio(nav=Decimal("83000.00"), daily_pnl=Decimal("-1700.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.DAILY_LOSS_CAP


def test_daily_loss_exactly_at_cap_rejects(rm: RiskManager) -> None:
    # Exactly at the threshold → also rejected
    port = _portfolio(nav=Decimal("83000.00"), daily_pnl=Decimal("-1660.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.DAILY_LOSS_CAP


def test_daily_loss_below_cap_is_allowed(rm: RiskManager) -> None:
    # daily_pnl=-1000 < 1660 → allowed
    port = _portfolio(nav=Decimal("83000.00"), daily_pnl=Decimal("-1000.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


# ---------------------------------------------------------------------------
# Weekly loss cap
# ---------------------------------------------------------------------------


def test_weekly_loss_cap_rejects(rm: RiskManager) -> None:
    # NAV=83000, max_weekly_loss=5% = 4150. weekly_pnl=-4200 → reject
    port = _portfolio(nav=Decimal("83000.00"), weekly_pnl=Decimal("-4200.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.WEEKLY_LOSS_CAP


def test_weekly_loss_exactly_at_cap_rejects(rm: RiskManager) -> None:
    port = _portfolio(nav=Decimal("83000.00"), weekly_pnl=Decimal("-4150.00"))
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.WEEKLY_LOSS_CAP


# ---------------------------------------------------------------------------
# Max drawdown / drawdown breach
# ---------------------------------------------------------------------------


def test_drawdown_breach_rejects(rm: RiskManager) -> None:
    # peak=100000, nav=91500 → drawdown=8.5% > 8% → reject
    port = _portfolio(
        nav=Decimal("91500.00"),
        cash=Decimal("30000.00"),
        peak_nav=Decimal("100000.00"),
    )
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.DRAWDOWN_BREACH


def test_drawdown_exactly_at_threshold_rejects(rm: RiskManager) -> None:
    # drawdown = 8.0% exactly → reject (>= threshold)
    port = _portfolio(
        nav=Decimal("92000.00"),
        cash=Decimal("30000.00"),
        peak_nav=Decimal("100000.00"),
    )
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.DRAWDOWN_BREACH


def test_drawdown_below_threshold_is_allowed(rm: RiskManager) -> None:
    # peak=100000, nav=95000 → drawdown=5% < 8% → allowed
    port = _portfolio(
        nav=Decimal("95000.00"),
        cash=Decimal("30000.00"),
        peak_nav=Decimal("100000.00"),
    )
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


# ---------------------------------------------------------------------------
# Max orders per day
# ---------------------------------------------------------------------------


def test_max_orders_today_rejects(rm: RiskManager) -> None:
    # 4 orders placed today; max is 4 → reject the 5th
    port = _portfolio(orders_today=4)
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.MAX_ORDERS_TODAY


def test_three_orders_today_allows_fourth(rm: RiskManager) -> None:
    port = _portfolio(orders_today=3)
    decision = rm.evaluate(_signal(), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


# ---------------------------------------------------------------------------
# Symbol cooldown
# ---------------------------------------------------------------------------


def test_symbol_in_cooldown_rejects(rm: RiskManager) -> None:
    # Last order on HDFCBANK was 15 min ago; cooldown is 30 min
    eval_time = datetime(2024, 1, 2, 10, 0, tzinfo=IST)
    last_time = eval_time - timedelta(minutes=15)
    port = _portfolio(
        last_order_time={"HDFCBANK": last_time},
        evaluation_time=eval_time,
    )
    decision = rm.evaluate(_signal(symbol="HDFCBANK"), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.SYMBOL_COOLDOWN


def test_symbol_cooldown_expired_allows_order(rm: RiskManager) -> None:
    # Last order was 31 min ago; cooldown is 30 min → allowed
    eval_time = datetime(2024, 1, 2, 10, 0, tzinfo=IST)
    last_time = eval_time - timedelta(minutes=31)
    port = _portfolio(
        last_order_time={"HDFCBANK": last_time},
        evaluation_time=eval_time,
    )
    decision = rm.evaluate(_signal(symbol="HDFCBANK"), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_different_symbol_not_in_cooldown(rm: RiskManager) -> None:
    # INFY was recently traded but HDFCBANK was not → HDFCBANK allowed
    eval_time = datetime(2024, 1, 2, 10, 0, tzinfo=IST)
    port = _portfolio(
        last_order_time={"INFY": eval_time - timedelta(minutes=5)},
        evaluation_time=eval_time,
    )
    decision = rm.evaluate(_signal(symbol="HDFCBANK"), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


# ---------------------------------------------------------------------------
# EXIT_LONG bypass
# ---------------------------------------------------------------------------


def test_exit_long_bypasses_entry_checks(rm: RiskManager) -> None:
    """EXIT_LONG is approved even when most ENTER_LONG rules would reject."""
    eval_time = datetime(2024, 1, 2, 15, 30, tzinfo=IST)  # after window → ENTER_LONG rejected
    port = _portfolio(
        orders_today=10,  # over daily limit
        evaluation_time=eval_time,
        positions={
            "HDFCBANK": Position(
                symbol="HDFCBANK",
                quantity=5,
                average_price=Decimal("1700.00"),
                product="MIS",
            )
        },
    )
    decision = rm.evaluate(_signal(action=Action.EXIT_LONG), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED


def test_exit_long_quantity_matches_existing_position(rm: RiskManager) -> None:
    port = _portfolio(
        positions={
            "HDFCBANK": Position(
                symbol="HDFCBANK",
                quantity=7,
                average_price=Decimal("1700.00"),
                product="MIS",
            )
        }
    )
    decision = rm.evaluate(_signal(action=Action.EXIT_LONG), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED
    assert decision.quantity == 7


def test_exit_long_with_no_position_returns_zero_quantity(rm: RiskManager) -> None:
    """EXIT_LONG for a symbol with no open position → approved with quantity=0."""
    port = _portfolio(positions={})
    decision = rm.evaluate(_signal(action=Action.EXIT_LONG), port, Decimal("1700.00"))
    assert decision.verdict == RiskVerdict.APPROVED
    assert decision.quantity == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/tatsatshah/Desktop/yegedge && source .venv/bin/activate && pytest tests/risk/test_manager.py -v --no-cov 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'agent.risk.manager'`

- [ ] **Step 3: Create `agent/risk/manager.py`**

```python
from __future__ import annotations

from datetime import time
from decimal import Decimal
from zoneinfo import ZoneInfo

import structlog

from agent.data.types import DataQuality
from agent.risk.rules import RiskRules
from agent.risk.types import (
    PortfolioState,
    RejectionReason,
    RiskDecision,
    RiskVerdict,
)
from agent.strategies.types import Action, Signal

logger = structlog.get_logger()

IST = ZoneInfo("Asia/Kolkata")

_ALLOWED_QUALITIES = {DataQuality.OK, DataQuality.PARTIAL}


class RiskManager:
    """Evaluate a Signal against all risk rules and return a RiskDecision.

    evaluate() is a pure function of (signal, portfolio, entry_price).
    Rules are checked in priority order — first rejection wins.
    """

    def __init__(self, rules: RiskRules) -> None:
        self._rules = rules
        self._window_start = _parse_time(rules.windows.trade_start_ist)
        self._window_end = _parse_time(rules.windows.trade_end_ist)

    def evaluate(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        entry_price: Decimal,
    ) -> RiskDecision:
        """Apply all risk rules to the signal. Returns first rejection encountered.

        EXIT_LONG bypasses all entry rules except the kill switch.
        """
        # 1. Kill switch — blocks everything including exits
        if portfolio.kill_switch_active:
            return self._reject(
                signal, entry_price, RejectionReason.KILL_SWITCH_ACTIVE,
                "Kill switch is active — manual restart required"
            )

        # 2. EXIT_LONG bypasses all entry-specific checks
        if signal.action == Action.EXIT_LONG:
            return self._approve_exit(signal, portfolio, entry_price)

        # 3. Data quality gate
        if signal.data_quality not in _ALLOWED_QUALITIES:
            return self._reject(
                signal, entry_price, RejectionReason.SUSPECT_DATA_QUALITY,
                f"Data quality is {signal.data_quality.value!r} — skipping"
            )

        # 4. Reward/risk from actual entry vs signal stop/target
        risk = entry_price - signal.suggested_stop
        reward = signal.suggested_target - entry_price
        min_rr = Decimal(str(self._rules.per_trade.min_reward_risk))
        if risk <= 0 or reward / risk < min_rr:
            actual_rr = (reward / risk).quantize(Decimal("0.01")) if risk > 0 else Decimal("0")
            return self._reject(
                signal, entry_price, RejectionReason.INSUFFICIENT_REWARD_RISK,
                f"R/R={actual_rr} below minimum {min_rr}"
            )

        # 5. Trading window
        eval_ist = portfolio.evaluation_time.astimezone(IST)
        t = eval_ist.time().replace(second=0, microsecond=0)
        if not (self._window_start <= t <= self._window_end):
            return self._reject(
                signal, entry_price, RejectionReason.OUTSIDE_TRADING_WINDOW,
                f"Current time {t} outside [{self._window_start}, {self._window_end}] IST"
            )

        # 6. Max concurrent positions
        if len(portfolio.positions) >= self._rules.portfolio.max_concurrent_positions:
            return self._reject(
                signal, entry_price, RejectionReason.MAX_POSITIONS_REACHED,
                f"Already at max {self._rules.portfolio.max_concurrent_positions} positions"
            )

        # 7. Minimum cash buffer
        min_cash = portfolio.nav * Decimal(str(self._rules.portfolio.min_cash_fraction))
        if portfolio.cash < min_cash:
            return self._reject(
                signal, entry_price, RejectionReason.INSUFFICIENT_CASH,
                f"Cash {portfolio.cash} below minimum buffer {min_cash}"
            )

        # 8. Daily loss cap (loss at-or-beyond cap → reject)
        max_daily_loss = portfolio.nav * Decimal(str(self._rules.loss_caps.max_daily_loss_fraction))
        if portfolio.daily_pnl <= -max_daily_loss:
            return self._reject(
                signal, entry_price, RejectionReason.DAILY_LOSS_CAP,
                f"Daily P&L {portfolio.daily_pnl} hit cap -{max_daily_loss}"
            )

        # 9. Weekly loss cap
        max_weekly_loss = portfolio.nav * Decimal(str(self._rules.loss_caps.max_weekly_loss_fraction))
        if portfolio.weekly_pnl <= -max_weekly_loss:
            return self._reject(
                signal, entry_price, RejectionReason.WEEKLY_LOSS_CAP,
                f"Weekly P&L {portfolio.weekly_pnl} hit cap -{max_weekly_loss}"
            )

        # 10. Max drawdown
        if portfolio.peak_nav > 0:
            drawdown = (portfolio.peak_nav - portfolio.nav) / portfolio.peak_nav
            max_dd = Decimal(str(self._rules.loss_caps.max_drawdown_fraction))
            if drawdown >= max_dd:
                return self._reject(
                    signal, entry_price, RejectionReason.DRAWDOWN_BREACH,
                    f"Drawdown {drawdown:.2%} >= max {max_dd:.2%} — kill switch required"
                )

        # 11. Max orders per day
        if portfolio.orders_today >= self._rules.frequency.max_new_orders_per_day:
            return self._reject(
                signal, entry_price, RejectionReason.MAX_ORDERS_TODAY,
                f"Already placed {portfolio.orders_today} orders today "
                f"(max {self._rules.frequency.max_new_orders_per_day})"
            )

        # 12. Symbol cooldown
        last_time = portfolio.last_order_time.get(signal.symbol)
        if last_time is not None:
            elapsed_secs = (portfolio.evaluation_time - last_time).total_seconds()
            cooldown_secs = self._rules.frequency.symbol_cooldown_minutes * 60
            if elapsed_secs < cooldown_secs:
                remaining = int((cooldown_secs - elapsed_secs) / 60)
                return self._reject(
                    signal, entry_price, RejectionReason.SYMBOL_COOLDOWN,
                    f"{signal.symbol} in cooldown — {remaining}m remaining"
                )

        # 13. Position sizing — last check, computes approved quantity
        return self._size_position(signal, portfolio, entry_price)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reject(
        self,
        signal: Signal,
        entry_price: Decimal,
        reason: RejectionReason,
        detail: str,
    ) -> RiskDecision:
        logger.info(
            "risk_manager.rejected",
            symbol=signal.symbol,
            reason=reason.value,
            detail=detail,
        )
        return RiskDecision(
            verdict=RiskVerdict.REJECTED,
            quantity=0,
            entry_price=entry_price,
            stop_price=signal.suggested_stop,
            target_price=signal.suggested_target,
            risk_per_share=Decimal("0"),
            position_value=Decimal("0"),
            rejection_reason=reason,
            rejection_detail=detail,
            signal=signal,
        )

    def _approve_exit(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        entry_price: Decimal,
    ) -> RiskDecision:
        existing = portfolio.positions.get(signal.symbol)
        quantity = existing.quantity if existing is not None else 0
        risk_per_share = max(entry_price - signal.suggested_stop, Decimal("0.01"))
        logger.info(
            "risk_manager.exit_approved",
            symbol=signal.symbol,
            quantity=quantity,
        )
        return RiskDecision(
            verdict=RiskVerdict.APPROVED,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=signal.suggested_stop,
            target_price=signal.suggested_target,
            risk_per_share=risk_per_share,
            position_value=Decimal(str(quantity)) * entry_price,
            rejection_reason=RejectionReason.NONE,
            rejection_detail="",
            signal=signal,
        )

    def _size_position(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        entry_price: Decimal,
    ) -> RiskDecision:
        """Compute approved position size.

        quantity = min(qty_by_risk, qty_by_size)
        qty_by_risk  = floor(nav * max_risk_fraction  / risk_per_share)
        qty_by_size  = floor(nav * max_position_fraction / entry_price)
        """
        risk_per_share = entry_price - signal.suggested_stop
        if risk_per_share <= 0:
            return self._reject(
                signal, entry_price, RejectionReason.ZERO_QUANTITY,
                "risk_per_share <= 0 (entry_price <= stop_price)"
            )

        max_risk_amount = portfolio.nav * Decimal(str(self._rules.per_trade.max_risk_fraction))
        qty_by_risk = int(max_risk_amount / risk_per_share)

        max_position_value = portfolio.nav * Decimal(str(self._rules.per_trade.max_position_fraction))
        qty_by_size = int(max_position_value / entry_price)

        quantity = min(qty_by_risk, qty_by_size)

        if quantity == 0:
            return self._reject(
                signal, entry_price, RejectionReason.ZERO_QUANTITY,
                f"Position sizing produced 0 shares "
                f"(max_risk={max_risk_amount:.2f}, risk_per_share={risk_per_share})"
            )

        position_value = Decimal(str(quantity)) * entry_price
        logger.info(
            "risk_manager.approved",
            symbol=signal.symbol,
            quantity=quantity,
            entry_price=str(entry_price),
            risk_per_share=str(risk_per_share),
            position_value=str(position_value),
        )
        return RiskDecision(
            verdict=RiskVerdict.APPROVED,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=signal.suggested_stop,
            target_price=signal.suggested_target,
            risk_per_share=risk_per_share,
            position_value=position_value,
            rejection_reason=RejectionReason.NONE,
            rejection_detail="",
            signal=signal,
        )


def _parse_time(hhmm: str) -> time:
    """Parse 'HH:MM' string into a datetime.time object."""
    h, m = hhmm.split(":")
    return time(int(h), int(m))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/tatsatshah/Desktop/yegedge && source .venv/bin/activate && pytest tests/risk/test_manager.py -v --no-cov
```

Expected: `38 passed`

If any test fails, read the failure and fix `manager.py`. Do not modify the test file.

- [ ] **Step 5: Lint and format**

```bash
ruff check agent/risk/manager.py tests/risk/test_manager.py
black agent/risk/manager.py tests/risk/test_manager.py
```

Fix any ruff violations. Re-run tests after fixes.

- [ ] **Step 6: Commit**

```bash
git add agent/risk/manager.py tests/risk/test_manager.py
git commit -m "feat(risk): add RiskManager enforcing all 12 risk rules from risk_rules.yaml"
```

---

## Task 3: Full test suite + coverage gate

**Files:**
- Run full suite, add targeted tests if `agent/risk/` coverage falls below 70%.

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/tatsatshah/Desktop/yegedge && source .venv/bin/activate
pytest tests/ -v --cov=agent --cov-report=term-missing
```

Expected: all 190+ tests pass. Coverage for `agent/risk/` ≥ 70%.

Check the `term-missing` column for `agent/risk/manager.py`. Typical uncovered lines:
- The `risk_per_share <= 0` guard in `_size_position` (unreachable in normal use since Signal validates stop < target, so entry − stop > 0 when entry > stop; but cover it anyway).

If coverage is below 70%, add a targeted test to `tests/risk/test_manager.py`:

```python
def test_size_position_with_inverted_stop_rejects(rm: RiskManager) -> None:
    """If entry_price <= stop (abnormal), _size_position rejects with ZERO_QUANTITY."""
    # entry=1660, stop=1660 → risk_per_share=0 → ZERO_QUANTITY
    sig = _signal(suggested_stop=Decimal("1660.00"), suggested_target=Decimal("1780.00"))
    decision = rm.evaluate(sig, _portfolio(), Decimal("1660.00"))
    assert decision.verdict == RiskVerdict.REJECTED
    assert decision.rejection_reason == RejectionReason.ZERO_QUANTITY
```

- [ ] **Step 2: Run linters on all risk files**

```bash
ruff check agent/risk/ tests/risk/
black --check agent/risk/ tests/risk/
```

Fix any issues:

```bash
black agent/risk/ tests/risk/
ruff check --fix agent/risk/ tests/risk/
```

- [ ] **Step 3: Spot-check — full pipeline → strategy → risk**

```bash
source .venv/bin/activate && python - <<'EOF'
import math
from decimal import Decimal
from datetime import datetime
from zoneinfo import ZoneInfo
import polars as pl

from tests.features.conftest import make_ohlcv_df
from agent.features.pipeline import FeaturePipeline
from agent.features.regime import RegimeDetector
from agent.strategies.trend_following import TrendFollowingStrategy
from agent.risk.rules import load_risk_rules
from agent.risk.manager import RiskManager
from agent.risk.types import PortfolioState
from agent.data.types import Position

IST = ZoneInfo("Asia/Kolkata")

# Build enriched data
closes = [1500.0 + 200.0 * math.sin(i * 0.15) for i in range(200)]
df = make_ohlcv_df(closes)
enriched = FeaturePipeline().run(df)
rd = RegimeDetector()
rd.fit(enriched)
final = FeaturePipeline(regime_detector=rd).run(df)
final = final.with_columns(pl.lit("ok").alias("data_quality"))

# Generate signals
signals = TrendFollowingStrategy().generate(final)
print(f"Signals: {len(signals)}")

# Risk-evaluate each signal
rm = RiskManager(load_risk_rules())
portfolio = PortfolioState(
    nav=Decimal("83000"),
    cash=Decimal("60000"),
    positions={},
    daily_pnl=Decimal("0"),
    weekly_pnl=Decimal("0"),
    peak_nav=Decimal("83000"),
    orders_today=0,
    last_order_time={},
    kill_switch_active=False,
    evaluation_time=datetime(2024, 1, 2, 10, 0, tzinfo=IST),
)
for sig in signals[:5]:
    entry = float(final.filter(pl.col("timestamp") == sig.timestamp)["close"][0])
    decision = rm.evaluate(sig, portfolio, Decimal(str(round(entry, 2))))
    print(f"  {sig.action} {sig.symbol} → {decision.verdict} qty={decision.quantity} reason={decision.rejection_reason}")
EOF
```

Expected: each signal prints a verdict and quantity (or rejection reason). No exceptions.

- [ ] **Step 4: Commit**

```bash
git add agent/risk/ tests/risk/
git commit -m "test(risk): Phase 4 test suite passes coverage gate; all rules trigger-tested"
```

---

## Self-Review

**Spec coverage:**
- [x] Kill switch active → KILL_SWITCH_ACTIVE rejection — Task 2
- [x] Suspect/missing data quality → SUSPECT_DATA_QUALITY rejection — Task 2
- [x] R/R < min_reward_risk (1.5) → INSUFFICIENT_REWARD_RISK rejection — Task 2
- [x] Outside trading window 09:45–14:45 IST → OUTSIDE_TRADING_WINDOW rejection — Task 2
- [x] Max concurrent positions (6) → MAX_POSITIONS_REACHED rejection — Task 2
- [x] Cash below 10% of NAV → INSUFFICIENT_CASH rejection — Task 2
- [x] Daily loss ≥ 2% of NAV → DAILY_LOSS_CAP rejection — Task 2
- [x] Weekly loss ≥ 5% of NAV → WEEKLY_LOSS_CAP rejection — Task 2
- [x] Drawdown ≥ 8% → DRAWDOWN_BREACH rejection — Task 2
- [x] Max orders today (4) → MAX_ORDERS_TODAY rejection — Task 2
- [x] Symbol cooldown (30 min) → SYMBOL_COOLDOWN rejection — Task 2
- [x] Sizing → 0 shares → ZERO_QUANTITY rejection — Task 2
- [x] EXIT_LONG bypasses entry checks — Task 2
- [x] EXIT_LONG blocked by kill switch — Task 2
- [x] Position sizing formula: min(qty_by_risk, qty_by_size) — Task 2
- [x] Config loaded from risk_rules.yaml into typed Pydantic models — Task 1
- [x] One test per rule that deliberately triggers it — Task 2 test file

**Placeholder scan:** None found. All code is complete.

**Type consistency:**
- `RiskDecision` constructor called with all 10 fields in `manager.py` — matches `types.py`
- `RejectionReason.NONE` used for approved decisions — consistent
- `Decimal(str(...))` pattern used for all float→Decimal conversions — consistent
- `_parse_time("09:45")` returns `datetime.time(9, 45)` — used in `__init__`, not per-call — consistent
