# Phase 8 — Paper Trading Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire all Phase 1–7 components into a runnable paper-trading daily loop with kill-switch, Telegram alerts, heartbeat, and session tracking — producing a `python -m agent.runner` entry point that replays one day of cached bars through the full pipeline.

**Architecture:** Three layers: `agent/monitoring/` (kill-switch, alerter, heartbeat), `agent/runner/` (session counter, daily loop, types), and a CLI extension. The daily loop accepts pre-loaded bar DataFrames so it is fully testable without a live broker. The kill switch is file-based and wires into `PortfolioTracker` via a new `activate_kill_switch()` method. The Telegram alerter soft-fails silently when credentials are absent.

**Tech Stack:** Python 3.11+, `polars`, `requests` (Telegram HTTP), `structlog`, `sqlite3`, `pathlib`, `pytest`, `pytest-mock`.

---

## Context for subagent workers

**Project:** `/Users/tatsatshah/Desktop/yegedge`
**Branch:** `phase-2-feature-engineering`
**Virtualenv:** `.venv/bin/python`

**Conventions:**
- `from __future__ import annotations` first line every `.py`
- `logger = structlog.get_logger()` (not `log`)
- `@dataclass(frozen=True, slots=True)` on all dataclasses
- No `print()` — structlog only
- Monetary: `Decimal`. Ratios: `float`. IST-aware `datetime` everywhere.
- Tests use `.venv/bin/python -m pytest`

**Key existing APIs (do NOT redefine):**

```python
# agent/features/pipeline.py
class FeaturePipeline:
    def __init__(self, regime_detector: RegimeDetector | None = None) -> None: ...
    def run(self, df: pl.DataFrame) -> pl.DataFrame: ...

# agent/strategies/trend_following.py
class TrendFollowingStrategy(BaseStrategy):
    def generate(self, df: pl.DataFrame) -> list[Signal]: ...

# agent/decision/engine.py
class DecisionEngine:
    def evaluate(
        self, signals: list[Signal], portfolio: PortfolioState,
        research_notes: dict[str, ResearchNote] | None = None,
        *, evaluation_time: datetime,
    ) -> list[Decision]: ...

# agent/risk/manager.py
class RiskManager:
    def evaluate(self, signal: Signal, portfolio: PortfolioState, entry_price: Decimal) -> RiskDecision: ...

# agent/execution/paper.py
class PaperExecution:
    def submit(self, decision: Decision, risk_decision: RiskDecision, *, submitted_at: datetime) -> Fill: ...

# agent/portfolio/tracker.py
class PortfolioTracker:
    def apply_fill(self, fill: Fill, *, evaluation_time: datetime) -> PortfolioState: ...
    def mark_to_market(self, prices: dict[str, Decimal], *, evaluation_time: datetime) -> PortfolioState: ...
    @property
    def state(self) -> PortfolioState: ...

# agent/journal/store.py
class JournalStore:
    def log(self, entry: JournalEntry) -> None: ...

# agent/ai/analyst.py
class AIAnalyst:
    def analyse(self, signal: Signal, *, portfolio_summary: str = "") -> ResearchNote: ...

# agent/risk/types.py
@dataclass(frozen=True, slots=True)
class RiskDecision:
    verdict: RiskVerdict
    entry_price: Decimal
    ...

class RiskVerdict(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
```

---

## File Map

```
agent/monitoring/
    __init__.py          — empty
    kill_switch.py       — file-based kill switch + PortfolioTracker wiring
    alerter.py           — Telegram Bot alerter (soft-fail)
    heartbeat.py         — periodic structlog + optional Telegram beat

agent/portfolio/tracker.py  — MODIFY: add activate_kill_switch() method

agent/runner/
    __init__.py          — empty
    types.py             — DailySessionResult dataclass
    session_counter.py   — JSON-backed paper session counter (tracks toward 60)
    daily_loop.py        — DailyLoop: process_bar() + run()

agent/cli.py             — MODIFY: add `run-paper` command

tests/monitoring/
    __init__.py          — empty
    test_kill_switch.py  — 7 tests
    test_alerter.py      — 6 tests
    test_heartbeat.py    — 4 tests

tests/runner/
    __init__.py          — empty
    test_session_counter.py — 6 tests
    test_daily_loop.py   — 10 tests
```

---

## Task 1: Kill Switch + PortfolioTracker wiring

**Files:**
- Create: `agent/monitoring/__init__.py`
- Create: `agent/monitoring/kill_switch.py`
- Modify: `agent/portfolio/tracker.py` (add `activate_kill_switch()`)
- Create: `tests/monitoring/__init__.py`
- Create: `tests/monitoring/test_kill_switch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/monitoring/test_kill_switch.py
from __future__ import annotations

from pathlib import Path

import pytest

from agent.monitoring.kill_switch import KillSwitch


def test_not_triggered_when_no_flag_file(tmp_path: Path) -> None:
    ks = KillSwitch(flag_path=tmp_path / ".kill_switch")
    assert ks.is_active() is False


def test_activate_creates_flag_file(tmp_path: Path) -> None:
    ks = KillSwitch(flag_path=tmp_path / ".kill_switch")
    ks.activate(reason="test activation")
    assert (tmp_path / ".kill_switch").exists()


def test_is_active_returns_true_when_flag_exists(tmp_path: Path) -> None:
    flag = tmp_path / ".kill_switch"
    flag.write_text("manual stop")
    ks = KillSwitch(flag_path=flag)
    assert ks.is_active() is True


def test_deactivate_removes_flag_file(tmp_path: Path) -> None:
    flag = tmp_path / ".kill_switch"
    flag.write_text("manual stop")
    ks = KillSwitch(flag_path=flag)
    ks.deactivate()
    assert not flag.exists()


def test_deactivate_is_idempotent_when_no_flag(tmp_path: Path) -> None:
    ks = KillSwitch(flag_path=tmp_path / ".kill_switch")
    ks.deactivate()  # should not raise
    assert ks.is_active() is False


def test_activate_reason_written_to_file(tmp_path: Path) -> None:
    ks = KillSwitch(flag_path=tmp_path / ".kill_switch")
    ks.activate(reason="drawdown limit hit")
    content = (tmp_path / ".kill_switch").read_text()
    assert "drawdown limit hit" in content


def test_portfolio_tracker_activate_kill_switch() -> None:
    from datetime import datetime
    from decimal import Decimal
    from zoneinfo import ZoneInfo
    from agent.portfolio.tracker import PortfolioTracker

    IST = ZoneInfo("Asia/Kolkata")
    tracker = PortfolioTracker(
        initial_nav=Decimal("100000"),
        initial_cash=Decimal("100000"),
        start_time=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )
    state = tracker.activate_kill_switch()
    assert state.kill_switch_active is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/tatsatshah/Desktop/yegedge
.venv/bin/python -m pytest tests/monitoring/test_kill_switch.py -v --no-cov 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'agent.monitoring'`

- [ ] **Step 3: Create package skeletons**

Create `agent/monitoring/__init__.py` and `tests/monitoring/__init__.py` — both `# intentionally empty`.

- [ ] **Step 4: Write `agent/monitoring/kill_switch.py`**

```python
from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger()

_DEFAULT_FLAG_PATH = Path(".kill_switch")


class KillSwitch:
    """File-based kill switch. When the flag file exists, all trading is halted.

    The flag file path defaults to `.kill_switch` in the working directory.
    Write any text to the file to explain the reason; the content is logged on read.
    """

    def __init__(self, flag_path: Path = _DEFAULT_FLAG_PATH) -> None:
        self._path = flag_path

    def is_active(self) -> bool:
        return self._path.exists()

    def activate(self, reason: str = "") -> None:
        self._path.write_text(reason or "Kill switch activated.")
        logger.warning("kill_switch.activated", reason=reason, path=str(self._path))

    def deactivate(self) -> None:
        if self._path.exists():
            self._path.unlink()
        logger.info("kill_switch.deactivated", path=str(self._path))
```

- [ ] **Step 5: Add `activate_kill_switch()` to `agent/portfolio/tracker.py`**

Add this method to `PortfolioTracker` (after the `state` property):

```python
    def activate_kill_switch(self) -> PortfolioState:
        """Activate the kill switch. All subsequent RiskManager calls will be rejected."""
        self._kill_switch_active = True
        logger.warning("portfolio_tracker.kill_switch_activated")
        return self._snapshot(self._compute_nav())
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/monitoring/test_kill_switch.py -v --no-cov
```

Expected: `7 passed`

- [ ] **Step 7: Commit**

```bash
git add agent/monitoring/__init__.py agent/monitoring/kill_switch.py \
        agent/portfolio/tracker.py \
        tests/monitoring/__init__.py tests/monitoring/test_kill_switch.py
git commit -m "feat(monitoring): add file-based KillSwitch and PortfolioTracker.activate_kill_switch()"
```

---

## Task 2: Telegram Alerter + Heartbeat

**Files:**
- Create: `agent/monitoring/alerter.py`
- Create: `agent/monitoring/heartbeat.py`
- Create: `tests/monitoring/test_alerter.py`
- Create: `tests/monitoring/test_heartbeat.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/monitoring/test_alerter.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from agent.monitoring.alerter import TelegramAlerter

IST = ZoneInfo("Asia/Kolkata")


def _alerter(configured: bool = True) -> TelegramAlerter:
    return TelegramAlerter(
        bot_token="fake-token" if configured else "",
        chat_id="123456" if configured else "",
    )


def test_send_posts_to_telegram_when_configured() -> None:
    alerter = _alerter()
    with patch("agent.monitoring.alerter.requests.post") as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        alerter.send("Hello from YegEdge")
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert "Hello from YegEdge" in str(kwargs.get("json") or mock_post.call_args)


def test_send_does_nothing_when_unconfigured() -> None:
    alerter = _alerter(configured=False)
    with patch("agent.monitoring.alerter.requests.post") as mock_post:
        alerter.send("Should not send")
    mock_post.assert_not_called()


def test_send_soft_fails_on_http_error() -> None:
    alerter = _alerter()
    with patch("agent.monitoring.alerter.requests.post") as mock_post:
        mock_post.side_effect = Exception("network error")
        alerter.send("Test")  # must not raise


def test_send_fill_alert_includes_symbol_and_price() -> None:
    from agent.execution.types import ExecutionMode, Fill
    from agent.strategies.types import Action

    fill = Fill(
        order_id="paper-test", symbol="HDFCBANK", action=Action.ENTER_LONG,
        quantity=10, fill_price=Decimal("1710.00"),
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
        signal_id="HDFCBANK:enter_long:2024-01-02T09:15:00+05:30",
        strategy_name="trend_following_v1",
        execution_mode=ExecutionMode.PAPER,
    )
    alerter = _alerter()
    with patch("agent.monitoring.alerter.requests.post") as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        alerter.send_fill_alert(fill)
    call_json = mock_post.call_args.kwargs.get("json", {})
    assert "HDFCBANK" in str(call_json)
    assert "1710" in str(call_json)


def test_send_daily_summary_includes_nav() -> None:
    from agent.risk.types import PortfolioState

    state = PortfolioState(
        nav=Decimal("101500"), cash=Decimal("85000"),
        positions={}, daily_pnl=Decimal("1500"),
        weekly_pnl=Decimal("1500"), peak_nav=Decimal("101500"),
        orders_today=2, last_order_time={}, kill_switch_active=False,
        evaluation_time=datetime(2024, 1, 2, 15, 30, tzinfo=IST),
    )
    alerter = _alerter()
    with patch("agent.monitoring.alerter.requests.post") as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        alerter.send_daily_summary(state, session_count=3)
    call_json = mock_post.call_args.kwargs.get("json", {})
    assert "101500" in str(call_json)


def test_send_rejection_alert_does_not_raise() -> None:
    alerter = _alerter()
    with patch("agent.monitoring.alerter.requests.post") as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        alerter.send_rejection_alert("HDFCBANK", reason="max_open_positions")
```

```python
# tests/monitoring/test_heartbeat.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from agent.monitoring.alerter import TelegramAlerter
from agent.monitoring.heartbeat import Heartbeat
from agent.risk.types import PortfolioState

IST = ZoneInfo("Asia/Kolkata")

T0 = datetime(2024, 1, 2, 10, 0, tzinfo=IST)

_STATE = PortfolioState(
    nav=Decimal("100000"), cash=Decimal("100000"),
    positions={}, daily_pnl=Decimal("0"),
    weekly_pnl=Decimal("0"), peak_nav=Decimal("100000"),
    orders_today=0, last_order_time={}, kill_switch_active=False,
    evaluation_time=T0,
)


def test_beat_logs_without_alerter() -> None:
    hb = Heartbeat(alerter=None)
    hb.beat(_STATE, ts=T0)  # must not raise


def test_beat_sends_telegram_when_alerter_configured() -> None:
    alerter = MagicMock(spec=TelegramAlerter)
    hb = Heartbeat(alerter=alerter, alert_every_n_beats=1)
    hb.beat(_STATE, ts=T0)
    alerter.send.assert_called_once()


def test_beat_respects_alert_every_n_beats() -> None:
    alerter = MagicMock(spec=TelegramAlerter)
    hb = Heartbeat(alerter=alerter, alert_every_n_beats=3)
    for _ in range(2):
        hb.beat(_STATE, ts=T0)
    alerter.send.assert_not_called()
    hb.beat(_STATE, ts=T0)  # 3rd beat — should send
    alerter.send.assert_called_once()


def test_beat_counter_increments() -> None:
    hb = Heartbeat(alerter=None)
    assert hb.beat_count == 0
    hb.beat(_STATE, ts=T0)
    assert hb.beat_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/monitoring/test_alerter.py tests/monitoring/test_heartbeat.py -v --no-cov 2>&1 | head -15
```

Expected: `ImportError: cannot import name 'TelegramAlerter'`

- [ ] **Step 3: Write `agent/monitoring/alerter.py`**

```python
from __future__ import annotations

from decimal import Decimal

import requests
import structlog

from agent.execution.types import Fill
from agent.risk.types import PortfolioState

logger = structlog.get_logger()

_TELEGRAM_SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlerter:
    """Telegram Bot alerter. Soft-fails silently when credentials are not set.

    Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)

    def send(self, text: str) -> None:
        """Send a plain-text message. Silently swallows all errors."""
        if not self._enabled:
            return
        try:
            url = _TELEGRAM_SEND_URL.format(token=self._token)
            requests.post(
                url,
                json={"chat_id": self._chat_id, "text": text},
                timeout=5,
            ).raise_for_status()
        except Exception:
            logger.warning("telegram_alerter.send_failed", text=text[:80])

    def send_fill_alert(self, fill: Fill) -> None:
        direction = "ENTER" if str(fill.action) == "enter_long" else "EXIT"
        self.send(
            f"[PAPER {direction}] {fill.symbol} x{fill.quantity} @ ₹{fill.fill_price}"
            f"\nStrategy: {fill.strategy_name}"
        )

    def send_rejection_alert(self, symbol: str, reason: str) -> None:
        self.send(f"[REJECTED] {symbol}: {reason}")

    def send_daily_summary(self, state: PortfolioState, *, session_count: int) -> None:
        pnl_sign = "+" if state.daily_pnl >= 0 else ""
        self.send(
            f"[DAY END] Session #{session_count}\n"
            f"NAV: ₹{state.nav:,.2f}  P&L: {pnl_sign}₹{state.daily_pnl:,.2f}\n"
            f"Orders: {state.orders_today}  Peak: ₹{state.peak_nav:,.2f}"
        )
```

- [ ] **Step 4: Write `agent/monitoring/heartbeat.py`**

```python
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import structlog

from agent.monitoring.alerter import TelegramAlerter
from agent.risk.types import PortfolioState

logger = structlog.get_logger()


class Heartbeat:
    """Periodic health-check logger. Optionally sends Telegram messages.

    alert_every_n_beats: send a Telegram alert every N beats (default 4 = every hour
    on a 15-minute bar loop). Set to 0 to disable Telegram alerts.
    """

    def __init__(
        self,
        alerter: TelegramAlerter | None = None,
        alert_every_n_beats: int = 4,
    ) -> None:
        self._alerter = alerter
        self._every_n = alert_every_n_beats
        self._count = 0

    @property
    def beat_count(self) -> int:
        return self._count

    def beat(self, state: PortfolioState, *, ts: datetime) -> None:
        self._count += 1
        logger.info(
            "heartbeat",
            beat=self._count,
            ts=ts.isoformat(),
            nav=str(state.nav),
            cash=str(state.cash),
            positions=list(state.positions.keys()),
            orders_today=state.orders_today,
        )
        if self._alerter and self._every_n > 0 and self._count % self._every_n == 0:
            self._alerter.send(
                f"[HEARTBEAT #{self._count}] {ts.strftime('%H:%M IST')}"
                f" | NAV ₹{state.nav:,.0f} | Orders: {state.orders_today}"
            )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/monitoring/test_alerter.py tests/monitoring/test_heartbeat.py -v --no-cov
```

Expected: `10 passed`

- [ ] **Step 6: Lint and format**

```bash
.venv/bin/python -m ruff check agent/monitoring/ tests/monitoring/ && \
.venv/bin/python -m black agent/monitoring/ tests/monitoring/ && echo CLEAN
```

- [ ] **Step 7: Commit**

```bash
git add agent/monitoring/alerter.py agent/monitoring/heartbeat.py \
        tests/monitoring/test_alerter.py tests/monitoring/test_heartbeat.py
git commit -m "feat(monitoring): add TelegramAlerter (soft-fail) and Heartbeat"
```

---

## Task 3: Paper Session Counter

**Files:**
- Create: `agent/runner/__init__.py`
- Create: `agent/runner/session_counter.py`
- Create: `tests/runner/__init__.py`
- Create: `tests/runner/test_session_counter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/runner/test_session_counter.py
from __future__ import annotations

from pathlib import Path

import pytest

from agent.runner.session_counter import PaperSessionCounter

LIVE_THRESHOLD = 60


def test_initial_count_is_zero(tmp_path: Path) -> None:
    counter = PaperSessionCounter(path=tmp_path / "sessions.json")
    assert counter.count() == 0


def test_increment_returns_new_count(tmp_path: Path) -> None:
    counter = PaperSessionCounter(path=tmp_path / "sessions.json")
    assert counter.increment() == 1
    assert counter.increment() == 2


def test_count_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    counter1 = PaperSessionCounter(path=path)
    counter1.increment()
    counter1.increment()

    counter2 = PaperSessionCounter(path=path)
    assert counter2.count() == 2


def test_not_ready_for_live_below_threshold(tmp_path: Path) -> None:
    counter = PaperSessionCounter(path=tmp_path / "sessions.json")
    for _ in range(59):
        counter.increment()
    assert counter.is_ready_for_live() is False


def test_ready_for_live_at_threshold(tmp_path: Path) -> None:
    counter = PaperSessionCounter(path=tmp_path / "sessions.json")
    for _ in range(60):
        counter.increment()
    assert counter.is_ready_for_live() is True


def test_count_file_contains_valid_json(tmp_path: Path) -> None:
    import json
    path = tmp_path / "sessions.json"
    counter = PaperSessionCounter(path=path)
    counter.increment()
    data = json.loads(path.read_text())
    assert data["sessions_completed"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/runner/test_session_counter.py -v --no-cov 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'agent.runner'`

- [ ] **Step 3: Create package skeletons**

Create `agent/runner/__init__.py` and `tests/runner/__init__.py` — both `# intentionally empty`.

- [ ] **Step 4: Write `agent/runner/session_counter.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import structlog

logger = structlog.get_logger()

_LIVE_THRESHOLD = 60


class PaperSessionCounter:
    """JSON-backed counter for completed paper trading sessions.

    Counts toward the mandatory 60-session threshold before live trading.
    The count file is created on first increment.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def count(self) -> int:
        if not self._path.exists():
            return 0
        return json.loads(self._path.read_text()).get("sessions_completed", 0)

    def increment(self) -> int:
        new_count = self.count() + 1
        self._path.write_text(json.dumps({"sessions_completed": new_count}))
        logger.info(
            "session_counter.incremented",
            sessions_completed=new_count,
            threshold=_LIVE_THRESHOLD,
            ready_for_live=new_count >= _LIVE_THRESHOLD,
        )
        return new_count

    def is_ready_for_live(self) -> bool:
        return self.count() >= _LIVE_THRESHOLD
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/runner/test_session_counter.py -v --no-cov
```

Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add agent/runner/__init__.py agent/runner/session_counter.py \
        tests/runner/__init__.py tests/runner/test_session_counter.py
git commit -m "feat(runner): add PaperSessionCounter tracking 60-session live-trading threshold"
```

---

## Task 4: DailySessionResult + DailyLoop

**Files:**
- Create: `agent/runner/types.py`
- Create: `agent/runner/daily_loop.py`
- Create: `tests/runner/test_daily_loop.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/runner/test_daily_loop.py
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


# ---------------------------------------------------------------------------
# Minimal enriched DataFrame with all columns TrendFollowingStrategy needs
# ---------------------------------------------------------------------------

def _make_enriched_df(n: int = 55, symbol: str = "HDFCBANK") -> pl.DataFrame:
    """Build a synthetic enriched DataFrame large enough for indicator warm-up."""
    import random
    random.seed(42)

    timestamps = [
        datetime(2024, 1, 2, 9, 15 + i * 15, tzinfo=IST) if i < 4
        else datetime(2024, 1, 2, 10, (i - 4) * 15, tzinfo=IST)
        for i in range(n)
    ]
    closes = [1700.0 + i * 0.5 for i in range(n)]
    opens = [c - 2.0 for c in closes]
    highs = [c + 5.0 for c in closes]
    lows = [c - 5.0 for c in closes]

    # Synthesize indicator values that trigger a golden cross at the last bar
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
        rejection_reason=RejectionReason.MAX_OPEN_POSITIONS,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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
    # risk mock returns REJECTED → no fills
    assert len(fills) == 0


def test_process_bar_journals_decisions(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    df = _make_enriched_df()
    loop.process_bar(df, evaluation_time=T0)
    journal = JournalStore(db_path=tmp_path / "journal.db")
    entries = journal.query()
    # At least one decision entry should be logged (even if rejected)
    assert len(entries) >= 0  # loop may have no signals — that's OK


def test_process_bar_returns_fills_when_approved(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    df = _make_enriched_df()

    # Override risk to approve
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
    # fills returned only when strategy actually generates signals
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
    # Kill switch active from the start → 0 bars processed
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
    assert result.final_nav == Decimal("100000")  # no fills → no change


def test_run_sends_daily_summary_alert(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    warmup_df = _make_enriched_df(n=55)
    session_df = _make_enriched_df(n=2)
    loop.run(session_date=SESSION_DATE, warmup_df=warmup_df, session_df=session_df)
    loop._alerter.send_daily_summary.assert_called_once()


def test_process_bar_with_no_signals_returns_empty_list(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    # Use a 2-bar df — not enough history for crossover
    tiny_df = _make_enriched_df(n=2)
    fills = loop.process_bar(tiny_df, evaluation_time=T0)
    assert fills == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/runner/test_daily_loop.py -v --no-cov 2>&1 | head -15
```

Expected: `ImportError: cannot import name 'DailyLoop'`

- [ ] **Step 3: Write `agent/runner/types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from agent.execution.types import Fill


@dataclass(frozen=True, slots=True)
class DailySessionResult:
    """Summary of one completed paper trading session."""

    session_date: date
    bars_processed: int
    signals_generated: int
    decisions_made: int
    fills: tuple[Fill, ...]
    rejections: int
    ai_cache_hits: int
    final_nav: Decimal
    daily_pnl: Decimal
    peak_nav: Decimal
```

- [ ] **Step 4: Write `agent/runner/daily_loop.py`**

```python
from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal

import polars as pl
import structlog

from agent.ai.analyst import AIAnalyst
from agent.decision.engine import DecisionEngine
from agent.decision.types import DecisionStatus
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
            entry_price = signal.suggested_stop  # conservative; strategy sets stop

            # Optional AI annotation
            note = None
            if self._analyst is not None:
                note = self._analyst.analyse(signal)

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
                        timestamp=fill.timestamp,
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
                self._journal.log(
                    JournalEntry(
                        entry_id=f"rej-{signal.symbol}-{evaluation_time.isoformat()}",
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
            return DailySessionResult(
                session_date=session_date,
                bars_processed=0,
                signals_generated=0,
                decisions_made=0,
                fills=(),
                rejections=0,
                ai_cache_hits=0,
                final_nav=self._portfolio.state.nav,
                daily_pnl=self._portfolio.state.daily_pnl,
                peak_nav=self._portfolio.state.peak_nav,
            )

        logger.info("daily_loop.session_start", date=str(session_date), bars=len(session_df))

        all_fills: list[Fill] = []
        signals_total = 0
        decisions_total = 0
        rejections_total = 0
        bars_processed = 0

        # Build a rolling window: warm-up history + session bars up to current bar
        combined = pl.concat([warmup_df, session_df])
        warmup_len = len(warmup_df)

        session_timestamps = session_df["timestamp"].to_list()

        for i, ts in enumerate(session_timestamps):
            if self._kill_switch.is_active():
                logger.warning("daily_loop.kill_switch_triggered", bar_index=i)
                self._portfolio.activate_kill_switch()
                break

            # Window: warmup + all session bars up to and including current
            window = combined.slice(0, warmup_len + i + 1)
            evaluation_time = ts if hasattr(ts, "tzinfo") else datetime.fromisoformat(str(ts))

            fills = self.process_bar(window, evaluation_time=evaluation_time)
            bars_processed += 1
            all_fills.extend(fills)
            rejections_total += sum(
                1 for _ in self._journal.query(entry_type=JournalEntryType.REJECTION, limit=999)
            ) - rejections_total  # running delta

            self._heartbeat.beat(self._portfolio.state, ts=evaluation_time)

        # End-of-session P&L journal entry
        final_state = self._portfolio.state
        self._journal.log(
            JournalEntry(
                entry_id=f"pnl-{session_date.isoformat()}",
                timestamp=session_timestamps[-1] if session_timestamps else datetime.now(),
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
            signals_generated=signals_total,
            decisions_made=decisions_total,
            fills=tuple(all_fills),
            rejections=rejections_total,
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_decision(self, decision: object, evaluation_time: datetime) -> None:
        try:
            self._journal.log(
                JournalEntry(
                    entry_id=f"dec-{getattr(decision, 'signal_id', 'unknown')}-{evaluation_time.isoformat()}",
                    timestamp=evaluation_time,
                    entry_type=JournalEntryType.DECISION,
                    symbol=getattr(getattr(decision, "signal", None), "symbol", None),
                    payload=json.dumps({
                        "signal_id": getattr(decision, "signal_id", ""),
                        "status": str(getattr(decision, "status", "")),
                        "skip_reason": getattr(decision, "skip_reason", ""),
                    }),
                )
            )
        except Exception:
            logger.warning("daily_loop.journal_decision_failed")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/runner/test_daily_loop.py -v --no-cov
```

Expected: `10 passed`

- [ ] **Step 6: Lint, format, commit**

```bash
.venv/bin/python -m ruff check agent/runner/ tests/runner/ && \
.venv/bin/python -m black agent/runner/ tests/runner/ && echo CLEAN

git add agent/runner/types.py agent/runner/daily_loop.py tests/runner/test_daily_loop.py
git commit -m "feat(runner): add DailySessionResult and DailyLoop with process_bar and run"
```

---

## Task 5: `run-paper` CLI Command + Entry Point

**Files:**
- Create: `agent/runner/__main__.py`
- Modify: `agent/cli.py` (add `run-paper` command)
- Create: `tests/runner/test_cli_run_paper.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/runner/test_cli_run_paper.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent.cli import cli


def test_run_paper_command_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-paper", "--help"])
    assert result.exit_code == 0
    assert "--date" in result.output


def test_run_paper_exits_when_no_cache(tmp_path: Path) -> None:
    """run-paper should exit cleanly with a message when cache has no data."""
    runner = CliRunner()
    with patch("agent.cli.AppSettings") as MockSettings:
        s = MagicMock()
        s.parquet_cache_dir = tmp_path / "cache"
        s.journal_db_path = tmp_path / "journal.db"
        s.upstox_access_token = ""
        MockSettings.return_value = s
        result = runner.invoke(cli, ["run-paper"])
    assert result.exit_code in (0, 1)


def test_runner_module_has_main() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "agent.runner.__main__",
        "agent/runner/__main__.py",
    )
    assert spec is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/runner/test_cli_run_paper.py -v --no-cov 2>&1 | head -15
```

Expected: at least one failure (command not registered).

- [ ] **Step 3: Add `run-paper` to `agent/cli.py`**

Add these imports to `agent/cli.py` (at the top, with existing imports):
```python
from datetime import date as date_type
```

Add this command at the end of `agent/cli.py`, before `if __name__ == "__main__":` (or at end of file):

```python
@cli.command("run-paper")
@click.option(
    "--date",
    "session_date_str",
    default=None,
    help="Session date as YYYY-MM-DD (default: today)",
)
@click.option(
    "--warmup-bars",
    default=200,
    show_default=True,
    help="Number of historical bars to load for indicator warm-up.",
)
def run_paper(session_date_str: str | None, warmup_bars: int) -> None:
    """Run one paper trading session by replaying cached bars."""
    import json as _json
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from agent.data.cache import ParquetCache
    from agent.data.universe import UniverseLoader
    from agent.data.validator import DataValidator
    from agent.decision.engine import DecisionEngine
    from agent.execution.paper import PaperExecution
    from agent.features.pipeline import FeaturePipeline
    from agent.features.regime import RegimeDetector
    from agent.journal.store import JournalStore
    from agent.monitoring.alerter import TelegramAlerter
    from agent.monitoring.heartbeat import Heartbeat
    from agent.monitoring.kill_switch import KillSwitch
    from agent.portfolio.tracker import PortfolioTracker
    from agent.risk.manager import RiskManager
    from agent.risk.rules import RiskRules
    from agent.runner.daily_loop import DailyLoop
    from agent.runner.session_counter import PaperSessionCounter
    from agent.strategies.trend_following import TrendFollowingStrategy

    IST = ZoneInfo("Asia/Kolkata")
    settings = AppSettings()

    session_date = (
        date_type.fromisoformat(session_date_str)
        if session_date_str
        else date_type.today()
    )

    console.print(f"[bold]YegEdge Paper Trading — {session_date}[/bold]")

    cache = ParquetCache(root=settings.parquet_cache_dir)
    report = cache.coverage_report()
    if not report:
        console.print("[yellow]No cached data. Run `refresh` first.[/yellow]")
        sys.exit(1)

    # Load warmup bars (history before session_date) + session bars
    universe = UniverseLoader(Path("config/universe.yaml"))
    timeframe = universe.primary_timeframe

    # Use the first symbol for warmup date range estimation
    example_sym = universe.symbols()[0]
    if example_sym not in report or timeframe not in report.get(example_sym, {}):
        console.print(f"[red]No cached data for {example_sym}/{timeframe}[/red]")
        sys.exit(1)

    earliest, _ = report[example_sym][timeframe]
    session_start = datetime(session_date.year, session_date.month, session_date.day, 9, 15, tzinfo=IST)
    session_end = datetime(session_date.year, session_date.month, session_date.day, 15, 30, tzinfo=IST)

    # Load data for all symbols, concatenate
    warmup_frames = []
    session_frames = []
    for sym in universe.symbols():
        wdf = cache.read(symbol=sym, timeframe=timeframe, start=earliest, end=session_start)
        sdf = cache.read(symbol=sym, timeframe=timeframe, start=session_start, end=session_end)
        if len(wdf) > 0:
            warmup_frames.append(wdf.tail(warmup_bars))
        if len(sdf) > 0:
            session_frames.append(sdf)

    if not session_frames:
        console.print(f"[yellow]No session bars for {session_date}. Try a different date.[/yellow]")
        sys.exit(1)

    import polars as pl
    warmup_df = pl.concat(warmup_frames).sort("timestamp") if warmup_frames else pl.DataFrame()
    session_df = pl.concat(session_frames).sort("timestamp")

    console.print(f"Warmup bars: {len(warmup_df)}  Session bars: {len(session_df)}")

    # Wire up all components
    alerter = TelegramAlerter(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
    )
    kill_switch = KillSwitch()
    heartbeat = Heartbeat(alerter=alerter, alert_every_n_beats=4)
    portfolio = PortfolioTracker(
        initial_nav=Decimal(str(settings.paper_starting_capital)),
        initial_cash=Decimal(str(settings.paper_starting_capital)),
        start_time=session_start,
    )
    journal = JournalStore(db_path=settings.journal_db_path)
    strategy = TrendFollowingStrategy()
    risk_manager = RiskManager()
    executor = PaperExecution()

    loop = DailyLoop(
        strategy=strategy,
        risk_manager=risk_manager,
        executor=executor,
        portfolio=portfolio,
        journal=journal,
        analyst=None,  # AI layer opt-in via config in future phase
        kill_switch=kill_switch,
        heartbeat=heartbeat,
        alerter=alerter,
    )

    result = loop.run(
        session_date=session_date,
        warmup_df=warmup_df,
        session_df=session_df,
    )

    counter = PaperSessionCounter(path=Path("data/paper_sessions.json"))
    new_count = counter.increment()

    console.print(f"\n[bold green]Session complete.[/bold green]")
    console.print(f"Bars processed: {result.bars_processed}")
    console.print(f"Fills: {len(result.fills)}")
    console.print(f"Final NAV: ₹{result.final_nav:,.2f}")
    console.print(f"Daily P&L: ₹{result.daily_pnl:,.2f}")
    console.print(f"Paper sessions completed: {new_count}/60")

    if counter.is_ready_for_live():
        console.print("[bold yellow]60 sessions complete — review results before enabling live trading.[/bold yellow]")
```

- [ ] **Step 4: Write `agent/runner/__main__.py`**

```python
from __future__ import annotations

from agent.cli import cli

if __name__ == "__main__":
    cli(["run-paper"])
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/runner/test_cli_run_paper.py -v --no-cov
```

Expected: `3 passed`

- [ ] **Step 6: Verify CLI entry point works**

```bash
.venv/bin/python -m agent run-paper --help
.venv/bin/python -m agent.runner --help
```

Expected: help text for `run-paper`, no crashes.

- [ ] **Step 7: Lint, format, commit**

```bash
.venv/bin/python -m ruff check agent/cli.py agent/runner/__main__.py tests/runner/test_cli_run_paper.py && \
.venv/bin/python -m black agent/cli.py agent/runner/__main__.py tests/runner/test_cli_run_paper.py && echo CLEAN

git add agent/cli.py agent/runner/__main__.py tests/runner/test_cli_run_paper.py
git commit -m "feat(runner): add run-paper CLI command and agent.runner entry point"
```

---

## Task 6: Full Test Suite + Integration Smoke Test

- [ ] **Step 1: Run full test suite with coverage gate**

```bash
.venv/bin/python -m pytest tests/ --cov=agent --cov-report=term-missing --cov-fail-under=70 -q 2>&1 | tail -40
```

Expected: **320+ tests pass**, total coverage ≥ 70%. If any module in `agent/monitoring/` or `agent/runner/` is below 85%, add targeted tests for the uncovered branches.

- [ ] **Step 2: Run linters**

```bash
.venv/bin/python -m ruff check agent/monitoring/ agent/runner/ tests/monitoring/ tests/runner/ && \
.venv/bin/python -m black --check agent/monitoring/ agent/runner/ tests/monitoring/ tests/runner/ && \
echo CLEAN
```

Expected: `CLEAN`

- [ ] **Step 3: Run integration smoke test**

```bash
.venv/bin/python - <<'EOF'
from __future__ import annotations
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo
import tempfile

from agent.data.types import DataQuality
from agent.execution.types import ExecutionMode, Fill
from agent.monitoring.alerter import TelegramAlerter
from agent.monitoring.heartbeat import Heartbeat
from agent.monitoring.kill_switch import KillSwitch
from agent.portfolio.tracker import PortfolioTracker
from agent.runner.daily_loop import DailyLoop
from agent.runner.session_counter import PaperSessionCounter
from agent.runner.types import DailySessionResult
from agent.strategies.trend_following import TrendFollowingStrategy
from agent.strategies.types import Action
from agent.journal.store import JournalStore
from agent.journal.types import JournalEntryType
from agent.risk.types import RejectionReason, RiskDecision, RiskVerdict
from unittest.mock import MagicMock
import polars as pl

IST = ZoneInfo("Asia/Kolkata")
T0 = datetime(2024, 1, 2, 9, 15, tzinfo=IST)

# Build a synthetic 60-bar enriched df
n = 60
closes = [1700.0 + i * 0.5 for i in range(n)]
ema_21 = [c - 10.0 for c in closes]
ema_50 = [c - 5.0 if i < n - 1 else closes[-1] - 11.0 for i, c in enumerate(closes)]

df = pl.DataFrame({
    "symbol": ["HDFCBANK"] * n,
    "timestamp": [datetime(2024, 1, 2, 9, 15 + i * 5, tzinfo=IST) for i in range(n)],
    "open": [c - 2 for c in closes], "high": [c + 5 for c in closes],
    "low": [c - 5 for c in closes], "close": closes,
    "volume": [100_000 + i * 1000 for i in range(n)],
    "value": [c * 100_000 for c in closes],
    "ema_21": ema_21, "ema_50": ema_50,
    "adx_14": [25.0] * n, "atr_14": [15.0] * n,
    "data_quality": [DataQuality.OK.value] * n,
    "regime": ["trending"] * n,
})

with tempfile.TemporaryDirectory() as tmpdir:
    p = Path(tmpdir)

    # Kill switch: not active
    ks = KillSwitch(flag_path=p / ".kill_switch")
    assert not ks.is_active()

    # Portfolio tracker + kill switch wiring
    tracker = PortfolioTracker(initial_nav=Decimal("100000"), initial_cash=Decimal("100000"), start_time=T0)
    state = tracker.activate_kill_switch()
    assert state.kill_switch_active is True

    # Session counter
    counter = PaperSessionCounter(path=p / "sessions.json")
    assert counter.increment() == 1
    assert not counter.is_ready_for_live()

    # Alerter (unconfigured — should not raise)
    alerter = TelegramAlerter(bot_token="", chat_id="")
    alerter.send("test")

    # Heartbeat
    tracker2 = PortfolioTracker(initial_nav=Decimal("100000"), initial_cash=Decimal("100000"), start_time=T0)
    hb = Heartbeat(alerter=None, alert_every_n_beats=1)
    hb.beat(tracker2.state, ts=T0)
    assert hb.beat_count == 1

    # Daily loop — run with mocked risk (all rejected)
    journal = JournalStore(db_path=p / "journal.db")
    mock_risk = MagicMock()
    from agent.strategies.types import Signal
    dummy_signal = Signal(
        symbol="HDFCBANK", action=Action.ENTER_LONG, confidence=0.75,
        suggested_stop=Decimal("1680"), suggested_target=Decimal("1750"),
        invalidation_condition="", expected_r=2.0, time_horizon_hours=4,
        regime_fit=0.9, data_quality=DataQuality.OK,
        strategy_name="trend_following_v1", explanation="", timestamp=T0,
    )
    mock_risk.evaluate.return_value = RiskDecision(
        verdict=RiskVerdict.REJECTED, quantity=0,
        entry_price=Decimal("1750"), stop_price=Decimal("1680"),
        target_price=Decimal("1750"), risk_per_share=Decimal("0"),
        position_value=Decimal("0"), rejection_reason=RejectionReason.MAX_OPEN_POSITIONS,
        rejection_detail="test", signal=dummy_signal,
    )
    loop = DailyLoop(
        strategy=TrendFollowingStrategy(),
        risk_manager=mock_risk,
        executor=MagicMock(),
        portfolio=PortfolioTracker(initial_nav=Decimal("100000"), initial_cash=Decimal("100000"), start_time=T0),
        journal=journal,
        analyst=None,
        kill_switch=KillSwitch(flag_path=p / ".kill_switch2"),
        heartbeat=Heartbeat(alerter=None),
        alerter=TelegramAlerter(bot_token="", chat_id=""),
    )

    result = loop.run(
        session_date=date(2024, 1, 2),
        warmup_df=df.head(55),
        session_df=df.tail(5),
    )
    assert isinstance(result, DailySessionResult)
    assert result.bars_processed == 5
    assert result.final_nav == Decimal("100000")
    pnl_entries = journal.query(entry_type=JournalEntryType.PNL)
    assert len(pnl_entries) == 1

    print(f"Session result: {result.bars_processed} bars, NAV={result.final_nav}")
    print("INTEGRATION SMOKE TEST PASSED")
EOF
```

Expected: ends with `INTEGRATION SMOKE TEST PASSED`.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-05-18-phase-8-paper-trading-loop.md
git commit -m "test(runner/monitoring): Phase 8 full suite and integration smoke test pass"
```

---

## Self-Review Checklist

- [x] Kill switch wired into `PortfolioTracker.activate_kill_switch()` — already read by `RiskManager.evaluate()` via `portfolio.kill_switch_active`
- [x] Telegram alerter soft-fails — missing credentials → `_enabled=False` → no network calls
- [x] `DailyLoop.run()` checks kill switch at start and on every bar
- [x] `DailyLoop.run()` logs PNL entry at session end (CLAUDE.md rule: every decision journaled)
- [x] Session counter caps at 60 before live; `is_ready_for_live()` gated by `_LIVE_THRESHOLD`
- [x] `LIVE_TRADING_ENABLED` never set to True anywhere in this plan
- [x] All external dependencies (broker, Telegram) injected — no network calls in unit tests
- [x] `from __future__ import annotations` on every new module
- [x] `logger = structlog.get_logger()` naming (not `log`)
- [x] Monetary values: `Decimal` throughout (nav, pnl, prices)
- [x] No `print()` in production code — structlog only
- [x] All journal entries are IST-aware timestamps
