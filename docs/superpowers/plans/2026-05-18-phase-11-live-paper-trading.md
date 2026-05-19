# Phase 11 — Live Paper Trading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Upstox WebSocket LTPC ticks into the existing `DailyLoop` pipeline so the system can paper-trade in real time using live market data (not cached bars).

**Architecture:** Three layers added above the existing stack. (1) `BarBuilder` converts raw LTP ticks into closed OHLCV bars aligned to market-open boundaries. (2) `LiveSession` is an async orchestrator that subscribes to the WebSocket, passes ticks through `BarBuilder`, enriches closed bars with `FeaturePipeline`, and calls `DailyLoop.process_bar()`. (3) A new `live-paper` CLI command bootstraps the full session. LTPC mode delivers no per-tick volume; `tick_count` is used as a volume proxy (acceptable because `TrendFollowingStrategy` only uses price-based indicators).

**Tech Stack:** Python 3.11+, asyncio, polars, structlog, click, upstox-python-sdk (MarketDataStreamerV3), existing `BarBuilder`-free imports from `agent/`.

---

## File Map

```
agent/
  data/
    bar_builder.py          # NEW: ClosedBar + BarBuilder (tick → OHLCV aggregator)
  runner/
    live_session.py         # NEW: LiveSession async orchestrator
  cli.py                    # MODIFY: add live-paper command

tests/
  data/
    test_bar_builder.py     # NEW
  runner/
    test_live_session.py    # NEW
    test_cli_live_paper.py  # NEW
```

---

## Task 1: BarBuilder — Tick-to-OHLCV Aggregator

**Files:**
- Create: `agent/data/bar_builder.py`
- Create: `tests/data/test_bar_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_bar_builder.py
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.data.bar_builder import BarBuilder, ClosedBar

IST = ZoneInfo("Asia/Kolkata")


def ts(h: int, m: int, s: int = 0) -> datetime:
    return datetime(2024, 1, 2, h, m, s, tzinfo=IST)


def test_no_bar_returned_on_first_tick() -> None:
    bb = BarBuilder("HDFCBANK", "60m")
    result = bb.on_tick(1710.0, ts(9, 15))
    assert result is None


def test_bar_closes_when_next_slot_tick_arrives() -> None:
    bb = BarBuilder("HDFCBANK", "60m")
    bb.on_tick(1700.0, ts(9, 15))
    bb.on_tick(1720.0, ts(9, 45))
    closed = bb.on_tick(1715.0, ts(10, 15))  # new slot → closes 9:15 bar
    assert closed is not None
    assert closed.symbol == "HDFCBANK"
    assert closed.open == 1700.0
    assert closed.high == 1720.0
    assert closed.low == 1700.0
    assert closed.close == 1720.0
    assert closed.tick_count == 2


def test_bar_start_aligned_to_market_open_not_tick_time() -> None:
    bb = BarBuilder("HDFCBANK", "60m")
    bb.on_tick(1700.0, ts(9, 20))  # arrives after 9:15, but slot is 9:15
    closed = bb.on_tick(1705.0, ts(10, 16))  # new slot
    assert closed is not None
    assert closed.bar_open == ts(9, 15)


def test_15m_bar_closes_at_15_minute_boundary() -> None:
    bb = BarBuilder("TCS", "15m")
    bb.on_tick(3500.0, ts(9, 15))
    bb.on_tick(3510.0, ts(9, 25))
    closed = bb.on_tick(3520.0, ts(9, 30))  # new 15m slot
    assert closed is not None
    assert closed.bar_open == ts(9, 15)
    assert closed.tick_count == 2


def test_force_close_returns_current_bar() -> None:
    bb = BarBuilder("INFY", "60m")
    bb.on_tick(1600.0, ts(9, 15))
    bb.on_tick(1620.0, ts(9, 55))
    closed = bb.force_close()
    assert closed is not None
    assert closed.open == 1600.0
    assert closed.high == 1620.0
    assert closed.tick_count == 2


def test_force_close_returns_none_when_no_ticks() -> None:
    bb = BarBuilder("WIPRO", "60m")
    assert bb.force_close() is None


def test_closed_bar_to_dataframe_has_required_columns() -> None:
    bb = BarBuilder("HDFCBANK", "60m")
    bb.on_tick(1700.0, ts(9, 15))
    closed = bb.on_tick(1710.0, ts(10, 15))
    assert closed is not None
    df = closed.to_dataframe()
    for col in ("symbol", "timeframe", "timestamp", "open", "high", "low",
                "close", "volume", "value", "data_quality"):
        assert col in df.columns, f"Missing column: {col}"
    assert df["volume"][0] == 1  # 1 tick before close
    assert df["data_quality"][0] == "ok"


def test_tick_before_market_open_snaps_to_market_open_slot() -> None:
    bb = BarBuilder("SBIN", "60m")
    # Tick arrives at 9:10 — before 9:15 market open; should snap to 9:15 slot
    result = bb.on_tick(500.0, ts(9, 10))
    assert result is None  # no closed bar yet
    closed = bb.on_tick(505.0, ts(10, 16))
    assert closed is not None
    assert closed.bar_open == ts(9, 15)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/tatsatshah/Desktop/yegedge
python -m pytest tests/data/test_bar_builder.py -v --no-cov
```

Expected: `ImportError: cannot import name 'BarBuilder'`

- [ ] **Step 3: Implement `agent/data/bar_builder.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl

IST = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MINUTE = 15


@dataclass(frozen=True, slots=True)
class ClosedBar:
    symbol: str
    timeframe: str
    bar_open: datetime  # IST-aware bar-start timestamp
    open: float
    high: float
    low: float
    close: float
    tick_count: int  # volume proxy — LTPC mode has no per-tick volume

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "symbol": pl.Series([self.symbol], dtype=pl.Utf8),
                "timeframe": pl.Series([self.timeframe], dtype=pl.Utf8),
                "timestamp": pl.Series(
                    [self.bar_open],
                    dtype=pl.Datetime("us", "Asia/Kolkata"),
                ),
                "open": pl.Series([self.open], dtype=pl.Float64),
                "high": pl.Series([self.high], dtype=pl.Float64),
                "low": pl.Series([self.low], dtype=pl.Float64),
                "close": pl.Series([self.close], dtype=pl.Float64),
                "volume": pl.Series([self.tick_count], dtype=pl.Int64),
                "value": pl.Series([self.close * self.tick_count], dtype=pl.Float64),
                "data_quality": pl.Series(["ok"], dtype=pl.Utf8),
            }
        )


class BarBuilder:
    """Aggregates LTP ticks into OHLCV bars aligned to 9:15 IST market open.

    LTPC WebSocket mode delivers no per-tick volume; tick_count is used as a
    volume proxy. This is acceptable because TrendFollowingStrategy only uses
    price-based indicators (EMA, ATR, ADX).
    """

    _TIMEFRAME_MINUTES: Final[dict[str, int]] = {
        "15m": 15,
        "60m": 60,
        "1d": 375,  # 9:15 → 15:30 = 375 minutes
    }

    def __init__(self, symbol: str, timeframe: str) -> None:
        self._symbol = symbol
        self._timeframe = timeframe
        self._bar_minutes = self._TIMEFRAME_MINUTES[timeframe]
        self._current_slot: datetime | None = None
        self._open: float | None = None
        self._high: float | None = None
        self._low: float | None = None
        self._close: float | None = None
        self._tick_count: int = 0

    def on_tick(self, ltp: float, ts: datetime) -> ClosedBar | None:
        """Process one LTP tick. Returns a ClosedBar when a bar boundary is crossed."""
        slot = self._bar_start_for(ts)

        if self._current_slot is None:
            self._current_slot = slot
            self._start_bar(ltp)
            return None

        if slot == self._current_slot:
            self._update_bar(ltp)
            return None

        # New slot — close the current bar, start a new one
        closed = self._close_current()
        self._current_slot = slot
        self._start_bar(ltp)
        return closed

    def force_close(self) -> ClosedBar | None:
        """Close the current in-progress bar (call at session end: 15:30 IST)."""
        if self._current_slot is None or self._tick_count == 0:
            return None
        return self._close_current()

    def _bar_start_for(self, ts: datetime) -> datetime:
        ist = ts.astimezone(IST)
        market_open = ist.replace(
            hour=_MARKET_OPEN_HOUR,
            minute=_MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0,
        )
        if ist <= market_open:
            return market_open
        elapsed = int((ist - market_open).total_seconds()) // 60
        slot = elapsed // self._bar_minutes
        return market_open + timedelta(minutes=slot * self._bar_minutes)

    def _start_bar(self, ltp: float) -> None:
        self._open = ltp
        self._high = ltp
        self._low = ltp
        self._close = ltp
        self._tick_count = 1

    def _update_bar(self, ltp: float) -> None:
        assert self._high is not None and self._low is not None
        if ltp > self._high:
            self._high = ltp
        if ltp < self._low:
            self._low = ltp
        self._close = ltp
        self._tick_count += 1

    def _close_current(self) -> ClosedBar:
        assert self._current_slot is not None
        assert self._open is not None
        assert self._high is not None
        assert self._low is not None
        assert self._close is not None
        bar = ClosedBar(
            symbol=self._symbol,
            timeframe=self._timeframe,
            bar_open=self._current_slot,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            tick_count=self._tick_count,
        )
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self._tick_count = 0
        return bar
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/data/test_bar_builder.py -v --no-cov
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add agent/data/bar_builder.py tests/data/test_bar_builder.py
git commit -m "feat(data): add BarBuilder — tick-to-OHLCV aggregator with 9:15 IST alignment"
```

---

## Task 2: LiveSession — Async Orchestrator

**Files:**
- Create: `agent/runner/live_session.py`
- Create: `tests/runner/test_live_session.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/runner/test_live_session.py
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.data.bar_builder import BarBuilder, ClosedBar
from agent.runner.live_session import LiveSession

IST = ZoneInfo("Asia/Kolkata")


def _make_closed_bar(symbol: str = "HDFCBANK", bar_open: datetime | None = None) -> ClosedBar:
    if bar_open is None:
        bar_open = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
    return ClosedBar(
        symbol=symbol,
        timeframe="60m",
        bar_open=bar_open,
        open=1700.0,
        high=1720.0,
        low=1695.0,
        close=1710.0,
        tick_count=42,
    )


def _make_session(symbols: list[str] | None = None) -> LiveSession:
    from agent.portfolio.tracker import PortfolioTracker
    from decimal import Decimal

    if symbols is None:
        symbols = ["HDFCBANK"]
    portfolio = PortfolioTracker(initial_nav=Decimal("83000"))
    return LiveSession(
        symbols=symbols,
        timeframe="60m",
        portfolio=portfolio,
        warmup_df=pl.DataFrame(),
    )


def test_live_session_creates_bar_builder_per_symbol() -> None:
    session = _make_session(["HDFCBANK", "TCS"])
    assert len(session._builders) == 2
    assert "HDFCBANK" in session._builders
    assert "TCS" in session._builders


def test_on_tick_returns_none_before_bar_closes() -> None:
    session = _make_session()
    ts = datetime(2024, 1, 2, 9, 20, tzinfo=IST)
    result = session._on_tick("HDFCBANK", 1710.0, ts)
    assert result is None


def test_on_tick_returns_closed_bar_at_slot_boundary() -> None:
    session = _make_session()
    session._on_tick("HDFCBANK", 1700.0, datetime(2024, 1, 2, 9, 15, tzinfo=IST))
    session._on_tick("HDFCBANK", 1720.0, datetime(2024, 1, 2, 9, 45, tzinfo=IST))
    result = session._on_tick("HDFCBANK", 1715.0, datetime(2024, 1, 2, 10, 15, tzinfo=IST))
    assert result is not None
    assert isinstance(result, ClosedBar)
    assert result.symbol == "HDFCBANK"


def test_on_tick_unknown_symbol_is_ignored() -> None:
    session = _make_session(["HDFCBANK"])
    result = session._on_tick("UNKNOWN", 100.0, datetime(2024, 1, 2, 9, 15, tzinfo=IST))
    assert result is None


def test_on_bar_closed_calls_process_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.execution.paper import PaperExecution
    from agent.risk.manager import RiskManager
    from agent.strategies.trend_following import TrendFollowingStrategy
    from agent.decision.engine import DecisionEngine
    from agent.journal.store import JournalStore
    from agent.monitoring.alerter import TelegramAlerter
    from agent.monitoring.heartbeat import Heartbeat
    from agent.monitoring.kill_switch import KillSwitch
    from agent.runner.daily_loop import DailyLoop
    from agent.portfolio.tracker import PortfolioTracker
    from decimal import Decimal
    from pathlib import Path
    import tempfile

    portfolio = PortfolioTracker(initial_nav=Decimal("83000"))
    session = LiveSession(
        symbols=["HDFCBANK"],
        timeframe="60m",
        portfolio=portfolio,
        warmup_df=pl.DataFrame(),
    )

    called_with: list[pl.DataFrame] = []

    def fake_process_bar(df: pl.DataFrame, *, evaluation_time: datetime) -> list:
        called_with.append(df)
        return []

    with patch.object(session._loop, "process_bar", side_effect=fake_process_bar):
        bar = _make_closed_bar()
        session._on_bar_closed(bar)

    assert len(called_with) == 1


def test_is_within_session_returns_false_after_1530() -> None:
    session = _make_session()
    late = datetime(2024, 1, 2, 15, 31, tzinfo=IST)
    assert session._is_within_session(late) is False


def test_is_within_session_returns_true_at_1415() -> None:
    session = _make_session()
    mid = datetime(2024, 1, 2, 14, 15, tzinfo=IST)
    assert session._is_within_session(mid) is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/runner/test_live_session.py -v --no-cov
```

Expected: `ImportError: cannot import name 'LiveSession'`

- [ ] **Step 3: Implement `agent/runner/live_session.py`**

```python
from __future__ import annotations

import asyncio
from datetime import datetime, time
from decimal import Decimal
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import polars as pl
import structlog

from agent.data.bar_builder import BarBuilder, ClosedBar
from agent.decision.engine import DecisionEngine
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
    ) -> None:
        self._symbols = symbols
        self._timeframe = timeframe
        self._portfolio = portfolio
        self._warmup_df = warmup_df
        self._pipeline = FeaturePipeline()
        self._builders: dict[str, BarBuilder] = {
            s: BarBuilder(s, timeframe) for s in symbols
        }
        # Accumulated live bars per symbol (grows as bars close during session)
        self._live_bars: dict[str, pl.DataFrame] = {s: pl.DataFrame() for s in symbols}
        self._queue: asyncio.Queue[tuple[str, float, datetime]] = asyncio.Queue()
        self._loop = self._make_daily_loop(portfolio, alerter, heartbeat, kill_switch)
        self._kill_switch = kill_switch

    def put_tick(self, symbol: str, ltp: float, ts: datetime) -> None:
        """Thread-safe. Called from WebSocket callback thread."""
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(
            self._queue.put((symbol, ltp, ts)),
            loop,
        )

    async def run(self) -> None:
        """Main async loop. Processes ticks until 15:30 IST or kill switch."""
        logger.info("live_session.start", symbols=self._symbols, timeframe=self._timeframe)
        while True:
            try:
                symbol, ltp, ts = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
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
        raw_window = (
            pl.concat([sym_warmup, live_sym]) if len(sym_warmup) > 0 else live_sym
        )

        if len(raw_window) < 2:
            return

        enriched = self._pipeline.run(raw_window)
        self._loop.process_bar(enriched, evaluation_time=bar.bar_open)

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
        risk_manager = RiskManager()
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/runner/test_live_session.py -v --no-cov
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add agent/runner/live_session.py tests/runner/test_live_session.py
git commit -m "feat(runner): add LiveSession async orchestrator for real-time paper trading"
```

---

## Task 3: `live-paper` CLI Command

**Files:**
- Modify: `agent/cli.py`
- Create: `tests/runner/test_cli_live_paper.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/runner/test_cli_live_paper.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from agent.cli import cli


def test_live_paper_command_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["live-paper", "--help"])
    assert result.exit_code == 0
    assert "--timeframe" in result.output


def test_live_paper_exits_when_no_access_token(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch("agent.cli.AppSettings") as MockSettings:
        s = MagicMock()
        s.parquet_cache_dir = tmp_path / "cache"
        s.journal_db_path = tmp_path / "journal.db"
        s.upstox_access_token = ""  # empty token
        s.telegram_bot_token = ""
        s.telegram_chat_id = ""
        MockSettings.return_value = s
        result = runner.invoke(cli, ["live-paper"])
    assert result.exit_code == 1
    assert "access token" in result.output.lower() or "token" in result.output.lower()


def test_live_paper_exits_when_no_cache(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch("agent.cli.AppSettings") as MockSettings:
        s = MagicMock()
        s.parquet_cache_dir = tmp_path / "cache"
        s.journal_db_path = tmp_path / "journal.db"
        s.upstox_access_token = "fake-token"
        s.telegram_bot_token = ""
        s.telegram_chat_id = ""
        MockSettings.return_value = s
        result = runner.invoke(cli, ["live-paper"])
    # Should exit with an error about missing cache, not crash unhandled
    assert result.exit_code == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/runner/test_cli_live_paper.py -v --no-cov
```

Expected: All 3 tests fail — `live-paper` command not found in CLI.

- [ ] **Step 3: Add `live-paper` command to `agent/cli.py`**

Read the current end of `agent/cli.py` to find a good insertion point (after the `backtest` command, before any `if __name__` block). Then add:

```python
@cli.command("live-paper")
@click.option(
    "--timeframe",
    default="60m",
    type=click.Choice(["15m", "60m"]),
    show_default=True,
    help="Bar timeframe for live aggregation.",
)
@click.option(
    "--warmup-bars",
    default=100,
    show_default=True,
    help="Number of historical bars to prepend for indicator warm-up.",
)
def live_paper(timeframe: str, warmup_bars: int) -> None:
    """Paper-trade in real time using live Upstox WebSocket ticks."""
    import asyncio
    from datetime import date

    from agent.data.upstox_adapter import UpstoxAdapter
    from agent.data.universe import UniverseLoader
    from agent.monitoring.alerter import TelegramAlerter
    from agent.monitoring.kill_switch import KillSwitch
    from agent.portfolio.tracker import PortfolioTracker
    from agent.runner.live_session import LiveSession

    settings = AppSettings()

    if not settings.upstox_access_token:
        console.print("[red]UPSTOX_ACCESS_TOKEN not set. Run your daily login first.[/red]")
        sys.exit(1)

    cache = ParquetCache(root=settings.parquet_cache_dir)
    report = cache.coverage_report()

    if not report:
        console.print("[red]No cached data found. Run `refresh` first to load warmup bars.[/red]")
        sys.exit(1)

    universe = UniverseLoader(Path("config/universe.yaml"))
    symbols = universe.symbols()

    # Load warmup bars for all symbols
    today = datetime.now(tz=IST).date()
    session_start = datetime(today.year, today.month, today.day, 9, 15, tzinfo=IST)

    warmup_frames: list[pl.DataFrame] = []
    pipeline = FeaturePipeline()
    for sym in symbols:
        if sym not in report or timeframe not in report.get(sym, {}):
            continue
        sym_earliest, _ = report[sym][timeframe]
        all_sym = cache.read(symbol=sym, timeframe=timeframe, start=sym_earliest, end=session_start)
        if len(all_sym) == 0:
            continue
        enriched = pipeline.run(all_sym)
        warmup_frames.append(enriched.tail(warmup_bars))

    warmup_df = pl.concat(warmup_frames) if warmup_frames else pl.DataFrame()

    portfolio = PortfolioTracker(initial_nav=Decimal(str(settings.paper_starting_capital)))

    alerter = TelegramAlerter(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
    )

    kill_switch = KillSwitch(flag_path=Path("./data/.kill_switch"))

    live_session = LiveSession(
        symbols=symbols,
        timeframe=timeframe,
        portfolio=portfolio,
        warmup_df=warmup_df,
        alerter=alerter,
        kill_switch=kill_switch,
    )

    adapter = UpstoxAdapter(access_token=settings.upstox_access_token)

    def on_tick_df(df: pl.DataFrame) -> None:
        """Called from WebSocket thread — relay to LiveSession via thread-safe queue."""
        if len(df) == 0:
            return
        sym = df["symbol"][0]
        ltp = float(df["ltp"][0])
        ts = df["timestamp"][0]
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=IST)
        live_session.put_tick(sym, ltp, ts)

    async def _run() -> None:
        stream_task = asyncio.create_task(
            adapter.stream_live(symbols, callback=on_tick_df)
        )
        session_task = asyncio.create_task(live_session.run())
        await session_task
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass

    console.print(
        f"[bold green]Starting live paper trading session[/bold green] "
        f"({timeframe} bars, {len(symbols)} symbols)"
    )
    console.print("[dim]Press Ctrl+C to stop early.[/dim]")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted by user.[/yellow]")

    final_state = portfolio.state
    console.print(
        f"\n[bold]Session complete.[/bold] "
        f"NAV: ₹{final_state.nav:,.0f} | "
        f"P&L: ₹{final_state.daily_pnl:,.0f} | "
        f"Orders: {final_state.orders_today}"
    )
```

Also add these imports near the top of `agent/cli.py` if not already present:
```python
from decimal import Decimal
from agent.features.pipeline import FeaturePipeline
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/runner/test_cli_live_paper.py -v --no-cov
```

Expected: `3 passed`

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v --no-cov -x
```

Expected: All tests pass.

- [ ] **Step 6: Run linter**

```bash
python -m ruff check agent/cli.py agent/data/bar_builder.py agent/runner/live_session.py
python -m black --check agent/cli.py agent/data/bar_builder.py agent/runner/live_session.py
```

Expected: No issues.

- [ ] **Step 7: Commit**

```bash
git add agent/cli.py tests/runner/test_cli_live_paper.py
git commit -m "feat(cli): add live-paper command for real-time WebSocket paper trading"
```

---

## Verification

Manual test with real access token:

```bash
# 1. Ensure warmup data exists
python -m agent refresh --symbol HDFCBANK --timeframe 60m

# 2. Start live session (runs until 15:30 IST or Ctrl+C)
UPSTOX_ACCESS_TOKEN=<token> python -m agent live-paper --timeframe 60m

# Expected:
# - Connects to Upstox WebSocket
# - Prints "Starting live paper trading session (60m bars, N symbols)"
# - Emits structlog heartbeat JSON every ~4 bars
# - On bar close: strategy runs, risk manager evaluates, fills/rejections journaled
# - At 15:30 IST: session ends, final NAV printed
```

Unit test (no credentials needed):

```bash
python -m pytest tests/data/test_bar_builder.py tests/runner/test_live_session.py tests/runner/test_cli_live_paper.py -v --no-cov
```

---

## Review Priority

1. **Task 1** (BarBuilder) — correctness of bar boundary arithmetic is load-bearing; test `_bar_start_for()` edge cases carefully.
2. **Task 2** (LiveSession) — thread safety: `put_tick()` uses `run_coroutine_threadsafe`; verify the event loop passed is correct.
3. **Task 3** (CLI) — mostly integration glue; verify `on_tick_df` correctly relays from WebSocket DataFrame to `put_tick()`.
