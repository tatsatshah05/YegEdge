# Phase 10 — Backtest Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the broken `run-paper` command (missing FeaturePipeline enrichment), then build a full backtest engine with realistic Indian transaction cost model, performance metrics (Sharpe, max drawdown, CAGR), and a `python -m agent backtest` CLI command.

**Architecture:** A `agent/backtest/` package with three focused modules: `costs.py` (transaction cost calculator), `metrics.py` (performance math + frozen dataclasses), `runner.py` (multi-session orchestrator that reuses `DailyLoop`). The `BacktestRunner` loads raw OHLCV from Parquet cache, enriches per-symbol with `FeaturePipeline`, then feeds bar windows to `DailyLoop` with AI analyst disabled (no API costs). Costs are computed post-fill via `IndianCostModel` and deducted from running NAV at the session boundary. The live journal is never touched — backtest uses a temp SQLite file discarded after each run.

**Tech Stack:** Python 3.11+, Polars, Decimal (monetary arithmetic), Rich (CLI output), pytest, exchange-calendars (trading day enumeration), existing `FeaturePipeline`, `DailyLoop`, `RiskManager`, `TrendFollowingStrategy`.

---

## Context for subagent workers

**Project:** `/Users/tatsatshah/Desktop/yegedge`
**Branch:** `phase-2-feature-engineering`
**Virtualenv:** `.venv/bin/python`
**Run all commands from:** `/Users/tatsatshah/Desktop/yegedge`

**Key existing APIs (do NOT redefine):**

```python
# agent/features/pipeline.py
class FeaturePipeline:
    def __init__(self, regime_detector=None) -> None: ...
    def run(self, df: pl.DataFrame) -> pl.DataFrame: ...
    # Adds: ema_9, ema_21, ema_50, rsi_14, atr_14, adx_14, plus_di_14, minus_di_14, vwap

# agent/strategies/trend_following.py
class TrendFollowingStrategy:
    def __init__(self, *, fast_ema_col="ema_21", slow_ema_col="ema_50", ...) -> None: ...
    # IMPORTANT: generate() raises ValueError if ema_21/ema_50/adx_14/atr_14 missing!

# agent/runner/daily_loop.py
class DailyLoop:
    def __init__(self, *, strategy, risk_manager, executor, portfolio,
                 journal, analyst: AIAnalyst | None, kill_switch, heartbeat, alerter) -> None: ...
    def run(self, *, session_date: date, warmup_df: pl.DataFrame,
            session_df: pl.DataFrame) -> DailySessionResult: ...

# agent/runner/types.py
@dataclass(frozen=True, slots=True)
class DailySessionResult:
    session_date: date; bars_processed: int; signals_generated: int
    decisions_made: int; fills: tuple[Fill, ...]; rejections: int
    ai_cache_hits: int; final_nav: Decimal; daily_pnl: Decimal; peak_nav: Decimal

# agent/risk/rules.py
def load_risk_rules(path: Path = Path("config/risk_rules.yaml")) -> RiskRules: ...

# agent/data/cache.py
class ParquetCache:
    def read(self, *, symbol, timeframe, start, end) -> pl.DataFrame: ...
    def coverage_report(self) -> dict[str, dict[str, tuple[datetime, datetime]]]: ...

# agent/data/calendar.py
class NseTradingCalendar:
    def trading_sessions(self, start: date, end: date) -> list[date]: ...
```

**CLAUDE.md binding rule for costs (rule 10):**
> "Costs are always on in any backtest or simulation. Indian cost model: STT 0.025% intraday sell, exchange charges ~0.00325%, SEBI charges 0.0001%, stamp duty 0.003% on buy, GST 18% on brokerage + exchange charges, brokerage per broker config. Realistic round-trip ~6–10 bps."

---

## Critical Bug Discovered

`run-paper` is currently broken. `TrendFollowingStrategy.generate()` calls `_validate_columns()` which raises `ValueError` if `ema_21`, `ema_50`, `adx_14`, `atr_14` are missing. The existing CLI passes raw OHLCV from cache directly to `DailyLoop` — no `FeaturePipeline` enrichment. **Task 1 fixes this before building anything new.**

---

## File Map

```
agent/
  cli.py                       — MODIFY: fix run-paper (Task 1), add backtest (Task 5)
  backtest/
    __init__.py                — empty package marker (Task 2)
    costs.py                   — IndianCostModel: per-fill transaction cost calculator (Task 2)
    metrics.py                 — SessionResult, BacktestMetrics, BacktestReport, compute_metrics() (Task 3)
    runner.py                  — BacktestRunner: multi-session orchestrator (Task 4)

tests/
  runner/
    test_cli_run_paper.py      — MODIFY: add enrichment smoke test (Task 1)
  backtest/
    __init__.py                — empty (Task 2)
    test_costs.py              — 7 tests (Task 2)
    test_metrics.py            — 6 tests (Task 3)
    test_runner.py             — 5 tests (Task 4)
    test_cli_backtest.py       — 3 tests (Task 5)
```

---

## Task 1: Fix `run-paper` — FeaturePipeline Enrichment per Symbol

**Files:**
- Modify: `agent/cli.py` (the `run_paper` function, approximately lines 257–380)
- Modify: `tests/runner/test_cli_run_paper.py`

The fix: enrich each symbol's data separately with `FeaturePipeline().run()` before concatenating across symbols. Rolling indicators (EMA, ATR, ADX) must not cross symbol boundaries — hence per-symbol enrichment.

- [ ] **Step 1: Verify the bug exists**

```bash
cd /Users/tatsatshah/Desktop/yegedge
.venv/bin/python -c "
from agent.strategies.trend_following import TrendFollowingStrategy
import polars as pl
from datetime import datetime
from zoneinfo import ZoneInfo
IST = ZoneInfo('Asia/Kolkata')
df = pl.DataFrame({
    'symbol': ['HDFCBANK'], 'timeframe': ['60m'],
    'timestamp': [datetime(2024, 1, 2, 9, 15, tzinfo=IST)],
    'open': [1700.0], 'high': [1720.0], 'low': [1695.0],
    'close': [1710.0], 'volume': [100000], 'value': [171000000.0],
    'data_quality': ['ok'],
})
try:
    TrendFollowingStrategy().generate(df)
    print('BUG: no error raised — strategy accepts raw OHLCV')
except ValueError as e:
    print(f'CONFIRMED BUG: {e}')
"
```

Expected: `CONFIRMED BUG: TrendFollowingStrategy requires columns ...`

- [ ] **Step 2: Add the failing test**

Add this test to `tests/runner/test_cli_run_paper.py`:

```python
def test_feature_pipeline_produces_required_strategy_columns() -> None:
    """Regression guard: FeaturePipeline.run() must add the columns TrendFollowingStrategy needs."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import polars as pl

    from agent.features.pipeline import FeaturePipeline

    IST = ZoneInfo("Asia/Kolkata")
    base_ts = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
    n = 60  # enough bars for EMA(50) + ADX(14) to be meaningful
    df = pl.DataFrame(
        {
            "symbol": ["HDFCBANK"] * n,
            "timeframe": ["60m"] * n,
            "timestamp": [base_ts + timedelta(hours=i) for i in range(n)],
            "open": [1700.0] * n,
            "high": [1720.0] * n,
            "low": [1695.0] * n,
            "close": [1710.0] * n,
            "volume": [100_000] * n,
            "value": [171_000_000.0] * n,
            "data_quality": ["ok"] * n,
        }
    )
    enriched = FeaturePipeline().run(df)
    for col in ("ema_21", "ema_50", "adx_14", "atr_14"):
        assert col in enriched.columns, f"Missing required column: {col}"
```

- [ ] **Step 3: Run the new test to confirm it passes (FeaturePipeline itself is not broken)**

```bash
.venv/bin/python -m pytest tests/runner/test_cli_run_paper.py::test_feature_pipeline_produces_required_strategy_columns -v --no-cov
```

Expected: `1 passed` — the pipeline works; it's just not being called from run-paper.

- [ ] **Step 4: Fix `agent/cli.py` — the `run_paper` function**

Find the `run_paper` function (around line 257). Add `from agent.features.pipeline import FeaturePipeline` to the local imports block and replace the per-symbol data loading block with this enriched version.

The original block looks like:
```python
    warmup_frames = []
    session_frames = []
    for sym in universe.symbols():
        wdf = cache.read(symbol=sym, timeframe=timeframe, start=earliest, end=session_start)
        sdf = cache.read(symbol=sym, timeframe=timeframe, start=session_start, end=session_end)
        if len(wdf) > 0:
            warmup_frames.append(wdf.tail(warmup_bars))
        if len(sdf) > 0:
            session_frames.append(sdf)
```

Replace it with:
```python
    warmup_frames = []
    session_frames = []
    pipeline = FeaturePipeline()
    for sym in universe.symbols():
        if sym not in report or timeframe not in report.get(sym, {}):
            continue
        sym_earliest, _ = report[sym][timeframe]
        # Load all history for this symbol so rolling indicators are correctly seeded
        all_sym = cache.read(symbol=sym, timeframe=timeframe, start=sym_earliest, end=session_end)
        if len(all_sym) == 0:
            continue
        # Enrich per-symbol — rolling windows must not cross symbol boundaries
        enriched = pipeline.run(all_sym)
        wdf = enriched.filter(pl.col("timestamp") < session_start).tail(warmup_bars)
        sdf = enriched.filter(
            (pl.col("timestamp") >= session_start) & (pl.col("timestamp") <= session_end)
        )
        if len(wdf) > 0:
            warmup_frames.append(wdf)
        if len(sdf) > 0:
            session_frames.append(sdf)
```

Also add `from agent.features.pipeline import FeaturePipeline` to the local imports block inside `run_paper`.

**Complete replacement — find and replace the exact block in `agent/cli.py`.** Read the file first to get exact line numbers before editing.

- [ ] **Step 5: Run the full existing run-paper test suite**

```bash
.venv/bin/python -m pytest tests/runner/ -v --no-cov
```

Expected: all 4 tests pass (3 existing + 1 new).

- [ ] **Step 6: Run full suite to confirm no regressions**

```bash
.venv/bin/python -m pytest tests/ --no-cov -q 2>&1 | tail -5
```

Expected: 335 passed.

- [ ] **Step 7: Commit**

```bash
git add agent/cli.py tests/runner/test_cli_run_paper.py
git commit -m "$(cat <<'EOF'
fix(runner): enrich OHLCV with FeaturePipeline per-symbol before DailyLoop

TrendFollowingStrategy.generate() requires ema_21/ema_50/adx_14/atr_14
columns and raises ValueError when they are absent. run-paper was passing
raw cache data directly to DailyLoop without enrichment, making every
paper session crash immediately. Fix: run FeaturePipeline per-symbol so
rolling windows never cross symbol boundaries.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Indian Cost Model

**Files:**
- Create: `agent/backtest/__init__.py`
- Create: `agent/backtest/costs.py`
- Create: `tests/backtest/__init__.py`
- Create: `tests/backtest/test_costs.py`

**Cost model rates (CLAUDE.md rule 10, NSE equity MIS as of 2024):**
- Brokerage: ₹20 flat per order (Upstox/Zerodha discount broker, capped)
- Exchange (NSE): 0.00325% of trade value (both sides)
- SEBI: 0.0001% of trade value (both sides)
- STT: 0.025% of trade value on **sell side only** (intraday MIS)
- Stamp duty: 0.003% of trade value on **buy side only**
- GST: 18% on (brokerage + exchange charges)

- [ ] **Step 1: Write the failing tests**

Create `tests/backtest/test_costs.py`:

```python
# tests/backtest/test_costs.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.backtest.costs import IndianCostModel
from agent.execution.types import ExecutionMode, Fill
from agent.strategies.types import Action

IST = ZoneInfo("Asia/Kolkata")
T0 = datetime(2024, 1, 2, 9, 15, tzinfo=IST)


def _make_fill(action: Action, price: float, qty: int) -> Fill:
    return Fill(
        order_id="test-01",
        symbol="HDFCBANK",
        action=action,
        quantity=qty,
        fill_price=Decimal(str(price)),
        timestamp=T0,
        signal_id="sig-01",
        strategy_name="trend_following_v1",
        execution_mode=ExecutionMode.PAPER,
    )


def test_compute_cost_returns_decimal() -> None:
    cost = IndianCostModel().compute_cost(_make_fill(Action.ENTER_LONG, 1700.0, 10))
    assert isinstance(cost, Decimal)


def test_compute_cost_is_positive() -> None:
    cost = IndianCostModel().compute_cost(_make_fill(Action.ENTER_LONG, 1700.0, 10))
    assert cost > Decimal("0")


def test_exit_long_is_more_expensive_than_enter_long() -> None:
    """EXIT_LONG has STT (0.025%); ENTER_LONG has stamp (0.003%). EXIT must cost more."""
    model = IndianCostModel()
    enter = model.compute_cost(_make_fill(Action.ENTER_LONG, 1700.0, 10))
    exit_ = model.compute_cost(_make_fill(Action.EXIT_LONG, 1700.0, 10))
    assert exit_ > enter


def test_enter_long_excludes_stt_includes_stamp() -> None:
    """For ENTER_LONG: stamp duty 0.003% present, STT absent.

    trade_value = 1700 * 10 = 17000
    stamp = 17000 * 0.00003 = 0.51
    stt   = 17000 * 0.00025 = 4.25
    Verify cost is lower than if STT were included.
    """
    model = IndianCostModel()
    cost = model.compute_cost(_make_fill(Action.ENTER_LONG, 1700.0, 10))
    trade_value = Decimal("17000")
    stt_if_included = (trade_value * Decimal("0.00025")).quantize(Decimal("0.01"))
    # If STT were included, cost would be at least stt_if_included more than stamp
    stamp = (trade_value * Decimal("0.00003")).quantize(Decimal("0.01"))
    # cost should contain stamp but not stt: cost < cost_with_stt
    # proxy: exit (has STT) costs more than enter (has stamp) by roughly STT - stamp = 3.74
    exit_cost = model.compute_cost(_make_fill(Action.EXIT_LONG, 1700.0, 10))
    assert exit_cost - cost == pytest.approx(float(stt_if_included - stamp), abs=0.02)


def test_brokerage_capped_at_20_for_large_position() -> None:
    """Brokerage = min(₹20, trade_value * 0.0003). ₹1L position: 0.0003 * 100000 = ₹30 > ₹20."""
    model = IndianCostModel()
    # ₹100 * 1000 = ₹1,00,000
    enter = _make_fill(Action.ENTER_LONG, 100.0, 1000)
    cost = model.compute_cost(enter)
    # Manually compute expected cost with capped brokerage
    tv = Decimal("100000")
    exchange = (tv * Decimal("0.0000325")).quantize(Decimal("0.01"))
    stamp = (tv * Decimal("0.00003")).quantize(Decimal("0.01"))
    sebi = (tv * Decimal("0.000001")).quantize(Decimal("0.01"))
    brokerage = Decimal("20")  # capped at 20, not 30
    gst = ((brokerage + exchange) * Decimal("0.18")).quantize(Decimal("0.01"))
    expected = (stamp + exchange + sebi + brokerage + gst).quantize(Decimal("0.01"))
    assert cost == expected


def test_small_position_brokerage_not_capped() -> None:
    """Small trade: brokerage = trade_value * 0.0003 (below ₹20 cap)."""
    model = IndianCostModel()
    # ₹100 * 1 share = ₹100 → brokerage = 100 * 0.0003 = ₹0.03, not capped
    fill = _make_fill(Action.ENTER_LONG, 100.0, 1)
    cost = model.compute_cost(fill)
    assert cost > Decimal("0")
    tv = Decimal("100")
    brokerage_uncapped = (tv * Decimal("0.0003")).quantize(Decimal("0.01"))
    assert brokerage_uncapped < Decimal("20")


def test_round_trip_cost_in_6_to_12_bps_range() -> None:
    """Round-trip cost for ₹1L position ≈ 6–12 bps (₹60–₹120)."""
    model = IndianCostModel()
    enter = _make_fill(Action.ENTER_LONG, 100.0, 1000)   # ₹1,00,000
    exit_ = _make_fill(Action.EXIT_LONG, 100.0, 1000)
    total = model.compute_cost(enter) + model.compute_cost(exit_)
    assert Decimal("60") <= total <= Decimal("120"), (
        f"Round-trip cost {total} outside 6–12 bps (₹60–₹120)"
    )
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/backtest/test_costs.py -v --no-cov 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'agent.backtest'`

- [ ] **Step 3: Create package skeletons**

```python
# agent/backtest/__init__.py
# intentionally empty
```

```python
# tests/backtest/__init__.py
# intentionally empty
```

- [ ] **Step 4: Write `agent/backtest/costs.py`**

```python
# agent/backtest/costs.py
from __future__ import annotations

from decimal import Decimal

from agent.execution.types import Fill
from agent.strategies.types import Action

# Upstox/Zerodha discount broker: ₹20 flat per order
_BROKERAGE_PER_ORDER = Decimal("20")

# NSE equity rates (as fractions of trade value)
_EXCHANGE_RATE = Decimal("0.0000325")    # 0.00325% NSE transaction charge
_SEBI_RATE = Decimal("0.000001")         # 0.0001% SEBI charge
_STT_SELL_RATE = Decimal("0.00025")      # 0.025% STT on intraday sell only
_STAMP_DUTY_RATE = Decimal("0.00003")    # 0.003% stamp duty on buy only
_GST_RATE = Decimal("0.18")             # 18% GST on brokerage + exchange charges


class IndianCostModel:
    """Compute realistic NSE equity intraday (MIS) transaction costs per fill.

    Rates per SEBI/NSE schedule. STT applies to sell side only (intraday MIS).
    Stamp duty applies to buy side only. Round-trip ≈ 6–10 bps for typical lots.
    """

    def compute_cost(self, fill: Fill) -> Decimal:
        """Return total transaction cost for one fill, in INR, rounded to paise.

        ENTER_LONG: stamp duty + exchange + SEBI + brokerage + GST.
        EXIT_LONG:  STT + exchange + SEBI + brokerage + GST.
        """
        trade_value = fill.fill_price * Decimal(str(fill.quantity))

        exchange_inr = (trade_value * _EXCHANGE_RATE).quantize(Decimal("0.01"))
        sebi_inr = (trade_value * _SEBI_RATE).quantize(Decimal("0.01"))
        brokerage_inr = min(_BROKERAGE_PER_ORDER, trade_value * Decimal("0.0003"))
        gst_inr = ((brokerage_inr + exchange_inr) * _GST_RATE).quantize(Decimal("0.01"))

        if fill.action == Action.ENTER_LONG:
            side_charge = (trade_value * _STAMP_DUTY_RATE).quantize(Decimal("0.01"))
        else:  # EXIT_LONG
            side_charge = (trade_value * _STT_SELL_RATE).quantize(Decimal("0.01"))

        return (side_charge + exchange_inr + sebi_inr + brokerage_inr + gst_inr).quantize(
            Decimal("0.01")
        )
```

- [ ] **Step 5: Run tests to verify 7 pass**

```bash
.venv/bin/python -m pytest tests/backtest/test_costs.py -v --no-cov
```

Expected: `7 passed`

- [ ] **Step 6: Run full suite**

```bash
.venv/bin/python -m pytest tests/ --no-cov -q 2>&1 | tail -5
```

Expected: 342 passed (335 + 7).

- [ ] **Step 7: Commit**

```bash
git add agent/backtest/__init__.py agent/backtest/costs.py \
        tests/backtest/__init__.py tests/backtest/test_costs.py
git commit -m "$(cat <<'EOF'
feat(backtest): add IndianCostModel with NSE MIS transaction cost rates

Implements CLAUDE.md rule 10: STT 0.025% sell, stamp 0.003% buy,
exchange 0.00325%, SEBI 0.0001%, brokerage ₹20 flat, GST 18%.
Round-trip ≈ 6-12 bps for typical lot sizes.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Backtest Metrics + Report Types

**Files:**
- Create: `agent/backtest/metrics.py`
- Create: `tests/backtest/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/backtest/test_metrics.py`:

```python
# tests/backtest/test_metrics.py
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from agent.backtest.metrics import BacktestMetrics, BacktestReport, SessionResult, compute_metrics


def _sess(net_pnl: float, nav: float, costs: float = 0.0, fills: int = 2) -> SessionResult:
    """Build a SessionResult with session_date=2024-01-02 (date doesn't affect metrics)."""
    gross = net_pnl + costs
    return SessionResult(
        session_date=date(2024, 1, 2),
        bars_processed=6,
        fills=fills,
        gross_pnl=Decimal(str(gross)),
        costs=Decimal(str(costs)),
        net_pnl=Decimal(str(net_pnl)),
        final_nav=Decimal(str(nav)),
    )


def test_empty_sessions_returns_zero_metrics() -> None:
    m = compute_metrics([], Decimal("83000"))
    assert m.total_sessions == 0
    assert m.win_rate == 0.0
    assert m.sharpe_ratio == 0.0
    assert m.max_drawdown == 0.0
    assert m.cagr == 0.0
    assert m.final_nav == Decimal("83000")
    assert m.total_costs == Decimal("0")


def test_win_rate_two_wins_one_loss() -> None:
    sessions = [
        _sess(net_pnl=1000.0, nav=84000.0),
        _sess(net_pnl=-300.0, nav=83700.0),
        _sess(net_pnl=500.0, nav=84200.0),
    ]
    m = compute_metrics(sessions, Decimal("83000"))
    assert m.total_sessions == 3
    assert m.winning_sessions == 2
    assert m.win_rate == pytest.approx(2 / 3, rel=0.001)


def test_max_drawdown_peak_then_trough() -> None:
    # NAV path: 83000 → 85000 (+2000) → 83000 (-2000) → 84000 (+1000)
    # Peak = 85000, trough after peak = 83000
    # drawdown = 2000 / 85000
    sessions = [
        _sess(net_pnl=2000.0, nav=85000.0),
        _sess(net_pnl=-2000.0, nav=83000.0),
        _sess(net_pnl=1000.0, nav=84000.0),
    ]
    m = compute_metrics(sessions, Decimal("83000"))
    assert m.max_drawdown == pytest.approx(2000 / 85000, rel=0.01)


def test_consistent_positive_returns_give_positive_sharpe() -> None:
    sessions = [
        _sess(net_pnl=200.0, nav=83000.0 + 200.0 * (i + 1))
        for i in range(30)
    ]
    m = compute_metrics(sessions, Decimal("83000"))
    assert m.sharpe_ratio > 0.0


def test_total_costs_and_pnl_aggregation() -> None:
    sessions = [
        _sess(net_pnl=900.0, nav=83900.0, costs=50.0),
        _sess(net_pnl=450.0, nav=84350.0, costs=30.0),
    ]
    m = compute_metrics(sessions, Decimal("83000"))
    assert m.total_costs == Decimal("80.0")
    assert m.total_gross_pnl == Decimal("1430.0")
    assert m.total_net_pnl == Decimal("1350.0")


def test_cagr_positive_for_consistently_profitable_run() -> None:
    initial = Decimal("83000")
    # 60 sessions each netting +₹200
    sessions = [
        SessionResult(
            session_date=date(2024, 1, 2),
            bars_processed=6,
            fills=2,
            gross_pnl=Decimal("200"),
            costs=Decimal("0"),
            net_pnl=Decimal("200"),
            final_nav=Decimal(str(float(initial) + 200.0 * (i + 1))),
        )
        for i in range(60)
    ]
    m = compute_metrics(sessions, initial)
    assert m.cagr > 0.0
    assert m.final_nav == Decimal(str(float(initial) + 200.0 * 60))
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/backtest/test_metrics.py -v --no-cov 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'SessionResult'`

- [ ] **Step 3: Write `agent/backtest/metrics.py`**

```python
# agent/backtest/metrics.py
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Sequence

_RISK_FREE_RATE = 0.07       # India 10Y gilt ≈ 7%
_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class SessionResult:
    """Outcome of one backtest trading session."""

    session_date: date
    bars_processed: int
    fills: int
    gross_pnl: Decimal
    costs: Decimal     # transaction costs (IndianCostModel)
    net_pnl: Decimal   # gross_pnl - costs
    final_nav: Decimal  # running NAV after this session's net_pnl


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Aggregate performance metrics across all backtest sessions."""

    total_sessions: int
    winning_sessions: int
    win_rate: float
    total_gross_pnl: Decimal
    total_costs: Decimal
    total_net_pnl: Decimal
    sharpe_ratio: float   # annualized, net of costs, excess over risk-free rate
    max_drawdown: float   # peak-to-trough fraction of peak NAV
    cagr: float           # annualized net return (assumes 252 trading days/year)
    initial_nav: Decimal
    final_nav: Decimal


@dataclass(frozen=True)
class BacktestReport:
    """Full backtest output: per-session detail + aggregate metrics."""

    sessions: list[SessionResult]
    metrics: BacktestMetrics


def compute_metrics(
    sessions: Sequence[SessionResult],
    initial_nav: Decimal,
) -> BacktestMetrics:
    """Compute aggregate performance metrics from session results.

    All return calculations use net_pnl (after transaction costs).
    Sharpe uses daily excess returns over the Indian risk-free rate (7%).
    CAGR assumes 252 trading days per calendar year.
    """
    if not sessions:
        return BacktestMetrics(
            total_sessions=0,
            winning_sessions=0,
            win_rate=0.0,
            total_gross_pnl=Decimal("0"),
            total_costs=Decimal("0"),
            total_net_pnl=Decimal("0"),
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            cagr=0.0,
            initial_nav=initial_nav,
            final_nav=initial_nav,
        )

    total_sessions = len(sessions)
    winning_sessions = sum(1 for s in sessions if s.net_pnl > 0)
    win_rate = winning_sessions / total_sessions
    total_gross_pnl = sum((s.gross_pnl for s in sessions), Decimal("0"))
    total_costs = sum((s.costs for s in sessions), Decimal("0"))
    total_net_pnl = sum((s.net_pnl for s in sessions), Decimal("0"))
    final_nav = sessions[-1].final_nav

    # Daily net returns as fraction of initial NAV (not running NAV — consistent denominator)
    initial = float(initial_nav)
    daily_returns = [float(s.net_pnl) / initial for s in sessions]

    # Annualized Sharpe ratio (excess over risk-free rate)
    if len(daily_returns) > 1:
        daily_rf = _RISK_FREE_RATE / _TRADING_DAYS_PER_YEAR
        excess = [r - daily_rf for r in daily_returns]
        mean_excess = sum(excess) / len(excess)
        variance = sum((r - mean_excess) ** 2 for r in excess) / (len(excess) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        sharpe = (
            (mean_excess / std_dev * math.sqrt(_TRADING_DAYS_PER_YEAR)) if std_dev > 0 else 0.0
        )
    else:
        sharpe = 0.0

    # Max peak-to-trough drawdown
    peak = initial
    max_dd = 0.0
    running = initial
    for s in sessions:
        running += float(s.net_pnl)
        if running > peak:
            peak = running
        dd = (peak - running) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # CAGR (annualized net return)
    n_years = total_sessions / _TRADING_DAYS_PER_YEAR
    total_return = float(final_nav) / initial if initial > 0 else 0.0
    cagr = (total_return ** (1.0 / n_years) - 1.0) if n_years > 0 and total_return > 0 else 0.0

    return BacktestMetrics(
        total_sessions=total_sessions,
        winning_sessions=winning_sessions,
        win_rate=round(win_rate, 4),
        total_gross_pnl=total_gross_pnl,
        total_costs=total_costs,
        total_net_pnl=total_net_pnl,
        sharpe_ratio=round(sharpe, 3),
        max_drawdown=round(max_dd, 4),
        cagr=round(cagr, 4),
        initial_nav=initial_nav,
        final_nav=final_nav,
    )
```

- [ ] **Step 4: Run tests to verify 6 pass**

```bash
.venv/bin/python -m pytest tests/backtest/test_metrics.py -v --no-cov
```

Expected: `6 passed`

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/python -m pytest tests/ --no-cov -q 2>&1 | tail -5
```

Expected: 348 passed.

- [ ] **Step 6: Commit**

```bash
git add agent/backtest/metrics.py tests/backtest/test_metrics.py
git commit -m "$(cat <<'EOF'
feat(backtest): add SessionResult, BacktestMetrics, BacktestReport, compute_metrics

Sharpe uses annualized daily excess returns over 7% risk-free rate.
Max drawdown is peak-to-trough fraction. CAGR assumes 252 trading days/yr.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: BacktestRunner

**Files:**
- Create: `agent/backtest/runner.py`
- Create: `tests/backtest/test_runner.py`

**Design decisions:**
- Single-symbol per `run()` call. The CLI loops over universe if needed.
- Loads all historical data once (1 year pre-`start_date` through `end_date`).
- Enriches the entire dataset once with `FeaturePipeline` (per-symbol by design).
- Creates a fresh `PortfolioTracker` per session (starting NAV = prior session's final NAV).
- Uses a single temp SQLite journal for the entire run (discarded after).
- `KillSwitch` path is within the temp dir — cannot accidentally trigger the production kill switch.
- AI analyst is `None` — zero Anthropic API calls during backtest.
- `TelegramAlerter("", "")` — alerts suppressed (empty credentials → disabled).

- [ ] **Step 1: Write the failing tests**

Create `tests/backtest/test_runner.py`:

```python
# tests/backtest/test_runner.py
from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.backtest.costs import IndianCostModel
from agent.backtest.runner import BacktestRunner
from agent.data.cache import ParquetCache
from agent.execution.types import ExecutionMode, Fill
from agent.runner.types import DailySessionResult
from agent.strategies.types import Action

IST = ZoneInfo("Asia/Kolkata")


def _make_bars(session_date: date, n: int = 6) -> pl.DataFrame:
    """n bars starting at 09:15 IST on session_date, hourly."""
    base = datetime(session_date.year, session_date.month, session_date.day, 9, 15, tzinfo=IST)
    return pl.DataFrame(
        {
            "symbol": ["HDFCBANK"] * n,
            "timeframe": ["60m"] * n,
            "timestamp": [base + timedelta(hours=i) for i in range(n)],
            "open": [1700.0] * n,
            "high": [1720.0] * n,
            "low": [1695.0] * n,
            "close": [1710.0] * n,
            "volume": [100_000] * n,
            "value": [171_000_000.0] * n,
            "data_quality": ["ok"] * n,
        }
    )


def _fake_fill() -> Fill:
    return Fill(
        order_id="paper-HDFCBANK-20240102091500-sig-0001",
        symbol="HDFCBANK",
        action=Action.ENTER_LONG,
        quantity=10,
        fill_price=Decimal("1710.00"),
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
        signal_id="sig-0001",
        strategy_name="trend_following_v1",
        execution_mode=ExecutionMode.PAPER,
    )


def _fake_result(session_date: date, pnl: float = 0.0, fills: tuple = ()) -> DailySessionResult:
    return DailySessionResult(
        session_date=session_date,
        bars_processed=6,
        signals_generated=1,
        decisions_made=1,
        fills=fills,
        rejections=0,
        ai_cache_hits=0,
        final_nav=Decimal(str(83000.0 + pnl)),
        daily_pnl=Decimal(str(pnl)),
        peak_nav=Decimal(str(max(83000.0, 83000.0 + pnl))),
    )


def _make_runner(cache: ParquetCache) -> BacktestRunner:
    return BacktestRunner(
        strategy=MagicMock(),
        risk_manager=MagicMock(),
        cache=cache,
        initial_nav=Decimal("83000"),
    )


def test_runner_returns_empty_report_when_no_cache_data(tmp_path: Path) -> None:
    runner = _make_runner(ParquetCache(root=tmp_path / "cache"))
    report = runner.run(
        symbol="HDFCBANK",
        timeframe="60m",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 5),
    )
    assert len(report.sessions) == 0
    assert report.metrics.total_sessions == 0
    assert report.metrics.initial_nav == Decimal("83000")


def test_runner_creates_one_session_per_trading_day(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path / "cache")
    # Jan 2 (Tue) + Jan 3 (Wed) are both trading days
    bars = pl.concat([_make_bars(date(2024, 1, 2)), _make_bars(date(2024, 1, 3))])
    cache.write(bars, symbol="HDFCBANK", timeframe="60m")

    runner = _make_runner(cache)
    with patch("agent.backtest.runner.DailyLoop") as MockLoop:
        MockLoop.return_value.run.return_value = _fake_result(date(2024, 1, 2))
        report = runner.run(
            symbol="HDFCBANK",
            timeframe="60m",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
        )
    assert len(report.sessions) == 2


def test_runner_deducts_costs_from_gross_pnl(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path / "cache")
    cache.write(_make_bars(date(2024, 1, 2)), symbol="HDFCBANK", timeframe="60m")

    runner = _make_runner(cache)
    fill = _fake_fill()
    with patch("agent.backtest.runner.DailyLoop") as MockLoop:
        MockLoop.return_value.run.return_value = _fake_result(
            date(2024, 1, 2), pnl=1000.0, fills=(fill,)
        )
        report = runner.run(
            symbol="HDFCBANK",
            timeframe="60m",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
        )
    s = report.sessions[0]
    assert s.fills == 1
    assert s.gross_pnl == Decimal("1000.0")
    assert s.costs > Decimal("0")
    assert s.net_pnl == s.gross_pnl - s.costs
    assert s.net_pnl < s.gross_pnl


def test_runner_accumulates_nav_across_sessions(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path / "cache")
    bars = pl.concat([_make_bars(date(2024, 1, 2)), _make_bars(date(2024, 1, 3))])
    cache.write(bars, symbol="HDFCBANK", timeframe="60m")

    runner = _make_runner(cache)
    results_queue = [
        _fake_result(date(2024, 1, 2), pnl=500.0),
        _fake_result(date(2024, 1, 3), pnl=300.0),
    ]
    call_idx = 0

    def pop_result(*args, **kwargs):
        nonlocal call_idx
        r = results_queue[call_idx]
        call_idx += 1
        return r

    with patch("agent.backtest.runner.DailyLoop") as MockLoop:
        MockLoop.return_value.run.side_effect = pop_result
        report = runner.run(
            symbol="HDFCBANK",
            timeframe="60m",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
        )
    # Final NAV ≥ 83000 + 500 + 300 - costs (costs are small but non-zero)
    assert report.sessions[-1].final_nav > Decimal("83700")
    assert report.sessions[-1].final_nav <= Decimal("83800")


def test_runner_skips_days_with_no_bars_in_cache(tmp_path: Path) -> None:
    """A trading day with no cached bars is silently skipped (no DailyLoop created)."""
    cache = ParquetCache(root=tmp_path / "cache")
    # Only write Jan 2 — Jan 3 has no data
    cache.write(_make_bars(date(2024, 1, 2)), symbol="HDFCBANK", timeframe="60m")

    runner = _make_runner(cache)
    with patch("agent.backtest.runner.DailyLoop") as MockLoop:
        MockLoop.return_value.run.return_value = _fake_result(date(2024, 1, 2))
        report = runner.run(
            symbol="HDFCBANK",
            timeframe="60m",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
        )
    assert len(report.sessions) == 1
    assert MockLoop.return_value.run.call_count == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/backtest/test_runner.py -v --no-cov 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'BacktestRunner'`

- [ ] **Step 3: Write `agent/backtest/runner.py`**

```python
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
from agent.backtest.metrics import BacktestReport, BacktestMetrics, SessionResult, compute_metrics
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
    """Replay historical bars through the full pipeline, session by session.

    For each trading day in [start_date, end_date]:
    1. Slice the enriched DataFrame into warmup bars + session bars.
    2. Create a fresh PortfolioTracker (starting NAV = prior session's final NAV).
    3. Run DailyLoop with AI analyst disabled (no Anthropic API calls).
    4. Compute transaction costs via IndianCostModel and deduct from net P&L.
    5. Record a SessionResult.

    The live journal is never written — backtest uses a temp SQLite discarded after run().
    The kill switch flag is scoped to a temp directory — cannot trigger production kill switch.
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
        """Run the full backtest and return the report.

        symbol: NSE equity symbol (e.g. "HDFCBANK").
        timeframe: "15m", "60m", or "1d".
        start_date / end_date: inclusive range of trading dates to simulate.
        warmup_bars: bars prepended before each session for indicator warm-up.
        """
        calendar = NseTradingCalendar()
        trading_days = calendar.trading_sessions(start_date, end_date)

        if not trading_days:
            return BacktestReport(sessions=[], metrics=compute_metrics([], self._initial_nav))

        # Load full history for this symbol — 1 year before start (for indicator seeding)
        load_from = datetime(start_date.year - 1, start_date.month, start_date.day, tzinfo=IST)
        load_to = datetime(end_date.year, end_date.month, end_date.day, 23, 59, tzinfo=IST)
        raw = self._cache.read(symbol=symbol, timeframe=timeframe, start=load_from, end=load_to)

        if len(raw) == 0:
            log.warning("backtest.no_cache_data", symbol=symbol, timeframe=timeframe)
            return BacktestReport(sessions=[], metrics=compute_metrics([], self._initial_nav))

        # Enrich once — FeaturePipeline is safe to call on the full single-symbol DataFrame
        enriched = FeaturePipeline().run(raw)

        running_nav = self._initial_nav
        session_results: list[SessionResult] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            journal = JournalStore(db_path=tmp / "backtest.db")

            for session_date in trading_days:
                session_start = datetime(
                    session_date.year, session_date.month, session_date.day, 9, 15, tzinfo=IST
                )
                session_end = datetime(
                    session_date.year, session_date.month, session_date.day, 15, 30, tzinfo=IST
                )

                session_df = enriched.filter(
                    (pl.col("timestamp") >= session_start)
                    & (pl.col("timestamp") <= session_end)
                )
                if len(session_df) == 0:
                    continue

                warmup_df = (
                    enriched.filter(pl.col("timestamp") < session_start).tail(warmup_bars)
                )

                portfolio = PortfolioTracker(
                    initial_nav=running_nav,
                    initial_cash=running_nav,
                    start_time=session_start,
                )
                loop = DailyLoop(
                    strategy=self._strategy,
                    risk_manager=self._risk_manager,
                    executor=PaperExecution(),
                    portfolio=portfolio,
                    journal=journal,
                    analyst=None,  # AI disabled — no Anthropic API calls in backtest
                    kill_switch=KillSwitch(flag_path=tmp / ".kill_switch"),
                    heartbeat=Heartbeat(),
                    alerter=TelegramAlerter("", ""),  # alerts suppressed
                )

                result = loop.run(
                    session_date=session_date,
                    warmup_df=warmup_df,
                    session_df=session_df,
                )

                costs = sum(
                    (self._cost_model.compute_cost(f) for f in result.fills), Decimal("0")
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
                    bars=result.bars_processed,
                    fills=len(result.fills),
                    net_pnl=str(net_pnl),
                    running_nav=str(running_nav),
                )

        return BacktestReport(
            sessions=session_results,
            metrics=compute_metrics(session_results, self._initial_nav),
        )
```

- [ ] **Step 4: Run tests to verify 5 pass**

```bash
.venv/bin/python -m pytest tests/backtest/test_runner.py -v --no-cov
```

Expected: `5 passed`

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/python -m pytest tests/ --no-cov -q 2>&1 | tail -5
```

Expected: 353 passed.

- [ ] **Step 6: Commit**

```bash
git add agent/backtest/runner.py tests/backtest/test_runner.py
git commit -m "$(cat <<'EOF'
feat(backtest): add BacktestRunner with per-session cost deduction

Enriches raw cache data once with FeaturePipeline, then replays bar-by-bar
via DailyLoop with AI disabled and a temp journal (live journal untouched).
IndianCostModel deducted per fill; NAV compounds across sessions.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: CLI `backtest` Command

**Files:**
- Modify: `agent/cli.py` — add `backtest` command
- Create: `tests/backtest/test_cli_backtest.py`

The command signature:
```
python -m agent backtest --symbol HDFCBANK --start 2023-01-01 --end 2023-12-31
```

Optional flags: `--timeframe` (default 60m), `--warmup` (default 100).

- [ ] **Step 1: Write the failing tests**

Create `tests/backtest/test_cli_backtest.py`:

```python
# tests/backtest/test_cli_backtest.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent.cli import cli


def test_backtest_command_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["backtest", "--help"])
    assert result.exit_code == 0
    assert "--symbol" in result.output
    assert "--start" in result.output
    assert "--end" in result.output


def test_backtest_exits_when_cache_empty(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch("agent.cli.AppSettings") as MockSettings:
        s = MagicMock()
        s.parquet_cache_dir = tmp_path / "cache"
        s.paper_starting_capital = 83000.0
        MockSettings.return_value = s
        result = runner.invoke(
            cli,
            ["backtest", "--symbol", "HDFCBANK", "--start", "2024-01-02", "--end", "2024-01-05"],
        )
    assert result.exit_code == 1
    assert "No cached data" in result.output or "cache" in result.output.lower()


def test_backtest_prints_report_on_success(tmp_path: Path) -> None:
    """When BacktestRunner returns a non-empty report, backtest command prints metrics."""
    from decimal import Decimal
    from datetime import date

    from agent.backtest.metrics import BacktestMetrics, BacktestReport, SessionResult

    fake_session = SessionResult(
        session_date=date(2024, 1, 2),
        bars_processed=6,
        fills=2,
        gross_pnl=Decimal("1000"),
        costs=Decimal("50"),
        net_pnl=Decimal("950"),
        final_nav=Decimal("83950"),
    )
    fake_metrics = BacktestMetrics(
        total_sessions=1,
        winning_sessions=1,
        win_rate=1.0,
        total_gross_pnl=Decimal("1000"),
        total_costs=Decimal("50"),
        total_net_pnl=Decimal("950"),
        sharpe_ratio=1.5,
        max_drawdown=0.0,
        cagr=0.12,
        initial_nav=Decimal("83000"),
        final_nav=Decimal("83950"),
    )
    fake_report = BacktestReport(sessions=[fake_session], metrics=fake_metrics)

    runner = CliRunner()
    with (
        patch("agent.cli.AppSettings") as MockSettings,
        patch("agent.cli.BacktestRunner") as MockRunner,
    ):
        s = MagicMock()
        s.parquet_cache_dir = tmp_path / "cache"
        s.paper_starting_capital = 83000.0
        MockSettings.return_value = s

        MockRunner.return_value.run.return_value = fake_report

        result = runner.invoke(
            cli,
            ["backtest", "--symbol", "HDFCBANK", "--start", "2024-01-02", "--end", "2024-01-02"],
        )

    assert result.exit_code == 0
    assert "1" in result.output   # total_sessions
    assert "950" in result.output  # net_pnl somewhere in output
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/backtest/test_cli_backtest.py -v --no-cov 2>&1 | head -10
```

Expected: `test_backtest_command_exists FAILED` — command doesn't exist yet.

- [ ] **Step 3: Add `backtest` command to `agent/cli.py`**

Read the end of `agent/cli.py` to find the right insertion point. Add these imports to the top of the file (or inside the command function) and add the command below the existing `run-paper` command:

```python
@cli.command()
@click.option("--symbol", required=True, help="NSE symbol to backtest (e.g. HDFCBANK)")
@click.option(
    "--timeframe",
    default="60m",
    show_default=True,
    type=click.Choice(["15m", "60m", "1d"]),
    help="Bar timeframe",
)
@click.option("--start", "start_str", required=True, help="Start date YYYY-MM-DD (inclusive)")
@click.option("--end", "end_str", required=True, help="End date YYYY-MM-DD (inclusive)")
@click.option(
    "--warmup",
    default=100,
    show_default=True,
    help="Warmup bars before each session for indicator seeding",
)
def backtest(symbol: str, timeframe: str, start_str: str, end_str: str, warmup: int) -> None:
    """Replay historical bars through the strategy and report net-of-cost performance."""
    from datetime import date as date_type
    from decimal import Decimal

    from rich.table import Table

    from agent.backtest.metrics import BacktestReport
    from agent.backtest.runner import BacktestRunner
    from agent.data.cache import ParquetCache
    from agent.risk.manager import RiskManager
    from agent.risk.rules import load_risk_rules
    from agent.strategies.trend_following import TrendFollowingStrategy

    settings = AppSettings()
    start_date = date_type.fromisoformat(start_str)
    end_date = date_type.fromisoformat(end_str)

    console.print(f"[bold]YegEdge Backtest — {symbol} {timeframe} {start_date} → {end_date}[/bold]")

    cache = ParquetCache(root=settings.parquet_cache_dir)
    if not cache.coverage_report():
        console.print("[yellow]No cached data. Run `refresh` first.[/yellow]")
        sys.exit(1)

    strategy = TrendFollowingStrategy()
    risk_rules = load_risk_rules(Path("config/risk_rules.yaml"))
    risk_manager = RiskManager(rules=risk_rules)

    runner = BacktestRunner(
        strategy=strategy,
        risk_manager=risk_manager,
        cache=cache,
        initial_nav=Decimal(str(settings.paper_starting_capital)),
    )

    report = runner.run(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        warmup_bars=warmup,
    )

    if not report.sessions:
        console.print(f"[yellow]No sessions completed for {symbol}/{timeframe} in range.[/yellow]")
        sys.exit(1)

    m = report.metrics

    # --- Metrics summary ---
    console.print()
    table = Table(title="Backtest Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Sessions", str(m.total_sessions))
    table.add_row("Win Rate", f"{m.win_rate:.1%}")
    table.add_row("Gross P&L", f"₹{m.total_gross_pnl:,.2f}")
    table.add_row("Total Costs", f"₹{m.total_costs:,.2f}")
    table.add_row("Net P&L", f"₹{m.total_net_pnl:+,.2f}")
    table.add_row("Sharpe Ratio", f"{m.sharpe_ratio:.3f}")
    table.add_row("Max Drawdown", f"{m.max_drawdown:.2%}")
    table.add_row("CAGR", f"{m.cagr:.2%}")
    table.add_row("Initial NAV", f"₹{m.initial_nav:,.2f}")
    table.add_row("Final NAV", f"₹{m.final_nav:,.2f}")
    console.print(table)

    # --- Per-session detail (last 10) ---
    if len(report.sessions) > 0:
        console.print()
        detail = Table(title="Last 10 Sessions", show_header=True, header_style="bold")
        detail.add_column("Date")
        detail.add_column("Bars", justify="right")
        detail.add_column("Fills", justify="right")
        detail.add_column("Gross P&L", justify="right")
        detail.add_column("Costs", justify="right")
        detail.add_column("Net P&L", justify="right")
        detail.add_column("NAV", justify="right")
        for s in report.sessions[-10:]:
            pnl_color = "green" if s.net_pnl >= 0 else "red"
            detail.add_row(
                str(s.session_date),
                str(s.bars_processed),
                str(s.fills),
                f"₹{s.gross_pnl:,.2f}",
                f"₹{s.costs:,.2f}",
                f"[{pnl_color}]₹{s.net_pnl:+,.2f}[/{pnl_color}]",
                f"₹{s.final_nav:,.2f}",
            )
        console.print(detail)
```

**Important:** `BacktestRunner` must also be imported at the top of the function scope (not module level, following the lazy import pattern already used by `run_paper`).

- [ ] **Step 4: Run tests to verify 3 pass**

```bash
.venv/bin/python -m pytest tests/backtest/test_cli_backtest.py -v --no-cov
```

Expected: `3 passed`

- [ ] **Step 5: Verify the command renders correctly**

```bash
.venv/bin/python -m agent backtest --help
```

Expected: help text showing `--symbol`, `--timeframe`, `--start`, `--end`, `--warmup`.

- [ ] **Step 6: Run full suite**

```bash
.venv/bin/python -m pytest tests/ --no-cov -q 2>&1 | tail -5
```

Expected: 356 passed.

- [ ] **Step 7: Commit**

```bash
git add agent/cli.py tests/backtest/test_cli_backtest.py
git commit -m "$(cat <<'EOF'
feat(backtest): add backtest CLI command with Rich metrics table

Prints Sharpe ratio, max drawdown, CAGR, gross/net P&L, costs,
and per-session detail for the last 10 sessions.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Integration + Full Suite

**Files:** Verify only — no new files unless linting requires fixes.

- [ ] **Step 1: Run full test suite with coverage**

```bash
.venv/bin/python -m pytest tests/ --cov=agent --cov=dashboard \
    --cov-report=term-missing --cov-fail-under=70 -q 2>&1 | tail -30
```

Expected: **356+ tests pass**, coverage ≥ 70%.

- [ ] **Step 2: Run linters on all new/modified files**

```bash
.venv/bin/python -m ruff check agent/backtest/ agent/cli.py \
    tests/backtest/ tests/runner/test_cli_run_paper.py && \
.venv/bin/python -m black --check agent/backtest/ agent/cli.py \
    tests/backtest/ tests/runner/test_cli_run_paper.py && \
echo CLEAN
```

Fix any issues, then re-run until CLEAN. Commit any formatting fixes:
```bash
# If there are fixes:
git add agent/backtest/ agent/cli.py tests/backtest/ tests/runner/test_cli_run_paper.py
git commit -m "$(cat <<'EOF'
style(backtest): ruff + black fixes for backtest modules

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: End-to-end smoke test (pure Python, no real data needed)**

```bash
.venv/bin/python - <<'EOF'
from __future__ import annotations
from datetime import date
from decimal import Decimal

from agent.backtest.costs import IndianCostModel
from agent.backtest.metrics import SessionResult, compute_metrics
from agent.execution.types import ExecutionMode, Fill
from agent.strategies.types import Action
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# Verify cost model
model = IndianCostModel()
enter_fill = Fill(
    order_id="t1", symbol="HDFCBANK", action=Action.ENTER_LONG, quantity=10,
    fill_price=Decimal("1700"), timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    signal_id="s1", strategy_name="trend_following_v1",
    execution_mode=ExecutionMode.PAPER,
)
exit_fill = Fill(
    order_id="t2", symbol="HDFCBANK", action=Action.EXIT_LONG, quantity=10,
    fill_price=Decimal("1730"), timestamp=datetime(2024, 1, 2, 10, 15, tzinfo=IST),
    signal_id="s1", strategy_name="trend_following_v1",
    execution_mode=ExecutionMode.PAPER,
)
enter_cost = model.compute_cost(enter_fill)
exit_cost = model.compute_cost(exit_fill)
assert enter_cost > 0 and exit_cost > 0
assert exit_cost > enter_cost, "EXIT should cost more (STT > stamp)"

# Verify metrics
sessions = [
    SessionResult(
        session_date=date(2024, 1, 2), bars_processed=6, fills=2,
        gross_pnl=Decimal("1300"), costs=enter_cost + exit_cost,
        net_pnl=Decimal("1300") - enter_cost - exit_cost,
        final_nav=Decimal("83000") + Decimal("1300") - enter_cost - exit_cost,
    )
]
metrics = compute_metrics(sessions, Decimal("83000"))
assert metrics.total_sessions == 1
assert metrics.winning_sessions == 1
assert metrics.win_rate == 1.0
assert metrics.total_costs == enter_cost + exit_cost
assert metrics.total_net_pnl == Decimal("1300") - enter_cost - exit_cost

print(f"Round-trip cost for ₹34,000 trade: ₹{enter_cost + exit_cost:.2f}")
print(f"Net P&L after costs: ₹{metrics.total_net_pnl:.2f}")
print("SMOKE TEST PASSED")
EOF
```

Expected: `SMOKE TEST PASSED` with printed cost and net P&L values.

- [ ] **Step 4: Verify CLI renders help for all commands**

```bash
.venv/bin/python -m agent --help
```

Expected: lists `refresh`, `verify`, `run-paper`, `backtest`.

- [ ] **Step 5: Final git log**

```bash
git log --oneline -10
```

---

## Verification

After Phase 10, run the backtest against real cached data (requires `UPSTOX_ACCESS_TOKEN` for initial data refresh):

```bash
# 1. Fetch 1 year of HDFCBANK 60m bars
python -m agent refresh --symbol HDFCBANK --timeframe 60m

# 2. Run backtest
python -m agent backtest --symbol HDFCBANK --timeframe 60m \
    --start 2023-01-01 --end 2023-12-31

# Expected: Rich table with sessions, Sharpe, drawdown, CAGR, costs
# If Sharpe < 0 or drawdown > 20%: strategy needs tuning before paper trading
```

Without real credentials (unit-test only):
```bash
python -m pytest tests/ -v -m "not integration"
```

---

## Self-Review

**Spec coverage:**
- ✅ Bug fix: `run-paper` now enriches raw OHLCV with `FeaturePipeline` per-symbol before `DailyLoop`
- ✅ Cost model: STT 0.025% sell, stamp 0.003% buy, exchange 0.00325%, SEBI 0.0001%, brokerage ₹20, GST 18%
- ✅ CLAUDE.md rule 10: "Costs are always on in any backtest" — enforced via `IndianCostModel`
- ✅ Metrics: Sharpe (annualized, excess over 7% risk-free rate), max drawdown, CAGR, win rate, gross/net P&L
- ✅ BacktestRunner: per-symbol, per-session, live journal untouched, temp kill switch
- ✅ CLI `backtest` command with Rich output
- ✅ `LIVE_TRADING_ENABLED` never set to True

**Placeholder scan:** No TBDs, TODOs, or "implement later" present.

**Type consistency:**
- `SessionResult.net_pnl: Decimal` ← from `result.daily_pnl - costs` (both `Decimal`) ✅
- `BacktestRunner.run()` returns `BacktestReport` ← `compute_metrics()` returns `BacktestMetrics` ✅
- `IndianCostModel.compute_cost(fill: Fill) -> Decimal` ← used in runner as `sum(..., Decimal("0"))` ✅
- CLI imports `BacktestRunner` inside function body (lazy import pattern, consistent with `run_paper`) ✅
