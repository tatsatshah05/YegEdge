# Phase 3 — Strategy Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the strategy layer: `Signal` and `Action` types, a `BaseStrategy` ABC, and a `TrendFollowingStrategy` that fires on EMA21×EMA50 crossovers with ADX + volume confirmation, producing structured `Signal` records for downstream risk/decision modules.

**Architecture:** All strategies are pure functions (`enriched pl.DataFrame → list[Signal]`). The `TrendFollowingStrategy` reads parameters from its constructor (defaults match `config/strategies.yaml`); no YAML reads inside business logic. The `Signal` dataclass is owned by `agent/strategies/types.py` and referenced downstream by the decision engine, AI layer, and risk manager. `Signal` imports `DataQuality` from `agent/data/types` — that is the only cross-module dependency allowed in this direction.

**Tech Stack:** Python 3.11+, Polars (indicator columns from Phase 2 FeaturePipeline), `Decimal` for stop/target prices, `dataclasses` (frozen + slots), pytest.

**Critical conventions (same as Phase 2):**
- `from __future__ import annotations` at the top of every `.py` file
- `logger = structlog.get_logger()` (not `log`)
- `frozen=True, slots=True` on dataclasses
- No `print()` — use `structlog`
- All prices in Polars DataFrames are `Float64`; `Decimal` only in the `Signal` dataclass fields `suggested_stop` and `suggested_target`
- Temp columns in Polars are prefixed `_` and dropped before returning

**Key parameters (from `config/strategies.yaml`):**
| Parameter | Value |
|-----------|-------|
| Fast EMA | `ema_21` (period 21) |
| Slow EMA | `ema_50` (period 50) |
| ADX period / min | 14 / 20 |
| Volume lookback / ratio | 20 bars / 1.1× |
| Stop ATR multiple | 2.0 |
| Target R-multiple | 2.0 (target = 4× ATR above entry) |
| Long-only | Yes (V1) |

---

## File Map

```
agent/strategies/
    __init__.py             # empty package marker
    types.py                # Action(StrEnum), Signal dataclass
    base.py                 # BaseStrategy ABC
    trend_following.py      # TrendFollowingStrategy

tests/strategies/
    __init__.py             # empty package marker
    test_types.py           # Signal / Action validation tests
    test_trend_following.py # Strategy logic: crossover, filters, signals
```

**No modifications to existing files** except to add a `strategies/` directory alongside `data/` and `features/`.

---

## Task 1: Signal types + Action enum

**Files:**
- Create: `agent/strategies/__init__.py`
- Create: `agent/strategies/types.py`
- Create: `tests/strategies/__init__.py`
- Create: `tests/strategies/test_types.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_types.py`:

```python
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from agent.data.types import DataQuality
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")

_TS = datetime(2024, 1, 2, 9, 15, tzinfo=IST)


def _valid_signal(**overrides: object) -> Signal:
    """Return a Signal with valid defaults; override individual fields via kwargs."""
    kwargs: dict[str, object] = {
        "symbol": "HDFCBANK",
        "action": Action.ENTER_LONG,
        "confidence": 0.7,
        "suggested_stop": Decimal("1680.00"),
        "suggested_target": Decimal("1760.00"),
        "invalidation_condition": "Close below EMA21",
        "expected_r": 2.0,
        "time_horizon_hours": 4,
        "regime_fit": 0.9,
        "data_quality": DataQuality.OK,
        "strategy_name": "trend_following_v1",
        "explanation": "EMA21 crossed above EMA50 with ADX=28",
        "timestamp": _TS,
    }
    kwargs.update(overrides)
    return Signal(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------


def test_action_values() -> None:
    assert Action.ENTER_LONG == "enter_long"
    assert Action.EXIT_LONG == "exit_long"
    assert Action.HOLD == "hold"


# ---------------------------------------------------------------------------
# Signal — happy path
# ---------------------------------------------------------------------------


def test_signal_valid_construction() -> None:
    sig = _valid_signal()
    assert sig.symbol == "HDFCBANK"
    assert sig.action == Action.ENTER_LONG
    assert sig.confidence == 0.7
    assert sig.suggested_stop == Decimal("1680.00")
    assert sig.suggested_target == Decimal("1760.00")
    assert sig.expected_r == 2.0
    assert sig.regime_fit == 0.9
    assert sig.data_quality == DataQuality.OK
    assert sig.strategy_name == "trend_following_v1"
    assert sig.timestamp == _TS


def test_signal_is_frozen() -> None:
    sig = _valid_signal()
    with pytest.raises((AttributeError, TypeError)):
        sig.confidence = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Signal — validation errors
# ---------------------------------------------------------------------------


def test_signal_confidence_above_1_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _valid_signal(confidence=1.1)


def test_signal_confidence_below_0_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _valid_signal(confidence=-0.1)


def test_signal_regime_fit_above_1_raises() -> None:
    with pytest.raises(ValueError, match="regime_fit"):
        _valid_signal(regime_fit=1.5)


def test_signal_regime_fit_below_0_raises() -> None:
    with pytest.raises(ValueError, match="regime_fit"):
        _valid_signal(regime_fit=-0.1)


def test_signal_stop_gte_target_raises() -> None:
    with pytest.raises(ValueError, match="suggested_stop"):
        _valid_signal(
            suggested_stop=Decimal("1760.00"),
            suggested_target=Decimal("1680.00"),
        )


def test_signal_stop_equal_target_raises() -> None:
    with pytest.raises(ValueError, match="suggested_stop"):
        _valid_signal(
            suggested_stop=Decimal("1700.00"),
            suggested_target=Decimal("1700.00"),
        )


def test_signal_naive_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _valid_signal(timestamp=datetime(2024, 1, 2, 9, 15))  # naive
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/tatsatshah/Desktop/yegedge
source .venv/bin/activate
pytest tests/strategies/test_types.py -v --no-cov 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'agent.strategies'`

- [ ] **Step 3: Create package markers**

Create `agent/strategies/__init__.py` — empty:

```python
```

Create `tests/strategies/__init__.py` — empty:

```python
```

- [ ] **Step 4: Create `agent/strategies/types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from agent.data.types import DataQuality


class Action(StrEnum):
    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class Signal:
    """Structured output from a strategy — consumed by the decision engine, AI layer, and risk manager.

    All prices (suggested_stop, suggested_target) are Decimal to prevent float drift.
    Polars DataFrames use Float64 for indicators; Decimal only appears in Signal fields.
    """

    symbol: str
    action: Action
    confidence: float               # [0.0, 1.0]
    suggested_stop: Decimal         # ATR-based stop price
    suggested_target: Decimal       # R-multiple target price
    invalidation_condition: str     # human-readable description
    expected_r: float               # (target - entry) / (entry - stop)
    time_horizon_hours: int
    regime_fit: float               # [0.0, 1.0]
    data_quality: DataQuality
    strategy_name: str
    explanation: str                # ≤ 120 chars, structured
    timestamp: datetime             # bar-open time this signal was generated from

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if not 0.0 <= self.regime_fit <= 1.0:
            raise ValueError(
                f"regime_fit must be in [0.0, 1.0], got {self.regime_fit}"
            )
        if self.suggested_stop >= self.suggested_target:
            raise ValueError(
                f"suggested_stop ({self.suggested_stop}) must be < "
                f"suggested_target ({self.suggested_target})"
            )
        if self.timestamp.tzinfo is None:
            raise ValueError("Signal.timestamp must be timezone-aware (IST)")
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/strategies/test_types.py -v --no-cov
```

Expected: `10 passed`

- [ ] **Step 6: Commit**

```bash
git add agent/strategies/__init__.py agent/strategies/types.py \
        tests/strategies/__init__.py tests/strategies/test_types.py
git commit -m "feat(strategies): add Action enum and Signal dataclass"
```

---

## Task 2: BaseStrategy ABC

**Files:**
- Create: `agent/strategies/base.py`

No separate tests — the ABC is tested through its concrete implementation in Task 3.

- [ ] **Step 1: Create `agent/strategies/base.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

from agent.strategies.types import Signal


class BaseStrategy(ABC):
    """Abstract base for all strategy implementations.

    Strategies are pure functions: they receive an enriched DataFrame (output
    of FeaturePipeline.run()) and return a list of Signals.  No I/O, no broker
    calls, no side effects.

    The DataFrame is expected to have at minimum:
    symbol (Utf8), timestamp (Datetime[us, Asia/Kolkata]), open, high, low,
    close, volume (all Float64 / Int64), data_quality (Utf8), and any
    indicator columns the concrete strategy declares as required.

    Column requirements are validated at the start of generate() — if any
    required column is missing, a ValueError is raised immediately.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier — written into every Signal.strategy_name."""

    @abstractmethod
    def generate(self, df: pl.DataFrame) -> list[Signal]:
        """Generate signals from an enriched OHLCV+indicator DataFrame.

        Parameters
        ----------
        df:
            Enriched DataFrame from FeaturePipeline.run().

        Returns
        -------
        list[Signal]
            One Signal per bar where an actionable condition is met.
            Returns an empty list when no conditions are triggered.
            Never raises on normal market data (only on malformed input).
        """
```

- [ ] **Step 2: Commit**

```bash
git add agent/strategies/base.py
git commit -m "feat(strategies): add BaseStrategy ABC"
```

---

## Task 3: TrendFollowingStrategy

**Files:**
- Create: `agent/strategies/trend_following.py`
- Create: `tests/strategies/test_trend_following.py`

**Logic summary:**

Entry (`ENTER_LONG`):
1. `ema_21` crosses above `ema_50` (bar[i-1]: ema21 < ema50; bar[i]: ema21 >= ema50)
2. `adx_14` >= `min_adx` (default 20)
3. `volume` >= `min_volume_ratio` × 20-bar rolling mean volume (default 1.1×)
4. `data_quality` ∈ {`ok`, `partial`}

Exit (`EXIT_LONG`):
- `ema_21` crosses below `ema_50` (death cross), OR
- `close` drops below `ema_21`
- `data_quality` ∈ {`ok`, `partial`}

Stop: `close - stop_atr_multiple * atr_14` (default 2.0×)
Target: `close + stop_atr_multiple * target_r_multiple * atr_14` (default 2.0 × 2.0 = 4.0× ATR)
Expected R: `target_r_multiple` (default 2.0)

Confidence:  ADX linearly mapped from `[min_adx, 60]` → `[0.5, 0.9]`, clipped.

Regime fit:
| Regime | Fit |
|--------|-----|
| `trending` | 1.0 |
| `volatile` | 0.4 |
| `ranging` | 0.1 |
| `unknown` (or absent) | 0.5 |

The strategy computes a 20-bar rolling volume mean internally via Polars `.rolling_mean()` — this is a derived feature used only within the strategy.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_trend_following.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.data.types import DataQuality
from agent.strategies.trend_following import TrendFollowingStrategy
from agent.strategies.types import Action

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(i: int) -> datetime:
    return datetime(2024, 1, 2, 9, 15, tzinfo=IST) + timedelta(hours=i)


def _make_df(rows: list[dict]) -> pl.DataFrame:
    """Build an enriched-style DataFrame from a list of row dicts.

    Defaults: data_quality="ok", regime="trending", volume=100_000,
    atr_14=10.0.  Pass explicit values to override.
    """
    n = len(rows)
    filled: list[dict] = []
    for i, r in enumerate(rows):
        filled.append(
            {
                "symbol": r.get("symbol", "TEST"),
                "timeframe": "60m",
                "timestamp": r.get("timestamp", _ts(i)),
                "open": float(r.get("close", 1000.0)) * 0.999,
                "high": float(r.get("close", 1000.0)) * 1.005,
                "low": float(r.get("close", 1000.0)) * 0.995,
                "close": float(r.get("close", 1000.0)),
                "volume": int(r.get("volume", 100_000)),
                "value": float(r.get("close", 1000.0)) * float(r.get("volume", 100_000)),
                "ema_21": float(r["ema_21"]),
                "ema_50": float(r["ema_50"]),
                "adx_14": float(r.get("adx_14", 25.0)),
                "atr_14": float(r.get("atr_14", 10.0)),
                "data_quality": str(r.get("data_quality", "ok")),
                "regime": str(r.get("regime", "trending")),
            }
        )
    return pl.DataFrame(filled)


def _golden_cross_df(*, adx: float = 25.0, volume: int = 120_000) -> pl.DataFrame:
    """Two-bar DataFrame where bar[1] is a golden cross (EMA21 crosses above EMA50)."""
    return _make_df(
        [
            {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": adx, "volume": volume},  # bar 0: ema21 < ema50
            {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": adx, "volume": volume},  # bar 1: ema21 > ema50 → cross
        ]
    )


def _death_cross_df() -> pl.DataFrame:
    """Two-bar DataFrame where bar[1] is a death cross (EMA21 crosses below EMA50)."""
    return _make_df(
        [
            {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000},  # ema21 > ema50
            {"ema_21": 1005.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000},  # ema21 < ema50 → cross
        ]
    )


# ---------------------------------------------------------------------------
# Strategy instantiation
# ---------------------------------------------------------------------------


def test_strategy_name() -> None:
    assert TrendFollowingStrategy().name == "trend_following_v1"


# ---------------------------------------------------------------------------
# Edge cases: malformed input
# ---------------------------------------------------------------------------


def test_generate_raises_on_missing_required_columns() -> None:
    df = pl.DataFrame({"close": [1000.0]})  # missing ema_21, ema_50, etc.
    with pytest.raises(ValueError, match="ema_21"):
        TrendFollowingStrategy().generate(df)


def test_generate_returns_empty_for_single_bar() -> None:
    df = _make_df([{"ema_21": 1000.0, "ema_50": 1010.0}])
    assert TrendFollowingStrategy().generate(df) == []


def test_generate_returns_empty_for_empty_df() -> None:
    df = _make_df([{"ema_21": 1000.0, "ema_50": 1010.0}]).clear()
    assert TrendFollowingStrategy().generate(df) == []


# ---------------------------------------------------------------------------
# ENTER_LONG signals
# ---------------------------------------------------------------------------


def test_generate_enter_long_on_golden_cross() -> None:
    df = _golden_cross_df()
    signals = TrendFollowingStrategy().generate(df)
    assert len(signals) == 1
    assert signals[0].action == Action.ENTER_LONG
    assert signals[0].symbol == "TEST"


def test_generate_no_enter_long_when_adx_too_low() -> None:
    """Golden cross with ADX < min_adx must not produce a signal."""
    df = _golden_cross_df(adx=15.0)
    signals = TrendFollowingStrategy().generate(df)
    assert signals == []


def test_generate_no_enter_long_when_volume_too_low() -> None:
    """Golden cross with volume well below the rolling average must not fire."""
    # Use a very low volume on the signal bar so ratio < 1.1
    rows = [
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},  # avg bar
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        # bar 20: golden cross but volume = 50_000 << avg 100_000 → ratio = 0.5 < 1.1
        {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 50_000},
    ]
    df = _make_df(rows)
    signals = TrendFollowingStrategy().generate(df)
    enter_signals = [s for s in signals if s.action == Action.ENTER_LONG]
    assert enter_signals == [], f"Expected no ENTER_LONG, got {enter_signals}"


def test_generate_no_enter_long_when_ema21_already_above_ema50() -> None:
    """No crossover (EMA21 already above EMA50 on both bars) must not fire."""
    df = _make_df(
        [
            {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000},
            {"ema_21": 1020.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000},
        ]
    )
    signals = TrendFollowingStrategy().generate(df)
    assert signals == []


def test_generate_enter_long_stop_less_than_target() -> None:
    """The Signal invariant suggested_stop < suggested_target must always hold."""
    df = _golden_cross_df()
    signals = TrendFollowingStrategy().generate(df)
    for sig in signals:
        assert sig.suggested_stop < sig.suggested_target


def test_generate_enter_long_expected_r_equals_target_r_multiple() -> None:
    """expected_r must equal the target_r_multiple constructor parameter."""
    strategy = TrendFollowingStrategy(target_r_multiple=2.0)
    df = _golden_cross_df()
    signals = strategy.generate(df)
    enter = [s for s in signals if s.action == Action.ENTER_LONG]
    assert len(enter) == 1
    assert enter[0].expected_r == 2.0


def test_generate_enter_long_strategy_name() -> None:
    signals = TrendFollowingStrategy().generate(_golden_cross_df())
    assert signals[0].strategy_name == "trend_following_v1"


def test_generate_confidence_scales_with_adx() -> None:
    """Higher ADX on entry bar must produce higher confidence."""
    low_adx_df = _golden_cross_df(adx=21.0)
    high_adx_df = _golden_cross_df(adx=55.0)
    strategy = TrendFollowingStrategy()
    low_sig = strategy.generate(low_adx_df)
    high_sig = strategy.generate(high_adx_df)
    assert len(low_sig) == 1 and len(high_sig) == 1
    assert low_sig[0].confidence < high_sig[0].confidence


def test_generate_confidence_capped_at_0_9() -> None:
    """Confidence must never exceed 0.9 regardless of ADX value."""
    df = _golden_cross_df(adx=200.0)
    signals = TrendFollowingStrategy().generate(df)
    assert signals[0].confidence <= 0.9


def test_generate_confidence_floor_at_0_5() -> None:
    """Confidence must never go below 0.5 even at the minimum ADX threshold."""
    df = _golden_cross_df(adx=20.0)
    signals = TrendFollowingStrategy().generate(df)
    assert signals[0].confidence >= 0.5


# ---------------------------------------------------------------------------
# Regime fit
# ---------------------------------------------------------------------------


def test_generate_regime_fit_trending_is_1() -> None:
    df = _golden_cross_df()  # default regime="trending"
    signals = TrendFollowingStrategy().generate(df)
    assert signals[0].regime_fit == 1.0


def test_generate_regime_fit_volatile() -> None:
    rows = [
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000, "regime": "volatile"},
        {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000, "regime": "volatile"},
    ]
    signals = TrendFollowingStrategy().generate(_make_df(rows))
    assert signals[0].regime_fit == 0.4


def test_generate_regime_fit_ranging_is_low() -> None:
    rows = [
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000, "regime": "ranging"},
        {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000, "regime": "ranging"},
    ]
    signals = TrendFollowingStrategy().generate(_make_df(rows))
    assert signals[0].regime_fit == 0.1


def test_generate_works_without_regime_column() -> None:
    """DataFrame without a 'regime' column must still work; regime_fit defaults to 0.5."""
    rows = [
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000},
        {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000},
    ]
    df = _make_df(rows).drop("regime")
    signals = TrendFollowingStrategy().generate(df)
    assert len(signals) == 1
    assert signals[0].regime_fit == 0.5


# ---------------------------------------------------------------------------
# Data quality gating
# ---------------------------------------------------------------------------


def test_generate_skips_suspect_bars() -> None:
    """Bars with data_quality='suspect' must not produce any signal."""
    rows = [
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000, "data_quality": "suspect"},
        {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000, "data_quality": "suspect"},
    ]
    signals = TrendFollowingStrategy().generate(_make_df(rows))
    assert signals == []


def test_generate_skips_missing_bars() -> None:
    rows = [
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000, "data_quality": "missing"},
        {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000, "data_quality": "missing"},
    ]
    signals = TrendFollowingStrategy().generate(_make_df(rows))
    assert signals == []


def test_generate_accepts_partial_quality_bars() -> None:
    """data_quality='partial' must still be eligible for signal generation."""
    rows = [
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000, "data_quality": "partial"},
        {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000, "data_quality": "partial"},
    ]
    signals = TrendFollowingStrategy().generate(_make_df(rows))
    assert len(signals) == 1
    assert signals[0].data_quality == DataQuality.PARTIAL


# ---------------------------------------------------------------------------
# EXIT_LONG signals
# ---------------------------------------------------------------------------


def test_generate_exit_long_on_death_cross() -> None:
    df = _death_cross_df()
    signals = TrendFollowingStrategy().generate(df)
    assert len(signals) == 1
    assert signals[0].action == Action.EXIT_LONG


def test_generate_exit_long_when_close_below_ema21() -> None:
    """Close dropping below EMA21 must trigger EXIT_LONG even without a death cross."""
    rows = [
        {"ema_21": 1010.0, "ema_50": 1005.0, "close": 1012.0, "adx_14": 25.0, "volume": 120_000},  # ema21 > ema50, no cross
        {"ema_21": 1010.0, "ema_50": 1005.0, "close": 1008.0, "adx_14": 25.0, "volume": 120_000},  # close < ema21
    ]
    signals = TrendFollowingStrategy().generate(_make_df(rows))
    exits = [s for s in signals if s.action == Action.EXIT_LONG]
    assert len(exits) == 1


def test_generate_exit_long_stop_less_than_target() -> None:
    """EXIT_LONG signals must also satisfy the Signal invariant stop < target."""
    df = _death_cross_df()
    signals = TrendFollowingStrategy().generate(df)
    for sig in signals:
        if sig.action == Action.EXIT_LONG:
            assert sig.suggested_stop < sig.suggested_target
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/strategies/test_trend_following.py -v --no-cov 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'agent.strategies.trend_following'`

- [ ] **Step 3: Create `agent/strategies/trend_following.py`**

```python
from __future__ import annotations

from decimal import Decimal

import polars as pl
import structlog

from agent.data.types import DataQuality
from agent.strategies.base import BaseStrategy
from agent.strategies.types import Action, Signal

logger = structlog.get_logger()

_STRATEGY_NAME = "trend_following_v1"

_REGIME_FIT: dict[str, float] = {
    "trending": 1.0,
    "volatile": 0.4,
    "ranging": 0.1,
    "unknown": 0.5,
}

_ALLOWED_QUALITIES = {DataQuality.OK.value, DataQuality.PARTIAL.value}

# Columns the strategy requires in the input DataFrame.
_REQUIRED_COLS = frozenset(
    {"symbol", "timestamp", "close", "volume", "ema_21", "ema_50", "adx_14", "atr_14", "data_quality"}
)


class TrendFollowingStrategy(BaseStrategy):
    """EMA21 × EMA50 crossover with ADX and volume confirmation.

    Entry (ENTER_LONG):
    - EMA21 crosses above EMA50
    - ADX14 >= min_adx (default 20)
    - volume >= min_volume_ratio × 20-bar rolling mean volume (default 1.1×)
    - data_quality in {ok, partial}

    Exit (EXIT_LONG):
    - EMA21 crosses below EMA50 (death cross), OR
    - close drops below EMA21

    Stop: close - stop_atr_multiple * ATR14 (default 2.0×)
    Target: close + stop_atr_multiple * target_r_multiple * ATR14 (default 4.0×)
    Expected R: target_r_multiple (default 2.0)

    Parameters match config/strategies.yaml defaults.
    """

    def __init__(
        self,
        *,
        fast_ema_col: str = "ema_21",
        slow_ema_col: str = "ema_50",
        adx_col: str = "adx_14",
        atr_col: str = "atr_14",
        min_adx: float = 20.0,
        min_volume_ratio: float = 1.1,
        volume_lookback: int = 20,
        stop_atr_multiple: float = 2.0,
        target_r_multiple: float = 2.0,
        time_horizon_hours: int = 4,
    ) -> None:
        self._fast = fast_ema_col
        self._slow = slow_ema_col
        self._adx_col = adx_col
        self._atr_col = atr_col
        self._min_adx = min_adx
        self._min_vol_ratio = min_volume_ratio
        self._vol_lookback = volume_lookback
        self._stop_mult = stop_atr_multiple
        self._target_mult = stop_atr_multiple * target_r_multiple
        self._expected_r = target_r_multiple
        self._horizon = time_horizon_hours

    @property
    def name(self) -> str:
        return _STRATEGY_NAME

    def generate(self, df: pl.DataFrame) -> list[Signal]:
        """Scan *df* and return a Signal for every bar where entry/exit conditions are met.

        Callers should pass the full history for backtesting.
        For live trading, pass the last N bars (at minimum volume_lookback + 2).
        """
        self._validate_columns(df)

        if len(df) < 2:
            return []

        # Sort so crossover detection uses chronological order.
        df = df.sort("timestamp")

        # Add 20-bar rolling volume mean as a temp column for the volume filter.
        df = df.with_columns(
            pl.col("volume")
            .cast(pl.Float64)
            .rolling_mean(window_size=self._vol_lookback, min_periods=1)
            .alias("_vol_mean")
        )

        fast = df[self._fast].to_list()
        slow = df[self._slow].to_list()
        adx = df[self._adx_col].to_list()
        atr = df[self._atr_col].to_list()
        closes = df["close"].to_list()
        volumes = df["volume"].to_list()
        vol_means = df["_vol_mean"].to_list()
        timestamps = df["timestamp"].to_list()
        qualities = df["data_quality"].to_list()
        symbols = df["symbol"].to_list()
        regimes = df["regime"].to_list() if "regime" in df.columns else None

        signals: list[Signal] = []

        for i in range(1, len(df)):
            quality = qualities[i]
            if quality not in _ALLOWED_QUALITIES:
                continue

            prev_fast = fast[i - 1]
            prev_slow = slow[i - 1]
            curr_fast = fast[i]
            curr_slow = slow[i]
            curr_adx = adx[i]
            curr_atr = atr[i]
            curr_close = closes[i]
            curr_vol = float(volumes[i])
            curr_vol_mean = vol_means[i]
            ts = timestamps[i]
            symbol = symbols[i]

            # Skip warm-up bars where any indicator is None
            if any(v is None for v in [prev_fast, prev_slow, curr_fast, curr_slow, curr_adx, curr_atr]):
                continue

            regime_str = regimes[i] if regimes is not None else "unknown"
            regime_fit = _REGIME_FIT.get(str(regime_str), 0.5)

            # ---------------------------------------------------------------
            # ENTER_LONG: golden cross + ADX filter + volume confirmation
            # ---------------------------------------------------------------
            golden_cross = prev_fast < prev_slow and curr_fast >= curr_slow
            adx_ok = curr_adx >= self._min_adx
            vol_ratio = (curr_vol / curr_vol_mean) if curr_vol_mean and curr_vol_mean > 0 else 0.0
            volume_ok = vol_ratio >= self._min_vol_ratio

            if golden_cross and adx_ok and volume_ok:
                stop = Decimal(str(round(curr_close - self._stop_mult * curr_atr, 2)))
                target = Decimal(str(round(curr_close + self._target_mult * curr_atr, 2)))
                # Ensure stop < target (should always hold for positive ATR; guard anyway)
                if stop >= target:
                    logger.warning(
                        "trend_following.enter_long.invalid_stop_target",
                        symbol=symbol,
                        stop=str(stop),
                        target=str(target),
                    )
                    continue
                signals.append(
                    Signal(
                        symbol=symbol,
                        action=Action.ENTER_LONG,
                        confidence=self._confidence(curr_adx),
                        suggested_stop=stop,
                        suggested_target=target,
                        invalidation_condition=(
                            f"Close below EMA21 [{round(curr_fast, 2)}] or stop [{stop}]"
                        ),
                        expected_r=self._expected_r,
                        time_horizon_hours=self._horizon,
                        regime_fit=regime_fit,
                        data_quality=DataQuality(quality),
                        strategy_name=_STRATEGY_NAME,
                        explanation=(
                            f"EMA21 crossed above EMA50 "
                            f"(ADX={curr_adx:.1f}, vol_ratio={vol_ratio:.2f}); "
                            f"stop={stop}, target={target}"
                        ),
                        timestamp=ts,
                    )
                )
                continue  # don't also check exit on the same bar as entry

            # ---------------------------------------------------------------
            # EXIT_LONG: death cross OR close below EMA21
            # ---------------------------------------------------------------
            death_cross = prev_fast > prev_slow and curr_fast <= curr_slow
            close_below_fast = curr_close < curr_fast

            if death_cross or close_below_fast:
                # Dummy stop/target to satisfy Signal invariant (stop < target).
                # The decision engine treats EXIT_LONG as "close the position immediately".
                exit_stop = Decimal(str(round(curr_close * 0.99, 2)))
                exit_target = Decimal(str(round(curr_close * 1.01, 2)))
                signals.append(
                    Signal(
                        symbol=symbol,
                        action=Action.EXIT_LONG,
                        confidence=0.8 if death_cross else 0.6,
                        suggested_stop=exit_stop,
                        suggested_target=exit_target,
                        invalidation_condition="Exit signal — close position",
                        expected_r=0.0,
                        time_horizon_hours=0,
                        regime_fit=regime_fit,
                        data_quality=DataQuality(quality),
                        strategy_name=_STRATEGY_NAME,
                        explanation=(
                            "EMA21 crossed below EMA50 (death cross)"
                            if death_cross
                            else f"Close ({curr_close}) fell below EMA21 ({curr_fast:.2f})"
                        ),
                        timestamp=ts,
                    )
                )

        logger.debug(
            "trend_following.generate.done",
            symbol=df["symbol"][0] if len(df) > 0 else "?",
            signals=len(signals),
            bars=len(df),
        )
        return signals

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pl.DataFrame) -> None:
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"TrendFollowingStrategy requires columns {sorted(missing)} in df. "
                f"Run FeaturePipeline.run() on the DataFrame first."
            )

    def _confidence(self, adx: float) -> float:
        """Map ADX to confidence in [0.5, 0.9]. Linear between min_adx and 60."""
        adx_range = 60.0 - self._min_adx
        if adx_range <= 0:
            return 0.5
        normalized = (adx - self._min_adx) / adx_range
        return round(max(0.5, min(0.9, 0.5 + normalized * 0.4)), 4)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/strategies/test_trend_following.py -v --no-cov
```

Expected: `24 passed`

- [ ] **Step 5: Commit**

```bash
git add agent/strategies/trend_following.py tests/strategies/test_trend_following.py
git commit -m "feat(strategies): add TrendFollowingStrategy (EMA21×50 crossover, ADX+volume filters)"
```

---

## Task 4: Full test suite + coverage gate

**Files:**
- Run the full suite; add targeted tests if coverage for `agent/strategies/` falls below 70%.

- [ ] **Step 1: Run the complete test suite**

```bash
cd /Users/tatsatshah/Desktop/yegedge
source .venv/bin/activate
pytest tests/ -v --cov=agent --cov-report=term-missing
```

Expected: all tests pass (Phase 1 + 2 + 3). Coverage for `agent/strategies/` ≥ 70%.

If coverage for `trend_following.py` is below 70%, check the `term-missing` column for uncovered lines and add tests.

- [ ] **Step 2: Run linters**

```bash
ruff check agent/strategies/ tests/strategies/
black --check agent/strategies/ tests/strategies/
```

Expected: no issues. Fix any before committing:

```bash
black agent/strategies/ tests/strategies/
ruff check --fix agent/strategies/ tests/strategies/
```

- [ ] **Step 3: Spot-check a full pipeline → strategy run**

```bash
source .venv/bin/activate && python - <<'EOF'
from tests.features.conftest import make_ohlcv_df
from agent.features.pipeline import FeaturePipeline
from agent.features.regime import RegimeDetector
from agent.strategies.trend_following import TrendFollowingStrategy

# 200 bars of sinusoidal price — should produce crosses
import math
closes = [1500.0 + 200.0 * math.sin(i * 0.15) for i in range(200)]
df = make_ohlcv_df(closes)

# Enrich
pipeline = FeaturePipeline()
enriched = pipeline.run(df)
rd = RegimeDetector()
rd.fit(enriched)
pipeline_r = FeaturePipeline(regime_detector=rd)
final = pipeline_r.run(df)

# Generate signals
strategy = TrendFollowingStrategy()
signals = strategy.generate(final)
print(f"Bars: {len(final)}, Signals: {len(signals)}")
for s in signals[:5]:
    print(f"  {s.timestamp.strftime('%Y-%m-%d %H:%M')} | {s.action} | conf={s.confidence:.2f} | regime_fit={s.regime_fit} | stop={s.suggested_stop} target={s.suggested_target}")
EOF
```

Expected: multiple ENTER_LONG and EXIT_LONG signals printed; no exceptions.

- [ ] **Step 4: Commit**

```bash
git add agent/strategies/ tests/strategies/
git commit -m "test(strategies): Phase 3 test suite passes coverage gate"
```

---

## Self-Review

**Spec coverage:**
- [x] `Action(StrEnum)` with ENTER_LONG, EXIT_LONG, HOLD — Task 1
- [x] `Signal` dataclass with all architecture fields — Task 1
- [x] Signal validation: confidence/regime_fit range, stop < target, timezone-aware timestamp — Task 1
- [x] `BaseStrategy` ABC with `name` property and `generate()` method — Task 2
- [x] `TrendFollowingStrategy.generate()` → `list[Signal]` — Task 3
- [x] Entry: EMA21×EMA50 golden cross + ADX >= 20 + volume ratio >= 1.1 — Task 3
- [x] Exit: death cross OR close < EMA21 — Task 3
- [x] Stop: 2.0 × ATR14; target: 4.0 × ATR14 (2R); expected_r=2.0 — Task 3
- [x] Confidence: ADX→[0.5, 0.9] linear map — Task 3
- [x] Regime fit: trending=1.0, volatile=0.4, ranging=0.1, unknown=0.5 — Task 3
- [x] Data quality gate: suspect/missing bars skipped — Task 3
- [x] Parameters match `config/strategies.yaml` defaults — Task 3
- [x] Full test suite + linting — Task 4

**Placeholder scan:** None found.

**Type consistency:**
- `Action.ENTER_LONG / EXIT_LONG / HOLD` used consistently throughout
- `Signal` constructor called with all 13 fields in `trend_following.py` — matches `types.py` definition
- `DataQuality(quality)` conversion from string column value — consistent
- `Decimal(str(round(..., 2)))` pattern for all monetary values — consistent
