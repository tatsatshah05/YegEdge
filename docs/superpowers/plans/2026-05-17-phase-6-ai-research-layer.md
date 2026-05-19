# Phase 6 — AI Research Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `agent/ai/` — the Claude integration that takes a `Signal` and returns a schema-bounded `ResearchNote` with bullish/bearish case, dominant risk, and an optional veto flag. Includes in-memory caching and a hard budget cap.

**Architecture:** Three focused files: `prompt.py` (pure prompt builder, no I/O), `cache.py` (in-memory note cache keyed by regime+action+R/R bucket), and `analyst.py` (`AIAnalyst` class: budget check → cache lookup → Claude API call → ResearchNote). The `ResearchNote` type is already defined in `agent/decision/types.py` — this module only produces it. All Claude calls use tool use (function calling) for reliable structured output.

**Tech Stack:** Python 3.11+, `anthropic>=0.40.0` (already in requirements), `pydantic>=2.8.0`, `structlog`, `pytest` + `pytest-mock`. No real API calls in tests — Anthropic client is always mocked.

---

## Context for subagent workers

**Project:** `/Users/tatsatshah/Desktop/yegedge`
**Branch:** `phase-2-feature-engineering` — do not create a new branch.
**Virtualenv:** `source /Users/tatsatshah/Desktop/yegedge/.venv/bin/activate`

**Conventions (binding):**
- `from __future__ import annotations` first line of every `.py` file
- `logger = structlog.get_logger()` (not `log`)
- `@dataclass(frozen=True, slots=True)` on all dataclasses
- No `print()` — use structlog
- Monetary values: `Decimal`. Ratios/weights: `float`
- All `datetime` timezone-aware IST

**Key types already defined — do NOT redefine:**

```python
# agent/decision/types.py
@dataclass(frozen=True, slots=True)
class ResearchNote:
    signal_id: str          # "{symbol}:{action}:{timestamp.isoformat()}"
    bullish_case: str
    bearish_case: str
    dominant_risk: str
    regime_fit_assessment: str
    confidence_qualitative: str  # "LOW" | "MEDIUM" | "HIGH"
    veto: bool
    veto_reason: str | None
    model_used: str
    tokens_used: int
    cached: bool

# agent/strategies/types.py
@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    action: Action           # ENTER_LONG | EXIT_LONG | HOLD
    confidence: float        # [0, 1]
    suggested_stop: Decimal
    suggested_target: Decimal
    invalidation_condition: str
    expected_r: float
    time_horizon_hours: int
    regime_fit: float        # [0, 1]
    data_quality: DataQuality
    strategy_name: str
    explanation: str
    timestamp: datetime

# config/settings.py
class AppSettings(BaseSettings):
    anthropic_api_key: str = ""
    claude_model_primary: str = "claude-sonnet-4-6"
    claude_model_cheap: str = "claude-haiku-4-5-20251001"
    max_monthly_api_spend_inr: Decimal = Decimal("1500.00")
```

**signal_id convention:** `f"{signal.symbol}:{signal.action}:{signal.timestamp.isoformat()}"` — same format as in `Decision.signal_id` so the DecisionEngine can look it up.

---

## File Map

```
agent/ai/
    __init__.py     — empty package marker
    prompt.py       — build_prompt(signal, portfolio_summary) → str  (pure, no I/O)
    cache.py        — NoteCache: in-memory dict keyed by (action, regime_bucket, rr_bucket)
    analyst.py      — AIAnalyst: budget guard → cache → Claude tool use → ResearchNote

tests/ai/
    __init__.py     — empty
    test_prompt.py  — 6 tests (pure function, no mocking needed)
    test_cache.py   — 6 tests (pure in-memory, no mocking needed)
    test_analyst.py — 10 tests (all Anthropic client mocked via pytest-mock)
```

---

## Task 1: Prompt Builder

**Files:**
- Create: `agent/ai/__init__.py`
- Create: `agent/ai/prompt.py`
- Create: `tests/ai/__init__.py`
- Test: `tests/ai/test_prompt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ai/test_prompt.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from agent.ai.prompt import build_prompt
from agent.data.types import DataQuality
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")


def _make_signal(
    symbol: str = "HDFCBANK",
    action: Action = Action.ENTER_LONG,
    confidence: float = 0.75,
    regime_fit: float = 0.8,
    expected_r: float = 2.0,
) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        suggested_stop=Decimal("1680.00"),
        suggested_target=Decimal("1760.00"),
        invalidation_condition="Close below EMA21",
        expected_r=expected_r,
        time_horizon_hours=4,
        regime_fit=regime_fit,
        data_quality=DataQuality.OK,
        strategy_name="trend_following_v1",
        explanation="EMA21 crossed above EMA50 (ADX=28.5, vol_ratio=1.20)",
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )


def test_prompt_contains_symbol() -> None:
    prompt = build_prompt(_make_signal(symbol="TCS"))
    assert "TCS" in prompt


def test_prompt_contains_action() -> None:
    prompt = build_prompt(_make_signal(action=Action.ENTER_LONG))
    assert "ENTER_LONG" in prompt or "enter_long" in prompt or "long" in prompt.lower()


def test_prompt_contains_stop_and_target() -> None:
    prompt = build_prompt(_make_signal())
    assert "1680" in prompt
    assert "1760" in prompt


def test_prompt_contains_expected_r() -> None:
    prompt = build_prompt(_make_signal(expected_r=2.5))
    assert "2.5" in prompt


def test_prompt_contains_explanation() -> None:
    prompt = build_prompt(_make_signal())
    assert "EMA21" in prompt


def test_prompt_with_portfolio_summary_includes_it() -> None:
    summary = "Holding 2 positions: SBIN (10 shares), TCS (5 shares)"
    prompt = build_prompt(_make_signal(), portfolio_summary=summary)
    assert "SBIN" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/tatsatshah/Desktop/yegedge && source .venv/bin/activate
python -m pytest tests/ai/test_prompt.py -v --no-cov 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'agent.ai'`

- [ ] **Step 3: Create package skeletons**

Create `agent/ai/__init__.py` and `tests/ai/__init__.py` — both with content `# intentionally empty`.

- [ ] **Step 4: Write `agent/ai/prompt.py`**

```python
from __future__ import annotations

from agent.strategies.types import Signal

_SYSTEM_PROMPT = """You are a pre-trade risk analyst for an NSE equity intraday trading system.
Analyse the trade signal provided and return a structured research note.
Be concise: bullish_case and bearish_case must each be ≤ 80 words.
Veto the trade (veto: true) only when you identify a material risk that invalidates
the technical signal — for example, a major earnings announcement, RBI policy event,
or a clear contradiction between the signal direction and prevailing macro regime."""

_NOTE_SCHEMA = {
    "name": "submit_research_note",
    "description": "Submit the structured pre-trade research note. Call this once.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bullish_case": {
                "type": "string",
                "description": "Bullish thesis for the trade in ≤80 words.",
            },
            "bearish_case": {
                "type": "string",
                "description": "Key risk / bearish argument in ≤80 words.",
            },
            "dominant_risk": {
                "type": "string",
                "description": "The single most important risk factor (one sentence).",
            },
            "regime_fit_assessment": {
                "type": "string",
                "description": "One sentence on whether current regime suits the strategy.",
            },
            "confidence_qualitative": {
                "type": "string",
                "enum": ["LOW", "MEDIUM", "HIGH"],
                "description": "Overall qualitative confidence in the trade.",
            },
            "veto": {
                "type": "boolean",
                "description": "True only if a material risk invalidates the signal.",
            },
            "veto_reason": {
                "type": ["string", "null"],
                "description": "Required when veto is true; null otherwise.",
            },
        },
        "required": [
            "bullish_case",
            "bearish_case",
            "dominant_risk",
            "regime_fit_assessment",
            "confidence_qualitative",
            "veto",
            "veto_reason",
        ],
    },
}


def build_prompt(signal: Signal, portfolio_summary: str = "") -> str:
    """Build the user-turn prompt for pre-trade analysis.

    Returns a plain string. The system prompt and tool schema live in analyst.py.
    This function is pure — no I/O, no randomness, deterministic output.
    """
    lines = [
        f"Symbol: {signal.symbol}",
        f"Action: {signal.action}",
        f"Strategy: {signal.strategy_name}",
        f"Signal: {signal.explanation}",
        f"Confidence: {signal.confidence:.2f}  Regime fit: {signal.regime_fit:.2f}",
        f"Expected R: {signal.expected_r:.1f}x",
        f"Stop: {signal.suggested_stop}  Target: {signal.suggested_target}",
        f"Time horizon: {signal.time_horizon_hours}h",
        f"Data quality: {signal.data_quality}",
    ]
    if portfolio_summary:
        lines.append(f"Portfolio context: {portfolio_summary}")
    lines.append(
        "\nAnalyse this trade signal and submit your research note using the tool."
    )
    return "\n".join(lines)


SYSTEM_PROMPT = _SYSTEM_PROMPT
NOTE_SCHEMA = _NOTE_SCHEMA
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/ai/test_prompt.py -v --no-cov
```

Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add agent/ai/__init__.py agent/ai/prompt.py \
        tests/ai/__init__.py tests/ai/test_prompt.py
git commit -m "feat(ai): add build_prompt pure function and tool schema"
```

---

## Task 2: NoteCache

**Files:**
- Create: `agent/ai/cache.py`
- Test: `tests/ai/test_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ai/test_cache.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from agent.ai.cache import NoteCache
from agent.data.types import DataQuality
from agent.decision.types import ResearchNote
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")


def _make_signal(
    symbol: str = "HDFCBANK",
    action: Action = Action.ENTER_LONG,
    regime_fit: float = 0.8,
    expected_r: float = 2.0,
    confidence: float = 0.7,
) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        suggested_stop=Decimal("1680.00"),
        suggested_target=Decimal("1760.00"),
        invalidation_condition="Close below EMA21",
        expected_r=expected_r,
        time_horizon_hours=4,
        regime_fit=regime_fit,
        data_quality=DataQuality.OK,
        strategy_name="trend_following_v1",
        explanation="EMA21 crossed above EMA50",
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )


def _make_note(signal_id: str = "test-id") -> ResearchNote:
    return ResearchNote(
        signal_id=signal_id,
        bullish_case="Strong momentum.",
        bearish_case="Overbought short-term.",
        dominant_risk="FII selling pressure.",
        regime_fit_assessment="Trending regime suits strategy.",
        confidence_qualitative="HIGH",
        veto=False,
        veto_reason=None,
        model_used="claude-haiku-4-5-20251001",
        tokens_used=200,
        cached=False,
    )


def test_cache_miss_returns_none() -> None:
    cache = NoteCache()
    sig = _make_signal()
    assert cache.get(sig) is None


def test_cache_put_then_get_returns_note() -> None:
    cache = NoteCache()
    sig = _make_signal()
    note = _make_note()
    cache.put(sig, note)
    result = cache.get(sig)
    assert result is not None
    assert result.bullish_case == note.bullish_case


def test_cache_hit_returns_note_with_cached_true() -> None:
    cache = NoteCache()
    sig = _make_signal()
    note = _make_note()
    cache.put(sig, note)
    result = cache.get(sig)
    assert result is not None
    assert result.cached is True


def test_cache_key_same_action_regime_rr_hits() -> None:
    cache = NoteCache()
    # Two signals with same action, regime bucket, and R/R bucket → same cache key
    sig1 = _make_signal(symbol="HDFCBANK", action=Action.ENTER_LONG, regime_fit=0.85, expected_r=2.1)
    sig2 = _make_signal(symbol="TCS", action=Action.ENTER_LONG, regime_fit=0.82, expected_r=2.3)
    note = _make_note()
    cache.put(sig1, note)
    # TCS with same regime/rr bucket should hit the cache
    result = cache.get(sig2)
    assert result is not None


def test_cache_key_different_action_misses() -> None:
    cache = NoteCache()
    sig_enter = _make_signal(action=Action.ENTER_LONG)
    sig_exit = _make_signal(action=Action.EXIT_LONG)
    cache.put(sig_enter, _make_note())
    assert cache.get(sig_exit) is None


def test_cache_size() -> None:
    cache = NoteCache()
    for i in range(5):
        sig = _make_signal(expected_r=float(i + 1))
        cache.put(sig, _make_note())
    assert cache.size >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/ai/test_cache.py -v --no-cov 2>&1 | head -15
```

Expected: `ImportError: cannot import name 'NoteCache'`

- [ ] **Step 3: Write `agent/ai/cache.py`**

```python
from __future__ import annotations

import structlog

from agent.decision.types import ResearchNote
from agent.strategies.types import Signal

logger = structlog.get_logger()


def _regime_bucket(regime_fit: float) -> str:
    if regime_fit >= 0.8:
        return "strong"
    if regime_fit >= 0.5:
        return "moderate"
    return "weak"


def _rr_bucket(expected_r: float) -> str:
    if expected_r >= 3.0:
        return "high"
    if expected_r >= 2.0:
        return "medium"
    return "low"


def _cache_key(signal: Signal) -> str:
    """Deterministic cache key from (action, regime bucket, R/R bucket).

    Deliberately excludes symbol so that structurally identical signals for
    different symbols share a cache entry — this is the intended behaviour.
    The pre-trade research note captures regime and risk characteristics of
    the signal pattern, not symbol-specific news.
    """
    return (
        f"{signal.action}:"
        f"{_regime_bucket(signal.regime_fit)}:"
        f"{_rr_bucket(signal.expected_r)}"
    )


class NoteCache:
    """In-memory cache of ResearchNote objects keyed by signal pattern.

    Cache key: (action, regime_bucket, rr_bucket) — same key for structurally
    identical signals on different symbols. This is the intended semantics:
    the research note captures the trade *pattern*, not symbol-specific data.
    """

    def __init__(self) -> None:
        self._store: dict[str, ResearchNote] = {}

    def get(self, signal: Signal) -> ResearchNote | None:
        key = _cache_key(signal)
        note = self._store.get(key)
        if note is None:
            return None
        # Return a copy with cached=True so the caller knows it came from cache.
        return ResearchNote(
            signal_id=note.signal_id,
            bullish_case=note.bullish_case,
            bearish_case=note.bearish_case,
            dominant_risk=note.dominant_risk,
            regime_fit_assessment=note.regime_fit_assessment,
            confidence_qualitative=note.confidence_qualitative,
            veto=note.veto,
            veto_reason=note.veto_reason,
            model_used=note.model_used,
            tokens_used=note.tokens_used,
            cached=True,
        )

    def put(self, signal: Signal, note: ResearchNote) -> None:
        key = _cache_key(signal)
        self._store[key] = note
        logger.debug("note_cache.put", key=key, veto=note.veto)

    @property
    def size(self) -> int:
        return len(self._store)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/ai/test_cache.py -v --no-cov
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add agent/ai/cache.py tests/ai/test_cache.py
git commit -m "feat(ai): add NoteCache keyed by (action, regime, R/R bucket)"
```

---

## Task 3: AIAnalyst

**Files:**
- Create: `agent/ai/analyst.py`
- Test: `tests/ai/test_analyst.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ai/test_analyst.py
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from agent.ai.analyst import AIAnalyst
from agent.data.types import DataQuality
from agent.decision.types import ResearchNote
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")


def _make_signal(
    symbol: str = "HDFCBANK",
    confidence: float = 0.75,
    expected_r: float = 2.0,
    regime_fit: float = 0.8,
) -> Signal:
    return Signal(
        symbol=symbol,
        action=Action.ENTER_LONG,
        confidence=confidence,
        suggested_stop=Decimal("1680.00"),
        suggested_target=Decimal("1760.00"),
        invalidation_condition="Close below EMA21",
        expected_r=expected_r,
        time_horizon_hours=4,
        regime_fit=regime_fit,
        data_quality=DataQuality.OK,
        strategy_name="trend_following_v1",
        explanation="EMA21 crossed above EMA50",
        timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
    )


def _fake_tool_response(
    bullish: str = "Strong uptrend supported by high ADX.",
    bearish: str = "Potential reversal if volume drops off.",
    veto: bool = False,
    veto_reason: str | None = None,
    confidence_qualitative: str = "HIGH",
) -> MagicMock:
    """Build a mock Anthropic API response for the submit_research_note tool."""
    tool_use = MagicMock()
    tool_use.type = "tool_use"
    tool_use.input = {
        "bullish_case": bullish,
        "bearish_case": bearish,
        "dominant_risk": "FII selling pressure.",
        "regime_fit_assessment": "Trending regime suits strategy.",
        "confidence_qualitative": confidence_qualitative,
        "veto": veto,
        "veto_reason": veto_reason,
    }

    response = MagicMock()
    response.content = [tool_use]
    response.usage.input_tokens = 150
    response.usage.output_tokens = 100
    return response


def _make_analyst(api_key: str = "test-key", budget_inr: Decimal = Decimal("1500")) -> AIAnalyst:
    return AIAnalyst(api_key=api_key, max_budget_inr=budget_inr)


# --- Basic ---

def test_analyze_returns_research_note() -> None:
    analyst = _make_analyst()
    sig = _make_signal()
    with patch("agent.ai.analyst.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _fake_tool_response()
        note = analyst.analyze(sig)
    assert isinstance(note, ResearchNote)


def test_analyze_signal_id_matches_convention() -> None:
    analyst = _make_analyst()
    sig = _make_signal()
    with patch("agent.ai.analyst.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _fake_tool_response()
        note = analyst.analyze(sig)
    expected_id = f"{sig.symbol}:{sig.action}:{sig.timestamp.isoformat()}"
    assert note.signal_id == expected_id


def test_analyze_not_cached_on_first_call() -> None:
    analyst = _make_analyst()
    sig = _make_signal()
    with patch("agent.ai.analyst.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _fake_tool_response()
        note = analyst.analyze(sig)
    assert note.cached is False


def test_analyze_uses_cache_on_second_call() -> None:
    analyst = _make_analyst()
    sig = _make_signal()
    with patch("agent.ai.analyst.anthropic.Anthropic") as MockClient:
        mock_create = MockClient.return_value.messages.create
        mock_create.return_value = _fake_tool_response()
        analyst.analyze(sig)
        note2 = analyst.analyze(sig)
    assert note2.cached is True
    assert mock_create.call_count == 1  # second call hits cache, not API


def test_analyze_high_confidence_uses_cheap_model() -> None:
    analyst = _make_analyst()
    sig = _make_signal(confidence=0.85)  # ≥ 0.80 → Haiku
    with patch("agent.ai.analyst.anthropic.Anthropic") as MockClient:
        mock_create = MockClient.return_value.messages.create
        mock_create.return_value = _fake_tool_response()
        analyst.analyze(sig)
    call_kwargs = mock_create.call_args.kwargs
    assert "haiku" in call_kwargs["model"]


def test_analyze_low_confidence_uses_primary_model() -> None:
    analyst = _make_analyst()
    sig = _make_signal(confidence=0.65)  # < 0.80 → Sonnet
    with patch("agent.ai.analyst.anthropic.Anthropic") as MockClient:
        mock_create = MockClient.return_value.messages.create
        mock_create.return_value = _fake_tool_response()
        analyst.analyze(sig)
    call_kwargs = mock_create.call_args.kwargs
    assert "sonnet" in call_kwargs["model"] or "claude-sonnet" in call_kwargs["model"]


def test_analyze_veto_propagated() -> None:
    analyst = _make_analyst()
    sig = _make_signal()
    with patch("agent.ai.analyst.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _fake_tool_response(
            veto=True, veto_reason="Earnings tomorrow — avoid overnight gap risk"
        )
        note = analyst.analyze(sig)
    assert note.veto is True
    assert note.veto_reason == "Earnings tomorrow — avoid overnight gap risk"


def test_budget_exceeded_returns_fallback_note() -> None:
    analyst = _make_analyst(budget_inr=Decimal("0"))  # zero budget
    sig = _make_signal()
    with patch("agent.ai.analyst.anthropic.Anthropic") as MockClient:
        note = analyst.analyze(sig)
        MockClient.return_value.messages.create.assert_not_called()
    assert note.veto is False
    assert note.model_used == "none"
    assert note.tokens_used == 0


def test_budget_used_increases_after_api_call() -> None:
    analyst = _make_analyst()
    sig = _make_signal()
    before = analyst.budget_used_inr
    with patch("agent.ai.analyst.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _fake_tool_response()
        analyst.analyze(sig)
    assert analyst.budget_used_inr > before


def test_model_used_recorded_in_note() -> None:
    analyst = _make_analyst()
    sig = _make_signal(confidence=0.85)
    with patch("agent.ai.analyst.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _fake_tool_response()
        note = analyst.analyze(sig)
    assert note.model_used != ""
    assert note.model_used != "none"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/ai/test_analyst.py -v --no-cov 2>&1 | head -15
```

Expected: `ImportError: cannot import name 'AIAnalyst'`

- [ ] **Step 3: Write `agent/ai/analyst.py`**

```python
from __future__ import annotations

from decimal import Decimal

import anthropic
import structlog

from agent.ai.cache import NoteCache
from agent.ai.prompt import NOTE_SCHEMA, SYSTEM_PROMPT, build_prompt
from agent.decision.types import ResearchNote
from agent.strategies.types import Signal

logger = structlog.get_logger()

# Model thresholds
_HIGH_CONFIDENCE_THRESHOLD = 0.80

# Models (match config/settings.py defaults)
_MODEL_PRIMARY = "claude-sonnet-4-6"
_MODEL_CHEAP = "claude-haiku-4-5-20251001"

# Approximate cost in INR per 1 000 tokens (input + output blended).
# Haiku: ~$0.80/$4 per M → ₹0.04/1K. Sonnet: ~$3/$15 per M → ₹0.12/1K.
# Exchange rate ~83 INR/USD. These are upper estimates; adjust if pricing changes.
_COST_PER_1K_TOKENS: dict[str, Decimal] = {
    _MODEL_CHEAP: Decimal("0.04"),
    _MODEL_PRIMARY: Decimal("0.12"),
}

_FALLBACK_NOTE_TEXT = "[Budget cap reached — no AI analysis available]"


class AIAnalyst:
    """Claude integration for pre-trade research notes.

    Flow: budget check → cache lookup → Claude tool call → ResearchNote.
    When the monthly budget is exhausted the API call is skipped and a
    fallback ResearchNote (veto=False, model_used="none") is returned so
    the deterministic pipeline can proceed unblocked.
    """

    def __init__(
        self,
        api_key: str,
        *,
        max_budget_inr: Decimal = Decimal("1500.00"),
        model_primary: str = _MODEL_PRIMARY,
        model_cheap: str = _MODEL_CHEAP,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._max_budget = max_budget_inr
        self._model_primary = model_primary
        self._model_cheap = model_cheap
        self._cache = NoteCache()
        self._spend_inr: Decimal = Decimal("0")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        signal: Signal,
        *,
        portfolio_summary: str = "",
    ) -> ResearchNote:
        """Return a ResearchNote for signal.

        Uses cache when the same pattern (action, regime, R/R) was seen before.
        Skips API and returns a fallback note if the monthly budget is exhausted.
        """
        signal_id = f"{signal.symbol}:{signal.action}:{signal.timestamp.isoformat()}"

        # 1. Budget check
        if self._spend_inr >= self._max_budget:
            logger.warning(
                "ai_analyst.budget_exceeded",
                spend_inr=str(self._spend_inr),
                max_inr=str(self._max_budget),
                symbol=signal.symbol,
            )
            return self._fallback_note(signal_id)

        # 2. Cache lookup
        cached = self._cache.get(signal)
        if cached is not None:
            logger.debug("ai_analyst.cache_hit", signal_id=signal_id)
            return ResearchNote(
                signal_id=signal_id,
                bullish_case=cached.bullish_case,
                bearish_case=cached.bearish_case,
                dominant_risk=cached.dominant_risk,
                regime_fit_assessment=cached.regime_fit_assessment,
                confidence_qualitative=cached.confidence_qualitative,
                veto=cached.veto,
                veto_reason=cached.veto_reason,
                model_used=cached.model_used,
                tokens_used=cached.tokens_used,
                cached=True,
            )

        # 3. Model selection: high-confidence signals → cheap model
        model = (
            self._model_cheap
            if signal.confidence >= _HIGH_CONFIDENCE_THRESHOLD
            else self._model_primary
        )

        # 4. Claude API call
        prompt = build_prompt(signal, portfolio_summary=portfolio_summary)
        response = self._client.messages.create(
            model=model,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            tools=[NOTE_SCHEMA],
            tool_choice={"type": "tool", "name": "submit_research_note"},
            messages=[{"role": "user", "content": prompt}],
        )

        # 5. Parse tool use response
        tool_block = next(b for b in response.content if b.type == "tool_use")
        data = tool_block.input
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        # 6. Track spend
        cost_per_1k = _COST_PER_1K_TOKENS.get(model, Decimal("0.12"))
        self._spend_inr += Decimal(str(tokens_used)) / Decimal("1000") * cost_per_1k

        note = ResearchNote(
            signal_id=signal_id,
            bullish_case=data["bullish_case"],
            bearish_case=data["bearish_case"],
            dominant_risk=data["dominant_risk"],
            regime_fit_assessment=data["regime_fit_assessment"],
            confidence_qualitative=data["confidence_qualitative"],
            veto=data["veto"],
            veto_reason=data["veto_reason"],
            model_used=model,
            tokens_used=tokens_used,
            cached=False,
        )

        # 7. Store in cache (without signal_id — cache is keyed by pattern)
        self._cache.put(signal, note)

        logger.info(
            "ai_analyst.note_generated",
            signal_id=signal_id,
            model=model,
            veto=note.veto,
            tokens=tokens_used,
            spend_inr=str(self._spend_inr),
        )
        return note

    @property
    def budget_used_inr(self) -> Decimal:
        return self._spend_inr

    @property
    def budget_remaining_inr(self) -> Decimal:
        return max(Decimal("0"), self._max_budget - self._spend_inr)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fallback_note(self, signal_id: str) -> ResearchNote:
        return ResearchNote(
            signal_id=signal_id,
            bullish_case=_FALLBACK_NOTE_TEXT,
            bearish_case=_FALLBACK_NOTE_TEXT,
            dominant_risk=_FALLBACK_NOTE_TEXT,
            regime_fit_assessment=_FALLBACK_NOTE_TEXT,
            confidence_qualitative="LOW",
            veto=False,
            veto_reason=None,
            model_used="none",
            tokens_used=0,
            cached=False,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/ai/test_analyst.py -v --no-cov
```

Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add agent/ai/analyst.py tests/ai/test_analyst.py
git commit -m "feat(ai): add AIAnalyst with caching, budget guard, and Claude tool-use output"
```

---

## Task 4: Full Test Suite + Coverage Gate

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/tatsatshah/Desktop/yegedge && source .venv/bin/activate
python -m pytest tests/ -v --cov=agent --cov-report=term-missing 2>&1 | tail -30
```

Expected: **246+ tests pass** (224 from earlier phases + 22 new). Coverage ≥ 70% total; `agent/ai/` ≥ 85%.

- [ ] **Step 2: Run linters**

```bash
ruff check agent/ai/ tests/ai/ && black --check agent/ai/ tests/ai/ && echo CLEAN
```

Expected: no issues.

- [ ] **Step 3: End-to-end smoke test**

```bash
python - <<'EOF'
from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from agent.ai.analyst import AIAnalyst
from agent.ai.cache import NoteCache
from agent.ai.prompt import build_prompt
from agent.data.types import DataQuality
from agent.strategies.types import Action, Signal

IST = ZoneInfo("Asia/Kolkata")

sig = Signal(
    symbol="HDFCBANK", action=Action.ENTER_LONG, confidence=0.75,
    suggested_stop=Decimal("1680"), suggested_target=Decimal("1760"),
    invalidation_condition="Close below EMA21", expected_r=2.0,
    time_horizon_hours=4, regime_fit=0.8, data_quality=DataQuality.OK,
    strategy_name="trend_following_v1",
    explanation="EMA21 crossed above EMA50 (ADX=28.5)",
    timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
)

# Verify prompt
prompt = build_prompt(sig, portfolio_summary="No open positions")
assert "HDFCBANK" in prompt and "1680" in prompt

# Verify cache
cache = NoteCache()
assert cache.get(sig) is None  # cold cache

# Verify analyst with mocked client
tool_use = MagicMock()
tool_use.type = "tool_use"
tool_use.input = {
    "bullish_case": "Strong trend.", "bearish_case": "Overbought.",
    "dominant_risk": "FII selling.", "regime_fit_assessment": "Good fit.",
    "confidence_qualitative": "HIGH", "veto": False, "veto_reason": None,
}
mock_resp = MagicMock()
mock_resp.content = [tool_use]
mock_resp.usage.input_tokens = 120
mock_resp.usage.output_tokens = 80

analyst = AIAnalyst(api_key="test", max_budget_inr=Decimal("1500"))
with patch("agent.ai.analyst.anthropic.Anthropic") as MockClient:
    MockClient.return_value.messages.create.return_value = mock_resp
    note = analyst.analyze(sig)

assert note.veto is False
assert note.cached is False
assert analyst.budget_used_inr > 0

# Second call → cache hit
with patch("agent.ai.analyst.anthropic.Anthropic") as MockClient:
    note2 = analyst.analyze(sig)
    MockClient.return_value.messages.create.assert_not_called()
assert note2.cached is True

print("END-TO-END SPOT-CHECK PASSED")
EOF
```

Expected: prints `END-TO-END SPOT-CHECK PASSED`.

- [ ] **Step 4: Commit (plan doc + coverage confirmation)**

```bash
git add docs/superpowers/plans/2026-05-17-phase-6-ai-research-layer.md
git commit -m "test(ai): Phase 6 full suite passes coverage gate (≥70%), linters clean"
```

---

## Self-review checklist

- [ ] `signal_id` format `"{symbol}:{action}:{timestamp.isoformat()}"` matches `Decision.signal_id` convention from `agent/decision/types.py`
- [ ] Cache key excludes symbol (intentional — research pattern not symbol-specific)
- [ ] Budget check happens BEFORE cache lookup (prevents wasted API calls when over budget)
- [ ] `cached=False` on first API call, `cached=True` on cache hit
- [ ] Fallback note: `veto=False`, `model_used="none"`, `tokens_used=0`
- [ ] Model selection: `confidence >= 0.80` → Haiku; `< 0.80` → Sonnet
- [ ] `from __future__ import annotations` first line in every `.py`
- [ ] `logger = structlog.get_logger()` (not `log`) in all files that log
