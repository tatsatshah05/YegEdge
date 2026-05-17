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
