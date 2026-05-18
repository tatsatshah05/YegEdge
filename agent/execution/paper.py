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
