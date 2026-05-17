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
