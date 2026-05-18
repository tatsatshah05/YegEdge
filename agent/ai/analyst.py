from __future__ import annotations

import dataclasses
from zoneinfo import ZoneInfo

import anthropic
import structlog

from agent.ai.cache import NoteCache
from agent.ai.prompt import NOTE_SCHEMA, SYSTEM_PROMPT, build_prompt
from agent.decision.types import ResearchNote
from agent.strategies.types import Signal
from config.settings import AppSettings

log = structlog.get_logger()
IST = ZoneInfo("Asia/Kolkata")

# Use cheap model for high-confidence signals (≥0.80); primary model otherwise.
_CHEAP_CONFIDENCE_THRESHOLD = 0.80


class AIAnalyst:
    """Pre-trade research analyst backed by Claude.

    Responsibilities:
    - Build structured pre-trade notes via Claude tool use
    - Cache notes by (action, regime_bucket, rr_bucket) via NoteCache
    - Select model based on signal confidence
    - Enforce monthly API spend cap (INR)
    - Never veto a trade permanently — veto=True downgrades to WAIT_FOR_CONFIRMATION for 1 bar
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        cache: NoteCache | None = None,
        *,
        _client: anthropic.Anthropic | None = None,  # injection point for tests
    ) -> None:
        self._settings = settings or AppSettings()
        self._cache = cache or NoteCache()
        self._client = _client or anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        self._spend_inr: float = 0.0  # running tally for this session

    def analyse(
        self,
        signal: Signal,
        *,
        portfolio_summary: str = "",
    ) -> ResearchNote:
        """Return a ResearchNote for the signal. Uses cache when available.

        Budget check runs before cache lookup to avoid unnecessary computation
        when the monthly cap has been hit.
        """
        signal_id = f"{signal.symbol}:{signal.action}:{signal.timestamp.isoformat()}"

        # 1. Budget check — if over cap, return a degraded note without calling Claude
        if self._over_budget():
            log.warning(
                "ai_analyst.budget_exceeded",
                signal_id=signal_id,
                spend_inr=self._spend_inr,
                cap_inr=float(self._settings.max_monthly_api_spend_inr),
            )
            return self._budget_exceeded_note(signal_id)

        # 2. Cache lookup
        cached = self._cache.get(signal)
        if cached is not None:
            log.debug("ai_analyst.cache_hit", signal_id=signal_id)
            return dataclasses.replace(cached, signal_id=signal_id)

        # 3. Model selection
        model = (
            self._settings.claude_model_cheap
            if signal.confidence >= _CHEAP_CONFIDENCE_THRESHOLD
            else self._settings.claude_model_primary
        )

        # 4. Claude API call (tool use)
        prompt = build_prompt(signal, portfolio_summary)
        note = self._call_claude(signal_id=signal_id, prompt=prompt, model=model)

        # 5. Store in cache (keyed by pattern, not signal_id)
        self._cache.put(signal, note)

        return note

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _over_budget(self) -> bool:
        return self._spend_inr >= float(self._settings.max_monthly_api_spend_inr)

    def _call_claude(self, *, signal_id: str, prompt: str, model: str) -> ResearchNote:
        response = self._client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[NOTE_SCHEMA],
            tool_choice={"type": "tool", "name": "submit_research_note"},
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract tool use block
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"),
            None,
        )
        if tool_block is None:
            raise RuntimeError(f"Claude did not call submit_research_note for signal {signal_id}")

        args: dict[str, object] = tool_block.input  # type: ignore[attr-defined]
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        # Update spend tally (approximate: tokens * rough INR rate)
        # Actual cost tracking is a monitoring concern; this is a conservative floor.
        self._spend_inr += self._estimate_cost_inr(model, tokens_used)

        log.info(
            "ai_analyst.claude_response",
            signal_id=signal_id,
            model=model,
            tokens=tokens_used,
            veto=args.get("veto"),
            spend_inr=round(self._spend_inr, 4),
        )

        return ResearchNote(
            signal_id=signal_id,
            bullish_case=str(args["bullish_case"]),
            bearish_case=str(args["bearish_case"]),
            dominant_risk=str(args["dominant_risk"]),
            regime_fit_assessment=str(args["regime_fit_assessment"]),
            confidence_qualitative=str(args["confidence_qualitative"]),
            veto=bool(args["veto"]),
            veto_reason=args.get("veto_reason") or None,  # type: ignore[arg-type]
            model_used=model,
            tokens_used=tokens_used,
            cached=False,
        )

    def _budget_exceeded_note(self, signal_id: str) -> ResearchNote:
        return ResearchNote(
            signal_id=signal_id,
            bullish_case="Budget cap reached — no analysis available.",
            bearish_case="Budget cap reached — no analysis available.",
            dominant_risk="Monthly API spend cap exceeded.",
            regime_fit_assessment="Not assessed.",
            confidence_qualitative="LOW",
            veto=False,
            veto_reason=None,
            model_used="none",
            tokens_used=0,
            cached=False,
        )

    def _estimate_cost_inr(self, model: str, tokens: int) -> float:
        """Very rough token→INR estimate. Actual billing happens at Anthropic."""
        # Haiku: ~$0.25/1M input tokens. Sonnet: ~$3/1M input tokens.
        # 1 USD ≈ 83 INR. These are floor estimates; adjust as pricing changes.
        usd_per_1m = 0.25 if "haiku" in model else 3.0
        return (tokens / 1_000_000) * usd_per_1m * 83.0
