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
    lines.append("\nAnalyse this trade signal and submit your research note using the tool.")
    return "\n".join(lines)


SYSTEM_PROMPT = _SYSTEM_PROMPT
NOTE_SCHEMA = _NOTE_SCHEMA
