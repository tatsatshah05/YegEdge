---
description: Generate a structured pre-market morning brief
---

Generate today's pre-market morning brief for the trading agent.

Steps:

1. Run `python -m agent.cli regime` to fetch the current market regime label and confidence.
2. Run `python -m agent.cli news --hours 16` to pull overnight news headlines for the universe.
3. Run `python -m agent.cli calendar` to list earnings, RBI events, expiries, and macro releases.
4. Compose the brief in this exact structure:

```
# Morning Brief — {YYYY-MM-DD}

## Market regime
- Current: {regime_label} (confidence: {confidence_score})
- Change since yesterday: {regime_change_or_stable}
- Implications: {one-sentence-on-which-strategies-are-allowed}

## Overnight context
- Global markets: {brief}
- USD/INR: {value, change}
- Brent crude: {value, change}
- India VIX: {value, change}

## Today's calendar
- Earnings (held names): {list}
- Earnings (universe names): {list}
- Macro/event: {list}
- Expiry-related: {list}

## Sector tilt
- Strong sectors: {list}
- Weak sectors: {list}

## Risk posture for the day
- Allowed strategies: {list based on regime}
- Restricted sectors: {list based on calendar}
- Size adjustment: {if regime is high_vol or after recent drawdown}

## Top candidate setups (from rule-based pre-scan)
{list, ≤ 5 names, each with one-line thesis}
```

5. Save the brief to `data/briefs/{YYYY-MM-DD}.md` and print to stdout.
6. If Telegram is configured, send the brief summary to the configured chat.

Constraints:
- This is a *research* brief, not a recommendation. Do not phrase as "buy X" or "sell Y."
- Speak in probabilities and conditional logic ("if regime holds, trend-following sleeve is allowed").
- Never claim certainty.
- Token budget: ≤ 3K input + 1K output per brief.
