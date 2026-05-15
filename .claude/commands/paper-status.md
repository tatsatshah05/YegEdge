---
description: Summarize current paper-trading status vs backtest expectations
---

Summarize the paper-trading session metrics and compare against backtest expectations.

Steps:

1. Query the trade journal (`data/journal.db`) for all trades in paper mode.
2. Compute realized metrics on the paper journal:
   - Total trades, win rate, average win, average loss
   - Profit factor, expectancy, total P&L
   - Realized Sharpe (annualized; require ≥ 20 trades)
   - Realized max drawdown
   - Realized average slippage per trade
3. Load the most recent backtest result from `docs/reports/backtest_v1_*.json` (latest by filename).
4. Compare realized vs backtest:
   - Sharpe ratio (realized / backtest)
   - Drawdown (realized / backtest 95th-percentile MC)
   - Slippage (realized / modeled)
   - Trade frequency (realized / expected)
5. Report in this format:

```
# Paper Trading Status — {YYYY-MM-DD}

## Session count
- Sessions completed: {N} / 60 required for live consideration
- Days remaining (if 5 trading days/week): {N}

## Realized metrics
| Metric             | Realized   | Backtest   | Ratio  | Status         |
|--------------------|------------|------------|--------|----------------|
| Sharpe (annualized)| {x.xx}     | {x.xx}     | {0.xx} | {OK|WATCH|RED} |
| Profit factor      | {x.xx}     | {x.xx}     | {0.xx} | ...            |
| Max drawdown       | {x.xx%}    | {x.xx%}    | ...    | ...            |
| Slippage / trade   | {x.x bps}  | {x.x bps}  | ...    | ...            |
| Trades / month     | {n}        | {n}        | ...    | ...            |

## Risk-engine activity
- Trades approved: {n}
- Trades rejected (with reason breakdown): {n}
- Kill switch triggers: {n}
- Daily-loss cap hits: {n}

## Status verdicts
- Realized Sharpe ≥ 60% of backtest: {YES|NO}
- Realized drawdown ≤ MC 95th percentile: {YES|NO}
- Risk engine has fired ≥ 1 time: {YES|NO}
- No unresolved data-quality issues (past 30 sessions): {YES|NO}
- No unresolved execution errors (past 30 sessions): {YES|NO}

## Recommendation
{One paragraph: are we on track for live consideration? What needs attention?}
```

6. If any "Status" is RED, surface as CRITICAL — do not proceed to live regardless of session count.

7. Token-cap the realized journal text passed to LLM at 50 most recent trades (don't bloat context).
