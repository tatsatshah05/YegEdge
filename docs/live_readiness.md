# Live-Trading Readiness Checklist

**Live trading is disabled by default. Do not flip the `LIVE_TRADING_ENABLED` flag in `.env` to `true` until every item below passes.**

This checklist exists to protect the operator from the operator's own future impatience. The system architecture is designed so that flipping live without passing this checklist is *deliberate*, not accidental.

## Hard prerequisites (system invariants)

These cannot be waived.

1. ☐ **Paper trading sessions completed: ≥ 60.**
   - Counter in `.env` (`PAPER_SESSIONS_COMPLETED`) is incremented automatically by the agent on each paper session.
   - System refuses to enable live mode if counter < 60, regardless of flag.

2. ☐ **All risk-manager tests pass.**
   - `pytest tests/test_risk.py -m risk -v` returns green.
   - Every rule in `config/risk_rules.yaml` has a deliberate-trigger test that fires correctly.

3. ☐ **No unresolved data-quality bugs in past 30 paper sessions.**
   - Check `logs/data_quality_*.json` for any sessions with > 5% partial/suspect/missing bars.
   - If any exist, root-cause and fix before proceeding.

4. ☐ **No unresolved execution bugs in past 30 paper sessions.**
   - Check journal for order errors, reconciliation mismatches, duplicate submissions, stale state.

5. ☐ **Reconciliation runs clean.**
   - Local state vs broker state matches at end of every recent paper session.

6. ☐ **Kill switch has been tested.**
   - Manually trigger via `python -m agent.cli kill-switch-test`.
   - Confirm: open orders cancelled, system stands down, alert fires.

## Strategy validation (the math must support live)

7. ☐ **Backtest OOS performance is positive after costs.**
   - Profit factor ≥ 1.3.
   - Sharpe ≥ 0.8 on out-of-sample data.
   - Calmar ≥ 0.4.
   - ≥ 100 OOS trades.
   - Backtest report saved to `docs/reports/backtest_v1_final.html`.

8. ☐ **Walk-forward results are stable.**
   - No single OOS window provides > 50% of total OOS return.
   - Sharpe variance across walk-forward windows ≤ 0.5.

9. ☐ **Parameter sensitivity is gentle.**
   - ±20% parameter perturbation does not destroy performance.
   - Sensitivity heatmap shows a plateau, not a peak.
   - Plot saved to `docs/reports/param_sensitivity.png`.

10. ☐ **Strategy performs in ≥ 2 distinct market regimes.**
    - Backtest decomposed by regime; positive (or controlled-negative) in at least two of {bull_trend, bear_trend, sideways, mean_reverting}.

## Paper-trading evidence (it actually worked live-ish)

11. ☐ **Realized paper Sharpe ≥ 60% of backtested Sharpe.**
    - If realized is < 60% of backtested, the cost model is wrong or the strategy is degrading. Investigate before live.

12. ☐ **Realized paper max drawdown ≤ 95th-percentile Monte Carlo drawdown from backtest.**
    - If realized exceeds the MC band, the backtest's risk profile is wrong.

13. ☐ **Risk engine has actually fired (rejected trades or capped sizes) during paper.**
    - Zero rejections in 60 sessions means risk rules are misaligned with signal generation. Investigate.

14. ☐ **Realized slippage tracks modeled slippage.**
    - Per-trade comparison; deviation < 50% on average.

## Operational readiness (the operator can run this)

15. ☐ **Operator has read ≥ 20 journal entries end-to-end.**
    - Manual review confirms the agent's reasoning is sound.
    - Operator can articulate, in plain English, *why* each accepted trade was taken and why each rejected trade was rejected.

16. ☐ **Operator can articulate the current market regime and why.**
    - Without looking at the dashboard.

17. ☐ **VPS is set up, static IP whitelisted, system has run for ≥ 7 sessions on VPS in paper mode.**
    - Local development is not the same as VPS deployment. Confirm parity.

18. ☐ **Telegram alerts working.**
    - Deliberate fault injection (e.g., kill the data feed) triggers expected alert chain within 60 seconds.

19. ☐ **Monitoring dashboard is the morning check.**
    - Operator opens dashboard every morning before market open and can scan for issues in < 30 seconds.

20. ☐ **Tax planning conversation with CA completed.**
    - Operator understands intraday equity is speculative income (slab rate, ITR-3, audit threshold).
    - Deductible expenses (broker fees, VPS, API, data) tracked from day one.

## Final commitment

21. ☐ **Written go/no-go decision committed to repo.**
    - File: `docs/reports/go_no_go_YYYYMMDD.md`.
    - Contains: date, every checkbox above, brief justification for each pass.
    - Committed and signed (git commit by operator).

22. ☐ **Live capital is amount you can afford to lose entirely.**
    - Self-assessment. Honest.

23. ☐ **Live trading begins at 25–50% of intended size for first 30 sessions.**
    - Configured in `.env` via `PAPER_STARTING_CAPITAL` halved for live mode initially.

24. ☐ **Scale-up plan written.**
    - After 30 live sessions at half-size: review against paper expectations. If matches, scale to ¾. After another 30: review again, scale to full.

## How the system enforces the bar

- The `enable_live_trading()` function in `agent/cli.py` runs a `validate_readiness()` check that programmatically verifies items 1–6 (the ones that can be machine-checked). It refuses to enable live mode otherwise.
- Items 7–24 are honor-system items that the operator must self-certify. The go/no-go report (item 21) records this.

## How to fail this checklist gracefully

If you don't pass an item, the right response is *not* "loosen the requirement." It's "the strategy needs more work." The most common reasons V1 strategies don't pass on first attempt:

- Backtest OOS performance disappoints → cost model was too optimistic, or strategy was overfit. Re-validate.
- Paper Sharpe lags backtest Sharpe → slippage model is wrong, or regime drift. Investigate.
- Risk engine never fires → signal generation is producing weak signals that get rejected upstream, *or* risk thresholds are too loose. Recalibrate.

A failed checklist is good information. It means the system caught a real problem before real money was at risk. That's the value the architecture provides.
