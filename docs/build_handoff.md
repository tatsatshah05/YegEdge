# YegEdge — Build Handoff & Decision History

This document summarizes the decisions made during the research and planning phase (in Cowork mode) before code was written. Claude Code should read this to understand *why* the architecture and constraints are what they are — not just *what* they are.

## Project owner & context

- **Owner:** Tatsat Shah. Indian entrepreneur. Software-first. This is one of several projects, not a full-time pursuit.
- **Initial capital:** ~₹83,000 (≈ $1,000) for the first live deployment, after paper trading.
- **Market focus:** Indian equities (NSE), intraday MIS, swing-friendly 1-hour timeframe.
- **Build orientation:** Risk-first, paper-first, research-driven. Profitability is sought but never assumed.

## Key decisions and their reasoning

### 1. Why NSE equities and not crypto, F&O, or US markets

Crypto in India is effectively un-tradeable actively because of 30% capital-gains tax + 1% TDS per transaction, which mathematically erodes any retail edge. F&O was tightened by SEBI in October 2024 with lot-size increases and weekly-expiry restrictions; SEBI's own data shows 89–93% of retail F&O traders lose money. US equities have no PDT-equivalent issue in India, but require currency hedging and FEMA reporting. NSE equity intraday is the cleanest path for own-account algo at this capital scale.

### 2. Why daily/1-hour bars and not 1-minute scalping

Intraday HFT competition is severe. At 1-minute resolution, the trader competes with co-located firms with microsecond latency — a structural losing battle for solo retail. At 1-hour bars, the competition is other discretionary retail traders, which is an easier field. Costs also compound per trade: at ₹83k, even a few extra round-trips per day in scalp-mode wipe out monthly profit potential.

### 3. Why trend-following (not mean reversion or breakout) for V1

Trend-following has the most documented out-of-sample evidence across decades and markets (CTA literature, AQR research). Mean reversion has compressed edge due to HFT activity. Breakout strategies have high false-breakout rates in choppy regimes. V1.5 adds mean reversion as a small sleeve gated by regime; V2 considers breakout. V1 is one strategy done right.

### 4. Why Upstox API as default broker

Free API (₹10/order pricing until March 2026, then standard ₹20 or 0.03%). Good documentation. Reasonable Python SDK. The adapter pattern means swapping to Zerodha Kite (₹2,000/month, best community) or Dhan (algo-focused, free) is a small change. Best ROI for the development phase.

### 5. Why static IP is required (and not a soft preference)

SEBI's February 2025 algo-trading framework, mandatory from April 1, 2026, requires brokers to whitelist client-specific API keys to static IPs. Open APIs are disallowed. This is not a recommendation; it is a regulatory constraint enforced by compliant brokers. A VPS with static IP (Hetzner Frankfurt or DigitalOcean Bangalore, ~₹600/month) is therefore part of the production plan.

### 6. Why daily TOTP login is non-negotiable

SEBI requires manual login at least once per day for retail API users. Brokers enforce this via TOTP. Programmatic bypass exists (GitHub repos for TOTP automation), but violates broker T&C, can trigger account suspension, and creates a regulatory tail risk that is unacceptable for production. The system therefore assumes one ~30-second daily action by the owner before market open.

### 7. Why live trading is disabled by default

Most retail algo accounts blow up because the operator skipped paper trading or shortened it. The system architecture protects against the owner's own future impatience: `LIVE_TRADING_ENABLED=false` is the default in `.env.example` and in `config/settings.py`, and flipping it requires (a) 60+ paper sessions logged, (b) passing the 15-point checklist in `docs/live_readiness.md`, (c) a written go/no-go decision, (d) manual config edit (no CLI flag override).

### 8. Why the LLM does not initiate trades

A 2024–2025 academic paper (StockBench) rigorously benchmarked LLM trading agents and found most fail to outperform a simple buy-and-hold baseline. LLMs are not alpha generators. The LLM's role is therefore bounded: pre-trade research notes (annotation), post-trade journal entries (narrative), periodic reviews (oversight), and veto rights (it can flag a trade but cannot initiate one). The deterministic strategy code is the only source of trade decisions.

### 9. Why Sonnet/Haiku/Opus split for LLM calls

Cost discipline. Sonnet 4.6 is good enough for the morning brief and ambiguous pre-trade analysis. Haiku 4.5 is good enough for obvious-signal pre-trade and post-trade journal entries. Opus 4.6 is reserved for weekly deep reviews. At our trade frequency (~15–20 trades/month) with caching, total monthly API spend is estimated at ₹300–₹600. A hard monthly cap (`MAX_MONTHLY_API_SPEND_INR`) is enforced in code — if hit, deterministic logic proceeds without LLM annotation and an alert fires.

### 10. Why 60 paper sessions before live

Three months in one regime tells you almost nothing about regime adaptability. 60 sessions ≈ 3 calendar months and typically includes some regime variation. The bar is not "did paper make money" — it is "did paper behave the way backtest expected, and did all the system invariants hold." Failing the 60-session bar means the strategy doesn't have edge or the system has bugs; either way, more time is needed.

### 11. Why fixed-fractional 0.5% risk per trade in V1

Survival > return. A 0.5% per-trade risk caps max single-trade loss at ₹400 on ₹83K. With a 50% win rate and 1.5:1 R/R, expectancy per trade is ~₹100. Over 20 trades/month, that's ~₹2,000 gross / ~₹1,200 net of taxes — modest but real. Larger sizing (Kelly, fractional Kelly) is deferred until 30+ profitable paper sessions confirm the edge estimate. Most retail accounts that blow up did so by sizing aggressively too early.

### 12. Why a custom build rather than Streak / Tradetron / AlgoTest

No-code platforms can deliver a working automated system in days, at ₹300–₹500/month. The reasons the owner is doing a custom build anyway are: full control over strategy logic, no recurring platform fee that's a meaningful percentage of capital, learning value that compounds for an entrepreneur, and Claude integration as a pre-trade analyst (not currently supported by no-code platforms). Trade-off: custom takes 12–16 weeks vs. days. The owner has accepted this trade.

## Build phases (current location: Phase 1)

- **Phase 0 — Research & design.** ✅ Complete. Output: research document + this handoff.
- **Phase 1 — Data pipeline.** ◀ Current. Output: Upstox adapter, Parquet cache, validators, `refresh`/`verify` CLI.
- **Phase 2 — Feature engineering.** Indicators, regime detector, sentiment.
- **Phase 3 — Strategy (trend-following).** Signal generator producing structured records.
- **Phase 4 — Backtester.** Walk-forward, Monte Carlo, full Indian cost model. *Critical gate.*
- **Phase 5 — Risk manager.** Hard-rule gatekeeper. Every rule has a deliberate-trigger test.
- **Phase 6 — Decision engine.** Aggregator over signals + research notes.
- **Phase 7 — AI research layer.** Claude integration. Schema-bounded.
- **Phase 8 — Paper trading.** 60-session minimum. Telegram alerts.
- **Phase 9 — Dashboard.** Streamlit; market sentiment, regime, positions, journal, performance.
- **Phase 10 — Monitoring & alerts.** Cross-cutting. Kill-switch wired.
- **Phase 11 — Advanced strategies (V1.5+).** Mean reversion sleeve. New strategies clear OOS bar before joining.
- **Phase 12 — Live readiness gate.** 15-point checklist. Written go/no-go. Half-size ramp.

## Reference materials owned by the project

- `../Advanced AI Trading Agent — Research and Architecture.docx` — Full 54-page research and architecture document. Section numbers cited throughout the codebase comments refer to this.
- `../Advanced AI Trading Agent — Research and Architecture.md` — Same content as markdown.
- `docs/sebi_compliance.md` — Summary of SEBI April 2026 framework.
- `docs/live_readiness.md` — The 15-point checklist.
- `docs/architecture.md` — Condensed system architecture.

## Things Claude Code should never propose

- Removing the live-trading flag default.
- Auto-skipping paper sessions.
- ML-based primary strategies in V1.
- Auto-bypass of TOTP login.
- F&O, crypto, or currency derivatives.
- LLM-initiated trade decisions.
- Backtests without costs.
- Strategies that beat one symbol but not the universe.
- Code that bypasses the risk manager.

## Things Claude Code should always do

- Read `CLAUDE.md` first.
- Use plan mode for non-trivial changes.
- Spawn `code-reviewer` subagent before changes to `agent/risk/` or `agent/execution/`.
- Add a test for any risk rule it changes — the test must deliberately trigger the rule.
- Apply costs in every backtest.
- Log every decision (taken and rejected).
- Push back with reasoning when an owner request would violate the binding rules in CLAUDE.md.
