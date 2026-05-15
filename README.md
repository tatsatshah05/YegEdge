# YegEdge

**An AI-assisted, risk-first, paper-trading-first automated trading system for Indian equity markets.** Built for SEBI's 2025 retail algo framework. Designed for a single solo developer running a single own-account algorithm.

The name: `YEG` (the IATA code for Edmonton, Alberta — operator's hometown reference) + `Edge` (the statistical edge every component of this system is built to find, validate, and protect).

**Status:** Phase 1 (Data Pipeline). Scaffold ready; implementation in progress.

## Quick links

- [SETUP.md](./SETUP.md) — Step-by-step setup
- [CLAUDE.md](./CLAUDE.md) — Project context for Claude Code (binding rules)
- [docs/architecture.md](./docs/architecture.md) — System architecture
- [docs/build_handoff.md](./docs/build_handoff.md) — Decision history from research phase
- [docs/sebi_compliance.md](./docs/sebi_compliance.md) — SEBI April 2026 framework summary
- [docs/live_readiness.md](./docs/live_readiness.md) — 15-point checklist before live trading

## Important truths

This system does not guarantee profit. Trading involves substantial risk of loss. Live trading is disabled by default and stays disabled until the readiness checklist is passed. Real-money trading requires explicit manual approval. Past performance does not predict future results.

## What it does

A morning AI brief, intraday signal generation across configurable Nifty 100 universe, hard risk rules enforced in code, paper-trading simulator, Claude-generated pre-trade research notes, append-only audit journal, Streamlit dashboard for monitoring, and a kill switch that cannot be auto-reset.

## What it does NOT do

No F&O, no options, no futures, no commodity, no currency, no crypto. No ML/RL strategies in V1. No autonomous LLM trading. No daily-login bypass. No live trading by default.

## Tech stack

Python 3.11+ · pandas + Polars · NSE trading calendar · Upstox API (Dhan and Kite supported) · Anthropic SDK (Sonnet/Haiku/Opus) · Streamlit dashboard · SQLite + Parquet storage · pytest · structlog · Docker for deployment.

## License

Personal research software. Not investment advice. Not a financial product. Provided as-is with no warranty.
