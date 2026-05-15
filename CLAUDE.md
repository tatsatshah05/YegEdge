# CLAUDE.md — Project Context for Claude Code

This file is read on every Claude Code session. Treat its rules as binding.

## What this project is

**Project name: YegEdge.** A research-driven, risk-first, paper-trading-first automated trading agent for Indian equity markets (NSE), integrating Claude as a bounded pre-trade research analyst. Built for a single solo developer (Tatsat) running a single own-account algorithm under SEBI's 2025 retail framework.

The name combines `YEG` (IATA airport code for Edmonton, the owner's hometown reference) and `Edge` (the statistical edge every component of this system exists to find, validate, and protect).

The owner's full research and architecture document lives at `../Advanced AI Trading Agent — Research and Architecture.docx` and `../Advanced AI Trading Agent — Research and Architecture.md`. Read those before suggesting architectural changes.

## Binding rules — never violate these

1. **Live trading is disabled by default.** `LIVE_TRADING_ENABLED` in `config/settings.py` defaults to `False`. Never set it to `True` in code, in tests, or in documentation examples. The owner flips it manually after paper trading clears the readiness checklist.
2. **Paper trading is mandatory for 60 sessions before any live consideration.** Never add logic that allows skipping or shortening this.
3. **Risk rules in `config/risk_rules.yaml` are inviolable.** Do not add code paths that bypass, override, or weaken them. If a rule needs to change, change the YAML — never special-case in code.
4. **No F&O, no options, no futures, no commodities, no currency, no crypto in V1.** NSE equity intraday (MIS) only. If a strategy proposal involves any of these, push back and ask before building.
5. **No ML or RL strategies in V1.** Rules-based only. Clustering for regime detection is the one exception (interpretable).
6. **The LLM never initiates trades.** Claude's output is annotation, research notes, and veto-only. The deterministic strategy logic is the only source of trade decisions. If you find yourself writing code where an LLM response gates a trade entry directly, stop and surface it for review.
7. **Daily TOTP login is the owner's responsibility.** Do not write code that automates broker login bypassing TOTP. SEBI prohibits it; brokers enforce it.
8. **Static IP requirement.** The deployed system needs a whitelisted static IP. Account for this in deployment docs and config; do not assume dynamic IP is OK.
9. **Every decision (taken or rejected) is journaled.** Rejected trades are as informative as taken ones. Never silently drop a signal.
10. **Costs are always on in any backtest or simulation.** Never produce gross-return numbers. Indian cost model: STT 0.025% intraday sell, exchange charges ~0.00325%, SEBI charges 0.0001%, stamp duty 0.003% on buy, GST 18% on brokerage + exchange charges, brokerage per broker config. Realistic round-trip ~6–10 bps.

## Coding conventions

- **Language:** Python 3.11+. Type-hinted everywhere. `from __future__ import annotations` at the top of every module.
- **Style:** Black-formatted, isort-ordered imports, Ruff-linted. Pre-commit hook expected.
- **Structured data:** Dataclasses (`@dataclass(frozen=True, slots=True)` where possible) for `Signal`, `Decision`, `Order`, `Fill`, `JournalEntry`, etc. No untyped dicts crossing module boundaries.
- **Pure functions where possible.** Strategy logic, indicator math, and risk rules are deterministic pure functions of their inputs. Side effects (I/O, broker calls, time) live in adapters.
- **Time:** All timestamps timezone-aware, IST (`Asia/Kolkata`). Bar timestamps are bar-open by convention; document explicitly when otherwise.
- **No print() statements.** Use `structlog`. JSON-structured logs for the journal, human-readable for stdout.
- **Tests:** pytest. The risk manager has the strictest test bar — every rule in `risk_rules.yaml` has a test that *triggers* it and confirms rejection. No risk-manager code merges without its test.
- **Config:** YAML in `config/`. Loaded once via `config/settings.py` into typed Pydantic models. Never read YAML directly from business logic.
- **Secrets:** `.env` only, loaded via `python-dotenv`. Never commit secrets. `.env` is gitignored.

## Module boundaries

```
agent/data/        Broker adapter, Parquet cache, validators. Pure ingest + storage.
agent/features/    Indicators, regime detector, sentiment. Pure transforms.
agent/strategies/  Signal generators. Take features + regime + portfolio; emit Signals.
agent/risk/        Risk manager + kill switch. Gatekeeper. Returns approved order or rejection.
agent/decision/    Aggregates signals across strategies, deduplicates, integrates research notes.
agent/ai/          Claude API client. Schema-bounded outputs. Caching layer.
agent/execution/   Paper + live (live disabled). Idempotent orders, reconciliation.
agent/portfolio/   Position tracking, exposure aggregation, P&L.
agent/journal/     Append-only audit log.
agent/monitoring/  Heartbeats, alerts, kill-switch wiring.
dashboard/         Streamlit UI.
```

Cross-module imports go through interfaces, not implementations. The data layer doesn't know about strategies. The strategy layer doesn't know about brokers.

## Broker abstraction

Default broker: Upstox. Adapter pattern: `agent/data/broker_adapter.py` defines `BrokerAdapter` abstract base; concrete implementations in `upstox_adapter.py`, `dhan_adapter.py`, `kite_adapter.py`. The rest of the code only knows about `BrokerAdapter`.

## AI / Claude usage

- **Models:** Sonnet 4.6 (`claude-sonnet-4-6`) for morning brief + ambiguous pre-trade. Haiku 4.5 (`claude-haiku-4-5-20251001`) for obvious-signal pre-trade and journal entries. Opus 4.6 (`claude-opus-4-6`) only for weekly strategy reviews.
- **Always JSON-bounded.** Every Claude call has a Pydantic schema for input *and* output. Reject any free-form text in the decision path.
- **Caching.** Pre-trade notes cache by `(regime_label, signal_template, R/R_bucket, sector)`. Cache hit-rate target ≥ 60%.
- **Budget cap.** `config/settings.py` defines `MAX_MONTHLY_API_SPEND_INR`. The agent checks usage before each call; if cap is hit, the deterministic path proceeds without the LLM annotation and an alert fires.
- **Veto handling.** If Claude flags `veto: true` in a research note, the decision engine downgrades the trade to `wait_for_confirmation` for 1 bar — it does not auto-block forever, and it logs the rationale.

## SEBI compliance posture (April 2026 framework)

- White-box, self-account, sub-10-orders-per-second → falls under "regular API user", no separate SEBI/exchange registration.
- All algo orders must carry the exchange-issued Algo-ID — handled by the broker; ensure our adapters pass it through.
- Static-IP whitelisting required by every compliant broker — deployment doc must reflect this.
- Only trade for the owner's own account. Do not add features that route orders for any other person.

## Available slash commands (in `.claude/commands/`)

- `/morning-brief` — Trigger Claude to summarize overnight news, regime, earnings calendar for the day's universe.
- `/risk-audit` — Walk through `config/risk_rules.yaml` and verify every rule has a passing test and live-firing capability.
- `/paper-status` — Summarize current paper-trading metrics vs. backtest expectations.
- `/sebi-check` — Confirm code changes don't introduce a SEBI compliance regression (algo-tag, 10-OPS, no-bypass-login, static-IP).

## Available subagents (in `.claude/agents/`)

- `code-reviewer` — Reviews PRs for risk-rule integrity, type safety, and test coverage. Mandatory for changes to `agent/risk/` or `agent/execution/`.
- `backtest-validator` — Reads a backtest result and surfaces signs of overfit, lookahead, or unrealistic fills.
- `sebi-compliance-auditor` — Checks code changes against the SEBI 2026 framework rules.

## Build phase and current status

We are in **Phase 1 — Data Pipeline** of the 12-phase roadmap (see `docs/architecture.md`). Phase 0 (research and design) is complete; the research document and this CLAUDE.md represent that output. Phase 1 deliverables are:

- Upstox adapter (REST + websocket).
- NSE trading calendar with holidays.
- Point-in-time Nifty 100 universe loader.
- Parquet cache for 15-min and 1-hour bars (2 years history).
- Data validator with `data_quality` enum on every bar.
- `python -m agent.cli refresh` and `python -m agent.cli verify` commands.

After Phase 1: features → strategies → backtest → risk → AI → paper. Then 60 sessions of paper trading before any live consideration.

## What to do before writing code in any session

1. Read this file (you are).
2. Skim `docs/build_handoff.md` for the decision history.
3. Check `git status` and `git log -10`.
4. Use plan mode for non-trivial changes — confirm approach before editing.
5. If touching `agent/risk/` or `agent/execution/`, spawn the `code-reviewer` subagent before finalizing.

## What to never do

- Never commit a change that allows live trading by default.
- Never add a code path that bypasses the risk manager.
- Never silently catch exceptions in the strategy or risk modules.
- Never compute backtest performance without costs.
- Never use `pandas.read_csv` without explicit `dtype` — silent type coercion has lost retail traders real money.
- Never use floating-point for monetary calculations involving rupees; use `Decimal` or fixed-point cents.
- Never call `time.sleep()` in production code without an explicit comment justifying it.
- Never log API keys, TOTP secrets, or broker tokens. Scrub logs aggressively.

---

*Tatsat is an entrepreneur, not a full-time quant. Explain trade-offs when relevant. Push back on bad ideas with reasoning, not deference.*
