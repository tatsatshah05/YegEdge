# Architecture — Condensed Reference

Full architecture lives in `../Advanced AI Trading Agent — Research and Architecture.docx` (Sections 2 and 17). This file is the engineering-level summary.

## System data flow

```
[ Market Data ]                  Upstox websocket + REST (Dhan / Kite adapters available)
      |
      v
[ Validator ]                    quality_flag ∈ {ok, partial, suspect, missing}
      |
      v
[ Feature Layer ]                EMA, ATR, ADX, RSI, realized vol, India VIX, relative strength
      |
      v
[ Regime Detector ] ─────────▶  regime ∈ {bull_trend, bear_trend, sideways, high_vol, mean_reverting, crisis, avoid}
      |
      v
[ Strategies ]                   trend_following (V1), mean_reversion (V1.5)
      |
      v  Signals (typed)
      |
      v
[ AI Research Layer ] ───────▶  pre-trade research note (Sonnet/Haiku), optional veto
      |
      v
[ Decision Engine ]              aggregate, dedup, apply portfolio context
      |
      v
[ Risk Manager ]                 hard rules from config/risk_rules.yaml; rejects or sizes
      |
      v
[ Execution ]                    paper (default) | live (disabled by default)
      |
      v
[ Portfolio + Journal ]          append-only audit log; position state; P&L
      |
      v
[ Monitoring & Alerts ]          heartbeats, kill switch, Telegram
      |
      v
[ Dashboard ]                    Streamlit; the morning check
```

## Module contracts

### `agent/data/`

Inputs: broker config, universe config, timeframes.
Outputs: validated bars in Parquet cache; quality flags per bar.
Side effects: HTTP/websocket to broker; disk writes.
Owns: `BrokerAdapter` interface + concrete implementations; `Cache`; `Validator`.

```python
class BrokerAdapter(ABC):
    @abstractmethod
    def fetch_historical(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pl.DataFrame: ...
    @abstractmethod
    async def stream_live(self, symbols: list[str], callback: Callable) -> None: ...
    @abstractmethod
    def place_order(self, order: Order) -> OrderAck: ...
    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...
    @abstractmethod
    def get_positions(self) -> list[Position]: ...
    @abstractmethod
    def reconcile(self, expected: list[Position]) -> list[Discrepancy]: ...
```

### `agent/features/`

Inputs: validated bars, benchmark series.
Outputs: feature DataFrame per (symbol, timeframe); regime label + confidence.
Side effects: none (pure transforms).
Owns: indicator library; `RegimeDetector`.

Key invariants:
- Every feature has `time_of_validity`: bar t's feature is available for decisions at bar t+1's open.
- Indicators are deterministic and unit-tested against TradingView reference values.

### `agent/strategies/`

Inputs: feature DataFrame, regime, portfolio state, strategy config.
Outputs: `list[Signal]`.
Side effects: none (pure functions).

```python
@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    action: Action                     # ENTER_LONG | EXIT | HOLD | etc.
    confidence: float                  # [0, 1]
    suggested_stop: Decimal
    suggested_target: Decimal
    invalidation_condition: str
    expected_r: float
    time_horizon_hours: int
    regime_fit: float                  # [0, 1]
    data_quality: DataQuality
    strategy_name: str
    explanation: str                   # short, structured
    timestamp: datetime
```

### `agent/ai/`

Inputs: `Signal`, feature snapshot, regime, portfolio context.
Outputs: `ResearchNote` (schema-bounded).
Side effects: Anthropic API calls; cache reads/writes.

```python
@dataclass(frozen=True, slots=True)
class ResearchNote:
    signal_id: str
    bullish_case: str                  # ≤ 80 words
    bearish_case: str                  # ≤ 80 words
    dominant_risk: str
    regime_fit_assessment: str
    confidence_qualitative: Confidence # LOW | MEDIUM | HIGH
    veto: bool
    veto_reason: str | None
    model_used: str
    tokens_used: int
    cached: bool
```

### `agent/decision/`

Inputs: signals, research notes, portfolio state, recent strategy performance.
Outputs: `list[Decision]`.

Aggregation rules:
- Multiple signals on the same symbol from different strategies → single decision with combined weighting.
- LLM veto on any signal → downgrade to `WAIT_FOR_CONFIRMATION` for 1 bar.
- Decision is logged whether taken or rejected.

### `agent/risk/`

Inputs: `Decision`, portfolio state, account state, recent P&L.
Outputs: `ApprovedOrder` *or* `Rejection` (with reason).
Side effects: none on success path; kill-switch trigger on critical breach.

Every rule in `config/risk_rules.yaml` has a corresponding pytest in `tests/test_risk.py` that deliberately triggers it. Code review for changes to this module is mandatory (use `code-reviewer` subagent).

### `agent/execution/`

Two implementations: `PaperExecution` (default) and `LiveExecution` (gated by `LIVE_TRADING_ENABLED` flag).

Order lifecycle:
1. Idempotency key generated client-side.
2. Pre-submit re-check: buying power, symbol halted, market open.
3. Submit with backoff retry on transient errors (max 3).
4. Reconcile periodically: local state vs broker state. Discrepancy → alert.
5. Order timeout → cancel and re-evaluate (not auto-retry).

### `agent/portfolio/`

Position state, exposure aggregation, correlation matrix, beta, drawdown tracking.

### `agent/journal/`

Append-only audit log. Every state transition. Stored in SQLite (`data/journal.db`) and JSON-structured logs (`logs/`).

### `agent/monitoring/`

Heartbeats per module. Alert tiers: INFO / WARN / CRITICAL. CRITICAL triggers kill switch + Telegram alert.

### `dashboard/`

Streamlit. Pages:
- Overview: NAV, drawdown, current regime, positions, today's P&L.
- Market sentiment: regime confidence, sector heatmap, India VIX, FII/DII flows.
- Signals: today's signals (taken + rejected), per-signal research note.
- Journal: searchable trade history.
- Performance: equity curve, rolling Sharpe, per-regime breakdown.
- Risk: current utilization of each risk rule.
- System: heartbeats, alerts, API spend.

## Critical invariants enforced in code

1. **`LIVE_TRADING_ENABLED = False`** by default; only flippable by manual config edit.
2. **`PAPER_SESSIONS_COMPLETED < 60`** prevents live mode regardless of config flag.
3. **Costs always on** in any backtest or simulation.
4. **Idempotent orders** — same client-order-id on every retry.
5. **Reconciliation every cycle** — local state vs broker state.
6. **Kill switch is manual-reset only.**

## Currently incomplete

Everything in `agent/` directories — the scaffold exists but no implementation yet. Phase 1 starts with `agent/data/`.
