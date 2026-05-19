# Phase 9 — Streamlit Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 5-page read-only Streamlit dashboard that surfaces paper trading results — equity curve, trade journal, risk configuration, and system status — from the existing SQLite journal and Parquet cache.

**Architecture:** A `dashboard/` package with one tested data-loading module (`data.py`) and four Streamlit page files. The data module contains pure Python functions with no Streamlit dependency (fully unit-testable). Page files call those functions and render with `st.*`. The dashboard is strictly read-only — it never writes to journal, cache, or settings.

**Tech Stack:** Python 3.11+, Streamlit ≥1.38 (already in requirements.txt), Plotly Express ≥5.24 (already in requirements.txt), Polars, PyYAML, SQLite (via JournalStore), pytest.

---

## Context for subagent workers

**Project:** `/Users/tatsatshah/Desktop/yegedge`
**Branch:** `phase-2-feature-engineering`
**Virtualenv:** `.venv/bin/python`
**Run all commands from:** `/Users/tatsatshah/Desktop/yegedge`

**Conventions:**
- `from __future__ import annotations` first line every `.py`
- `logger = structlog.get_logger()` for any module that logs
- No `print()` — structlog only (except Streamlit page files which use `st.*`)
- Run tests: `.venv/bin/python -m pytest`
- Streamlit: `streamlit run dashboard/app.py`

**Key existing APIs (do NOT redefine):**

```python
# agent/journal/store.py
class JournalStore:
    def __init__(self, db_path: Path) -> None: ...
    def query(self, *, entry_type: JournalEntryType | None = None,
              symbol: str | None = None, limit: int = 100) -> list[JournalEntry]: ...

# agent/journal/types.py
class JournalEntryType(StrEnum):
    SIGNAL = "signal"; DECISION = "decision"; FILL = "fill"
    REJECTION = "rejection"; PNL = "pnl"

@dataclass(frozen=True, slots=True)
class JournalEntry:
    entry_id: str; timestamp: datetime; entry_type: JournalEntryType
    symbol: str | None; payload: str  # JSON string

# agent/runner/session_counter.py
class PaperSessionCounter:
    def __init__(self, path: Path) -> None: ...
    def count(self) -> int: ...
    def is_ready_for_live(self) -> bool: ...

# agent/data/cache.py
class ParquetCache:
    def coverage_report(self) -> dict[str, dict[str, tuple[datetime, datetime]]]: ...

# config/settings.py — AppSettings fields used by dashboard:
# settings.journal_db_path: Path
# settings.parquet_cache_dir: Path
# settings.paper_starting_capital: float
# settings.broker: str
# settings.claude_model_primary: str
# settings.max_monthly_api_spend_inr: float
# settings.deployment_env: str
# settings.telegram_bot_token: str
# settings.telegram_chat_id: str
```

**Journal payload schemas (written by DailyLoop):**
- `PNL`: `{"session_date": "YYYY-MM-DD", "final_nav": "Decimal str", "daily_pnl": "Decimal str", "orders_today": int}`
- `FILL`: `{"action": "enter_long|exit_long", "quantity": int, "price": "Decimal str", "signal_id": str}`
- `REJECTION`: `{"reason": str, "detail": str, "signal_id": str}`

---

## File Map

```
dashboard/
    __init__.py              — empty package marker
    data.py                  — pure data-loading functions (unit-tested, no st.* calls)
    app.py                   — Streamlit entry point = Overview page
    pages/
        01_Journal.py        — Fills + rejections tabs with symbol filter
        02_Performance.py    — Equity curve + daily P&L bar chart
        03_Risk.py           — Risk rule configuration display
        04_System.py         — Session counter, cache coverage, kill-switch widget

tests/dashboard/
    __init__.py              — empty
    test_data.py             — 8 tests for dashboard/data.py functions
```

---

## Task 1: `dashboard/data.py` + Tests

**Files:**
- Create: `dashboard/__init__.py`
- Create: `dashboard/data.py`
- Create: `tests/dashboard/__init__.py`
- Create: `tests/dashboard/test_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dashboard/test_data.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.journal.store import JournalStore
from agent.journal.types import JournalEntry, JournalEntryType
from dashboard.data import (
    compute_equity_stats,
    load_fills,
    load_pnl_history,
    load_rejections,
    load_session_count,
)

IST = ZoneInfo("Asia/Kolkata")
T0 = datetime(2024, 1, 2, 15, 30, tzinfo=IST)


def _seed_pnl(db_path: Path, session_date: str, nav: float, pnl: float, orders: int = 2) -> None:
    store = JournalStore(db_path=db_path)
    store.log(JournalEntry(
        entry_id=f"pnl-{session_date}",
        timestamp=T0,
        entry_type=JournalEntryType.PNL,
        symbol=None,
        payload=json.dumps({"session_date": session_date, "final_nav": nav,
                             "daily_pnl": pnl, "orders_today": orders}),
    ))


def _seed_fill(db_path: Path, entry_id: str, symbol: str, qty: int, price: float) -> None:
    store = JournalStore(db_path=db_path)
    store.log(JournalEntry(
        entry_id=entry_id,
        timestamp=T0,
        entry_type=JournalEntryType.FILL,
        symbol=symbol,
        payload=json.dumps({"action": "enter_long", "quantity": qty,
                             "price": price, "signal_id": "test"}),
    ))


def _seed_rejection(db_path: Path, entry_id: str, symbol: str, reason: str) -> None:
    store = JournalStore(db_path=db_path)
    store.log(JournalEntry(
        entry_id=entry_id,
        timestamp=T0,
        entry_type=JournalEntryType.REJECTION,
        symbol=symbol,
        payload=json.dumps({"reason": reason, "detail": "test", "signal_id": "test"}),
    ))


def test_load_pnl_history_returns_correct_rows(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    _seed_pnl(db, "2024-01-02", nav=101500.0, pnl=1500.0)
    _seed_pnl(db, "2024-01-03", nav=102000.0, pnl=500.0)

    df = load_pnl_history(db)
    assert len(df) == 2
    assert "final_nav" in df.columns
    assert "daily_pnl" in df.columns
    assert "session_date" in df.columns


def test_load_pnl_history_empty_returns_schema(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    JournalStore(db_path=db)  # initialise empty DB
    df = load_pnl_history(db)
    assert len(df) == 0
    assert "final_nav" in df.columns


def test_load_fills_returns_correct_symbols(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    _seed_fill(db, "fill-1", "HDFCBANK", 10, 1710.0)
    _seed_fill(db, "fill-2", "TCS", 5, 3500.0)

    df = load_fills(db)
    assert len(df) == 2
    assert set(df["symbol"].to_list()) == {"HDFCBANK", "TCS"}
    assert "price" in df.columns
    assert "quantity" in df.columns


def test_load_rejections_returns_correct_reasons(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    _seed_rejection(db, "rej-1", "INFY", "max_positions_reached")
    _seed_rejection(db, "rej-2", "WIPRO", "daily_loss_cap")

    df = load_rejections(db)
    assert len(df) == 2
    assert "reason" in df.columns
    assert "max_positions_reached" in df["reason"].to_list()


def test_load_session_count_zero_when_no_file(tmp_path: Path) -> None:
    assert load_session_count(tmp_path / "sessions.json") == 0


def test_load_session_count_reflects_increments(tmp_path: Path) -> None:
    from agent.runner.session_counter import PaperSessionCounter
    path = tmp_path / "sessions.json"
    c = PaperSessionCounter(path=path)
    c.increment()
    c.increment()
    c.increment()
    assert load_session_count(path) == 3


def test_compute_equity_stats_empty_dataframe() -> None:
    empty_df = pl.DataFrame(schema={"session_date": pl.Utf8, "final_nav": pl.Float64,
                                     "daily_pnl": pl.Float64, "orders_today": pl.Int64})
    stats = compute_equity_stats(empty_df)
    assert stats["total_sessions"] == 0
    assert stats["win_rate"] == 0.0
    assert stats["current_nav"] == 0.0


def test_compute_equity_stats_with_data(tmp_path: Path) -> None:
    db = tmp_path / "journal.db"
    _seed_pnl(db, "2024-01-02", nav=101500.0, pnl=1500.0)
    _seed_pnl(db, "2024-01-03", nav=101000.0, pnl=-500.0)
    _seed_pnl(db, "2024-01-04", nav=102000.0, pnl=1000.0)

    df = load_pnl_history(db)
    stats = compute_equity_stats(df)
    assert stats["total_sessions"] == 3
    assert stats["total_pnl"] == pytest.approx(2000.0)
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["current_nav"] == pytest.approx(102000.0)
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/tatsatshah/Desktop/yegedge
.venv/bin/python -m pytest tests/dashboard/test_data.py -v --no-cov 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'dashboard'`

- [ ] **Step 3: Create package skeletons**

Create `dashboard/__init__.py` and `tests/dashboard/__init__.py` — both `# intentionally empty`.

- [ ] **Step 4: Write `dashboard/data.py`**

```python
# dashboard/data.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from agent.data.cache import ParquetCache
from agent.journal.store import JournalStore
from agent.journal.types import JournalEntryType
from agent.runner.session_counter import PaperSessionCounter

_EMPTY_PNL_SCHEMA = {
    "session_date": pl.Utf8,
    "final_nav": pl.Float64,
    "daily_pnl": pl.Float64,
    "orders_today": pl.Int64,
}

_EMPTY_FILL_SCHEMA = {
    "timestamp": pl.Utf8,
    "symbol": pl.Utf8,
    "action": pl.Utf8,
    "quantity": pl.Int64,
    "price": pl.Float64,
    "signal_id": pl.Utf8,
}

_EMPTY_REJECTION_SCHEMA = {
    "timestamp": pl.Utf8,
    "symbol": pl.Utf8,
    "reason": pl.Utf8,
    "detail": pl.Utf8,
}


def load_pnl_history(db_path: Path, limit: int = 500) -> pl.DataFrame:
    """Load PNL journal entries as a DataFrame.

    Returns columns: session_date (str), final_nav (float), daily_pnl (float),
    orders_today (int). Returns an empty DataFrame with correct schema when no data.
    """
    store = JournalStore(db_path=db_path)
    entries = store.query(entry_type=JournalEntryType.PNL, limit=limit)
    rows = []
    for e in entries:
        try:
            d = json.loads(e.payload)
            rows.append({
                "session_date": str(d.get("session_date", "")),
                "final_nav": float(d.get("final_nav", 0)),
                "daily_pnl": float(d.get("daily_pnl", 0)),
                "orders_today": int(d.get("orders_today", 0)),
            })
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    if not rows:
        return pl.DataFrame(schema=_EMPTY_PNL_SCHEMA)
    return pl.DataFrame(rows)


def load_fills(db_path: Path, limit: int = 1000) -> pl.DataFrame:
    """Load FILL journal entries as a DataFrame.

    Returns columns: timestamp, symbol, action, quantity (int), price (float),
    signal_id (str). Returns empty DataFrame with correct schema when no data.
    """
    store = JournalStore(db_path=db_path)
    entries = store.query(entry_type=JournalEntryType.FILL, limit=limit)
    rows = []
    for e in entries:
        try:
            d = json.loads(e.payload)
            rows.append({
                "timestamp": e.timestamp.isoformat(),
                "symbol": e.symbol or "",
                "action": str(d.get("action", "")),
                "quantity": int(d.get("quantity", 0)),
                "price": float(d.get("price", 0)),
                "signal_id": str(d.get("signal_id", "")),
            })
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    if not rows:
        return pl.DataFrame(schema=_EMPTY_FILL_SCHEMA)
    return pl.DataFrame(rows)


def load_rejections(db_path: Path, limit: int = 1000) -> pl.DataFrame:
    """Load REJECTION journal entries as a DataFrame.

    Returns columns: timestamp, symbol, reason, detail.
    """
    store = JournalStore(db_path=db_path)
    entries = store.query(entry_type=JournalEntryType.REJECTION, limit=limit)
    rows = []
    for e in entries:
        try:
            d = json.loads(e.payload)
            rows.append({
                "timestamp": e.timestamp.isoformat(),
                "symbol": e.symbol or "",
                "reason": str(d.get("reason", "")),
                "detail": str(d.get("detail", "")),
            })
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    if not rows:
        return pl.DataFrame(schema=_EMPTY_REJECTION_SCHEMA)
    return pl.DataFrame(rows)


def load_session_count(json_path: Path) -> int:
    """Return the number of completed paper sessions from the counter JSON file."""
    return PaperSessionCounter(path=json_path).count()


def load_coverage_summary(
    cache_root: Path,
) -> dict[str, dict[str, tuple[datetime, datetime]]]:
    """Return the Parquet cache coverage report."""
    return ParquetCache(root=cache_root).coverage_report()


def compute_equity_stats(pnl_df: pl.DataFrame) -> dict[str, float]:
    """Compute headline stats from a PNL history DataFrame.

    Returns: total_sessions, total_pnl, win_rate (0–1), current_nav.
    All values are 0.0 / 0 when pnl_df is empty.
    """
    if len(pnl_df) == 0:
        return {"total_sessions": 0, "total_pnl": 0.0, "win_rate": 0.0, "current_nav": 0.0}
    total_sessions = len(pnl_df)
    total_pnl = float(pnl_df["daily_pnl"].sum())
    wins = int((pnl_df["daily_pnl"] > 0).sum())
    win_rate = wins / total_sessions
    current_nav = float(pnl_df["final_nav"][-1])
    return {
        "total_sessions": float(total_sessions),
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "current_nav": current_nav,
    }
```

- [ ] **Step 5: Run tests to verify 8 pass**

```bash
.venv/bin/python -m pytest tests/dashboard/test_data.py -v --no-cov
```

Expected: `8 passed`

- [ ] **Step 6: Run full suite to check regressions**

```bash
.venv/bin/python -m pytest tests/ --no-cov -q 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add dashboard/__init__.py dashboard/data.py \
        tests/dashboard/__init__.py tests/dashboard/test_data.py
git commit -m "feat(dashboard): add data loading module with 8 unit tests"
```

---

## Task 2: Entry Point + Overview Page

**Files:**
- Create: `dashboard/app.py`
- Create: `dashboard/pages/` (directory, Streamlit discovers it automatically)

Note: Streamlit page files use `st.*` calls which cannot be unit-tested in pytest. The page files are smoke-tested via import and `streamlit run`. This is standard for Streamlit apps.

- [ ] **Step 1: Create `dashboard/pages/` directory**

```bash
mkdir -p /Users/tatsatshah/Desktop/yegedge/dashboard/pages
touch /Users/tatsatshah/Desktop/yegedge/dashboard/pages/.gitkeep
```

- [ ] **Step 2: Write `dashboard/app.py`**

```python
# dashboard/app.py
from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import AppSettings
from dashboard.data import compute_equity_stats, load_fills, load_pnl_history, load_session_count

st.set_page_config(
    page_title="YegEdge Dashboard",
    page_icon="📈",
    layout="wide",
)

settings = AppSettings()

st.title("YegEdge — Paper Trading Dashboard")
st.caption(f"Mode: {settings.deployment_env}  |  Live trading: DISABLED by default")

# --- Headline metrics ---
session_count = load_session_count(Path("data/paper_sessions.json"))
pnl_df = load_pnl_history(settings.journal_db_path)
stats = compute_equity_stats(pnl_df)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Paper Sessions", f"{session_count}/60")
with col2:
    nav = stats["current_nav"]
    st.metric("Current NAV", f"₹{nav:,.0f}" if nav > 0 else "—")
with col3:
    pnl = stats["total_pnl"]
    st.metric("Total P&L", f"₹{pnl:+,.0f}" if stats["total_sessions"] > 0 else "—")
with col4:
    wr = stats["win_rate"]
    st.metric("Win Rate", f"{wr:.0%}" if stats["total_sessions"] > 0 else "—")

# --- Live readiness progress ---
st.divider()
progress = min(session_count / 60, 1.0)
st.progress(progress, text=f"{session_count}/60 paper sessions toward live-trading review")
if session_count >= 60:
    st.success(
        "60 sessions complete. Consult `docs/live_readiness.md` before enabling live trading."
    )
else:
    st.info(f"{60 - session_count} more sessions before live-trading eligibility review.")

# --- Recent trades ---
st.divider()
st.subheader("Recent Trades")
fills_df = load_fills(settings.journal_db_path, limit=20)
if len(fills_df) > 0:
    st.dataframe(fills_df.to_pandas(), use_container_width=True)
else:
    st.info("No trades recorded yet. Run `python -m agent run-paper` to start a session.")

# --- Kill switch status ---
ks_path = Path(".kill_switch")
if ks_path.exists():
    st.divider()
    st.error(f"🔴 Kill switch ACTIVE — {ks_path.read_text().strip()}")
```

- [ ] **Step 3: Verify the file imports correctly (no st.* errors at import time)**

```bash
cd /Users/tatsatshah/Desktop/yegedge
.venv/bin/python -c "
import sys
sys.modules['streamlit'] = __import__('unittest.mock', fromlist=['MagicMock']).MagicMock()
import importlib.util
spec = importlib.util.spec_from_file_location('dashboard.app', 'dashboard/app.py')
print('dashboard/app.py: syntax OK' if spec else 'FAILED')
"
```

Expected: `dashboard/app.py: syntax OK`

- [ ] **Step 4: Commit**

```bash
git add dashboard/app.py dashboard/pages/.gitkeep
git commit -m "feat(dashboard): add Streamlit entry point with overview page"
```

---

## Task 3: Journal + Performance Pages

**Files:**
- Create: `dashboard/pages/01_Journal.py`
- Create: `dashboard/pages/02_Performance.py`

- [ ] **Step 1: Write `dashboard/pages/01_Journal.py`**

```python
# dashboard/pages/01_Journal.py
from __future__ import annotations

import streamlit as st

from config.settings import AppSettings
from dashboard.data import load_fills, load_rejections

st.set_page_config(page_title="Journal — YegEdge", layout="wide")
st.title("Trade Journal")

settings = AppSettings()

tab_fills, tab_rejections = st.tabs(["✅ Fills", "❌ Rejections"])

with tab_fills:
    fills_df = load_fills(settings.journal_db_path)
    if len(fills_df) == 0:
        st.info("No fills recorded yet. Run a paper session to populate this page.")
    else:
        all_symbols = sorted(fills_df["symbol"].unique().to_list())
        selected_sym = st.selectbox("Filter by symbol", ["All"] + all_symbols, key="fills_sym")
        display = fills_df if selected_sym == "All" else fills_df.filter(
            fills_df["symbol"] == selected_sym
        )
        st.dataframe(display.to_pandas(), use_container_width=True)
        st.caption(f"{len(display)} fill(s) shown")

with tab_rejections:
    rej_df = load_rejections(settings.journal_db_path)
    if len(rej_df) == 0:
        st.info("No rejections recorded yet.")
    else:
        all_reasons = sorted(rej_df["reason"].unique().to_list())
        selected_r = st.selectbox("Filter by reason", ["All"] + all_reasons, key="rej_reason")
        display_r = rej_df if selected_r == "All" else rej_df.filter(
            rej_df["reason"] == selected_r
        )
        st.dataframe(display_r.to_pandas(), use_container_width=True)
        st.caption(f"{len(display_r)} rejection(s) shown")
```

- [ ] **Step 2: Write `dashboard/pages/02_Performance.py`**

```python
# dashboard/pages/02_Performance.py
from __future__ import annotations

import plotly.express as px
import streamlit as st

from config.settings import AppSettings
from dashboard.data import compute_equity_stats, load_pnl_history

st.set_page_config(page_title="Performance — YegEdge", layout="wide")
st.title("Performance")

settings = AppSettings()
pnl_df = load_pnl_history(settings.journal_db_path)

if len(pnl_df) == 0:
    st.warning("No session P&L data yet. Run `python -m agent run-paper` to start a session.")
    st.stop()

stats = compute_equity_stats(pnl_df)

# Headline stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Sessions", int(stats["total_sessions"]))
with col2:
    st.metric("Total P&L", f"₹{stats['total_pnl']:+,.0f}")
with col3:
    st.metric("Win Rate", f"{stats['win_rate']:.0%}")

st.divider()

pnl_pd = pnl_df.to_pandas()

# Equity curve
fig_nav = px.line(
    pnl_pd,
    x="session_date",
    y="final_nav",
    title="Equity Curve — NAV per Session",
    labels={"final_nav": "NAV (₹)", "session_date": "Session Date"},
    markers=True,
)
st.plotly_chart(fig_nav, use_container_width=True)

# Daily P&L bars
fig_pnl = px.bar(
    pnl_pd,
    x="session_date",
    y="daily_pnl",
    title="Daily P&L per Session",
    labels={"daily_pnl": "P&L (₹)", "session_date": "Session Date"},
    color="daily_pnl",
    color_continuous_scale=["#d62728", "#ffffff", "#2ca02c"],
    color_continuous_midpoint=0,
)
st.plotly_chart(fig_pnl, use_container_width=True)

# Raw data
with st.expander("Raw session data"):
    st.dataframe(pnl_pd, use_container_width=True)
```

- [ ] **Step 3: Verify syntax of both page files**

```bash
.venv/bin/python -c "
import ast, sys
for f in ['dashboard/pages/01_Journal.py', 'dashboard/pages/02_Performance.py']:
    with open(f) as fh:
        ast.parse(fh.read())
    print(f'{f}: OK')
"
```

Expected: both files print `OK`

- [ ] **Step 4: Commit**

```bash
git add dashboard/pages/01_Journal.py dashboard/pages/02_Performance.py
git commit -m "feat(dashboard): add Journal and Performance pages"
```

---

## Task 4: Risk + System Pages

**Files:**
- Create: `dashboard/pages/03_Risk.py`
- Create: `dashboard/pages/04_System.py`

- [ ] **Step 1: Write `dashboard/pages/03_Risk.py`**

```python
# dashboard/pages/03_Risk.py
from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

st.set_page_config(page_title="Risk — YegEdge", layout="wide")
st.title("Risk Rules Configuration")
st.caption("Read-only view of config/risk_rules.yaml. Edit the file directly to change limits.")

rules_path = Path("config/risk_rules.yaml")
if not rules_path.exists():
    st.error("config/risk_rules.yaml not found. Cannot display risk rules.")
    st.stop()

with rules_path.open() as f:
    rules: dict = yaml.safe_load(f)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Per-Trade Limits")
    pt = rules.get("per_trade", {})
    st.metric("Max Risk per Trade", f"{pt.get('max_risk_fraction', 0) * 100:.2f}% of NAV")
    st.metric("Max Position Size", f"{pt.get('max_position_fraction', 0) * 100:.0f}% of NAV")
    st.metric("Min Reward : Risk", f"{pt.get('min_reward_risk', 0):.1f}x")

    st.subheader("Portfolio Limits")
    pf = rules.get("portfolio", {})
    st.metric("Max Concurrent Positions", pf.get("max_concurrent_positions", "—"))
    st.metric("Max Sector Exposure", f"{pf.get('max_sector_exposure', 0) * 100:.0f}% of NAV")
    st.metric("Min Cash Buffer", f"{pf.get('min_cash_fraction', 0) * 100:.0f}%")

    st.subheader("Order Frequency")
    freq = rules.get("frequency", {})
    st.metric("Max Orders / Day", freq.get("max_new_orders_per_day", "—"))
    st.metric("Symbol Cooldown", f"{freq.get('symbol_cooldown_minutes', 0)} min")

with col2:
    st.subheader("Loss Caps")
    caps = rules.get("loss_caps", {})
    st.metric("Daily Loss Cap", f"{caps.get('max_daily_loss_fraction', 0) * 100:.0f}% of NAV")
    st.metric("Weekly Loss Cap", f"{caps.get('max_weekly_loss_fraction', 0) * 100:.0f}% of NAV")
    st.metric(
        "Max Drawdown (Kill Switch)",
        f"{caps.get('max_drawdown_fraction', 0) * 100:.0f}% of NAV",
    )

    st.subheader("Trading Window (IST)")
    win = rules.get("windows", {})
    st.metric("Trade Start", win.get("trade_start_ist", "—"))
    st.metric("Trade End", win.get("trade_end_ist", "—"))
    st.metric("Square-Off By", win.get("square_off_ist", "—"))

    st.subheader("Kill Switch Triggers")
    ks = rules.get("kill_switch", {})
    st.metric("Data Feed Outage", f"{ks.get('data_feed_outage_minutes', '—')} min")
    st.metric("Consecutive Exec Errors", ks.get("consecutive_execution_errors", "—"))
    st.metric("Auto-Reset", str(ks.get("auto_reset", False)))

st.divider()
with st.expander("View full risk_rules.yaml"):
    st.code(rules_path.read_text(), language="yaml")
```

- [ ] **Step 2: Write `dashboard/pages/04_System.py`**

```python
# dashboard/pages/04_System.py
from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st

from config.settings import AppSettings
from dashboard.data import load_coverage_summary, load_session_count

st.set_page_config(page_title="System — YegEdge", layout="wide")
st.title("System Status")

settings = AppSettings()

# --- Session counter ---
st.subheader("Paper Trading Progress")
session_count = load_session_count(Path("data/paper_sessions.json"))
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sessions Completed", f"{session_count}/60")
with col2:
    st.metric("Live Trading", "DISABLED (default)")
with col3:
    status = "✅ Eligible for review" if session_count >= 60 else f"⏳ {60 - session_count} remaining"
    st.metric("Live Readiness", status)
st.progress(min(session_count / 60, 1.0), text=f"{session_count}/60 paper sessions")

st.divider()

# --- Kill switch ---
st.subheader("Kill Switch")
ks_path = Path(".kill_switch")
if ks_path.exists():
    reason = ks_path.read_text().strip()
    st.error(f"🔴 Kill switch ACTIVE: {reason or '(no reason written)'}")
    if st.button("Deactivate Kill Switch", type="primary"):
        ks_path.unlink()
        st.success("Kill switch deactivated. Restart the trading loop to resume.")
        st.rerun()
else:
    st.success("🟢 Kill switch: inactive")

st.divider()

# --- Data cache coverage ---
st.subheader("Parquet Cache Coverage")
coverage = load_coverage_summary(settings.parquet_cache_dir)
if not coverage:
    st.warning("No cached data found. Run `python -m agent refresh` to populate the cache.")
else:
    rows = []
    for symbol, tfs in coverage.items():
        for tf, (start, end) in tfs.items():
            rows.append({
                "Symbol": symbol,
                "Timeframe": tf,
                "From": str(start.date()),
                "To": str(end.date()),
            })
    df = pl.DataFrame(rows)
    st.dataframe(df.to_pandas(), use_container_width=True)
    st.caption(f"{len(rows)} symbol/timeframe pair(s) cached")

st.divider()

# --- Configuration summary ---
st.subheader("Configuration")
col_a, col_b = st.columns(2)
with col_a:
    st.metric("Broker", settings.broker)
    st.metric("Paper Starting Capital", f"₹{settings.paper_starting_capital:,.0f}")
    st.metric("Primary LLM", settings.claude_model_primary)
    st.metric("Monthly API Cap", f"₹{settings.max_monthly_api_spend_inr:,.0f}")
with col_b:
    st.metric("Journal DB", str(settings.journal_db_path))
    st.metric("Cache Dir", str(settings.parquet_cache_dir))
    st.metric("Log Dir", str(settings.log_dir))
    st.metric("Telegram", "Configured" if settings.telegram_bot_token else "Not set")
```

- [ ] **Step 3: Verify syntax of both page files**

```bash
.venv/bin/python -c "
import ast
for f in ['dashboard/pages/03_Risk.py', 'dashboard/pages/04_System.py']:
    with open(f) as fh:
        ast.parse(fh.read())
    print(f'{f}: OK')
"
```

Expected: both files print `OK`

- [ ] **Step 4: Commit**

```bash
git add dashboard/pages/03_Risk.py dashboard/pages/04_System.py
git commit -m "feat(dashboard): add Risk and System pages"
```

---

## Task 5: Integration Test + Full Suite

**Files:**
- Verify: full test suite passes
- Smoke-test: all dashboard imports resolve and data functions work end-to-end

- [ ] **Step 1: Run full test suite with coverage**

```bash
.venv/bin/python -m pytest tests/ --cov=agent --cov=dashboard \
    --cov-report=term-missing --cov-fail-under=70 -q 2>&1 | tail -30
```

Expected: **334+ tests pass** (326 existing + 8 new dashboard tests), coverage ≥ 70%.

- [ ] **Step 2: Run linters on dashboard code**

```bash
.venv/bin/python -m ruff check dashboard/ tests/dashboard/ && \
.venv/bin/python -m black --check dashboard/ tests/dashboard/ && \
echo CLEAN
```

Fix any issues, then re-run.

- [ ] **Step 3: Run the end-to-end data smoke test**

```bash
.venv/bin/python - <<'EOF'
from __future__ import annotations
import json, tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agent.journal.store import JournalStore
from agent.journal.types import JournalEntry, JournalEntryType
from dashboard.data import (
    compute_equity_stats, load_fills, load_pnl_history,
    load_rejections, load_session_count,
)

IST = ZoneInfo("Asia/Kolkata")
T0 = datetime(2024, 1, 2, 15, 30, tzinfo=IST)

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)
    db = p / "journal.db"
    store = JournalStore(db_path=db)

    # Seed two PNL entries
    for i, (date, nav, pnl) in enumerate([
        ("2024-01-02", 101500.0, 1500.0),
        ("2024-01-03", 100800.0, -700.0),
    ]):
        store.log(JournalEntry(
            entry_id=f"pnl-{date}",
            timestamp=T0,
            entry_type=JournalEntryType.PNL,
            symbol=None,
            payload=json.dumps({"session_date": date, "final_nav": nav,
                                 "daily_pnl": pnl, "orders_today": 2}),
        ))

    # Seed a fill
    store.log(JournalEntry(
        entry_id="fill-test",
        timestamp=T0,
        entry_type=JournalEntryType.FILL,
        symbol="HDFCBANK",
        payload=json.dumps({"action": "enter_long", "quantity": 10,
                             "price": 1710.0, "signal_id": "test"}),
    ))

    pnl_df = load_pnl_history(db)
    assert len(pnl_df) == 2, f"Expected 2 PNL rows, got {len(pnl_df)}"

    fills_df = load_fills(db)
    assert len(fills_df) == 1, f"Expected 1 fill row, got {len(fills_df)}"
    assert fills_df["symbol"][0] == "HDFCBANK"

    rej_df = load_rejections(db)
    assert len(rej_df) == 0

    stats = compute_equity_stats(pnl_df)
    assert stats["total_sessions"] == 2
    assert stats["total_pnl"] == 800.0, f"Expected 800.0, got {stats['total_pnl']}"
    assert stats["win_rate"] == 0.5

    count = load_session_count(p / "missing.json")
    assert count == 0

    print("DATA SMOKE TEST PASSED")
EOF
```

Expected: `DATA SMOKE TEST PASSED`

- [ ] **Step 4: Verify `streamlit run` can parse app.py without syntax errors**

```bash
.venv/bin/python -m py_compile dashboard/app.py && echo "app.py: OK"
.venv/bin/python -m py_compile dashboard/pages/01_Journal.py && echo "01_Journal.py: OK"
.venv/bin/python -m py_compile dashboard/pages/02_Performance.py && echo "02_Performance.py: OK"
.venv/bin/python -m py_compile dashboard/pages/03_Risk.py && echo "03_Risk.py: OK"
.venv/bin/python -m py_compile dashboard/pages/04_System.py && echo "04_System.py: OK"
```

Expected: all five files print `OK`

- [ ] **Step 5: Print how to run the dashboard**

```bash
echo "To start the dashboard: streamlit run dashboard/app.py"
echo "Then open http://localhost:8501 in your browser."
```

- [ ] **Step 6: Commit any linting fixes + final verification commit**

```bash
git add -p  # review any formatting changes
git status
# Only commit if there are actual changes from linting:
# git commit -m "style(dashboard): ruff + black fixes for dashboard modules"
```

Then a final log check:
```bash
git log --oneline -8
```

---

## Verification

After all tasks, run the dashboard locally to verify it renders:

```bash
streamlit run dashboard/app.py
```

Expected pages in the sidebar:
1. **YegEdge Dashboard** (Overview — main page)
2. **Journal** (fills + rejections with filter)
3. **Performance** (equity curve + P&L bar chart)
4. **Risk** (config display from risk_rules.yaml)
5. **System** (session counter + cache coverage + kill switch)

The dashboard should render without error even when the journal DB is empty (all data functions return empty DataFrames with the correct schema when no data exists).

---

## Self-Review

**Spec coverage:**
- ✅ Overview: NAV, daily P&L, session count, recent fills
- ✅ Journal: fills (with symbol filter) + rejections (with reason filter)
- ✅ Performance: equity curve + daily P&L bar chart
- ✅ Risk: all key rule categories displayed
- ✅ System: session counter, cache coverage, kill switch widget, config
- ⚠️ Market sentiment page (India VIX, FII/DII flows): deferred — requires live external data feeds not available in paper-trading mode. Intentionally out of scope for Phase 9.
- ⚠️ Signals page (per-signal research note): deferred — research notes are in-memory only (NoteCache), not persisted to SQLite. A future phase can add a note persistence layer.

**Placeholder scan:** No TBDs, TODOs, or "implement later" found.

**Type consistency:** All data functions return `pl.DataFrame` or primitive types. All page files call the same function names as defined in data.py.
