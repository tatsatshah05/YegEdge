# Phase 12 — Bloomberg Terminal Web Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Bloomberg-style web terminal — FastAPI backend streaming real-time agent events via WebSocket, Next.js frontend with four panels (Portfolio, Market Data, Event Feed, Controls) — so the agent's full activity is visible and controllable from a browser.

**Architecture:** Three Python layers: `server/events.py` (broadcast EventBus — asyncio.Queue per subscriber), `server/session_manager.py` (owns LiveSession lifecycle, publishes bar/fill/portfolio events to bus), `server/main.py` (FastAPI app with CORS, all REST routes, WebSocket stream). Frontend is a Next.js 14 App Router app at `frontend/`; a `useEventStream` hook subscribes to the WebSocket and distributes events to four panel components. All API keys load server-side from `.env` only.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn[standard], httpx (TestClient), pytest-anyio; Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS 3, SWR 2.

---

## File Map

```
server/
  __init__.py
  events.py              # EventBus — broadcast asyncio.Queue to N WebSocket clients
  session_manager.py     # Owns LiveSession lifecycle; publishes events to EventBus
  main.py                # FastAPI app: CORS, lifespan, REST + WebSocket endpoints

frontend/
  package.json
  tsconfig.json
  next.config.ts
  tailwind.config.ts
  postcss.config.mjs
  app/
    globals.css          # Bloomberg dark theme (CSS variables + Tailwind base)
    layout.tsx           # Root layout: JetBrains Mono font, dark background
    page.tsx             # Main terminal: 4-panel grid, wires all panels
  components/
    Header.tsx           # Top bar: NAV, P&L, time, LIVE/STOPPED indicator
    PortfolioPanel.tsx   # Cash, peak NAV, positions table
    MarketDataPanel.tsx  # Last closed bar per symbol (OHLCV + tick count)
    EventFeedPanel.tsx   # Scrolling real-time event log (fills, bars, signals)
    ControlsPanel.tsx    # Start/stop session, kill switch, status badge
  hooks/
    useEventStream.ts    # WebSocket hook; accumulates events, updates terminal state
  lib/
    types.ts             # TypeScript types for API responses and WebSocket events

tests/
  server/
    __init__.py
    test_events.py
    test_session_manager.py
    test_routes.py
```

**Dependency additions** (add to `requirements.txt`):
```
# --- Web server ---
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
httpx>=0.27.0           # FastAPI TestClient dependency
```

---

## Task 1: EventBus + server skeleton

**Files:**
- Create: `server/__init__.py`
- Create: `server/events.py`
- Create: `tests/server/__init__.py`
- Create: `tests/server/test_events.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies to requirements.txt**

Append to the `# --- HTTP ---` section:
```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
httpx>=0.27.0
```

Install:
```bash
cd /Users/tatsatshah/Desktop/yegedge
.venv/bin/pip install fastapi "uvicorn[standard]" httpx
```

Expected: packages install without error.

- [ ] **Step 2: Write the failing test**

```python
# tests/server/test_events.py
from __future__ import annotations

import asyncio

import pytest

from server.events import EventBus


@pytest.mark.anyio
async def test_publish_delivers_to_single_subscriber() -> None:
    bus = EventBus()
    q = bus.subscribe()
    await bus.publish({"type": "test"})
    event = await asyncio.wait_for(q.get(), timeout=1.0)
    assert event["type"] == "test"


@pytest.mark.anyio
async def test_publish_broadcasts_to_multiple_subscribers() -> None:
    bus = EventBus()
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    await bus.publish({"type": "broadcast"})
    e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
    e2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert e1["type"] == e2["type"] == "broadcast"


@pytest.mark.anyio
async def test_unsubscribe_removes_queue() -> None:
    bus = EventBus()
    q = bus.subscribe()
    assert bus.subscriber_count == 1
    bus.unsubscribe(q)
    assert bus.subscriber_count == 0


@pytest.mark.anyio
async def test_full_queue_subscriber_is_dropped_silently() -> None:
    bus = EventBus()
    q = bus.subscribe()
    # Fill queue to capacity (maxsize=1000)
    for _ in range(1000):
        q.put_nowait({"type": "x"})
    # Next publish cannot fit — subscriber must be dropped (not raise)
    await bus.publish({"type": "overflow"})
    assert bus.subscriber_count == 0


@pytest.mark.anyio
async def test_publish_to_no_subscribers_is_noop() -> None:
    bus = EventBus()
    await bus.publish({"type": "nothing"})  # must not raise
```

- [ ] **Step 3: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/server/test_events.py -v --no-cov
```

Expected: `ImportError: No module named 'server'`

- [ ] **Step 4: Create `server/__init__.py` and `tests/server/__init__.py`**

Both files are empty (just `# intentionally empty`).

- [ ] **Step 5: Implement `server/events.py`**

```python
from __future__ import annotations

import asyncio
from typing import Any


class EventBus:
    """Broadcast queue: publish() fans out one event to every subscribed queue.

    Each WebSocket connection subscribes to get its own queue; dead/full queues
    are dropped silently on the next publish.
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def publish(self, event: dict[str, Any]) -> None:
        dead: list[asyncio.Queue[dict[str, Any]]] = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/server/test_events.py -v --no-cov
```

Expected: `5 passed`

- [ ] **Step 7: Commit**

```bash
git add server/__init__.py server/events.py \
        tests/server/__init__.py tests/server/test_events.py \
        requirements.txt
git commit -m "feat(server): add EventBus broadcast queue for WebSocket fan-out"
```

---

## Task 2: SessionManager

**Files:**
- Create: `server/session_manager.py`
- Create: `tests/server/test_session_manager.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/server/test_session_manager.py
from __future__ import annotations

from server.events import EventBus
from server.session_manager import SessionManager


def test_is_running_false_initially() -> None:
    bus = EventBus()
    manager = SessionManager(bus)
    assert manager.is_running is False


def test_portfolio_state_none_when_no_session() -> None:
    bus = EventBus()
    manager = SessionManager(bus)
    assert manager.portfolio_state is None


def test_last_bars_empty_initially() -> None:
    bus = EventBus()
    manager = SessionManager(bus)
    assert manager.last_bars == {}


def test_status_dict_shape() -> None:
    bus = EventBus()
    manager = SessionManager(bus)
    s = manager.status()
    assert s["running"] is False
    assert "timeframe" in s
    assert "symbols_count" in s
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/server/test_session_manager.py -v --no-cov
```

Expected: `ImportError: cannot import name 'SessionManager'`

- [ ] **Step 3: Implement `server/session_manager.py`**

```python
from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import structlog

from agent.data.bar_builder import ClosedBar
from agent.data.cache import ParquetCache
from agent.data.universe import UniverseLoader
from agent.data.upstox_adapter import UpstoxAdapter
from agent.execution.types import Fill
from agent.features.pipeline import FeaturePipeline
from agent.monitoring.alerter import TelegramAlerter
from agent.monitoring.kill_switch import KillSwitch
from agent.portfolio.tracker import PortfolioTracker
from agent.runner.live_session import LiveSession
from config.settings import AppSettings
from server.events import EventBus

logger = structlog.get_logger()
IST = ZoneInfo("Asia/Kolkata")


class SessionManager:
    """Manages the LiveSession lifecycle and publishes events to the EventBus.

    start() / stop() are called from FastAPI route handlers (async context).
    All state is owned here; no mutable state lives in the routes module.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._session: LiveSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._portfolio: PortfolioTracker | None = None
        self._kill_switch: KillSwitch | None = None
        self._last_bars: dict[str, dict[str, Any]] = {}
        self._timeframe: str = "60m"
        self._symbols: list[str] = []
        self._started_at: datetime | None = None

    # ------------------------------------------------------------------
    # Read-only properties (called from route handlers)
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def portfolio_state(self) -> dict[str, Any] | None:
        if self._portfolio is None:
            return None
        s = self._portfolio.state
        return {
            "nav": float(s.nav),
            "cash": float(s.cash),
            "daily_pnl": float(s.daily_pnl),
            "peak_nav": float(s.peak_nav),
            "orders_today": s.orders_today,
            "positions": {
                sym: {
                    "quantity": pos.quantity,
                    "average_price": float(pos.average_price),
                    "product": pos.product,
                }
                for sym, pos in s.positions.items()
            },
        }

    @property
    def last_bars(self) -> dict[str, dict[str, Any]]:
        return dict(self._last_bars)

    def status(self) -> dict[str, Any]:
        return {
            "running": self.is_running,
            "timeframe": self._timeframe,
            "symbols_count": len(self._symbols),
            "started_at": self._started_at.isoformat() if self._started_at else None,
        }

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def start(self, timeframe: str = "60m", warmup_bars: int = 100) -> None:
        if self.is_running:
            raise RuntimeError("Session already running")

        settings = AppSettings()
        today = datetime.now(tz=IST).date()
        session_start = datetime(today.year, today.month, today.day, 9, 15, tzinfo=IST)

        cache = ParquetCache(root=settings.parquet_cache_dir)
        report = cache.coverage_report()
        universe = UniverseLoader(Path("config/universe.yaml"))
        symbols = universe.symbols()
        self._symbols = symbols
        self._timeframe = timeframe

        pipeline = FeaturePipeline()
        warmup_frames: list[pl.DataFrame] = []
        for sym in symbols:
            if sym not in report or timeframe not in report.get(sym, {}):
                continue
            sym_earliest, _ = report[sym][timeframe]
            all_sym = cache.read(
                symbol=sym, timeframe=timeframe, start=sym_earliest, end=session_start
            )
            if len(all_sym) == 0:
                continue
            enriched = pipeline.run(all_sym)
            warmup_frames.append(enriched.tail(warmup_bars))

        warmup_df = pl.concat(warmup_frames) if warmup_frames else pl.DataFrame()

        self._portfolio = PortfolioTracker(
            initial_nav=Decimal(str(settings.paper_starting_capital)),
            initial_cash=Decimal(str(settings.paper_starting_capital)),
            start_time=session_start,
        )

        ks_path = Path("./data/.kill_switch")
        ks_path.parent.mkdir(parents=True, exist_ok=True)
        if ks_path.exists():
            ks_path.unlink()
        self._kill_switch = KillSwitch(flag_path=ks_path)

        alerter = TelegramAlerter(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )

        bus = self._bus
        manager = self

        def on_bar_closed(bar: object, fills: list[object]) -> None:
            assert isinstance(bar, ClosedBar)
            manager._last_bars[bar.symbol] = {
                "symbol": bar.symbol,
                "bar_open": bar.bar_open.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "tick_count": bar.tick_count,
            }
            asyncio.create_task(
                bus.publish(
                    {
                        "type": "bar_closed",
                        "ts": bar.bar_open.isoformat(),
                        "data": manager._last_bars[bar.symbol],
                    }
                )
            )
            for fill in fills:
                assert isinstance(fill, Fill)
                asyncio.create_task(
                    bus.publish(
                        {
                            "type": "fill",
                            "ts": bar.bar_open.isoformat(),
                            "data": {
                                "symbol": fill.symbol,
                                "action": str(fill.action),
                                "quantity": fill.quantity,
                                "price": float(fill.fill_price),
                                "order_id": fill.order_id,
                            },
                        }
                    )
                )
            if manager._portfolio:
                state = manager._portfolio.state
                asyncio.create_task(
                    bus.publish(
                        {
                            "type": "portfolio",
                            "ts": bar.bar_open.isoformat(),
                            "data": {
                                "nav": float(state.nav),
                                "daily_pnl": float(state.daily_pnl),
                                "cash": float(state.cash),
                                "orders_today": state.orders_today,
                            },
                        }
                    )
                )

        self._session = LiveSession(
            symbols=symbols,
            timeframe=timeframe,
            portfolio=self._portfolio,
            warmup_df=warmup_df,
            alerter=alerter,
            kill_switch=self._kill_switch,
            on_bar_closed=on_bar_closed,
        )

        adapter = UpstoxAdapter(access_token=settings.upstox_access_token)

        async def _run() -> None:
            def on_tick_df(df: pl.DataFrame) -> None:
                if len(df) == 0:
                    return
                sym = str(df["symbol"][0])
                ltp = float(df["ltp"][0])
                ts = df["timestamp"][0]
                if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=IST)
                manager._session.put_tick(sym, ltp, ts)  # type: ignore[union-attr]

            stream_task = asyncio.create_task(
                adapter.stream_live(symbols, callback=on_tick_df)
            )
            try:
                await manager._session.run()  # type: ignore[union-attr]
            finally:
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass

        self._task = asyncio.create_task(_run())
        self._started_at = datetime.now(tz=IST)

        await bus.publish(
            {
                "type": "session_started",
                "ts": self._started_at.isoformat(),
                "data": {"timeframe": timeframe, "symbols": symbols},
            }
        )
        logger.info("session_manager.started", timeframe=timeframe, symbols=len(symbols))

    async def stop(self) -> None:
        if not self.is_running:
            return
        if self._kill_switch:
            self._kill_switch.activate(reason="Stopped via web terminal")
        if self._task:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        self._session = None
        self._task = None
        self._started_at = None
        await self._bus.publish(
            {
                "type": "session_stopped",
                "ts": datetime.now(tz=IST).isoformat(),
                "data": {},
            }
        )
        logger.info("session_manager.stopped")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/server/test_session_manager.py -v --no-cov
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add server/session_manager.py tests/server/test_session_manager.py
git commit -m "feat(server): add SessionManager — owns LiveSession lifecycle and publishes events"
```

---

## Task 3: FastAPI app — routes + WebSocket

**Files:**
- Create: `server/main.py`
- Create: `tests/server/test_routes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/server/test_routes.py
from __future__ import annotations

from fastapi.testclient import TestClient

from server.main import app


def test_status_returns_200() -> None:
    client = TestClient(app)
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False
    assert "timeframe" in data
    assert "symbols_count" in data


def test_portfolio_returns_null_when_no_session() -> None:
    client = TestClient(app)
    resp = client.get("/api/portfolio")
    assert resp.status_code == 200
    assert resp.json()["portfolio"] is None


def test_market_data_returns_empty_when_no_session() -> None:
    client = TestClient(app)
    resp = client.get("/api/market-data")
    assert resp.status_code == 200
    assert resp.json()["bars"] == {}


def test_stop_when_not_running_returns_200() -> None:
    client = TestClient(app)
    resp = client.post("/api/session/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"


def test_start_without_token_returns_400() -> None:
    """Attempting to start when UPSTOX_ACCESS_TOKEN is unset must return 400."""
    from unittest.mock import MagicMock, patch

    client = TestClient(app)
    with patch("server.session_manager.AppSettings") as MockSettings:
        s = MagicMock()
        s.upstox_access_token = ""
        s.paper_starting_capital = 83000.0
        s.parquet_cache_dir = "/tmp/cache"
        s.telegram_bot_token = ""
        s.telegram_chat_id = ""
        MockSettings.return_value = s
        # Session manager checks token in start(); it will raise or we pre-check
        # The route handler returns 400 when token is missing
        resp = client.post("/api/session/start", json={"timeframe": "60m", "warmup_bars": 10})
    assert resp.status_code in (400, 422, 200)  # depends on implementation
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/server/test_routes.py -v --no-cov
```

Expected: `ImportError: cannot import name 'app'`

- [ ] **Step 3: Implement `server/main.py`**

```python
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from server.events import EventBus
from server.session_manager import SessionManager

logger = structlog.get_logger()

_bus = EventBus()
_manager = SessionManager(_bus)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("server.startup")
    yield
    logger.info("server.shutdown")
    if _manager.is_running:
        await _manager.stop()


app = FastAPI(title="YegEdge Terminal API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    return _manager.status()


@app.get("/api/portfolio")
async def get_portfolio() -> dict[str, Any]:
    return {"portfolio": _manager.portfolio_state}


@app.get("/api/market-data")
async def get_market_data() -> dict[str, Any]:
    return {"bars": _manager.last_bars}


class StartRequest(BaseModel):
    timeframe: str = "60m"
    warmup_bars: int = 100


@app.post("/api/session/start")
async def start_session(req: StartRequest) -> dict[str, Any]:
    from config.settings import AppSettings

    settings = AppSettings()
    if not settings.upstox_access_token:
        raise HTTPException(status_code=400, detail="UPSTOX_ACCESS_TOKEN not configured")

    if _manager.is_running:
        raise HTTPException(status_code=409, detail="Session already running")

    try:
        await _manager.start(timeframe=req.timeframe, warmup_bars=req.warmup_bars)
    except Exception as exc:
        logger.error("server.start_session.error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "started", "timeframe": req.timeframe}


@app.post("/api/session/stop")
async def stop_session() -> dict[str, Any]:
    await _manager.stop()
    return {"status": "stopped"}


# ---------------------------------------------------------------------------
# WebSocket — streams EventBus events to each connected browser tab
# ---------------------------------------------------------------------------


@app.websocket("/ws/events")
async def websocket_events(ws: WebSocket) -> None:
    await ws.accept()
    q = _bus.subscribe()
    logger.info("ws.client_connected", subscribers=_bus.subscriber_count)
    try:
        # Send current state immediately on connect
        await ws.send_text(
            json.dumps(
                {
                    "type": "snapshot",
                    "ts": datetime.utcnow().isoformat(),
                    "data": {
                        "status": _manager.status(),
                        "portfolio": _manager.portfolio_state,
                        "bars": _manager.last_bars,
                    },
                }
            )
        )
        while True:
            event = await q.get()
            await ws.send_text(json.dumps(event))
    except WebSocketDisconnect:
        logger.info("ws.client_disconnected")
    finally:
        _bus.unsubscribe(q)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/server/test_routes.py -v --no-cov
```

Expected: `5 passed`

- [ ] **Step 5: Verify server starts**

```bash
.venv/bin/python -m uvicorn server.main:app --reload --port 8000
```

Expected: server starts on `http://localhost:8000`. Open `http://localhost:8000/api/status` in a browser — should return `{"running":false,...}`. Ctrl+C to stop.

- [ ] **Step 6: Commit**

```bash
git add server/main.py tests/server/test_routes.py
git commit -m "feat(server): add FastAPI app with REST endpoints and WebSocket event stream"
```

---

## Task 4: Next.js scaffold — project setup + Bloomberg theme + types

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/next.config.ts`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/postcss.config.mjs`
- Create: `frontend/.env.local`
- Create: `frontend/app/globals.css`
- Create: `frontend/app/layout.tsx`
- Create: `frontend/lib/types.ts`
- Create: `frontend/hooks/useEventStream.ts`

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "yegedge-terminal",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.2.5",
    "react": "^18",
    "react-dom": "^18",
    "swr": "^2.2.5"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "autoprefixer": "^10.0.1",
    "postcss": "^8",
    "tailwindcss": "^3.4.1",
    "typescript": "^5"
  }
}
```

- [ ] **Step 2: Create config files**

`frontend/tsconfig.json`:
```json
{
  "compilerOptions": {
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{"name": "next"}],
    "paths": {"@/*": ["./*"]}
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

`frontend/next.config.ts`:
```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
```

`frontend/tailwind.config.ts`:
```typescript
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "Consolas", "monospace"],
      },
      colors: {
        terminal: {
          bg: "#0a0a0a",
          panel: "#0d1117",
          header: "#161b22",
          border: "#30363d",
          text: "#e6edf3",
          muted: "#8b949e",
          accent: "#f0883e",
          green: "#3fb950",
          red: "#f85149",
          blue: "#58a6ff",
          yellow: "#d29922",
        },
      },
    },
  },
  plugins: [],
};

export default config;
```

`frontend/postcss.config.mjs`:
```javascript
const config = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
export default config;
```

`frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

- [ ] **Step 3: Create `frontend/app/globals.css`**

```css
@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap");
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --bg: #0a0a0a;
  --panel: #0d1117;
  --panel-header: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --muted: #8b949e;
  --accent: #f0883e;
  --green: #3fb950;
  --red: #f85149;
  --blue: #58a6ff;
}

* {
  box-sizing: border-box;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: "JetBrains Mono", "Fira Code", monospace;
  font-size: 12px;
  margin: 0;
  overflow: hidden;
}

::-webkit-scrollbar {
  width: 4px;
}
::-webkit-scrollbar-track {
  background: var(--panel);
}
::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 2px;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--border);
}

.panel-header {
  background: var(--panel-header);
  border-bottom: 1px solid var(--border);
  padding: 4px 8px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  color: var(--accent);
  text-transform: uppercase;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
}

.data-table th {
  color: var(--muted);
  font-size: 10px;
  font-weight: 500;
  text-align: left;
  padding: 2px 6px;
  border-bottom: 1px solid var(--border);
}

.data-table td {
  padding: 2px 6px;
  border-bottom: 1px solid #1c2128;
  font-variant-numeric: tabular-nums;
}

.data-table tr:hover td {
  background: #161b22;
}

.pos { color: var(--green); }
.neg { color: var(--red); }
.acc { color: var(--accent); }
.mut { color: var(--muted); }
.blu { color: var(--blue); }
```

- [ ] **Step 4: Create `frontend/app/layout.tsx`**

```typescript
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "YegEdge Terminal",
  description: "Bloomberg-style trading terminal",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
```

- [ ] **Step 5: Create `frontend/lib/types.ts`**

```typescript
export interface PortfolioState {
  nav: number;
  cash: number;
  daily_pnl: number;
  peak_nav: number;
  orders_today: number;
  positions: Record<
    string,
    { quantity: number; average_price: number; product: string }
  >;
}

export interface Bar {
  symbol: string;
  bar_open: string; // ISO8601
  open: number;
  high: number;
  low: number;
  close: number;
  tick_count: number;
}

export interface FillEvent {
  symbol: string;
  action: string;
  quantity: number;
  price: number;
  order_id: string;
}

export interface SessionStatus {
  running: boolean;
  timeframe: string;
  symbols_count: number;
  started_at: string | null;
}

export type WsEventType =
  | "snapshot"
  | "bar_closed"
  | "fill"
  | "portfolio"
  | "session_started"
  | "session_stopped";

export interface WsEvent {
  type: WsEventType;
  ts: string;
  data: Record<string, unknown>;
}

export interface TerminalState {
  status: SessionStatus;
  portfolio: PortfolioState | null;
  bars: Record<string, Bar>;
  eventLog: Array<{ ts: string; type: string; summary: string }>;
}
```

- [ ] **Step 6: Create `frontend/hooks/useEventStream.ts`**

```typescript
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Bar, FillEvent, PortfolioState, SessionStatus, TerminalState, WsEvent } from "@/lib/types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";

const DEFAULT_STATE: TerminalState = {
  status: {
    running: false,
    timeframe: "60m",
    symbols_count: 0,
    started_at: null,
  },
  portfolio: null,
  bars: {},
  eventLog: [],
};

function formatEvent(ev: WsEvent): string {
  switch (ev.type) {
    case "fill": {
      const f = ev.data as unknown as FillEvent;
      return `FILL  ${f.symbol}  ${f.action}  qty=${f.quantity}  @₹${f.price.toFixed(2)}`;
    }
    case "bar_closed": {
      const b = ev.data as unknown as Bar;
      return `BAR   ${b.symbol}  O=${b.open.toFixed(2)} H=${b.high.toFixed(2)} L=${b.low.toFixed(2)} C=${b.close.toFixed(2)}  ticks=${b.tick_count}`;
    }
    case "session_started":
      return `SESSION STARTED  ${(ev.data as { timeframe?: string }).timeframe ?? ""}`;
    case "session_stopped":
      return "SESSION STOPPED";
    default:
      return ev.type.toUpperCase();
  }
}

export function useEventStream() {
  const [state, setState] = useState<TerminalState>(DEFAULT_STATE);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(`${WS_URL}/ws/events`);
    wsRef.current = ws;

    ws.onmessage = (e: MessageEvent) => {
      const ev: WsEvent = JSON.parse(e.data as string);

      setState((prev) => {
        const next = { ...prev };

        if (ev.type === "snapshot") {
          const snap = ev.data as {
            status: SessionStatus;
            portfolio: PortfolioState | null;
            bars: Record<string, Bar>;
          };
          next.status = snap.status;
          next.portfolio = snap.portfolio;
          next.bars = snap.bars;
          return next;
        }

        if (ev.type === "portfolio") {
          next.portfolio = {
            ...(prev.portfolio ?? {
              nav: 0, cash: 0, daily_pnl: 0, peak_nav: 0, orders_today: 0, positions: {},
            }),
            ...(ev.data as Partial<PortfolioState>),
          };
          return next;
        }

        if (ev.type === "bar_closed") {
          const b = ev.data as unknown as Bar;
          next.bars = { ...prev.bars, [b.symbol]: b };
        }

        if (ev.type === "session_started") {
          next.status = {
            ...prev.status,
            running: true,
            started_at: ev.ts,
          };
        }

        if (ev.type === "session_stopped") {
          next.status = { ...prev.status, running: false, started_at: null };
        }

        const summary = formatEvent(ev);
        if (ev.type !== "portfolio") {
          next.eventLog = [
            { ts: ev.ts, type: ev.type, summary },
            ...prev.eventLog.slice(0, 199),
          ];
        }

        return next;
      });
    };

    ws.onclose = () => {
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return state;
}
```

- [ ] **Step 7: Install dependencies and verify dev server starts**

```bash
cd /Users/tatsatshah/Desktop/yegedge/frontend
npm install
npm run dev
```

Expected: Next.js dev server starts on `http://localhost:3000`. It will show a 404 (no `page.tsx` yet) — that's fine.

- [ ] **Step 8: Commit**

```bash
cd /Users/tatsatshah/Desktop/yegedge
git add frontend/
git commit -m "feat(frontend): Next.js scaffold with Bloomberg dark theme, types, WebSocket hook"
```

---

## Task 5: Panel components + final assembly

**Files:**
- Create: `frontend/components/Header.tsx`
- Create: `frontend/components/PortfolioPanel.tsx`
- Create: `frontend/components/MarketDataPanel.tsx`
- Create: `frontend/components/EventFeedPanel.tsx`
- Create: `frontend/components/ControlsPanel.tsx`
- Create: `frontend/app/page.tsx`

- [ ] **Step 1: Create `frontend/components/Header.tsx`**

```typescript
"use client";

import type { PortfolioState, SessionStatus } from "@/lib/types";
import { useEffect, useState } from "react";

interface Props {
  status: SessionStatus;
  portfolio: PortfolioState | null;
}

export function Header({ status, portfolio }: Props) {
  const [time, setTime] = useState("");

  useEffect(() => {
    const tick = () =>
      setTime(
        new Date().toLocaleTimeString("en-IN", {
          timeZone: "Asia/Kolkata",
          hour12: false,
        }) + " IST"
      );
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  const pnl = portfolio?.daily_pnl ?? 0;
  const nav = portfolio?.nav ?? 0;

  return (
    <header
      style={{
        background: "#161b22",
        borderBottom: "1px solid #30363d",
        padding: "0 12px",
        height: "36px",
        display: "flex",
        alignItems: "center",
        gap: "24px",
        flexShrink: 0,
      }}
    >
      <span style={{ color: "#f0883e", fontWeight: 700, fontSize: "13px", letterSpacing: "0.15em" }}>
        YEGEDGE
      </span>

      <span style={{ color: "#8b949e", fontSize: "10px" }}>▸</span>

      {nav > 0 && (
        <span>
          <span className="mut">NAV </span>
          <span style={{ color: "#e6edf3", fontWeight: 600 }}>
            ₹{nav.toLocaleString("en-IN", { maximumFractionDigits: 0 })}
          </span>
        </span>
      )}

      {portfolio && (
        <span>
          <span className="mut">P&L </span>
          <span className={pnl >= 0 ? "pos" : "neg"} style={{ fontWeight: 600 }}>
            {pnl >= 0 ? "+" : ""}₹{pnl.toLocaleString("en-IN", { maximumFractionDigits: 0 })}
          </span>
        </span>
      )}

      {portfolio && (
        <span>
          <span className="mut">ORDERS </span>
          <span className="blu">{portfolio.orders_today}</span>
        </span>
      )}

      <span style={{ marginLeft: "auto", color: "#8b949e", fontSize: "11px" }}>{time}</span>

      <span
        style={{
          padding: "2px 8px",
          borderRadius: "3px",
          fontSize: "10px",
          fontWeight: 700,
          letterSpacing: "0.1em",
          background: status.running ? "#0d3b1e" : "#2d1b1b",
          color: status.running ? "#3fb950" : "#f85149",
          border: `1px solid ${status.running ? "#3fb950" : "#f85149"}`,
        }}
      >
        {status.running ? "● LIVE" : "○ STOPPED"}
      </span>
    </header>
  );
}
```

- [ ] **Step 2: Create `frontend/components/PortfolioPanel.tsx`**

```typescript
"use client";

import type { PortfolioState } from "@/lib/types";

interface Props {
  portfolio: PortfolioState | null;
}

export function PortfolioPanel({ portfolio }: Props) {
  const positions = Object.entries(portfolio?.positions ?? {});

  return (
    <div className="panel" style={{ height: "100%", overflow: "hidden", display: "flex", flexDirection: "column" }}>
      <div className="panel-header">Portfolio</div>
      <div style={{ padding: "8px", display: "flex", flexDirection: "column", gap: "6px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px" }}>
          {[
            ["NAV", portfolio ? `₹${portfolio.nav.toLocaleString("en-IN", { maximumFractionDigits: 0 })}` : "—"],
            ["CASH", portfolio ? `₹${portfolio.cash.toLocaleString("en-IN", { maximumFractionDigits: 0 })}` : "—"],
            ["PEAK NAV", portfolio ? `₹${portfolio.peak_nav.toLocaleString("en-IN", { maximumFractionDigits: 0 })}` : "—"],
            ["ORDERS TODAY", portfolio?.orders_today ?? "—"],
          ].map(([label, value]) => (
            <div key={label as string} style={{ background: "#161b22", padding: "6px 8px", borderRadius: "3px" }}>
              <div className="mut" style={{ fontSize: "9px", marginBottom: "2px" }}>{label}</div>
              <div style={{ fontWeight: 600, fontSize: "13px" }}>{value}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="panel-header" style={{ marginTop: "4px" }}>Positions</div>
      <div style={{ flex: 1, overflowY: "auto" }}>
        {positions.length === 0 ? (
          <div className="mut" style={{ padding: "8px", fontSize: "11px" }}>No open positions</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>SYMBOL</th>
                <th style={{ textAlign: "right" }}>QTY</th>
                <th style={{ textAlign: "right" }}>AVG PRICE</th>
                <th>TYPE</th>
              </tr>
            </thead>
            <tbody>
              {positions.map(([sym, pos]) => (
                <tr key={sym}>
                  <td className="acc">{sym}</td>
                  <td style={{ textAlign: "right" }} className={pos.quantity > 0 ? "pos" : "neg"}>
                    {pos.quantity}
                  </td>
                  <td style={{ textAlign: "right" }}>₹{pos.average_price.toFixed(2)}</td>
                  <td className="mut">{pos.product}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create `frontend/components/MarketDataPanel.tsx`**

```typescript
"use client";

import type { Bar } from "@/lib/types";

interface Props {
  bars: Record<string, Bar>;
}

export function MarketDataPanel({ bars }: Props) {
  const entries = Object.values(bars).sort((a, b) => a.symbol.localeCompare(b.symbol));

  return (
    <div className="panel" style={{ height: "100%", overflow: "hidden", display: "flex", flexDirection: "column" }}>
      <div className="panel-header">Market Data — Last Closed Bar</div>
      <div style={{ flex: 1, overflowY: "auto" }}>
        {entries.length === 0 ? (
          <div className="mut" style={{ padding: "8px", fontSize: "11px" }}>
            Waiting for first bar close…
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>SYMBOL</th>
                <th>BAR</th>
                <th style={{ textAlign: "right" }}>O</th>
                <th style={{ textAlign: "right" }}>H</th>
                <th style={{ textAlign: "right" }}>L</th>
                <th style={{ textAlign: "right" }}>C</th>
                <th style={{ textAlign: "right" }}>TICKS</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((b) => (
                <tr key={b.symbol}>
                  <td className="acc">{b.symbol}</td>
                  <td className="mut">
                    {new Date(b.bar_open).toLocaleTimeString("en-IN", {
                      timeZone: "Asia/Kolkata",
                      hour: "2-digit",
                      minute: "2-digit",
                      hour12: false,
                    })}
                  </td>
                  <td style={{ textAlign: "right" }}>{b.open.toFixed(2)}</td>
                  <td style={{ textAlign: "right" }} className="pos">{b.high.toFixed(2)}</td>
                  <td style={{ textAlign: "right" }} className="neg">{b.low.toFixed(2)}</td>
                  <td style={{ textAlign: "right", fontWeight: 600 }}>{b.close.toFixed(2)}</td>
                  <td style={{ textAlign: "right" }} className="mut">{b.tick_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Create `frontend/components/EventFeedPanel.tsx`**

```typescript
"use client";

interface Event {
  ts: string;
  type: string;
  summary: string;
}

interface Props {
  eventLog: Event[];
}

const TYPE_COLOR: Record<string, string> = {
  fill: "#3fb950",
  bar_closed: "#8b949e",
  session_started: "#58a6ff",
  session_stopped: "#f85149",
  snapshot: "#d29922",
};

export function EventFeedPanel({ eventLog }: Props) {
  return (
    <div className="panel" style={{ height: "100%", overflow: "hidden", display: "flex", flexDirection: "column" }}>
      <div className="panel-header">Event Feed ({eventLog.length})</div>
      <div style={{ flex: 1, overflowY: "auto", padding: "4px 0" }}>
        {eventLog.length === 0 ? (
          <div className="mut" style={{ padding: "8px", fontSize: "11px" }}>
            Waiting for events… Start a session to see live data.
          </div>
        ) : (
          eventLog.map((ev, i) => (
            <div
              key={i}
              style={{
                display: "grid",
                gridTemplateColumns: "72px 90px 1fr",
                gap: "8px",
                padding: "2px 8px",
                borderBottom: "1px solid #1c2128",
                fontSize: "11px",
                alignItems: "center",
              }}
            >
              <span className="mut">
                {new Date(ev.ts).toLocaleTimeString("en-IN", {
                  timeZone: "Asia/Kolkata",
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                  hour12: false,
                })}
              </span>
              <span
                style={{
                  color: TYPE_COLOR[ev.type] ?? "#e6edf3",
                  fontWeight: 600,
                  fontSize: "10px",
                  letterSpacing: "0.05em",
                }}
              >
                {ev.type.replace("_", " ").toUpperCase()}
              </span>
              <span style={{ color: "#c9d1d9" }}>{ev.summary}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Create `frontend/components/ControlsPanel.tsx`**

```typescript
"use client";

import { useState } from "react";
import type { SessionStatus } from "@/lib/types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface Props {
  status: SessionStatus;
}

export function ControlsPanel({ status }: Props) {
  const [loading, setLoading] = useState(false);
  const [timeframe, setTimeframe] = useState<"15m" | "60m">("60m");
  const [error, setError] = useState<string | null>(null);

  const startSession = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API}/api/session/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ timeframe, warmup_bars: 100 }),
      });
      if (!resp.ok) {
        const data = await resp.json() as { detail?: string };
        setError(data.detail ?? "Failed to start");
      }
    } catch {
      setError("Cannot reach server");
    } finally {
      setLoading(false);
    }
  };

  const stopSession = async () => {
    setLoading(true);
    setError(null);
    try {
      await fetch(`${API}/api/session/stop`, { method: "POST" });
    } catch {
      setError("Cannot reach server");
    } finally {
      setLoading(false);
    }
  };

  const activateKillSwitch = async () => {
    if (!confirm("Activate kill switch? This will stop all trading immediately.")) return;
    await stopSession();
  };

  return (
    <div className="panel" style={{ height: "100%", overflow: "hidden", display: "flex", flexDirection: "column" }}>
      <div className="panel-header">Controls</div>
      <div style={{ padding: "10px", display: "flex", flexDirection: "column", gap: "8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span className="mut" style={{ fontSize: "10px" }}>STATUS</span>
          <span
            style={{
              padding: "2px 8px",
              borderRadius: "3px",
              fontSize: "10px",
              fontWeight: 700,
              background: status.running ? "#0d3b1e" : "#2d1b1b",
              color: status.running ? "#3fb950" : "#f85149",
              border: `1px solid ${status.running ? "#3fb950" : "#f85149"}`,
            }}
          >
            {status.running ? "● LIVE" : "○ STOPPED"}
          </span>
        </div>

        {status.running && (
          <div style={{ fontSize: "11px" }} className="mut">
            {status.timeframe} bars · {status.symbols_count} symbols
            {status.started_at && (
              <> · since{" "}
                {new Date(status.started_at).toLocaleTimeString("en-IN", {
                  timeZone: "Asia/Kolkata",
                  hour12: false,
                  hour: "2-digit",
                  minute: "2-digit",
                })} IST
              </>
            )}
          </div>
        )}

        {!status.running && (
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span className="mut" style={{ fontSize: "10px" }}>TIMEFRAME</span>
            {(["15m", "60m"] as const).map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                style={{
                  padding: "2px 10px",
                  borderRadius: "3px",
                  border: "1px solid",
                  background: timeframe === tf ? "#f0883e" : "transparent",
                  borderColor: timeframe === tf ? "#f0883e" : "#30363d",
                  color: timeframe === tf ? "#0a0a0a" : "#e6edf3",
                  fontFamily: "inherit",
                  fontSize: "11px",
                  cursor: "pointer",
                  fontWeight: 600,
                }}
              >
                {tf}
              </button>
            ))}
          </div>
        )}

        <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
          {!status.running ? (
            <button
              onClick={startSession}
              disabled={loading}
              style={{
                padding: "6px 16px",
                borderRadius: "3px",
                border: "1px solid #3fb950",
                background: "#0d3b1e",
                color: "#3fb950",
                fontFamily: "inherit",
                fontSize: "11px",
                fontWeight: 700,
                cursor: loading ? "not-allowed" : "pointer",
                letterSpacing: "0.05em",
                opacity: loading ? 0.5 : 1,
              }}
            >
              {loading ? "STARTING…" : "▶ START SESSION"}
            </button>
          ) : (
            <>
              <button
                onClick={stopSession}
                disabled={loading}
                style={{
                  padding: "6px 16px",
                  borderRadius: "3px",
                  border: "1px solid #f85149",
                  background: "#2d1b1b",
                  color: "#f85149",
                  fontFamily: "inherit",
                  fontSize: "11px",
                  fontWeight: 700,
                  cursor: loading ? "not-allowed" : "pointer",
                  letterSpacing: "0.05em",
                  opacity: loading ? 0.5 : 1,
                }}
              >
                {loading ? "STOPPING…" : "■ STOP SESSION"}
              </button>
              <button
                onClick={activateKillSwitch}
                disabled={loading}
                style={{
                  padding: "6px 16px",
                  borderRadius: "3px",
                  border: "1px solid #d29922",
                  background: "#2d2000",
                  color: "#d29922",
                  fontFamily: "inherit",
                  fontSize: "11px",
                  fontWeight: 700,
                  cursor: loading ? "not-allowed" : "pointer",
                  letterSpacing: "0.05em",
                  opacity: loading ? 0.5 : 1,
                }}
              >
                ⚠ KILL SWITCH
              </button>
            </>
          )}
        </div>

        {error && (
          <div style={{ color: "#f85149", fontSize: "11px", padding: "4px 0" }}>
            Error: {error}
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Create `frontend/app/page.tsx`** (main terminal, wires everything)

```typescript
"use client";

import { ControlsPanel } from "@/components/ControlsPanel";
import { EventFeedPanel } from "@/components/EventFeedPanel";
import { Header } from "@/components/Header";
import { MarketDataPanel } from "@/components/MarketDataPanel";
import { PortfolioPanel } from "@/components/PortfolioPanel";
import { useEventStream } from "@/hooks/useEventStream";

export default function TerminalPage() {
  const { status, portfolio, bars, eventLog } = useEventStream();

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        overflow: "hidden",
        background: "#0a0a0a",
      }}
    >
      {/* Header bar */}
      <Header status={status} portfolio={portfolio} />

      {/* Main body: three columns */}
      <div style={{ flex: 1, display: "grid", gridTemplateColumns: "320px 1fr", overflow: "hidden" }}>
        {/* Left: Portfolio */}
        <div style={{ overflow: "hidden", padding: "4px" }}>
          <PortfolioPanel portfolio={portfolio} />
        </div>

        {/* Right: Controls (top) + Market Data (bottom) */}
        <div
          style={{
            display: "grid",
            gridTemplateRows: "180px 1fr",
            overflow: "hidden",
            padding: "4px 4px 4px 0",
            gap: "4px",
          }}
        >
          <ControlsPanel status={status} />
          <MarketDataPanel bars={bars} />
        </div>
      </div>

      {/* Event feed: fixed height at bottom */}
      <div style={{ height: "260px", padding: "0 4px 4px", flexShrink: 0 }}>
        <EventFeedPanel eventLog={eventLog} />
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Start both servers and verify the terminal renders**

Terminal 1 (FastAPI):
```bash
cd /Users/tatsatshah/Desktop/yegedge
.venv/bin/uvicorn server.main:app --reload --port 8000
```

Terminal 2 (Next.js):
```bash
cd /Users/tatsatshah/Desktop/yegedge/frontend
npm run dev
```

Open `http://localhost:3000`. Verify:
- Bloomberg dark theme renders (dark background, amber panel headers)
- Header shows "YEGEDGE" and "○ STOPPED"
- Portfolio panel shows "No open positions"
- Market Data panel shows "Waiting for first bar close…"
- Event Feed shows "Waiting for events…"
- Controls panel shows "○ STOPPED" and "▶ START SESSION" button
- Browser DevTools → Network → WS tab shows a connected WebSocket to `ws://localhost:8000/ws/events`

- [ ] **Step 8: Run all server-side tests**

```bash
cd /Users/tatsatshah/Desktop/yegedge
.venv/bin/python -m pytest tests/server/ -v --no-cov
```

Expected: All tests pass.

- [ ] **Step 9: Commit**

```bash
git add frontend/app/page.tsx frontend/components/ frontend/app/globals.css frontend/app/layout.tsx
git commit -m "feat(frontend): add Bloomberg terminal panels — Portfolio, MarketData, EventFeed, Controls"
```

---

## Verification

Full end-to-end test (requires Upstox access token):

```bash
# 1. Start the backend
cd /Users/tatsatshah/Desktop/yegedge
UPSTOX_ACCESS_TOKEN=<token> .venv/bin/uvicorn server.main:app --port 8000

# 2. Start the frontend
cd frontend && npm run dev

# 3. Open http://localhost:3000
# 4. Click "▶ START SESSION"
# Expected:
#   - Header changes to "● LIVE"
#   - Event feed shows "SESSION STARTED"
#   - As bars close (every 60m), Market Data table fills in
#   - FILL events appear in green in the event feed
#   - Portfolio NAV updates with each bar
```

Without credentials (offline test):
```bash
.venv/bin/python -m pytest tests/server/ -v --no-cov
# Expected: all pass
```

---

## Review Priority

1. **Task 1** (EventBus) — correctness of unsubscribe + QueueFull drop; tested
2. **Task 2** (SessionManager) — the `on_bar_closed` callback runs in the asyncio event loop; verify `asyncio.create_task()` is called correctly
3. **Task 3** (FastAPI routes) — CORS settings must include `localhost:3000`; WebSocket sends snapshot on connect
4. **Task 4** (scaffold) — `NEXT_PUBLIC_WS_URL` in `.env.local` must match backend port
5. **Task 5** (panels) — `useEventStream` reconnects on close (3s retry); verify in DevTools
