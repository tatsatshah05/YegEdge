from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import AppSettings
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://134.209.150.61:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    settings = AppSettings()
    if settings.broker != "yfinance" and not settings.upstox_access_token:
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


@app.websocket("/ws/events")
async def websocket_events(ws: WebSocket) -> None:
    await ws.accept()
    q = _bus.subscribe()
    logger.info("ws.client_connected", subscribers=_bus.subscriber_count)
    try:
        await ws.send_text(
            json.dumps(
                {
                    "type": "snapshot",
                    "ts": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
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
