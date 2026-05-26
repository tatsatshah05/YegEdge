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
          const d = ev.data as { timeframe?: string; symbols?: string[] };
          next.status = {
            ...prev.status,
            running: true,
            started_at: ev.ts,
            ...(d.timeframe && { timeframe: d.timeframe }),
            ...(d.symbols && { symbols_count: d.symbols.length }),
          };
        }

        if (ev.type === "session_stopped") {
          next.status = { ...prev.status, running: false, started_at: null };
        }

        const summary = formatEvent(ev);
        // "portfolio" events are handled above via early-return; this branch is always true here
        next.eventLog = [
          { ts: ev.ts, type: ev.type, summary },
          ...prev.eventLog.slice(0, 199),
        ];

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
