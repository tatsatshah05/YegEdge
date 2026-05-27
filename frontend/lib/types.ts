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
  exchange: string;
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
