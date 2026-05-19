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
