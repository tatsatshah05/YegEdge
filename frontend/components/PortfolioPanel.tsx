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
