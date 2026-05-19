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

      <span style={{ color: "#8b949e", fontSize: "10px" }}>&#9658;</span>

      {nav > 0 && (
        <span>
          <span className="mut">NAV </span>
          <span style={{ color: "#e6edf3", fontWeight: 600 }}>
            &#8377;{nav.toLocaleString("en-IN", { maximumFractionDigits: 0 })}
          </span>
        </span>
      )}

      {portfolio && (
        <span>
          <span className="mut">P&amp;L </span>
          <span className={pnl >= 0 ? "pos" : "neg"} style={{ fontWeight: 600 }}>
            {pnl >= 0 ? "+" : ""}&#8377;{pnl.toLocaleString("en-IN", { maximumFractionDigits: 0 })}
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
