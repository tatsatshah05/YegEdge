"use client";

import { useState } from "react";
import type { SessionStatus } from "@/lib/types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface Props {
  status: SessionStatus;
}

export function ControlsPanel({ status }: Props) {
  const [loading, setLoading] = useState(false);
  const [timeframe, setTimeframe] = useState<"5m" | "15m" | "60m">("5m");
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
            {(["5m", "15m", "60m"] as const).map((tf) => (
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
