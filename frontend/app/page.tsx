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

      {/* Main body: two columns */}
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
