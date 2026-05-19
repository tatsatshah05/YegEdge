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
