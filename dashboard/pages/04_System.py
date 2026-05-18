from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st

from config.settings import AppSettings
from dashboard.data import load_coverage_summary, load_session_count

st.set_page_config(page_title="System — YegEdge", layout="wide")
st.title("System Status")

settings = AppSettings()

# --- Session counter ---
st.subheader("Paper Trading Progress")
session_count = load_session_count(Path("data/paper_sessions.json"))
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sessions Completed", f"{session_count}/60")
with col2:
    st.metric("Live Trading", "DISABLED (default)")
with col3:
    status = (
        "✅ Eligible for review" if session_count >= 60 else f"⏳ {60 - session_count} remaining"
    )
    st.metric("Live Readiness", status)
st.progress(min(session_count / 60, 1.0), text=f"{session_count}/60 paper sessions")

st.divider()

# --- Kill switch ---
st.subheader("Kill Switch")
ks_path = Path(".kill_switch")
if ks_path.exists():
    reason = ks_path.read_text().strip()
    st.error(f"🔴 Kill switch ACTIVE: {reason or '(no reason written)'}")
    if st.button("Deactivate Kill Switch", type="primary"):
        ks_path.unlink()
        st.success("Kill switch deactivated. Restart the trading loop to resume.")
        st.rerun()
else:
    st.success("🟢 Kill switch: inactive")

st.divider()

# --- Data cache coverage ---
st.subheader("Parquet Cache Coverage")
coverage = load_coverage_summary(settings.parquet_cache_dir)
if not coverage:
    st.warning("No cached data found. Run `python -m agent refresh` to populate the cache.")
else:
    rows = []
    for symbol, tfs in coverage.items():
        for tf, (start, end) in tfs.items():
            rows.append(
                {
                    "Symbol": symbol,
                    "Timeframe": tf,
                    "From": str(start.date()),
                    "To": str(end.date()),
                }
            )
    df = pl.DataFrame(rows)
    st.dataframe(df.to_pandas(), use_container_width=True)
    st.caption(f"{len(rows)} symbol/timeframe pair(s) cached")

st.divider()

# --- Configuration summary ---
st.subheader("Configuration")
col_a, col_b = st.columns(2)
with col_a:
    st.metric("Broker", settings.broker)
    st.metric("Paper Starting Capital", f"₹{settings.paper_starting_capital:,.0f}")
    st.metric("Primary LLM", settings.claude_model_primary)
    st.metric("Monthly API Cap", f"₹{settings.max_monthly_api_spend_inr:,.0f}")
with col_b:
    st.metric("Journal DB", str(settings.journal_db_path))
    st.metric("Cache Dir", str(settings.parquet_cache_dir))
    st.metric("Log Dir", str(settings.log_dir))
    st.metric("Telegram", "Configured" if settings.telegram_bot_token else "Not set")
