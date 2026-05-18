# dashboard/app.py
from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import AppSettings
from dashboard.data import compute_equity_stats, load_fills, load_pnl_history, load_session_count

st.set_page_config(
    page_title="YegEdge Dashboard",
    page_icon="📈",
    layout="wide",
)

settings = AppSettings()

st.title("YegEdge — Paper Trading Dashboard")
st.caption(f"Mode: {settings.deployment_env}  |  Live trading: DISABLED by default")

# --- Headline metrics ---
session_count = load_session_count(Path("data/paper_sessions.json"))
pnl_df = load_pnl_history(settings.journal_db_path)
stats = compute_equity_stats(pnl_df)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Paper Sessions", f"{session_count}/60")
with col2:
    nav = stats["current_nav"]
    st.metric("Current NAV", f"₹{nav:,.0f}" if nav > 0 else "—")
with col3:
    pnl = stats["total_pnl"]
    st.metric("Total P&L", f"₹{pnl:+,.0f}" if stats["total_sessions"] > 0 else "—")
with col4:
    wr = stats["win_rate"]
    st.metric("Win Rate", f"{wr:.0%}" if stats["total_sessions"] > 0 else "—")

# --- Live readiness progress ---
st.divider()
progress = min(session_count / 60, 1.0)
st.progress(progress, text=f"{session_count}/60 paper sessions toward live-trading review")
if session_count >= 60:
    st.success(
        "60 sessions complete. Consult `docs/live_readiness.md` before enabling live trading."
    )
else:
    st.info(f"{60 - session_count} more sessions before live-trading eligibility review.")

# --- Recent trades ---
st.divider()
st.subheader("Recent Trades")
fills_df = load_fills(settings.journal_db_path, limit=20)
if len(fills_df) > 0:
    st.dataframe(fills_df.to_pandas(), use_container_width=True)
else:
    st.info("No trades recorded yet. Run `python -m agent run-paper` to start a session.")

# --- Kill switch status ---
ks_path = Path(".kill_switch")
if ks_path.exists():
    st.divider()
    st.error(f"🔴 Kill switch ACTIVE — {ks_path.read_text().strip()}")
