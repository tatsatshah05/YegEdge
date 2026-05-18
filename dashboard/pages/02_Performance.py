from __future__ import annotations

import plotly.express as px
import streamlit as st

from config.settings import AppSettings
from dashboard.data import compute_equity_stats, load_pnl_history

st.set_page_config(page_title="Performance — YegEdge", layout="wide")
st.title("Performance")

settings = AppSettings()
pnl_df = load_pnl_history(settings.journal_db_path)

if len(pnl_df) == 0:
    st.warning("No session P&L data yet. Run `python -m agent run-paper` to start a session.")
    st.stop()

stats = compute_equity_stats(pnl_df)

# Headline stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Sessions", int(stats["total_sessions"]))
with col2:
    st.metric("Total P&L", f"₹{stats['total_pnl']:+,.0f}")
with col3:
    st.metric("Win Rate", f"{stats['win_rate']:.0%}")

st.divider()

pnl_pd = pnl_df.to_pandas()

# Equity curve
fig_nav = px.line(
    pnl_pd,
    x="session_date",
    y="final_nav",
    title="Equity Curve — NAV per Session",
    labels={"final_nav": "NAV (₹)", "session_date": "Session Date"},
    markers=True,
)
st.plotly_chart(fig_nav, use_container_width=True)

# Daily P&L bars
fig_pnl = px.bar(
    pnl_pd,
    x="session_date",
    y="daily_pnl",
    title="Daily P&L per Session",
    labels={"daily_pnl": "P&L (₹)", "session_date": "Session Date"},
    color="daily_pnl",
    color_continuous_scale=["#d62728", "#ffffff", "#2ca02c"],
    color_continuous_midpoint=0,
)
st.plotly_chart(fig_pnl, use_container_width=True)

# Raw data
with st.expander("Raw session data"):
    st.dataframe(pnl_pd, use_container_width=True)
