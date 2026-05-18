from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

st.set_page_config(page_title="Risk — YegEdge", layout="wide")
st.title("Risk Rules Configuration")
st.caption("Read-only view of config/risk_rules.yaml. Edit the file directly to change limits.")

rules_path = Path("config/risk_rules.yaml")
if not rules_path.exists():
    st.error("config/risk_rules.yaml not found. Cannot display risk rules.")
    st.stop()

with rules_path.open() as f:
    rules: dict = yaml.safe_load(f)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Per-Trade Limits")
    pt = rules.get("per_trade", {})
    st.metric("Max Risk per Trade", f"{pt.get('max_risk_fraction', 0) * 100:.2f}% of NAV")
    st.metric("Max Position Size", f"{pt.get('max_position_fraction', 0) * 100:.0f}% of NAV")
    st.metric("Min Reward : Risk", f"{pt.get('min_reward_risk', 0):.1f}x")

    st.subheader("Portfolio Limits")
    pf = rules.get("portfolio", {})
    st.metric("Max Concurrent Positions", pf.get("max_concurrent_positions", "—"))
    st.metric("Max Sector Exposure", f"{pf.get('max_sector_exposure', 0) * 100:.0f}% of NAV")
    st.metric("Min Cash Buffer", f"{pf.get('min_cash_fraction', 0) * 100:.0f}%")

    st.subheader("Order Frequency")
    freq = rules.get("frequency", {})
    st.metric("Max Orders / Day", freq.get("max_new_orders_per_day", "—"))
    st.metric("Symbol Cooldown", f"{freq.get('symbol_cooldown_minutes', 0)} min")

with col2:
    st.subheader("Loss Caps")
    caps = rules.get("loss_caps", {})
    st.metric("Daily Loss Cap", f"{caps.get('max_daily_loss_fraction', 0) * 100:.0f}% of NAV")
    st.metric("Weekly Loss Cap", f"{caps.get('max_weekly_loss_fraction', 0) * 100:.0f}% of NAV")
    st.metric(
        "Max Drawdown (Kill Switch)",
        f"{caps.get('max_drawdown_fraction', 0) * 100:.0f}% of NAV",
    )

    st.subheader("Trading Window (IST)")
    win = rules.get("windows", {})
    st.metric("Trade Start", win.get("trade_start_ist", "—"))
    st.metric("Trade End", win.get("trade_end_ist", "—"))
    st.metric("Square-Off By", win.get("square_off_ist", "—"))

    st.subheader("Kill Switch Triggers")
    ks = rules.get("kill_switch", {})
    st.metric("Data Feed Outage", f"{ks.get('data_feed_outage_minutes', '—')} min")
    st.metric("Consecutive Exec Errors", ks.get("consecutive_execution_errors", "—"))
    st.metric("Auto-Reset", str(ks.get("auto_reset", False)))

st.divider()
with st.expander("View full risk_rules.yaml"):
    st.code(rules_path.read_text(), language="yaml")
