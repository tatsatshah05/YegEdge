from __future__ import annotations

import streamlit as st

from config.settings import AppSettings
from dashboard.data import load_fills, load_rejections

st.set_page_config(page_title="Journal — YegEdge", layout="wide")
st.title("Trade Journal")

settings = AppSettings()

tab_fills, tab_rejections = st.tabs(["✅ Fills", "❌ Rejections"])

with tab_fills:
    fills_df = load_fills(settings.journal_db_path)
    if len(fills_df) == 0:
        st.info("No fills recorded yet. Run a paper session to populate this page.")
    else:
        all_symbols = sorted(fills_df["symbol"].unique().to_list())
        selected_sym = st.selectbox(
            "Filter by symbol", ["All", *all_symbols], key="fills_sym"
        )
        display = (
            fills_df
            if selected_sym == "All"
            else fills_df.filter(fills_df["symbol"] == selected_sym)
        )
        st.dataframe(display.to_pandas(), use_container_width=True)
        st.caption(f"{len(display)} fill(s) shown")

with tab_rejections:
    rej_df = load_rejections(settings.journal_db_path)
    if len(rej_df) == 0:
        st.info("No rejections recorded yet.")
    else:
        all_reasons = sorted(rej_df["reason"].unique().to_list())
        selected_r = st.selectbox(
            "Filter by reason", ["All", *all_reasons], key="rej_reason"
        )
        display_r = rej_df if selected_r == "All" else rej_df.filter(rej_df["reason"] == selected_r)
        st.dataframe(display_r.to_pandas(), use_container_width=True)
        st.caption(f"{len(display_r)} rejection(s) shown")
