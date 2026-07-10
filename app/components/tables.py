"""
Reusable tables.
"""

from __future__ import annotations

import streamlit as st


def render_dataframe(df):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )