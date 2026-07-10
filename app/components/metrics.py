"""
Metric cards.
"""

from __future__ import annotations

import streamlit as st


def render_metrics(result):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Credit Score",
            result.score,
        )

    with col2:
        st.metric(
            "Probability of Default",
            f"{result.probability:.2%}",
        )

    with col3:
        st.metric(
            "Risk Band",
            result.risk_band,
        )

    with col4:
        st.metric(
            "Decision",
            result.decision,
        )