"""
Application sidebar.
"""

from __future__ import annotations

import streamlit as st


def render_sidebar(metadata):

    with st.sidebar:

        st.title("🏦 Credit Risk")

        st.markdown("---")

        st.caption("Model")

        st.write(metadata.get("model_name"))

        st.caption("Version")

        st.write(metadata.get("version", "1.0"))

        st.caption("Threshold")

        st.write(metadata.get("threshold", 0.50))

        st.markdown("---")

        st.info(
            "This application predicts the probability of default "
            "using the production scorecard model."
        )