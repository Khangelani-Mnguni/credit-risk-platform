"""
Footer.
"""

from __future__ import annotations

import streamlit as st


def render_footer():

    st.markdown("---")

    st.caption(
        "Credit Risk Scorecard • Built with Streamlit • "
        "Machine Learning • Explainable AI"
    )