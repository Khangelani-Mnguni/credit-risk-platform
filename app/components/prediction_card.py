"""
Prediction summary card.
"""

from __future__ import annotations

import streamlit as st


def render_prediction_card(result):

    if result.risk_band == "Low":

        st.success("✅ Low Risk Applicant")

    elif result.risk_band == "Medium":

        st.warning("⚠️ Medium Risk Applicant")

    else:

        st.error("❌ High Risk Applicant")

    st.markdown("---")

    st.write(f"**Credit Score:** {result.score}")

    st.write(
        f"**Probability of Default:** {result.probability:.2%}"
    )

    st.write(f"**Decision:** {result.decision}")