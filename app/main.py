"""
main.py

Main entry point for the Credit Risk Scorecard application.

Author: Your Name
"""

from __future__ import annotations

import streamlit as st

# Removed the "app." prefix since we are already inside the app folder
from config import (
    APP_NAME,
    APP_ICON,
    LAYOUT,
)

from services import (
    load_artifacts,
    PredictionService,
    validate_inputs,
    ModelMetadataService,
)

from components import (
    render_sidebar,
    render_applicant_form,
    render_metrics,
    render_prediction_card,
    render_footer,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

# =============================================================================
# LOAD MODEL ARTIFACTS
# =============================================================================

try:
    artifacts = load_artifacts()

except Exception as e:

    st.error("Failed to load model artifacts.")

    st.exception(e)

    st.stop()

# =============================================================================
# INITIALIZE SERVICES
# =============================================================================

predictor = PredictionService(artifacts)

metadata = ModelMetadataService(
    artifacts.metadata
)

# =============================================================================
# SIDEBAR
# =============================================================================

render_sidebar(artifacts.metadata)

# =============================================================================
# HEADER
# =============================================================================

st.title("Credit Risk Scorecard Evaluation")

st.markdown(
    """
Predict the probability of default (PD) and calculate an
application credit score using the production logistic scorecard.
"""
)

st.divider()

# =============================================================================
# MODEL INFORMATION
# =============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Model",
        metadata.model_name,
    )

with col2:

    st.metric(
        "Version",
        metadata.version,
    )

with col3:

    auc = metadata.auc

    st.metric(
        "ROC AUC",
        "-" if auc is None else f"{auc:.3f}",
    )

with col4:

    ks = metadata.ks

    st.metric(
        "KS Statistic",
        "-" if ks is None else f"{ks:.3f}",
    )

st.divider()

# =============================================================================
# CREDIT APPLICATION FORM
# =============================================================================

applicant = render_applicant_form()

# =============================================================================
# RUN PREDICTION
# =============================================================================

if applicant is not None:
    try:
        validate_inputs(applicant)

        with st.spinner("Scoring applicant..."):
            result = predictor.predict(applicant)
            # Save the result to session state so the Chatbot page can read it!
            st.session_state.current_prediction = result

        st.divider()
        st.subheader("Prediction Results")
        render_metrics(result)
        st.write("")
        render_prediction_card(result)

    except ValueError as e:
        st.warning(str(e))
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

# =============================================================================
# FOOTER
# =============================================================================

render_footer()

