"""
pages/1_single_applicant.py

Single Applicant Entry Form for the Credit Risk Scorecard Platform.
"""

from __future__ import annotations

import streamlit as st

from config import (
    APP_NAME,
    APP_ICON,
)

from services import (
    load_artifacts,
    PredictionService,
    validate_inputs,
    ModelMetadataService,
)

from components import (
    render_applicant_form,
    render_metrics,
    render_prediction_card,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# Note: set_page_config must be the first Streamlit command
st.set_page_config(
    page_title=f"Single Applicant - {APP_NAME}",
    page_icon=APP_ICON,
    layout="wide",
)

# =============================================================================
# LOAD MODEL ARTIFACTS
# =============================================================================

@st.cache_resource
def get_services():
    try:
        artifacts = load_artifacts()
        predictor = PredictionService(artifacts)
        metadata = ModelMetadataService(artifacts.metadata)
        return predictor, metadata
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

predictor, metadata = get_services()

# =============================================================================
# HEADER
# =============================================================================

st.title("👤 Single Applicant Evaluation")

st.markdown(
    """
Enter individual applicant details below to instantly predict their Probability of Default (PD) 
and calculate their Credit Score using the production logistic scorecard.
"""
)

st.divider()

# =============================================================================
# MODEL INFORMATION
# =============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", metadata.model_name)
with col2:
    st.metric("Version", metadata.version)
with col3:
    auc = metadata.auc
    st.metric("ROC AUC", "-" if auc is None else f"{auc:.3f}")
with col4:
    ks = metadata.ks
    st.metric("KS Statistic", "-" if ks is None else f"{ks:.3f}")

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