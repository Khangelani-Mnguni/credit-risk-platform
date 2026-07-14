"""
main.py

Batch-Processing Entry Point for the Credit Risk Scorecard Platform.
Allows users to upload a CSV/Excel/TXT file of applicants and generate scorecard decisions.
"""

from __future__ import annotations

import os
import math
import random
import streamlit as st
import pandas as pd

from config import (
    APP_NAME,
    APP_ICON,
    LAYOUT,
    DEFAULT_VALUES
)
from services import (
    load_artifacts,
    PredictionService,
)
from models.applicant import Applicant
from pydantic import ValidationError

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
# INITIALIZE SERVICES & STATE
# =============================================================================

@st.cache_resource
def get_prediction_service():
    """Caches the model loading so it doesn't reload on every UI click."""
    try:
        artifacts = load_artifacts()
        return PredictionService(artifacts)
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

predictor = get_prediction_service()

# Initialize a random seed in session state to prevent the data from 
# shuffling every time the user interacts with a UI widget (like a filter)
if "sample_seed" not in st.session_state:
    st.session_state.sample_seed = 42

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def process_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Processes a dataframe of applicants through the prediction pipeline."""
    results = []
    
    # 0. Ensure revol_util exists (the raw dataset uses revol_util_pct)
    if 'revol_util' not in df.columns and 'revol_util_pct' in df.columns:
        df['revol_util'] = df['revol_util_pct']
    
    for idx, row in df.iterrows():
        app_dict = row.to_dict()
        
        # 1. Fill missing required dummy fields with defaults from config to avoid pipeline crashes
        for key, default_val in DEFAULT_VALUES.items():
            if key not in app_dict or pd.isna(app_dict[key]):
                app_dict[key] = default_val

        # --- CRITICAL FIX: Safe float casting ---
        # Ensures that if a CSV has a string like "50.0", it becomes a real number
        def safe_float(val, fallback):
            try:
                return float(val)
            except (ValueError, TypeError):
                return fallback

        # 2. Dynamically calculate installment_income_ratio if missing
        if 'installment_income_ratio' not in app_dict or pd.isna(app_dict['installment_income_ratio']):
            annual_inc = safe_float(app_dict.get('annual_inc', 0), 0)
            installment = safe_float(app_dict.get('installment', 0), 0)
            app_dict['installment_income_ratio'] = (installment / annual_inc) if annual_inc > 0 else 0.0
            
        # Extract numerical values safely
        dti_val = safe_float(app_dict.get('dti', 20.0), 20.0)
        all_util_val = safe_float(app_dict.get('all_util', 50.0), 50.0)
        bc_util_val = safe_float(app_dict.get('bc_util', 50.0), 50.0)
        revol_util_val = safe_float(app_dict.get('revol_util', 50.0), 50.0)
        emp_length_val = safe_float(app_dict.get('emp_length_years', 5), 5)
            
        # 3. Apply Boundary Clamps (using the safe float values!)
        app_dict['dti'] = min(max(dti_val, 0.0), 100.0)
        app_dict['all_util'] = min(max(all_util_val, 0.0), 200.0)
        app_dict['bc_util'] = min(max(bc_util_val, 0.0), 200.0)
        app_dict['revol_util'] = min(max(revol_util_val, 0.0), 200.0)
        app_dict['emp_length_years'] = int(min(max(emp_length_val, 0), 50))
        
        # 4. Final NaN sweep (Pydantic crashes if it encounters a NaN float for a required field)
        for k, v in app_dict.items():
            if isinstance(v, float) and math.isnan(v):
                app_dict[k] = 0

        # 5. Predict
        try:
            applicant = Applicant(**app_dict)
            pred = predictor.predict(applicant)
            
            app_dict['Probability of Default'] = f"{pred.probability:.2%}"
            app_dict['Credit Score'] = pred.score
            app_dict['Risk Band'] = pred.risk_band
            app_dict['Decision'] = pred.decision
            
        except ValidationError as e:
            failed_field = str(e).split('\n')[1] if '\n' in str(e) else "Unknown"
            app_dict['Probability of Default'] = "Error"
            app_dict['Credit Score'] = "Error"
            app_dict['Risk Band'] = "Invalid Data"
            app_dict['Decision'] = f"Validation Failed: {failed_field}"
            
        results.append(app_dict)
        
    return pd.DataFrame(results)


def create_executive_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts the most important columns for the executive summary."""
    columns_to_show = [
        'id', 'fico_score', 'annual_inc', 'tot_cur_bal', 'home_ownership', 
        'Credit Score', 'Probability of Default', 'Risk Band', 'Decision'
    ]
    # Only keep columns that actually exist in the dataframe
    existing_cols = [c for c in columns_to_show if c in df.columns]
    
    summary = df[existing_cols].copy()
    
    # Rename columns for presentation
    summary.rename(columns={
        'id': 'Applicant ID',
        'fico_score': 'FICO',
        'annual_inc': 'Income ($)',
        'tot_cur_bal': 'Total Balance ($)',
        'home_ownership': 'Home Ownership'
    }, inplace=True)
    
    return summary


def create_final_report(row: pd.Series) -> str:
    """Generates a Markdown executive report for a specific customer."""
    
    # Format the header based on the risk band
    band = row.get('Risk Band', 'Unknown')
    if band == "Low":
        tier_color = "🟢"
    elif band == "Medium":
        tier_color = "🟡"
    else:
        tier_color = "🔴"
        
    return f"""
### {tier_color} Applicant ID: {row.get('id', 'N/A')}
**Decision:** {row.get('Decision', 'N/A')}  |  **Probability of Default:** {row.get('Probability of Default', 'N/A')}  |  **Credit Score:** {row.get('Credit Score', 'N/A')}

| **Financial Profile** | **Details** | **Credit Behavior** | **Details** |
|-----------------------|-------------|---------------------|-------------|
| **FICO Score** | {row.get('fico_score', 'N/A')} | **Total Bankcard Limit** | ${safe_float(row.get('total_bc_limit', 0), 0):,.2f} |
| **Annual Income** | ${safe_float(row.get('annual_inc', 0), 0):,.2f} | **Bankcard Utilization** | {row.get('bc_util', 0)}% |
| **Proposed Installment** | ${safe_float(row.get('installment', 0), 0):,.2f} | **Total Credit Utilization** | {row.get('all_util', 0)}% |
| **Debt-to-Income (DTI)** | {row.get('dti', 0)}% | **Inquiries (Last 12m)** | {row.get('inq_last_12m', 0)} |
    """

# Utility function for the report block below
def safe_float(val, fallback):
    try:
        return float(val)
    except (ValueError, TypeError):
        return fallback

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("💳 Batch Credit Evaluation")

# Updated file_uploader to accept txt files
uploaded_file = st.sidebar.file_uploader("Upload Applicants (CSV, TXT, Excel)", type=["csv", "txt", "xlsx"])

st.sidebar.markdown("""
### User Guide:

This platform allows you to evaluate multiple loan applicants at once using the Logistic Scorecard Model.

1. **Upload a File**:
   - Upload a CSV, TXT (comma or semicolon separated), or Excel file containing applicant data.
   - If you don't upload a file, the app will auto-load your local test dataset.

2. **Select Sample Size**:
   - Choose how many applicants to score to prevent memory limits, and randomize the selection at any time.

3. **Automated Predictions**:
   - The app instantly runs the pipeline, calculating the Probability of Default (PD), Credit Score, Risk Band, and Final Decision.

4. **Executive Summary**:
   - Filter applicants by risk band to quickly isolate high or low-risk decisions.

5. **Individual Customer Report**:
   - Use the row index selector at the bottom to generate a detailed, printable Markdown report for any specific applicant in the batch.
""")

# =============================================================================
# MAIN UI
# =============================================================================

st.title("Predicting Credit Risk & Loan Approvals")

# 1. Load Data Securely
raw_data = None

if uploaded_file is not None:
    try:
        # Treat both CSV and TXT files with python engine to detect delimiters and skip bad lines
        if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
            raw_data = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
        else:
            raw_data = pd.read_excel(uploaded_file)
        st.success(f"Successfully loaded {len(raw_data):,} applicants from {uploaded_file.name}")
    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file is completely empty.")
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")

if raw_data is None or len(raw_data) == 0:
    # Attempt to automatically find the 'test_applicants.csv' we generated earlier
    app_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(app_dir)
    
    file_in_root = os.path.join(project_root, "test_applicants.csv")
    file_in_app = os.path.join(app_dir, "test_applicants.csv")
    
    try:
        # Only read the file if it actually exists AND has more than 0 bytes
        if os.path.exists(file_in_root) and os.path.getsize(file_in_root) > 0:
            raw_data = pd.read_csv(file_in_root, sep=None, engine='python', on_bad_lines='skip')
            st.info(f"No file uploaded. Automatically loaded local test data ({len(raw_data):,} applicants).")
        elif os.path.exists(file_in_app) and os.path.getsize(file_in_app) > 0:
            raw_data = pd.read_csv(file_in_app, sep=None, engine='python', on_bad_lines='skip')
            st.info(f"No file uploaded. Automatically loaded local test data ({len(raw_data):,} applicants).")
    except pd.errors.EmptyDataError:
        st.warning("⚠️ The local 'test_applicants.csv' file is empty or corrupted.")
    except Exception as e:
        st.warning(f"⚠️ Could not load local test data: {e}")

# Final Fallback - If all files are empty, bad, or missing, inject safe dummy data
if raw_data is None or len(raw_data) == 0:
    st.info("Using default fallback sample data to display application features.")
    raw_data = pd.DataFrame([
        {**DEFAULT_VALUES, "id": 10001, "fico_score": 780, "annual_inc": 120000, "installment": 400, "dti": 12.5, "all_util": 15.0, "bc_util": 10.0, "total_bc_limit": 55000, "inq_last_12m": 0},
        {**DEFAULT_VALUES, "id": 10002, "fico_score": 670, "annual_inc": 65000, "installment": 600, "dti": 25.0, "all_util": 45.0, "bc_util": 50.0, "total_bc_limit": 15000, "inq_last_12m": 2},
        {**DEFAULT_VALUES, "id": 10003, "fico_score": 580, "annual_inc": 45000, "installment": 800, "dti": 40.0, "all_util": 85.0, "bc_util": 90.0, "total_bc_limit": 5000, "inq_last_12m": 6},
    ])

# =============================================================================
# DATA SAMPLING & RANDOMIZATION UI
# =============================================================================
st.divider()
st.subheader("⚙️ Data Sampling & Randomization")
st.info("💡 **Recommendation:** We recommend evaluating **100 applicants or fewer** at a time to prevent browser memory limits (MessageSizeError) and ensure fast scorecard processing.")

col_samp1, col_samp2 = st.columns([1, 1])

with col_samp1:
    sample_size = st.number_input(
        "Number of applicants to evaluate:", 
        min_value=1, 
        max_value=len(raw_data), 
        value=min(100, len(raw_data)), 
        step=1
    )

with col_samp2:
    # Use HTML to vertically align the button with the input field
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    if st.button("🔀 Randomize Selection", use_container_width=True):
        st.session_state.sample_seed = random.randint(0, 1000000)

# Apply the sampling (using the locked session seed)
raw_data = raw_data.sample(n=sample_size, random_state=st.session_state.sample_seed).reset_index(drop=True)

# 2. Display Raw Data
st.subheader("Raw Applicant Data")
st.dataframe(raw_data, use_container_width=True)

# 3. Process Predictions
with st.spinner("Scoring applicants via ML Pipeline..."):
    results_df = process_batch(raw_data)

st.divider()

# 4. Display Full Results
st.subheader("Results: Full Evaluation Output")
st.dataframe(results_df, use_container_width=True)

st.divider()

# =============================================================================
# FILTERED EXECUTIVE SUMMARY & INDIVIDUAL REPORT
# =============================================================================

st.subheader("Executive Summary")

# --- NEW RISK FILTER BUTTONS ---
risk_filter = st.radio(
    "Filter applicants by Risk Band:",
    options=["All", "Low", "Medium", "High", "Invalid Data"],
    horizontal=True
)

# Filter the dataframe based on the button selection
if risk_filter != "All":
    filtered_df = results_df[results_df['Risk Band'] == risk_filter].reset_index(drop=True)
else:
    filtered_df = results_df.reset_index(drop=True)

# 5. Display the filtered Executive Summary
executive_summary = create_executive_summary(filtered_df)
st.dataframe(executive_summary, use_container_width=True)

st.divider()

# 6. Final Report Generator
st.subheader("Individual Applicant Report")

# Safety check in case a filter returns 0 results
if len(filtered_df) == 0:
    st.info(f"No applicants found in the '{risk_filter}' risk band.")
else:
    st.write(f"Enter the row index (0 to {len(filtered_df) - 1}) to view a detailed breakdown for a specific applicant in this view.")
    
    max_index = len(filtered_df) - 1
    row_index = st.number_input("Row Index:", min_value=0, max_value=max_index, value=0, step=1)

    # Generate and display the markdown report for the selected row
    final_report = create_final_report(filtered_df.iloc[row_index])
    st.markdown(final_report)