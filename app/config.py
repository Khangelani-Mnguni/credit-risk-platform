"""
config.py

Central configuration for the Credit Risk Scorecard application.

This module contains:
- Project paths
- Application settings
- Model settings
- Risk thresholds
- Default applicant values

Author: Your Name
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# App directory (credit-risk-platform/app/)
APP_DIR = Path(__file__).resolve().parent

# True project root (credit-risk-platform/) - goes one level up from app/
PROJECT_ROOT = APP_DIR.parent

# Directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ASSETS_DIR = APP_DIR / "assets"
PAGES_DIR = APP_DIR / "pages"
COMPONENTS_DIR = APP_DIR / "components"
SERVICES_DIR = APP_DIR / "services"

# =============================================================================
# MODEL ARTIFACTS
# =============================================================================

PIPELINE_PATH = OUTPUTS_DIR / "preprocessing_pipeline.pkl"
MODEL_PATH = OUTPUTS_DIR / "logistic_model.pkl"
SCALER_PATH = OUTPUTS_DIR / "score_scaler.pkl"
METADATA_PATH = OUTPUTS_DIR / "model_metadata.json"
BENCHMARK_PATH = OUTPUTS_DIR / "model_comparison_summary.csv"
SCORECARD_PATH = OUTPUTS_DIR / "final_scorecard.xlsx"

# =============================================================================
# STREAMLIT SETTINGS
# =============================================================================

APP_NAME = "Credit Risk Scorecard"
APP_ICON = "🏦"
LAYOUT = "wide"

# =============================================================================
# MODEL SETTINGS
# =============================================================================

DEFAULT_THRESHOLD = 0.50

# =============================================================================
# RISK BANDS
# =============================================================================

LOW_RISK_MAX = 0.10
MEDIUM_RISK_MAX = 0.20

EXCELLENT_SCORE = 750
GOOD_SCORE = 700
FAIR_SCORE = 650
POOR_SCORE = 600

# =============================================================================
# DEFAULT APPLICANT VALUES (ALL 50 TRAINING FEATURES)
# =============================================================================

DEFAULT_VALUES = {
    # --- Identifiers & Dates ---
    "id": "99999999",
    "issue_date": "2015-01-01",
    "earliest_credit_line_date": "2000-01-01",
    "term": " 36 months",  # Usually in the data but not in your schema list; keeping safe

    # --- Core Loan Details ---
    "loan_amnt": 15000.0,
    "funded_amnt": 15000.0,
    "funded_amnt_inv": 15000.0,
    "int_rate_pct": "10.5",  # Object type in your schema
    "installment": 450.0,
    "grade": "C",
    "sub_grade": "C1",
    "purpose": "debt_consolidation",
    "application_type": "Individual",

    # --- Borrower Profile ---
    "emp_length_years": 5,
    "annual_inc": 65000.0,
    "home_ownership": "MORTGAGE",
    "dti": 20.0,
    
    # --- Credit Scores ---
    "fico_score": 700.0,
    "last_fico_score": 700.0,
    "fico_change": 0.0,
    "fico_band": "700-749",
    
    # --- Credit History & Derogatory Marks ---
    "delinq_2yrs": 0.0,
    "acc_now_delinq": 0.0,
    "inq_last_6mths": 0.0,
    "inq_last_12m": 1.0,
    "pub_rec": 0.0,
    "mort_acc": 1.0,
    "credit_history_months": 120,

    # --- Accounts & Balances ---
    "open_acc": 10.0,
    "total_acc": 20.0,
    "revol_bal": 15000.0,
    "tot_cur_bal": 100000.0,
    "total_rev_hi_lim": 25000.0,
    "total_bc_limit": 10000.0,

    # --- Utilization Metrics ---
    "revol_util_pct": "50.0",  # Object type in your schema
    "bc_util": 50.0,
    "all_util": 50.0,
    "revolving_balance_ratio": 0.50,
    "balance_to_limit_ratio": 0.50,

    # --- Income Ratios ---
    "loan_to_income_ratio": 0.20,
    "installment_income_ratio": 0.05,
    "funded_to_income_ratio": 0.20,

    # --- Custom Engineered Flags (Int64 types) ---
    "delinquency_flag": 0,
    "recent_inquiry_flag": 0,
    "high_utilization_flag": 0,
    "thin_file_flag": 0,
    "mortgage_flag": 1,
    "employment_stability_flag": 1,
    "high_dti_flag": 0,
    "subprime_flag": 0,
    "default_flag": 0,  # Target variable (just in case the pipeline strictly validates X shape)
}

# =============================================================================
# UI OPTIONS
# =============================================================================

TERM_OPTIONS = [
    " 36 months",
    " 60 months",
]

HOME_OWNERSHIP_OPTIONS = [
    "MORTGAGE",
    "RENT",
    "OWN",
    "ANY",
]

PURPOSE_OPTIONS = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "small_business",
    "other",
]

GRADE_OPTIONS = list("ABCDEFG")

SUBGRADE_OPTIONS = [
    f"{grade}{number}"
    for grade in "ABCDEFG"
    for number in range(1, 6)
]

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

SCORE_DECIMALS = 0
PROBABILITY_DECIMALS = 2

# =============================================================================
# CACHE SETTINGS
# =============================================================================

CACHE_TTL = None

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"

# =============================================================================
# VALIDATION LIMITS
# =============================================================================

VALIDATION_LIMITS = {
    "annual_inc": (0, 10_000_000),
    "dti": (0.0, 100.0),
    "emp_length_years": (0, 50),
    "revol_util": (0.0, 200.0),
    "tot_cur_bal": (0, 100_000_000),
    "credit_history_months": (0, 800),
    "inq_last_6mths": (0, 50),
}