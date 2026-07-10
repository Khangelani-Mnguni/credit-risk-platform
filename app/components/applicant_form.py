"""
components/applicant_form.py

Reusable applicant input form.
"""

from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

# Removed unused imports (TERM_OPTIONS, PURPOSE_OPTIONS, etc.)
from config import HOME_OWNERSHIP_OPTIONS

from models.applicant import Applicant


def render_applicant_form() -> Applicant | None:
    """
    Render the loan application form.

    Returns
    -------
    Applicant | None
        Applicant object if submitted successfully,
        otherwise None.
    """

    with st.form(
        key="credit_application_form",
        clear_on_submit=False,
    ):

        st.subheader("Applicant Information")

        col1, col2 = st.columns(2)

        # ==============================================================
        # LEFT COLUMN (Financial Profile)
        # ==============================================================

        with col1:

            fico_score = st.number_input(
                "FICO Score",
                min_value=300.0,
                max_value=850.0,
                value=720.0,
                step=1.0,
            )

            annual_inc = st.number_input(
                "Annual Income ($)",
                min_value=0.0,
                value=65000.0,
                step=1000.0,
                format="%.2f",
            )

            installment = st.number_input(
                "Proposed Monthly Installment ($)",
                min_value=0.0,
                value=450.0,
                step=10.0,
                format="%.2f",
            )

            tot_cur_bal = st.number_input(
                "Total Current Balance ($)",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
                format="%.2f",
            )

            total_bc_limit = st.number_input(
                "Total Bankcard Limit ($)",
                min_value=0.0,
                value=25000.0,
                step=1000.0,
                format="%.2f",
            )

            home_ownership = st.selectbox(
                "Home Ownership",
                HOME_OWNERSHIP_OPTIONS,
            )

        # ==============================================================
        # RIGHT COLUMN (Credit Utilization & Behavior)
        # ==============================================================

        with col2:

            dti = st.slider(
                "Debt-to-Income Ratio (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
            )

            all_util = st.slider(
                "Total Credit Utilization (%)",
                min_value=0.0,
                max_value=200.0,
                value=35.0,
            )

            bc_util = st.slider(
                "Bankcard Utilization (%)",
                min_value=0.0,
                max_value=200.0,
                value=40.0,
            )

            inq_last_6mths = st.number_input(
                "Credit Inquiries (Last 6 Months)",
                min_value=0,
                max_value=50,
                value=0,
            )

            inq_last_12m = st.number_input(
                "Credit Inquiries (Last 12 Months)",
                min_value=0,
                max_value=50,
                value=1,
            )

        submitted = st.form_submit_button(
            "Calculate Credit Score",
            use_container_width=True,
        )

    if not submitted:
        return None

    try:
        # Calculate the derived ratio that the ML pipeline expects
        # Formula: Installment / Annual Income
        calc_inst_inc_ratio = installment / annual_inc if annual_inc > 0 else 0.0

        return Applicant(
            id=99999999,
            
            # ---------------------------------------------------------
            # HIGH-IMPACT PREDICTIVE FIELDS (Captured from UI)
            # ---------------------------------------------------------
            fico_score=fico_score,
            annual_inc=annual_inc,
            installment=installment,
            tot_cur_bal=tot_cur_bal,
            total_bc_limit=total_bc_limit,
            home_ownership=home_ownership,
            dti=dti,
            all_util=all_util,
            bc_util=bc_util,
            inq_last_6mths=inq_last_6mths,
            inq_last_12m=inq_last_12m,
            
            # ---------------------------------------------------------
            # DERIVED RATIO
            # ---------------------------------------------------------
            installment_income_ratio=calc_inst_inc_ratio,
            
            # ---------------------------------------------------------
            # HIDDEN / DUMMY FIELDS (To satisfy Pydantic & Pipeline)
            # ---------------------------------------------------------
            term=" 36 months",
            purpose="debt_consolidation",
            grade="C",
            sub_grade="C1",
            emp_length_years=5,
            revol_util=50.0,
            credit_history_months=120,
        )

    except ValidationError as e:

        st.error("Input validation failed. Please check the values entered.")

        st.exception(e)

        return None