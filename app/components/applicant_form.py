"""
components/applicant_form.py

Reusable applicant input form.
"""

from __future__ import annotations

import streamlit as st
from pydantic import ValidationError

# Removed "app." prefix
from config import (
    TERM_OPTIONS,
    HOME_OWNERSHIP_OPTIONS,
    PURPOSE_OPTIONS,
    GRADE_OPTIONS,
    SUBGRADE_OPTIONS,
)

# Removed "app." prefix
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
        # LEFT COLUMN
        # ==============================================================

        with col1:

            term = st.selectbox(
                "Loan Term",
                TERM_OPTIONS,
            )

            home_ownership = st.selectbox(
                "Home Ownership",
                HOME_OWNERSHIP_OPTIONS,
            )

            purpose = st.selectbox(
                "Loan Purpose",
                PURPOSE_OPTIONS,
            )

            grade = st.selectbox(
                "Grade",
                GRADE_OPTIONS,
            )

            sub_grade = st.selectbox(
                "Sub Grade",
                SUBGRADE_OPTIONS,
            )

            annual_inc = st.number_input(
                "Annual Income ($)",
                min_value=0.0,
                value=65000.0,
                step=1000.0,
                format="%.2f",
            )

        # ==============================================================
        # RIGHT COLUMN
        # ==============================================================

        with col2:

            dti = st.slider(
                "Debt-to-Income Ratio (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
            )

            emp_length_years = st.slider(
                "Employment Length (Years)",
                min_value=0,
                max_value=50,
                value=5,
            )

            revol_util = st.slider(
                "Revolving Utilization (%)",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
            )

            tot_cur_bal = st.number_input(
                "Total Current Balance ($)",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
                format="%.2f",
            )

            credit_history_months = st.slider(
                "Credit History (Months)",
                min_value=0,
                max_value=600,
                value=120,
            )

            inq_last_6mths = st.number_input(
                "Credit Inquiries (Last 6 Months)",
                min_value=0,
                max_value=50,
                value=0,
            )

        submitted = st.form_submit_button(
            "🔍 Calculate Credit Score",
            use_container_width=True,
        )

    if not submitted:
        return None

    try:

        return Applicant(
            id=99999999,  # Perfect placement!
            term=term,
            home_ownership=home_ownership,
            purpose=purpose,
            grade=grade,
            sub_grade=sub_grade,
            annual_inc=annual_inc,
            dti=dti,
            emp_length_years=emp_length_years,
            revol_util=revol_util,
            tot_cur_bal=tot_cur_bal,
            credit_history_months=credit_history_months,
            inq_last_6mths=inq_last_6mths,
        )

    except ValidationError as e:

        st.error("Input validation failed.")

        st.exception(e)

        return None