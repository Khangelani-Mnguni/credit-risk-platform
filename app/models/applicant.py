"""
models/applicant.py

Pydantic model representing a loan applicant.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class Applicant(BaseModel):
    """
    Loan applicant used throughout the application.

    This object is shared by:
        • Streamlit
        • FastAPI
        • PredictionService
        • LangChain Agent
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",  # CRITICAL: Allows pipeline defaults and new UI fields to pass through safely!
        frozen=False,
    )

    id: int

    # ------------------------------------------------------------------
    # High-Impact Predictive Features (Captured from UI)
    # ------------------------------------------------------------------
    
    fico_score: float = Field(..., ge=300, le=850)
    
    annual_inc: float = Field(..., ge=0)
    
    installment: float = Field(..., ge=0)
    
    tot_cur_bal: float = Field(..., ge=0)
    
    total_bc_limit: float = Field(..., ge=0)

    home_ownership: Literal[
        "MORTGAGE",
        "RENT",
        "OWN",
        "ANY",
    ]

    dti: float = Field(..., ge=0, le=100)
    
    all_util: float = Field(..., ge=0, le=200)
    
    bc_util: float = Field(..., ge=0, le=200)

    inq_last_6mths: int = Field(..., ge=0)
    
    inq_last_12m: int = Field(..., ge=0)

    # ------------------------------------------------------------------
    # Derived Ratios (Calculated in the UI form)
    # ------------------------------------------------------------------
    
    installment_income_ratio: float = Field(..., ge=0)

    # ------------------------------------------------------------------
    # Dummy / Legacy Fields (To satisfy pipeline shape requirements)
    # ------------------------------------------------------------------

    term: str

    purpose: str

    grade: str

    sub_grade: str

    emp_length_years: int = Field(..., ge=0, le=50)

    revol_util: float = Field(..., ge=0, le=200)

    credit_history_months: int = Field(..., ge=0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert applicant to a single-row DataFrame.
        """
        return pd.DataFrame([self.model_dump()])

    def to_dict(self) -> dict:
        """
        Convert applicant to dictionary.
        """
        return self.model_dump()