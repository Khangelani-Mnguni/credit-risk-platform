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
        extra="ignore",
        frozen=False,
    )

    # ------------------------------------------------------------------
    # Categorical Features
    # ------------------------------------------------------------------

    term: Literal[" 36 months", " 60 months"]

    home_ownership: Literal[
        "MORTGAGE",
        "RENT",
        "OWN",
        "ANY",
    ]

    purpose: str

    grade: Literal[
        "A", "B", "C", "D", "E", "F", "G"
    ]

    sub_grade: str

    # ------------------------------------------------------------------
    # Numerical Features
    # ------------------------------------------------------------------

    annual_inc: float = Field(..., ge=0)

    dti: float = Field(..., ge=0, le=100)

    emp_length_years: int = Field(..., ge=0, le=50)

    revol_util: float = Field(..., ge=0, le=200)

    tot_cur_bal: float = Field(..., ge=0)

    credit_history_months: int = Field(..., ge=0)

    inq_last_6mths: int = Field(..., ge=0)

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