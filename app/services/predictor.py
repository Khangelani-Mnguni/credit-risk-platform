"""
predictor.py

Prediction service used by Streamlit, FastAPI and AI agents.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Removed "app." prefix for relative import safety
from config import (
    DEFAULT_VALUES,
    LOW_RISK_MAX,
    MEDIUM_RISK_MAX,
)


@dataclass
class PredictionResult:

    probability: float

    score: int

    risk_band: str

    decision: str


class PredictionService:

    def __init__(self, artifacts):

        self.pipeline = artifacts.pipeline

        self.model = artifacts.model

        self.scaler = artifacts.scaler

    def predict(self, applicant) -> PredictionResult:

        # 1. Safely convert Pydantic Applicant object to a dictionary
        if hasattr(applicant, "model_dump"):
            app_dict = applicant.model_dump()
        elif hasattr(applicant, "dict"):
            app_dict = applicant.dict()
        else:
            app_dict = dict(applicant)

        # 2. Start with default values and update with applicant data
        row = DEFAULT_VALUES.copy()
        row.update(app_dict)

        # 3. Convert to DataFrame
        X = pd.DataFrame([row])

        # 4. The ULTIMATE Fallback: Loop through the pipeline's exact config
        if hasattr(self.pipeline, "config"):
            # Catch all missing categorical features
            if hasattr(self.pipeline.config, "categorical_features"):
                for col in self.pipeline.config.categorical_features:
                    if col not in X.columns:
                        X[col] = "Missing"
            
            # Catch all missing numerical features (like loan_amnt, id, etc.)
            if hasattr(self.pipeline.config, "numerical_features"):
                for col in self.pipeline.config.numerical_features:
                    if col not in X.columns:
                        X[col] = 0.0

        # 5. Transform and Predict
        X_processed = self.pipeline.transform(X)

        probability = float(self.model.predict_proba(X_processed)[0, 1])

        score = int(
            self.scaler.calculate_score(
                X_processed,
                round_scores=True,
            ).iloc[0]
        )

        # 6. Apply Business Logic
        if probability <= LOW_RISK_MAX:
            risk = "Low"
            decision = "Approve"
        elif probability <= MEDIUM_RISK_MAX:
            risk = "Medium"
            decision = "Review"
        else:
            risk = "High"
            decision = "Decline"

        return PredictionResult(
            probability=probability,
            score=score,
            risk_band=risk,
            decision=decision,
        )