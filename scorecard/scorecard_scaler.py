"""
score_scaler.py

Scorecard scaling engine.

Converts logistic regression probabilities
or coefficients into credit scores.

Implements standard scorecard scaling:
- PDO (Points to Double Odds)
- Base Score
- Base Odds

Author
------
Credit Risk Platform
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScorecardConfig:
    """Configuration parameters for Scorecard scaling."""
    pdo: int = 50
    base_score: int = 600
    base_odds: float = 50.0  # odds of good:bad


class ScoreScaler:
    """
    Converts logistic regression outputs into scorecard points.
    
    Parameters
    ----------
    config : ScorecardConfig, optional
        Configuration containing PDO, Base Score, and Base Odds.
    """

    def __init__(self, config: ScorecardConfig | None = None):
        self.config = config or ScorecardConfig()

    # ---------------------------------------------------------------
    # FIT
    # ---------------------------------------------------------------

    def fit(self, model: Any, feature_names: List[str]) -> "ScoreScaler":
        """
        Extract coefficients and calculate scaling factors.

        Parameters
        ----------
        model : object
            A fitted scikit-learn LogisticRegression model (or similar) 
            exposing `coef_` and `intercept_`.
        feature_names : list of str
            Names of the features in the exact order they were fed to the model.
        """
        if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
            raise ValueError(
                "Model must have 'coef_' and 'intercept_' attributes. "
                "Ensure it is a fitted scikit-learn linear model."
            )

        self.model_ = model
        self.feature_names_ = feature_names

        # Standard scorecard mathematics
        self.factor_ = self.config.pdo / np.log(2)
        self.offset_ = self.config.base_score - (self.factor_ * np.log(self.config.base_odds))

        # Extract weights (handling 2D arrays from sklearn)
        coef_array = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        self.coefficients_ = pd.Series(coef_array, index=feature_names)
        
        intercept_val = model.intercept_[0] if isinstance(model.intercept_, (list, np.ndarray)) else model.intercept_
        self.intercept_ = float(intercept_val)

        logger.info(
            "Score scaler fitted. Factor: %.4f, Offset: %.4f", 
            self.factor_, 
            self.offset_
        )

        return self

    # ---------------------------------------------------------------
    # SCORE CALCULATION
    # ---------------------------------------------------------------

    def calculate_score(self, X: pd.DataFrame, round_scores: bool = True) -> pd.Series:
        """
        Transform features into final credit scores.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data (must be WoE transformed).
        round_scores : bool, default=True
            Whether to round the output to the nearest integer.

        Returns
        -------
        pandas.Series
            Calculated credit scores.
        """
        self._check_fitted()
        
        if (missing_cols := [col for col in self.feature_names_ if col not in X.columns]):
            raise ValueError(f"Input DataFrame is missing required features: {missing_cols}")

        # Use Pandas .dot() for safe index/column alignment
        log_odds = self.intercept_ + X[self.feature_names_].dot(self.coefficients_)
        
        # Standard formulation: Score = Offset - Factor * ln(Odds)
        scores = self.offset_ - (self.factor_ * log_odds)

        if round_scores:
            scores = scores.round().astype(int)

        return pd.Series(scores, index=X.index, name="Score")

    # ---------------------------------------------------------------
    # PROBABILITY TO SCORE
    # ---------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Return the raw probability of the default/bad event.
        """
        self._check_fitted()
        
        # Assumes target '1' is the event of interest (Default/Bad)
        probs = self.model_.predict_proba(X[self.feature_names_])[:, 1]
        
        return pd.Series(probs, index=X.index, name="Probability_Bad")

    # ---------------------------------------------------------------
    # FEATURE SCORE CONTRIBUTION
    # ---------------------------------------------------------------

    def feature_contribution(self) -> pd.DataFrame:
        """
        Display how each model coefficient scales into points.

        Returns
        -------
        pandas.DataFrame
        """
        self._check_fitted()

        # Calculate per-unit feature impact
        df = pd.DataFrame({
            "Feature": self.feature_names_,
            "Coefficient": self.coefficients_.values,
            "Score_Impact_Per_Unit": -self.coefficients_.values * self.factor_,
        })
        
        # Add the global intercept contribution row
        intercept_row = pd.DataFrame([{
            "Feature": "_INTERCEPT_",
            "Coefficient": self.intercept_,
            "Score_Impact_Per_Unit": self.offset_ - (self.factor_ * self.intercept_)
        }])

        df = pd.concat([df, intercept_row], ignore_index=True)
        return df.sort_values("Score_Impact_Per_Unit", ascending=False).reset_index(drop=True)

    # ---------------------------------------------------------------
    # SERIALIZATION
    # ---------------------------------------------------------------

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize the fitted scaler to disk."""
        self._check_fitted()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("ScoreScaler saved to %s", filepath)
        return filepath

    @staticmethod
    def load(filepath: Union[str, Path]) -> "ScoreScaler":
        """Load a fitted scaler from disk."""
        filepath = Path(filepath)
        logger.info("Loading ScoreScaler from %s", filepath)
        return joblib.load(filepath)

    # ---------------------------------------------------------------
    # UTILS
    # ---------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Ensure scaler has been fitted before use."""
        if not hasattr(self, "model_"):
            raise RuntimeError("ScoreScaler has not been fitted.")

    def __repr__(self) -> str:
        if hasattr(self, "factor_"):
            return (
                f"ScoreScaler(pdo={self.config.pdo}, "
                f"base_score={self.config.base_score}, "
                f"fitted_features={len(self.feature_names_)})"
            )
        return f"ScoreScaler(pdo={self.config.pdo}, base_score={self.config.base_score})"