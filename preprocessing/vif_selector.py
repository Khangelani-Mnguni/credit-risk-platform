"""
vif_selector.py

Variance Inflation Factor (VIF) based feature selector.

Purpose
-------
Removes multicollinearity after:
- IV filtering
- Correlation filtering

This ensures stable Logistic Regression coefficients
for scorecard development.

Method
------
Iteratively removes the feature with the highest VIF
until all remaining features are below threshold.

Author
------
Credit Risk Platform
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class VIFSelector(BaseEstimator, TransformerMixin):
    """
    Iterative VIF-based feature selector.

    Parameters
    ----------
    threshold : float, default=5.0
        Maximum allowed VIF before removal.

    Attributes
    ----------
    selected_features_ : list of str
        Features that survived the VIF threshold.
    removed_features_ : list of str
        Features removed due to high multicollinearity.
    vif_history_ : list of dict
        History of VIF scores at each iteration.
    feature_vif_scores_ : dict
        The final VIF score for selected features, or the VIF score 
        at the time of removal for dropped features.
    """

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold

    # ---------------------------------------------------------------
    # FIT
    # ---------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "VIFSelector":
        """
        Iteratively compute VIF and drop features exceeding the threshold.
        """
        self._validate_input(X)
        X = X.copy()

        selected = list(X.columns)
        removed: List[str] = []
        
        self.vif_history_: List[Dict[str, float]] = []
        self.feature_vif_scores_: Dict[str, float] = {}

        logger.info("Starting iterative VIF selection (threshold=%.2f)...", self.threshold)

        while True:
            vif_values = self._compute_vif(X[selected])
            self.vif_history_.append(vif_values.to_dict())

            max_vif = vif_values.max()
            max_feature = vif_values.idxmax()

            if max_vif <= self.threshold:
                # Record final VIFs for surviving features
                for feat, score in vif_values.items():
                    self.feature_vif_scores_[feat] = score
                break

            # Record the VIF score that caused the feature to be dropped
            self.feature_vif_scores_[max_feature] = max_vif

            # Remove worst feature
            selected.remove(max_feature)
            removed.append(max_feature)

            logger.debug("Removing %s with VIF %.3f", max_feature, max_vif)

            if len(selected) <= 1:
                # If 1 or 0 features left, record the final one (if any) and break
                if selected:
                    self.feature_vif_scores_[selected[0]] = 1.0
                break

        self.selected_features_ = selected
        self.removed_features_ = removed

        logger.info(
            "VIF selector retained %d features, removed %d",
            len(self.selected_features_),
            len(self.removed_features_),
        )

        return self

    # ---------------------------------------------------------------
    # TRANSFORM
    # ---------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the DataFrame to retain only features with low VIF.
        """
        self._check_is_fitted()

        if (missing := [col for col in self.selected_features_ if col not in X.columns]):
            raise ValueError(f"Selected features missing from input: {missing}")

        return X.loc[:, self.selected_features_].copy()

    # ---------------------------------------------------------------
    # VIF COMPUTATION
    # ---------------------------------------------------------------

    def _compute_vif(self, X: pd.DataFrame) -> pd.Series:
        """Calculate VIF for the current set of features."""
        vif_dict = {}

        for col in X.columns:
            y = X[col]
            X_other = X.drop(columns=[col])

            # If only one feature is left, its VIF is 1.0 by definition
            if X_other.shape[1] == 0:
                vif_dict[col] = 1.0
                continue

            model = LinearRegression()
            model.fit(X_other, y)

            r2 = model.score(X_other, y)

            # Safeguard against floating point precision issues
            vif = np.inf if r2 >= 1.0 - 1e-9 else 1 / (1 - r2)

            vif_dict[col] = vif

        return pd.Series(vif_dict)

    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if X.shape[1] < 2:
            raise ValueError("Need at least 2 features to compute VIF.")

    def _check_is_fitted(self) -> None:
        """Internal check to ensure the selector has been fitted."""
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("VIFSelector has not been fitted.")

    # ---------------------------------------------------------------
    # SUMMARY & GETTERS
    # ---------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Return a comprehensive summary of feature selection status and VIF scores.
        """
        self._check_is_fitted()

        rows = [
            {
                "Feature": feature,
                "Status": "Selected",
                "VIF": self.feature_vif_scores_.get(feature, np.nan),
            }
            for feature in self.selected_features_
        ] + [
            {
                "Feature": feature,
                "Status": "Removed",
                "VIF": self.feature_vif_scores_.get(feature, np.nan),
            }
            for feature in self.removed_features_
        ]

        return pd.DataFrame(rows).sort_values(by="Status", ascending=True).reset_index(drop=True)

    def get_selected_features(self) -> List[str]:
        self._check_is_fitted()
        return self.selected_features_

    def get_removed_features(self) -> List[str]:
        self._check_is_fitted()
        return self.removed_features_

    # ---------------------------------------------------------------
    # REPRESENTATION
    # ---------------------------------------------------------------

    def __repr__(self) -> str:
        if hasattr(self, "selected_features_"):
            return (
                f"VIFSelector("
                f"selected={len(self.selected_features_)}, "
                f"removed={len(self.removed_features_)}, "
                f"threshold={self.threshold})"
            )
        return f"VIFSelector(threshold={self.threshold})"