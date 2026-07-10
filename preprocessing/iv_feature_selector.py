"""
iv_feature_selector.py

Sklearn-compatible transformer for selecting variables using
Information Value (IV).

The selector removes:
- Variables with IV below the minimum threshold
- Variables with IV above the maximum threshold (possible target leakage)

Author
------
Credit Risk Platform
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class IVFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector based on Information Value.

    Parameters
    ----------
    min_iv : float, default=0.02
        Minimum acceptable IV.
    max_iv : float, default=0.50
        Maximum acceptable IV.

    Attributes
    ----------
    selected_features_ : list of str
        Features that met the IV criteria.
    removed_features_ : list of str
        Features that were dropped.
    iv_summary_ : pandas.DataFrame
        A copy of the evaluated IV summary.
    """

    def __init__(
        self,
        min_iv: float = 0.02,
        max_iv: float = 0.50,
    ):
        self.min_iv = min_iv
        self.max_iv = max_iv

    ####################################################################
    # FIT
    ####################################################################

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        iv_summary: Optional[pd.DataFrame] = None,
    ) -> "IVFeatureSelector":
        """
        Learn selected features based on IV thresholds.

        Parameters
        ----------
        X : pandas.DataFrame
            WoE transformed dataframe.
        y : pandas.Series, optional
            Target variable (ignored, required for sklearn compatibility).
        iv_summary : pandas.DataFrame
            Output from IVCalculator.summary() or WOETransformer.iv_table_

        Returns
        -------
        self
        """
        if iv_summary is None:
            raise ValueError("iv_summary must be supplied to fit the selector.")

        required_cols = {"Feature", "IV"}
        if not required_cols.issubset(iv_summary.columns):
            raise ValueError(f"iv_summary must contain columns {required_cols}.")

        self.iv_summary_ = iv_summary.copy().reset_index(drop=True)

        # Filter features based on min and max IV thresholds
        selected_mask = (
            (self.iv_summary_["IV"] >= self.min_iv) & 
            (self.iv_summary_["IV"] <= self.max_iv)
        )
        
        self.selected_features_ = self.iv_summary_.loc[selected_mask, "Feature"].tolist()
        self.removed_features_ = [c for c in X.columns if c not in self.selected_features_]

        logger.info("Selected %d variables.", len(self.selected_features_))
        logger.info("Removed %d variables.", len(self.removed_features_))

        return self

    ####################################################################
    # TRANSFORM
    ####################################################################

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the DataFrame to retain only selected features.
        """
        self._check_is_fitted()

        if missing_cols := [col for col in self.selected_features_ if col not in X.columns]:
            raise ValueError(
                f"The following selected features are missing from X: {missing_cols}"
            )

        return X.loc[:, self.selected_features_].copy()

    ####################################################################
    # FIT TRANSFORM
    ####################################################################

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        iv_summary: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Fit the selector and transform the data in one step.
        """
        return self.fit(X, y, iv_summary=iv_summary).transform(X)

    ####################################################################
    # SUMMARY & GETTERS
    ####################################################################

    def summary(self) -> pd.DataFrame:
        """Return IV summary with a selection flag."""
        self._check_is_fitted()
        summary = self.iv_summary_.copy()
        summary["Selected"] = summary["Feature"].isin(self.selected_features_)
        return summary

    def get_selected_features(self) -> List[str]:
        """Return the list of retained features."""
        self._check_is_fitted()
        return self.selected_features_

    def get_removed_features(self) -> List[str]:
        """Return the list of dropped features."""
        self._check_is_fitted()
        return self.removed_features_

    def _check_is_fitted(self) -> None:
        """Internal check to ensure the selector has been fitted."""
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("IVFeatureSelector has not been fitted.")

    ####################################################################
    # SERIALIZATION & EXPORT
    ####################################################################

    def export(self, filepath: Union[str, Path]) -> None:
        """Export selection report to CSV or Excel."""
        self._check_is_fitted()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.summary()
        suffix = filepath.suffix.lower()

        if suffix == ".csv":
            summary.to_csv(filepath, index=False)
        elif suffix == ".xlsx":
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                summary.to_excel(writer, index=False, sheet_name="IV Selection")
        else:
            raise ValueError("Unsupported file type. Please use .csv or .xlsx")

        logger.info("Selection report saved to %s", filepath)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the fitted selector to disk."""
        joblib.dump(self, filepath)
        logger.info("Selector saved to %s", filepath)

    @staticmethod
    def load(filepath: Union[str, Path]) -> "IVFeatureSelector":
        """Load a fitted selector from disk."""
        return joblib.load(filepath)

    ####################################################################
    # REPRESENTATION
    ####################################################################

    def __repr__(self) -> str:
        if hasattr(self, "selected_features_"):
            return (
                f"IVFeatureSelector("
                f"selected={len(self.selected_features_)}, "
                f"removed={len(self.removed_features_)})"
            )
        return f"IVFeatureSelector(min_iv={self.min_iv}, max_iv={self.max_iv})"