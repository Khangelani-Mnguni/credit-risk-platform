"""
woe_transformer.py

Production-quality transformer for Weight of Evidence (WoE) encoding.

This module applies WoE transformations to both numerical and categorical
features using the OptimalBinner wrapper. It evaluates Information Value (IV),
stores binning diagnostics, and integrates seamlessly into scikit-learn
pipelines.

Example
-------
transformer = WOETransformer(max_n_bins=5, min_bin_size=0.05)
X_train_woe = transformer.fit_transform(X_train, y_train)
X_test_woe = transformer.transform(X_test)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Assuming optimal_binner is accessible in the same module structure
from .optimal_binner import OptimalBinner

logger = logging.getLogger(__name__)


class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Apply Weight of Evidence (WoE) transformation to dataset features.

    Parameters
    ----------
    max_n_bins : int, default=6
        Maximum number of bins to create per feature.
    min_bin_size : float, default=0.05
        Minimum fraction of observations allowed in a single bin.
    monotonic_trend : str, default="auto"
        Monotonicity constraint for numerical features (e.g., "auto", "ascending", "descending").
    special_codes : dict or list, optional
        Values to be treated as special codes (e.g., missing indicators).
    features : list of str, optional
        Specific features to transform. If None, all columns in X are used.

    Attributes
    ----------
    binners_ : dict
        Dictionary mapping feature names to their fitted OptimalBinner instances.
    feature_types_ : dict
        Dictionary mapping feature names to their detected types ('numerical' or 'categorical').
    binning_tables_ : dict
        Dictionary containing the binning table DataFrames for each feature.
    statistics_ : dict
        Dictionary storing summary statistics (e.g., IV) for each feature.
    diagnostics_ : dict
        Dictionary of diagnostic information (e.g., optimization status) per feature.
    iv_table_ : pandas.DataFrame
        Summary table of Information Values for all fitted features.
    selected_features_ : list of str
        List of features that were successfully fitted and stored.
    """

    def __init__(
        self,
        max_n_bins: int = 6,
        min_bin_size: float = 0.05,
        monotonic_trend: str = "auto",
        special_codes: Optional[Union[Dict, List]] = None,
        features: Optional[List[str]] = None,
    ):
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.monotonic_trend = monotonic_trend
        self.special_codes = special_codes
        self.features = features

    ####################################################################
    # CORE METHODS
    ####################################################################

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        """
        Fit OptimalBinners for all specified features.

        Parameters
        ----------
        X : pandas.DataFrame
            Training features.
        y : pandas.Series
            Target variable (binary: 0 or 1).

        Returns
        -------
        self : WOETransformer
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")

        # Initialize storage attributes
        self.binners_: Dict[str, OptimalBinner] = {}
        self.binning_tables_: Dict[str, pd.DataFrame] = {}
        self.statistics_: Dict[str, float] = {}
        self.diagnostics_: Dict[str, Any] = {}
        self.selected_features_: List[str] = self.features or list(X.columns)

        logger.info("Detecting feature types...")
        self._detect_feature_types(X[self.selected_features_])

        logger.info("Fitting WoE binners for %d features...", len(self.selected_features_))
        
        iv_records = []

        for feature in self.selected_features_:
            binner = self._fit_feature(feature, X[feature], y)
            
            # Store fitted objects and metadata
            self.binners_[feature] = binner
            self.binning_tables_[feature] = binner.get_binning_table()
            self.statistics_[feature] = binner.iv
            self.diagnostics_[feature] = binner.status

            iv_records.append({
                "Feature": feature,
                "Type": self.feature_types_[feature],
                "IV": binner.iv,
                "Status": binner.status
            })

        self.iv_table_ = pd.DataFrame(iv_records).sort_values(by="IV", ascending=False).reset_index(drop=True)
        
        logger.info("Finished fitting WOETransformer.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to WoE values.

        Parameters
        ----------
        X : pandas.DataFrame
            Data to transform.

        Returns
        -------
        pandas.DataFrame
            DataFrame with WoE transformed features.
        """
        if not hasattr(self, "binners_"):
            raise RuntimeError("WOETransformer has not been fitted.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        X_transformed = X.copy()
        logger.info("Applying WoE transformations...")

        for feature in self.selected_features_:
            if feature not in X.columns:
                raise ValueError(f"Feature '{feature}' was fitted but is missing from input data.")
            
            X_transformed[feature] = self._transform_feature(feature, X[feature])

        logger.info("WoE transformation complete.")
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the transformer and transform the data in one step.

        Parameters
        ----------
        X : pandas.DataFrame
            Training features.
        y : pandas.Series
            Target variable.

        Returns
        -------
        pandas.DataFrame
            WoE transformed DataFrame.
        """
        return self.fit(X, y).transform(X)

    ####################################################################
    # INTERNAL HELPERS
    ####################################################################

    def _detect_feature_types(self, X: pd.DataFrame) -> None:
        """
        Detect whether features are numerical or categorical.

        Parameters
        ----------
        X : pandas.DataFrame
        """
        self.feature_types_ = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.feature_types_[col] = "numerical"
            else:
                self.feature_types_[col] = "categorical"

    def _fit_feature(self, feature: str, X_feat: pd.Series, y: pd.Series) -> OptimalBinner:
        """
        Initialize and fit an OptimalBinner for a single feature.

        Parameters
        ----------
        feature : str
            Feature name.
        X_feat : pandas.Series
            Feature data.
        y : pandas.Series
            Target data.

        Returns
        -------
        OptimalBinner
            Fitted OptimalBinner instance.
        """
        dtype = self.feature_types_[feature]
        
        binner = OptimalBinner(
            name=feature,
            dtype=dtype,
            max_n_bins=self.max_n_bins,
            min_bin_size=self.min_bin_size,
            monotonic_trend=self.monotonic_trend if dtype == "numerical" else "auto",
            special_codes=self.special_codes,
        )
        return binner.fit(X_feat, y)

    def _transform_feature(self, feature: str, X_feat: pd.Series) -> pd.Series:
        """
        Transform a single feature using its fitted binner.

        Parameters
        ----------
        feature : str
            Feature name.
        X_feat : pandas.Series
            Feature data.

        Returns
        -------
        pandas.Series
            WoE transformed feature.
        """
        binner = self.binners_[feature]
        return binner.transform(X_feat, metric="woe")

    ####################################################################
    # UTILITY METHODS
    ####################################################################

    def get_binner(self, feature: str) -> OptimalBinner:
        """Get the fitted OptimalBinner for a specific feature."""
        self._check_is_fitted()
        if feature not in self.binners_:
            raise KeyError(f"Feature '{feature}' not found in fitted binners.")
        return self.binners_[feature]

    def get_binning_table(self, feature: str) -> pd.DataFrame:
        """Get the detailed binning table for a specific feature."""
        self._check_is_fitted()
        if feature not in self.binning_tables_:
            raise KeyError(f"Feature '{feature}' not found in binning tables.")
        return self.binning_tables_[feature]

    def get_statistics(self) -> Dict[str, float]:
        """Get the IV statistics for all features."""
        self._check_is_fitted()
        return self.statistics_

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get optimization statuses for all features."""
        self._check_is_fitted()
        return self.diagnostics_

    def plot(self, feature: str, metric: str = "woe", figsize: tuple = (8, 5)) -> None:
        """
        Plot the binning results for a specific feature.

        Parameters
        ----------
        feature : str
            Feature name to plot.
        metric : str, default="woe"
            Metric to plot ("woe", "event_rate").
        figsize : tuple, default=(8, 5)
            Figure size.
        """
        binner = self.get_binner(feature)
        binner.plot(metric=metric, figsize=figsize)

    def summary(self) -> pd.DataFrame:
        """
        Return the overall IV and status summary table.

        Returns
        -------
        pandas.DataFrame
        """
        self._check_is_fitted()
        return self.iv_table_.copy()

    def _check_is_fitted(self):
        """Internal check to ensure transformer has been fit."""
        if not hasattr(self, "binners_"):
            raise RuntimeError("WOETransformer has not been fitted.")

    ####################################################################
    # SERIALIZATION
    ####################################################################

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Serialize the fitted transformer to disk.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to save the model.
        """
        self._check_is_fitted()
        joblib.dump(self, filepath)
        logger.info("Saved WOETransformer to %s", filepath)

    @staticmethod
    def load(filepath: Union[str, Path]) -> "WOETransformer":
        """
        Load a fitted transformer from disk.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the saved model.

        Returns
        -------
        WOETransformer
        """
        logger.info("Loading WOETransformer from %s", filepath)
        return joblib.load(filepath)