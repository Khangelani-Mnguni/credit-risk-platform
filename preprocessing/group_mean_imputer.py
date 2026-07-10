"""
group_mean_imputer.py

Feature-specific grouped mean imputer for credit risk modelling.

This transformer imputes missing numerical values using the mean calculated
within a predefined grouping variable (e.g. grade, fico_band).

Example
-------
FEATURE_GROUP_MAP = {
    "annual_inc": "grade",
    "dti": "grade",
    "bc_util": "fico_band",
    "mort_acc": "home_ownership"
}

imputer = FeatureSpecificGroupMeanImputer(FEATURE_GROUP_MAP)

X_train = imputer.fit_transform(X_train)
X_valid = imputer.transform(X_valid)
X_test = imputer.transform(X_test)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


logger = logging.getLogger(__name__)


class FeatureSpecificGroupMeanImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing numerical values using feature-specific group means.

    Parameters
    ----------
    feature_group_map : dict
        Dictionary mapping numerical features to grouping variables.

        Example
        -------
        {
            "annual_inc": "grade",
            "dti": "grade",
            "bc_util": "fico_band"
        }

    Attributes
    ----------
    group_means_ : dict
        Nested dictionary containing group means.

    global_means_ : dict
        Global feature means used as fallback.

    feature_group_map_ : dict
        Validated feature-group mapping.
    """

    def __init__(self, feature_group_map: Dict[str, str]):

        self.feature_group_map = feature_group_map

    ####################################################################
    # FIT
    ####################################################################

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ):
        """
        Learn grouped means from the training data.

        Parameters
        ----------
        X : pandas.DataFrame
            Training features.

        y : ignored

        Returns
        -------
        self
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        self.feature_group_map_ = {}
        self.group_means_ = {}
        self.global_means_ = {}

        logger.info("Learning grouped mean statistics...")

        for feature, group in self.feature_group_map.items():

            if feature not in X.columns:
                raise ValueError(
                    f"Feature '{feature}' not found."
                )

            if group not in X.columns:
                raise ValueError(
                    f"Grouping column '{group}' not found."
                )

            self.feature_group_map_[feature] = group

            ###########################################################
            # Global mean
            ###########################################################

            global_mean = X[feature].mean()

            self.global_means_[feature] = global_mean

            ###########################################################
            # Group means
            ###########################################################

            means = (
                X.groupby(group)[feature]
                .mean()
                .to_dict()
            )

            self.group_means_[feature] = means

            logger.info(
                "Stored %d group means for %s",
                len(means),
                feature
            )

        logger.info("Finished fitting Group Mean Imputer.")

        return self

    ####################################################################
    # TRANSFORM
    ####################################################################

    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Impute missing values.
        """
        if not hasattr(self, "group_means_"):
            raise RuntimeError(
                "The imputer has not been fitted."
            )

        X = X.copy()
        logger.info("Applying grouped mean imputation...")

        for feature, group in self.feature_group_map_.items():

            global_mean = self.global_means_[feature]
            group_means = self.group_means_[feature]

            missing_mask = X[feature].isna()

            if missing_mask.sum() == 0:
                continue

            # -------------------------------------------------------------
            # NEW: Cast integer columns to Float64 so they can hold the mean
            # -------------------------------------------------------------
            if pd.api.types.is_integer_dtype(X[feature]):
                X[feature] = X[feature].astype("Float64")

            # -------------------------------------------------------------
            # Row-wise imputation
            # -------------------------------------------------------------
            for idx in X.index[missing_mask]:
                group_value = X.at[idx, group]

                if group_value in group_means:
                    value = group_means[group_value]
                    X.at[idx, feature] = value if pd.notna(value) else global_mean
                else:
                    X.at[idx, feature] = global_mean

            logger.info(
                "%s: imputed %d observations.",
                feature,
                missing_mask.sum()
            )

        return X

    ####################################################################
    # FIT TRANSFORM
    ####################################################################

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:

        return self.fit(X, y).transform(X)

    ####################################################################
    # UTILITIES
    ####################################################################

    def get_group_means(self) -> Dict:
        """
        Return learned group means.
        """

        if not hasattr(self, "group_means_"):
            raise RuntimeError(
                "Transformer has not been fitted."
            )

        return self.group_means_

    def get_global_means(self) -> Dict:
        """
        Return learned global means.
        """

        if not hasattr(self, "global_means_"):
            raise RuntimeError(
                "Transformer has not been fitted."
            )

        return self.global_means_

    def summary(self) -> pd.DataFrame:
        """
        Return a summary table.

        Returns
        -------
        pandas.DataFrame
        """

        if not hasattr(self, "group_means_"):
            raise RuntimeError(
                "Transformer has not been fitted."
            )

        rows = []
        rows.extend(
            {
                "feature": feature,
                "group_variable": self.feature_group_map_[feature],
                "global_mean": self.global_means_[feature],
                "number_of_groups": len(self.group_means_[feature]),
            }
            for feature in self.feature_group_map_
        )

        return pd.DataFrame(rows)