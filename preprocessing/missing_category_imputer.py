"""
missing_category_imputer.py

Sklearn-compatible transformer for imputing missing categorical values.

Missing values are replaced with the explicit category "Missing"
rather than the most frequent category. This preserves potentially
predictive information and is particularly suitable for
Weight of Evidence (WoE) encoding in credit scorecards.

Example
-------
categorical_features = [
    "grade",
    "sub_grade",
    "purpose",
    "application_type",
    "home_ownership",
    "fico_band"
]

imputer = MissingCategoryImputer(categorical_features)

X_train = imputer.fit_transform(X_train)
X_valid = imputer.transform(X_valid)
X_test = imputer.transform(X_test)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


logger = logging.getLogger(__name__)


class MissingCategoryImputer(BaseEstimator, TransformerMixin):
    """
    Replace missing categorical values with a specified category.

    Parameters
    ----------
    categorical_features : list[str]
        List of categorical variables to impute.

    fill_value : str, default="Missing"
        Category used to replace missing values.

    Attributes
    ----------
    categorical_features_ : list
        Validated list of categorical variables.

    fill_value_ : str
        Learned fill value.
    """

    def __init__(
        self,
        categorical_features: List[str],
        fill_value: str = "Missing"
    ):

        self.categorical_features = categorical_features
        self.fill_value = fill_value

    ####################################################################
    # FIT
    ####################################################################

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ):
        """
        Validate the categorical variables.

        Parameters
        ----------
        X : pandas.DataFrame

        y : ignored

        Returns
        -------
        self
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        self.categorical_features_ = []

        for feature in self.categorical_features:

            if feature not in X.columns:
                raise ValueError(
                    f"Categorical feature '{feature}' not found."
                )

            self.categorical_features_.append(feature)

        self.fill_value_ = self.fill_value

        logger.info(
            "Validated %d categorical features.",
            len(self.categorical_features_)
        )

        return self

    ####################################################################
    # TRANSFORM
    ####################################################################

    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Replace missing categorical values.

        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """

        if not hasattr(self, "categorical_features_"):
            raise RuntimeError(
                "Transformer has not been fitted."
            )

        X = X.copy()

        logger.info("Applying missing category imputation...")

        for feature in self.categorical_features_:

            before = X[feature].isna().sum()

            ###########################################################
            # Ensure categorical dtype supports new category
            ###########################################################

            if (
                pd.api.types.is_categorical_dtype(X[feature])
                and self.fill_value_ not in X[feature].cat.categories
            ):
                X[feature] = X[feature].cat.add_categories(
                    [self.fill_value_]
                )

            ###########################################################
            # Fill missing values
            ###########################################################

            X[feature] = X[feature].fillna(self.fill_value_)

            after = X[feature].isna().sum()

            logger.info(
                "%s: %d missing values imputed.",
                feature,
                before - after
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
    # SUMMARY
    ####################################################################

    def summary(self) -> pd.DataFrame:
        """
        Return transformer summary.

        Returns
        -------
        pandas.DataFrame
        """

        if not hasattr(self, "categorical_features_"):
            raise RuntimeError(
                "Transformer has not been fitted."
            )

        return pd.DataFrame(
            {
                "feature": self.categorical_features_,
                "fill_value": self.fill_value_
            }
        )

    ####################################################################
    # GET PARAMETERS
    ####################################################################

    def get_fill_value(self) -> str:
        """
        Return fill value.
        """

        if not hasattr(self, "fill_value_"):
            raise RuntimeError(
                "Transformer has not been fitted."
            )

        return self.fill_value_

    def get_features(self) -> List[str]:
        """
        Return categorical features.
        """

        if not hasattr(self, "categorical_features_"):
            raise RuntimeError(
                "Transformer has not been fitted."
            )

        return self.categorical_features_