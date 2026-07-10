"""
logistic_model.py

Production-grade Logistic Regression model for Probability of Default (PD)
estimation.

Features
--------
- Inherits from BaseCreditRiskModel
- Cross-validation support
- Hyperparameter tuning (GridSearchCV)
- Feature importance (coefficients)
- Model evaluation
- Model persistence
- Scorecard-compatible coefficients

Author
------
Credit Risk Platform
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)

from .base_model import BaseCreditRiskModel
from .model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class LogisticRegressionConfig:
    """Configuration for the Logistic Regression classifier."""
    penalty: str = "l2"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 500
    class_weight: str | dict | None = "balanced"
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5


# ---------------------------------------------------------------------
# Logistic Regression Model
# ---------------------------------------------------------------------

class LogisticRegressionModel(BaseCreditRiskModel):
    """
    Production-grade Logistic Regression implementation.

    Parameters
    ----------
    config : LogisticRegressionConfig
        Configuration dataclass.
    """

    def __init__(
        self,
        config: LogisticRegressionConfig = LogisticRegressionConfig(),
    ):
        super().__init__(
            model_name="Logistic Regression",
            model_version="1.0.0",
            random_state=config.random_state,
        )
        self.config = config

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str = "default_flag",
    ) -> "LogisticRegressionModel":

        logger.info("Training Logistic Regression model...")
        start = time.time()

        self.feature_names_ = list(X.columns)
        self.target_name_ = target_name
        self.training_samples_ = len(X)
        self.training_features_ = X.shape[1]
        self.training_timestamp_ = datetime.utcnow().isoformat()

        self.class_distribution_ = (
            y.value_counts(normalize=True)
            .sort_index()
            .to_dict()
        )

        self.model_ = LogisticRegression(
            penalty=self.config.penalty,
            C=self.config.C,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        self.model_.fit(X, y)

        self.training_time_ = time.time() - start
        self.is_fitted_ = True

        logger.info("Training completed in %.2f seconds", self.training_time_)

        return self

    # -----------------------------------------------------------------
    # Cross Validation
    # -----------------------------------------------------------------

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str = "roc_auc",
    ) -> pd.DataFrame:
        
        # Instantiate a fresh estimator for clean cross-validation
        estimator = LogisticRegression(
            penalty=self.config.penalty,
            C=self.config.C,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        scores = cross_val_score(
            estimator,
            X,
            y,
            scoring=scoring,
            cv=cv,
            n_jobs=self.config.n_jobs,
        )

        return pd.DataFrame(
            {
                "Fold": np.arange(1, len(scores) + 1),
                "Score": scores,
            }
        )

    # -----------------------------------------------------------------
    # Hyperparameter Tuning
    # -----------------------------------------------------------------

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[dict] = None,
        scoring: str = "roc_auc",
    ):

        if param_grid is None:
            param_grid = {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"], 
            }
            
        # Instantiate a base estimator for grid search
        estimator = LogisticRegression(
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
        )

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=self.config.n_jobs,
            refit=True,
        )

        logger.info("Starting GridSearchCV for Logistic Regression...")
        search.fit(X, y)

        self.model_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        
        # Sync config with the newly discovered best parameters
        for key, value in self.best_params_.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                
        logger.info("Best ROC-AUC: %.4f", self.best_score_)

        return search.cv_results_

    # -----------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> ModelEvaluator:

        evaluator = ModelEvaluator()
        probabilities = self.predict_proba(X)[:, 1]

        return evaluator.evaluate(
            y_true=y,
            y_prob=probabilities,
        )

    # -----------------------------------------------------------------
    # Feature Importance
    # -----------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_is_fitted()

        # Handle 1D or 2D coefficient array safely
        coef_array = self.model_.coef_[0] if self.model_.coef_.ndim > 1 else self.model_.coef_

        importance = pd.DataFrame(
            {
                "Feature": self.feature_names_,
                "Coefficient": coef_array,
                "AbsCoefficient": np.abs(coef_array),
                "OddsRatio": np.exp(coef_array),
            }
        )

        return importance.sort_values(
            "AbsCoefficient",
            ascending=False,
        ).reset_index(drop=True)

    # -----------------------------------------------------------------
    # Coefficients
    # -----------------------------------------------------------------

    def coefficients(self) -> pd.Series:
        self._check_is_fitted()
        coef_array = self.model_.coef_[0] if self.model_.coef_.ndim > 1 else self.model_.coef_
        return pd.Series(coef_array, index=self.feature_names_)

    # -----------------------------------------------------------------
    # Odds Ratios
    # -----------------------------------------------------------------

    def odds_ratios(self) -> pd.DataFrame:
        self._check_is_fitted()
        coef_array = self.model_.coef_[0] if self.model_.coef_.ndim > 1 else self.model_.coef_

        return pd.DataFrame(
            {
                "Feature": self.feature_names_,
                "OddsRatio": np.exp(coef_array),
            }
        ).sort_values("OddsRatio", ascending=False).reset_index(drop=True)

    # -----------------------------------------------------------------
    # Intercept
    # -----------------------------------------------------------------

    @property
    def intercept_(self) -> float:
        self._check_is_fitted()
        val = self.model_.intercept_[0] if isinstance(self.model_.intercept_, (list, np.ndarray)) else self.model_.intercept_
        return float(val)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        base = super().summary()

        extra = pd.DataFrame(
            {
                "Property": [
                    "Penalty",
                    "Regularization (C)",
                    "Solver",
                    "Max Iterations",
                ],
                "Value": [
                    self.config.penalty,
                    self.config.C,
                    self.config.solver,
                    self.config.max_iter,
                ],
            }
        )

        return pd.concat([base, extra], ignore_index=True)