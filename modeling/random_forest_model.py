"""
random_forest_model.py

Production-grade Random Forest model for Probability of Default (PD) estimation.

Features
--------
- Inherits from BaseCreditRiskModel
- Cross-validation support
- Hyperparameter tuning
- Out-of-Bag (OOB) scoring
- Gini feature importance
- Permutation feature importance
- Tree architecture diagnostics (depth, leaves)
- Model evaluation
- Model persistence

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)

from .base_model import BaseCreditRiskModel
from .model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RandomForestConfig:
    """Configuration for the Random Forest classifier."""

    n_estimators: int = 500
    max_depth: Optional[int] = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 5
    max_features: str | float = "sqrt"
    bootstrap: bool = True
    oob_score: bool = True
    class_weight: str | dict | None = "balanced"
    
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5


# =============================================================================
# Random Forest Model
# =============================================================================

class RandomForestModel(BaseCreditRiskModel):
    """
    Production-grade Random Forest implementation.

    Parameters
    ----------
    config : RandomForestConfig
        Configuration dataclass.
    """

    def __init__(
        self,
        config: RandomForestConfig = RandomForestConfig(),
    ):

        super().__init__(
            model_name="Random Forest",
            model_version="1.0.0",
            random_state=config.random_state,
        )

        self.config = config

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str = "default_flag",
    ):

        logger.info("Training Random Forest model...")

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

        self.model_ = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            oob_score=self.config.oob_score,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        self.model_.fit(X, y)

        self.training_time_ = time.time() - start
        self.is_fitted_ = True

        logger.info(
            "Training completed in %.2f seconds",
            self.training_time_,
        )

        if self.config.oob_score:
            logger.info("OOB Score (Accuracy): %.4f", self.model_.oob_score_)

        return self

    # -------------------------------------------------------------------------
    # Cross Validation
    # -------------------------------------------------------------------------

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str = "roc_auc",
    ) -> pd.DataFrame:

        estimator = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
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
            cv=cv,
            scoring=scoring,
            n_jobs=self.config.n_jobs,
        )

        return pd.DataFrame(
            {
                "Fold": np.arange(1, len(scores) + 1),
                "Score": scores,
            }
        )

    # -------------------------------------------------------------------------
    # Hyperparameter Tuning
    # -------------------------------------------------------------------------

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[dict] = None,
        scoring: str = "roc_auc",
    ):

        if param_grid is None:

            param_grid = {
                "max_depth": [5, 10, 15, None],
                "min_samples_leaf": [5, 10, 20],
                "max_features": ["sqrt", "log2", 0.5],
                "n_estimators": [300, 500],
            }

        estimator = RandomForestClassifier(
            bootstrap=self.config.bootstrap,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
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
            refit=True,
            n_jobs=self.config.n_jobs,
        )

        search.fit(X, y)

        self.model_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        
        # Update config to match the best parameters found
        for key, value in self.best_params_.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        logger.info("Best ROC-AUC: %.4f", self.best_score_)

        return search.cv_results_

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):

        evaluator = ModelEvaluator()

        probabilities = self.predict_proba(X)[:, 1]

        return evaluator.evaluate(
            y_true=y,
            y_prob=probabilities,
        )

    # -------------------------------------------------------------------------
    # Feature Importance (Gini Impurity)
    # -------------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """Return standard Random Forest impurity-based feature importances."""
        self._check_is_fitted()

        importance = pd.DataFrame(
            {
                "Feature": self.feature_names_,
                "Importance": self.model_.feature_importances_,
            }
        )

        return importance.sort_values(
            "Importance",
            ascending=False,
        ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Permutation Feature Importance
    # -------------------------------------------------------------------------

    def feature_importance_permutation(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_repeats: int = 5,
        scoring: str = "roc_auc",
    ) -> pd.DataFrame:
        """
        Calculates permutation feature importance, which is often more 
        reliable than Gini impurity for high-cardinality features.
        """
        self._check_is_fitted()
        logger.info("Calculating permutation feature importance...")

        result = permutation_importance(
            self.model_, 
            X, 
            y, 
            n_repeats=n_repeats, 
            random_state=self.config.random_state, 
            scoring=scoring,
            n_jobs=self.config.n_jobs
        )

        importance = pd.DataFrame(
            {
                "Feature": self.feature_names_,
                "Importance_Mean": result.importances_mean,
                "Importance_Std": result.importances_std,
            }
        )

        return importance.sort_values(
            "Importance_Mean", 
            ascending=False
        ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Tree Architecture Diagnostics
    # -------------------------------------------------------------------------

    def tree_depth_statistics(self) -> pd.DataFrame:
        """Returns statistics regarding the depths of the constituent trees."""
        self._check_is_fitted()
        
        depths = [estimator.tree_.max_depth for estimator in self.model_.estimators_]
        
        return pd.DataFrame({
            "Metric": ["Min Depth", "Max Depth", "Mean Depth", "Median Depth"],
            "Value": [
                np.min(depths), 
                np.max(depths), 
                np.mean(depths), 
                np.median(depths)
            ]
        })

    def leaf_statistics(self) -> pd.DataFrame:
        """Returns statistics regarding the number of leaves across trees."""
        self._check_is_fitted()
        
        leaves = [estimator.tree_.n_leaves for estimator in self.model_.estimators_]
        
        return pd.DataFrame({
            "Metric": ["Min Leaves", "Max Leaves", "Mean Leaves", "Median Leaves"],
            "Value": [
                np.min(leaves), 
                np.max(leaves), 
                np.mean(leaves), 
                np.median(leaves)
            ]
        })

    # -------------------------------------------------------------------------
    # Out of Bag Score
    # -------------------------------------------------------------------------

    @property
    def oob_score(self) -> float:
        """Returns the out-of-bag score if bootstrap=True and oob_score=True."""
        self._check_is_fitted()
        
        if not self.config.oob_score:
            raise ValueError(
                "oob_score was set to False during initialization. "
                "Cannot retrieve OOB score."
            )
            
        return self.model_.oob_score_

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:

        base = super().summary()

        params = pd.DataFrame(
            {
                "Property": [
                    "Estimators",
                    "Max Depth",
                    "Min Samples Split",
                    "Min Samples Leaf",
                    "Max Features",
                    "Bootstrap",
                    "OOB Score Enabled",
                    "Class Weight",
                ],
                "Value": [
                    self.config.n_estimators,
                    self.config.max_depth,
                    self.config.min_samples_split,
                    self.config.min_samples_leaf,
                    self.config.max_features,
                    self.config.bootstrap,
                    self.config.oob_score,
                    self.config.class_weight,
                ],
            }
        )

        return pd.concat(
            [base, params],
            ignore_index=True,
        )