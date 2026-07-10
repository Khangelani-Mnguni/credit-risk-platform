"""
xgboost_model.py

Production-grade XGBoost model for Probability of Default (PD) estimation.

Features
--------
- Inherits from BaseCreditRiskModel
- Cross-validation support
- Hyperparameter tuning
- Early stopping
- Feature importance
- SHAP-ready
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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from xgboost import XGBClassifier

from .base_model import BaseCreditRiskModel
from .model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class XGBoostConfig:
    """Configuration for the XGBoost classifier."""
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 5
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    objective: str = "binary:logistic"
    eval_metric: str = "auc"
    tree_method: str = "hist"

    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5

    early_stopping_rounds: Optional[int] = 50


# =============================================================================
# XGBoost Model
# =============================================================================

class XGBoostModel(BaseCreditRiskModel):
    """
    Production-grade XGBoost implementation.

    Parameters
    ----------
    config : XGBoostConfig
        Configuration dataclass.
    """

    def __init__(
        self,
        config: XGBoostConfig = XGBoostConfig(),
    ):
        super().__init__(
            model_name="XGBoost",
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
        eval_set: Optional[list] = None,
        **kwargs,
    ) -> "XGBoostModel":

        logger.info("Training XGBoost model...")
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

        # Setup base initialization arguments
        init_kwargs = {
            "n_estimators": self.config.n_estimators,
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.max_depth,
            "min_child_weight": self.config.min_child_weight,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "gamma": self.config.gamma,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "objective": self.config.objective,
            "eval_metric": self.config.eval_metric,
            "tree_method": self.config.tree_method,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
        }

        fit_kwargs: Dict[str, Any] = {}

        # Handle early stopping for XGBoost >= 1.6.0 API
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False
            if self.config.early_stopping_rounds:
                init_kwargs["early_stopping_rounds"] = self.config.early_stopping_rounds

        self.model_ = XGBClassifier(**init_kwargs)
        self.model_.fit(X, y, **fit_kwargs)

        self.training_time_ = time.time() - start
        self.is_fitted_ = True

        logger.info("Training completed in %.2f seconds", self.training_time_)

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

        estimator = XGBClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            gamma=self.config.gamma,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            objective=self.config.objective,
            eval_metric=self.config.eval_metric,
            tree_method=self.config.tree_method,
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

        return pd.DataFrame({
            "Fold": np.arange(1, len(scores) + 1),
            "Score": scores,
        })

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
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.10],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "n_estimators": [300, 500],
            }

        estimator = XGBClassifier(
            objective=self.config.objective,
            eval_metric=self.config.eval_metric,
            tree_method=self.config.tree_method,
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

        logger.info("Starting GridSearchCV for XGBoost...")
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

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Feature Importance (Default Sklearn API)
    # -------------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_is_fitted()

        importance = pd.DataFrame({
            "Feature": self.feature_names_,
            "Importance": self.model_.feature_importances_,
        })

        return importance.sort_values(
            "Importance",
            ascending=False,
        ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Gain, Weight, Cover Importance (XGBoost Native APIs)
    # -------------------------------------------------------------------------
    
    def _get_booster_importance(self, importance_type: str) -> pd.DataFrame:
        """Helper to extract and align specific XGBoost importances."""
        self._check_is_fitted()
        booster = self.model_.get_booster()
        score_dict = booster.get_score(importance_type=importance_type)
        
        # Ensure features that were not used to split are mapped to 0.0
        aligned_scores = {feat: score_dict.get(feat, 0.0) for feat in self.feature_names_}
        
        return pd.DataFrame(
            aligned_scores.items(),
            columns=["Feature", importance_type.capitalize()]
        ).sort_values(importance_type.capitalize(), ascending=False).reset_index(drop=True)

    def gain_importance(self) -> pd.DataFrame:
        return self._get_booster_importance("gain")

    def weight_importance(self) -> pd.DataFrame:
        return self._get_booster_importance("weight")

    def cover_importance(self) -> pd.DataFrame:
        return self._get_booster_importance("cover")

    # -------------------------------------------------------------------------
    # Evaluation History & Booster
    # -------------------------------------------------------------------------

    def evaluation_history(self) -> Optional[Dict]:
        self._check_is_fitted()
        try:
            return self.model_.evals_result()
        except Exception:
            return None

    @property
    def booster_(self):
        self._check_is_fitted()
        return self.model_.get_booster()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        base = super().summary()

        params = pd.DataFrame({
            "Property": [
                "Learning Rate",
                "Max Depth",
                "Estimators",
                "Min Child Weight",
                "Subsample",
                "Column Sample",
                "Gamma",
                "L1 Regularization",
                "L2 Regularization",
                "Tree Method",
            ],
            "Value": [
                self.config.learning_rate,
                self.config.max_depth,
                self.config.n_estimators,
                self.config.min_child_weight,
                self.config.subsample,
                self.config.colsample_bytree,
                self.config.gamma,
                self.config.reg_alpha,
                self.config.reg_lambda,
                self.config.tree_method,
            ],
        })

        return pd.concat([base, params], ignore_index=True)