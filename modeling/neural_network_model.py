"""
neural_network_model.py

Production-grade Neural Network model for Probability of Default (PD) estimation.

Features
--------
- Inherits from BaseCreditRiskModel
- Scikit-learn MLPClassifier backend
- Cross-validation support
- Hyperparameter tuning
- Permutation feature importance (model-agnostic explainability)
- Early stopping support
- Loss curve extraction
- Model evaluation and persistence

Author
------
Credit Risk Platform
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neural_network import MLPClassifier

from .base_model import BaseCreditRiskModel
from .model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class NeuralNetworkConfig:
    """Configuration for the Neural Network (MLP) classifier."""

    hidden_layer_sizes: Tuple[int, ...] = (64, 32)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001  # L2 penalty parameter
    batch_size: str | int = "auto"
    learning_rate: str = "constant"
    learning_rate_init: float = 0.001
    max_iter: int = 500
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5


# =============================================================================
# Neural Network Model
# =============================================================================

class NeuralNetworkModel(BaseCreditRiskModel):
    """
    Production-grade Neural Network (Multi-Layer Perceptron) implementation.

    Parameters
    ----------
    config : NeuralNetworkConfig
        Configuration dataclass.
    """

    def __init__(
        self,
        config: NeuralNetworkConfig = NeuralNetworkConfig(),
    ):

        super().__init__(
            model_name="Neural Network (MLP)",
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
    ) -> "NeuralNetworkModel":

        logger.info("Training Neural Network model...")

        start = time.time()

        self.feature_names_ = list(X.columns)
        self.target_name_ = target_name
        self.training_samples_ = len(X)
        self.training_features_ = X.shape[1]
        self.training_timestamp_ = datetime.now(timezone.utc).isoformat()

        self.class_distribution_ = (
            y.value_counts(normalize=True)
            .sort_index()
            .to_dict()
        )

        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.config.hidden_layer_sizes,
            activation=self.config.activation,
            solver=self.config.solver,
            alpha=self.config.alpha,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            learning_rate_init=self.config.learning_rate_init,
            max_iter=self.config.max_iter,
            early_stopping=self.config.early_stopping,
            validation_fraction=self.config.validation_fraction,
            n_iter_no_change=self.config.n_iter_no_change,
            random_state=self.config.random_state,
        )

        self.model_.fit(X, y)

        self.training_time_ = time.time() - start
        self.is_fitted_ = True

        logger.info("Training completed in %.2f seconds", self.training_time_)
        logger.info("Epochs elapsed: %d", self.model_.n_iter_)

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

        estimator = MLPClassifier(
            hidden_layer_sizes=self.config.hidden_layer_sizes,
            activation=self.config.activation,
            solver=self.config.solver,
            alpha=self.config.alpha,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            learning_rate_init=self.config.learning_rate_init,
            max_iter=self.config.max_iter,
            early_stopping=self.config.early_stopping,
            validation_fraction=self.config.validation_fraction,
            n_iter_no_change=self.config.n_iter_no_change,
            random_state=self.config.random_state,
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
                "hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01],
                "activation": ["relu", "tanh"]
            }

        estimator = MLPClassifier(
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            early_stopping=self.config.early_stopping,
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
            refit=True,
            n_jobs=self.config.n_jobs,
        )

        logger.info("Starting GridSearchCV for Neural Network...")
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
    ) -> ModelEvaluator:

        evaluator = ModelEvaluator()
        probabilities = self.predict_proba(X)[:, 1]

        return evaluator.evaluate(
            y_true=y,
            y_prob=probabilities,
        )

    # -------------------------------------------------------------------------
    # Feature Importance (Permutation)
    # -------------------------------------------------------------------------

    def get_feature_importance(
        self, 
        X: Optional[pd.DataFrame] = None, 
        y: Optional[pd.Series] = None,
        n_repeats: int = 5,
        scoring: str = "roc_auc",
    ) -> pd.DataFrame:
        """
        Calculates permutation feature importance. 
        Neural networks do not natively expose feature weights per input.
        
        Note: Requires X and y to evaluate the drop in performance when 
        features are randomly shuffled.
        """
        self._check_is_fitted()
        
        if X is None or y is None:
            raise ValueError(
                "Neural networks require evaluation data (X, y) to compute "
                "permutation feature importance."
            )
            
        logger.info("Calculating permutation feature importance (this may take a moment)...")

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
    # Loss Curve Diagnostics
    # -------------------------------------------------------------------------

    def get_loss_curve(self) -> List[float]:
        """
        Returns the loss value evaluated at the end of each training step.
        """
        self._check_is_fitted()
        return self.model_.loss_curve_

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:

        base = super().summary()

        params = pd.DataFrame(
            {
                "Property": [
                    "Hidden Layers",
                    "Activation Function",
                    "Solver",
                    "L2 Penalty (Alpha)",
                    "Initial Learning Rate",
                    "Max Iterations",
                    "Early Stopping",
                    "Final Loss",
                    "Total Epochs",
                ],
                "Value": [
                    str(self.config.hidden_layer_sizes),
                    self.config.activation,
                    self.config.solver,
                    self.config.alpha,
                    self.config.learning_rate_init,
                    self.config.max_iter,
                    self.config.early_stopping,
                    getattr(self.model_, "loss_", np.nan) if hasattr(self, "model_") else "Not Fitted",
                    getattr(self.model_, "n_iter_", np.nan) if hasattr(self, "model_") else "Not Fitted",
                ],
            }
        )

        return pd.concat(
            [base, params],
            ignore_index=True,
        )