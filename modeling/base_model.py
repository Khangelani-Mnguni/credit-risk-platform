"""
base_model.py

Abstract base class for all Credit Risk models.

Provides:
- sklearn compatibility
- serialization
- metadata tracking
- prediction interface
- governance support
- feature importance interface

This class should be inherited by:
- LogisticRegressionModel
- RandomForestModel
- XGBoostModel
- NeuralNetworkModel

Author
------
Credit Risk Platform
"""

from __future__ import annotations

import json
import logging
import platform
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class BaseCreditRiskModel(BaseEstimator, ABC):
    """
    Abstract base class for all supervised credit risk models.
    """

    ####################################################################
    # INITIALIZATION
    ####################################################################

    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0.0",
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.random_state = random_state

    ####################################################################
    # ABSTRACT METHODS
    ####################################################################

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str = "default_flag",
        **kwargs,
    ) -> "BaseCreditRiskModel":
        """
        Train the model.
        """
        pass

    @abstractmethod
    def get_feature_importance(self, *args, **kwargs) -> pd.DataFrame:
        """
        Return feature importance.
        """
        pass

    ####################################################################
    # FITTED CHECK
    ####################################################################

    def _check_is_fitted(self) -> None:
        """Ensure the model has been fitted before making predictions or retrieving metadata."""
        check_is_fitted(
            self,
            attributes=["is_fitted_"],
        )

    ####################################################################
    # PREDICT
    ####################################################################

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for samples in X."""
        self._check_is_fitted()
        return self.model_.predict(X)

    ####################################################################
    # PREDICT PROBA
    ####################################################################

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        self._check_is_fitted()

        if not hasattr(self.model_, "predict_proba"):
            raise AttributeError(
                f"{self.model_name} does not support probability prediction."
            )

        return self.model_.predict_proba(X)

    ####################################################################
    # MODEL METADATA
    ####################################################################

    def metadata(self) -> Dict[str, Any]:
        """Return governance and audit metadata for the fitted model."""
        self._check_is_fitted()

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "training_timestamp": getattr(self, "training_timestamp_", None),
            "training_samples": getattr(self, "training_samples_", None),
            "training_features": getattr(self, "training_features_", None),
            "target_name": getattr(self, "target_name_", None),
            "random_state": self.random_state,
            "training_time_seconds": getattr(self, "training_time_", None),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
        }

    ####################################################################
    # SUMMARY
    ####################################################################

    def summary(self) -> pd.DataFrame:
        """Return the metadata as a formatted pandas DataFrame."""
        self._check_is_fitted()
        meta = self.metadata()

        return pd.DataFrame({
            "Property": list(meta.keys()),
            "Value": list(meta.values()),
        })

    ####################################################################
    # SAVE & LOAD
    ####################################################################

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize the fitted model to disk."""
        self._check_is_fitted()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, filepath)
        logger.info("%s saved to %s", self.model_name, filepath)

        return filepath

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseCreditRiskModel":
        """Load a serialized model from disk."""
        filepath = Path(filepath)
        logger.info("Loading model from %s", filepath)
        return joblib.load(filepath)

    ####################################################################
    # EXPORT METADATA
    ####################################################################

    def export_metadata(self, filepath: Union[str, Path]) -> Path:
        """Export model metadata to a JSON file for governance documentation."""
        self._check_is_fitted()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.metadata(), f, indent=4)

        logger.info("Metadata exported to %s", filepath)
        return filepath

    ####################################################################
    # GETTERS
    ####################################################################

    def get_feature_names(self) -> List[str]:
        """Return the list of features the model was trained on."""
        self._check_is_fitted()
        return list(self.feature_names_)

    @property
    def n_features_(self) -> int:
        """Return the number of features the model was trained on."""
        self._check_is_fitted()
        return len(self.feature_names_)

    @property
    def n_samples_(self) -> int:
        """Return the number of observations the model was trained on."""
        self._check_is_fitted()
        return self.training_samples_

    ####################################################################
    # REPR
    ####################################################################

    def __repr__(self) -> str:
        if hasattr(self, "is_fitted_"):
            return (
                f"{self.__class__.__name__}("
                f"name={self.model_name}, "
                f"features={self.n_features_}, "
                f"samples={self.n_samples_})"
            )
        return f"{self.__class__.__name__}(name={self.model_name})"