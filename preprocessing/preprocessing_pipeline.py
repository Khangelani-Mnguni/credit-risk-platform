"""
preprocessing_pipeline.py

End-to-end credit risk preprocessing pipeline.

This pipeline orchestrates:
1. Missing value imputation
2. Optimal binning
3. WoE transformation
4. IV-based feature selection
5. Correlation filtering
6. VIF filtering

Designed for:
- Train / validation / test consistency
- Sklearn compatibility
- Production reproducibility

Author
------
Credit Risk Platform
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union

# Assuming these are available in your local module structure
from .group_mean_imputer import FeatureSpecificGroupMeanImputer
from .missing_category_imputer import MissingCategoryImputer
from .woe_transformer import WOETransformer
from .iv_feature_selector import IVFeatureSelector
from .correlation_filter import CorrelationFilter
from .vif_selector import VIFSelector

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration parameters for the preprocessing pipeline."""
    
    # Required parameters for imputers
    feature_group_map: Dict[str, str]
    categorical_features: List[str]
    
    # Thresholds for feature selection
    iv_min: float = 0.02
    iv_max: float = 0.50
    corr_threshold: float = 0.70
    vif_threshold: float = 5.0


class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Full credit risk preprocessing pipeline orchestrator.

    Parameters
    ----------
    config : PreprocessingConfig
        Dataclass containing thresholds and feature maps.
    woe_binning_strategy : str, default="optimal"
        Strategy for binning (e.g., optimal).
    """

    def __init__(
        self,
        config: PreprocessingConfig,
        woe_binning_strategy: str = "optimal",
    ):
        self.config = config
        self.woe_binning_strategy = woe_binning_strategy

    # ---------------------------------------------------------------
    # FIT
    # ---------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PreprocessingPipeline":
        """
        Fit all steps in the preprocessing pipeline sequentially.
        """
        logger.info("Starting preprocessing pipeline fit...")

        # 1. Missing value imputation
        logger.info("Step 1: Imputing missing values...")
        self.num_imputer_ = FeatureSpecificGroupMeanImputer(
            feature_group_map=self.config.feature_group_map
        )
        self.cat_imputer_ = MissingCategoryImputer(
            categorical_features=self.config.categorical_features
        )

        X_num = self.num_imputer_.fit_transform(X, y)
        X_cat = self.cat_imputer_.fit_transform(X_num, y)

        # 2. WoE transformation
        logger.info("Step 2: Fitting Optimal Binning and WoE...")
        self.woe_ = WOETransformer()
        X_woe = self.woe_.fit_transform(X_cat, y)

        # 3. IV selection
        logger.info("Step 3: Filtering features by Information Value...")
        self.iv_selector_ = IVFeatureSelector(
            min_iv=self.config.iv_min,
            max_iv=self.config.iv_max,
        )
        # Pass the IV summary from the WOE transformer into the IV selector
        X_iv = self.iv_selector_.fit_transform(
            X_woe, 
            y, 
            iv_summary=self.woe_.summary()
        )

        # 4. Correlation filtering
        logger.info("Step 4: Filtering correlated features...")
        self.corr_filter_ = CorrelationFilter(
            threshold=self.config.corr_threshold,
            iv_source=self.woe_,
        )
        X_corr = self.corr_filter_.fit_transform(X_iv)

        # 5. VIF filtering
        logger.info("Step 5: Applying VIF selection...")
        self.vif_selector_ = VIFSelector(
            threshold=self.config.vif_threshold,
        )
        # VIFSelector handles the target variable internally via X inter-correlations
        self.vif_selector_.fit(X_corr)

        # Store final selected features
        self.selected_features_ = self.vif_selector_.selected_features_
        self.is_fitted_ = True

        logger.info(
            "Preprocessing pipeline successfully fitted. Final feature count: %d",
            len(self.selected_features_),
        )

        return self

    # ---------------------------------------------------------------
    # TRANSFORM
    # ---------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted pipeline to new data.
        """
        self._check_is_fitted()
        logger.info("Applying preprocessing pipeline transformations...")

        X_num = self.num_imputer_.transform(X)
        X_cat = self.cat_imputer_.transform(X_num)
        
        X_woe = self.woe_.transform(X_cat)
        X_iv = self.iv_selector_.transform(X_woe)
        X_corr = self.corr_filter_.transform(X_iv)

        return self.vif_selector_.transform(X_corr)

    # ---------------------------------------------------------------
    # FIT TRANSFORM
    # ---------------------------------------------------------------

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit pipeline and transform the training data.
        """
        return self.fit(X, y).transform(X)

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Returns a basic summary of the final features retained.
        """
        self._check_is_fitted()
        
        return pd.DataFrame(
            {"Final Features": self.selected_features_}
        )

    # ---------------------------------------------------------------
    # INTERNAL CHECK
    # ---------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Ensure pipeline is fitted before calling transform."""
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("PreprocessingPipeline has not been fitted.")
        
        
    # ---------------------------------------------------------------
    # SERIALIZATION
    # ---------------------------------------------------------------

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize the fitted pipeline to disk."""
        self._check_is_fitted()
        filepath = Path(filepath)
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        logger.info("PreprocessingPipeline saved to %s", filepath)
        return filepath

    @staticmethod
    def load(filepath: Union[str, Path]) -> "PreprocessingPipeline":
        """Load a fitted pipeline from disk."""
        filepath = Path(filepath)
        logger.info("Loading PreprocessingPipeline from %s", filepath)
        return joblib.load(filepath)