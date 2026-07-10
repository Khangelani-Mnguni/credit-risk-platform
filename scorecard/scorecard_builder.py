"""
scorecard_builder.py

Converts a trained Logistic Regression model + WoE transformer
into a fully interpretable credit scorecard.

This module produces:
1. Feature-level score contributions
2. Bin-level score allocation
3. Full scorecard table (regulatory format)
4. Exportable governance artifacts

Assumes:
- Features are WoE transformed
- Model is LogisticRegression
- Feature names align with WoE output

Author
------
Credit Risk Platform
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScorecardBuilder:
    """
    Builds a scorecard from a trained logistic regression model
    and a fitted WoE transformer.
    """

    def __init__(self):
        self.scorecard_: Optional[pd.DataFrame] = None
        self.feature_summary_: Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------
    # FIT
    # ---------------------------------------------------------------

    def fit(
        self,
        model: Any,
        woe_transformer: Any,
        feature_names: List[str],
        score_scaler: Optional[Any] = None,
    ) -> "ScorecardBuilder":
        """
        Construct the scorecard mapping from the model and scaler.
        """
        self.model_ = model
        self.woe_ = woe_transformer
        self.feature_names_ = feature_names
        self.score_scaler_ = score_scaler

        # Handle scikit-learn 2D array structures
        coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        intercept = model.intercept_[0] if isinstance(model.intercept_, (list, np.ndarray)) else model.intercept_

        self.coeff_map_ = dict(zip(feature_names, coefficients))
        self.intercept_ = float(intercept)

        # -----------------------------------------------------------
        # SCALING MATH SETUP
        # -----------------------------------------------------------
        n_features = len(feature_names)
        
        if self.score_scaler_ is not None:
            factor = self.score_scaler_.factor_
            offset = self.score_scaler_.offset_
            # Distribute the base intercept points evenly across all features
            base_points_per_feature = (offset - (factor * self.intercept_)) / n_features
        else:
            factor = 1.0
            base_points_per_feature = self.intercept_ / n_features

        logger.info("Building scorecard for %d features...", n_features)
        rows = []

        # -----------------------------------------------------------
        # FEATURE LOOP
        # -----------------------------------------------------------
        for feature in feature_names:
            binner = self._get_binner(feature)
            if binner is None:
                logger.warning("Binner not found for feature: %s", feature)
                continue

            # Use the getter method we defined in optimal_binner.py
            bin_table = binner.get_binning_table()
            feature_coef = self.coeff_map_[feature]

            for _, row in bin_table.iterrows():
                bin_label = row["Bin"]
                
                # 1. Skip the "Totals" row that optbinning appends
                if str(bin_label).strip().lower() == "totals" or "totals" in str(row.values).lower():
                    continue

                # 2. Safely parse WoE (Catch strings like "" or "-")
                raw_woe = row.get("WoE", 0.0)
                try:
                    woe = float(raw_woe)
                except (ValueError, TypeError):
                    woe = 0.0  # Default to 0 impact if empty/invalid
                
                # Standard scorecard formula: Base/N - (WoE * Coef * Factor)
                points = base_points_per_feature - (woe * feature_coef * factor)

                rows.append({
                    "Feature": feature,
                    "Bin": bin_label,
                    "Count": row.get("Count", np.nan),
                    "Event_Rate": row.get("Event rate", np.nan),
                    "WOE": woe,
                    "Coefficient": feature_coef,
                    "Points": points,
                })

        self.scorecard_ = pd.DataFrame(rows)

        # Round points to nearest integer for standard scorecard presentation
        if self.score_scaler_ is not None:
            self.scorecard_["Points"] = self.scorecard_["Points"].round().astype(int)

        # -----------------------------------------------------------
        # FEATURE SUMMARY
        # -----------------------------------------------------------
        
        # Calculate scorecard impact by range (Max points - Min points)
        self.feature_summary_ = (
            self.scorecard_
            .groupby("Feature")
            .agg(
                Min_Points=("Points", "min"),
                Max_Points=("Points", "max"),
                Point_Range=("Points", lambda x: x.max() - x.min()),
                Avg_WOE=("WOE", "mean"),
            )
            .reset_index()
            .sort_values("Point_Range", ascending=False)
        )

        return self

    # ---------------------------------------------------------------
    # TRANSFORM SCORECARD
    # ---------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the scorecard points directly to raw incoming data.
        """
        self._check_is_fitted()
        df = X.copy()
        
        total_scores = np.zeros(len(df))

        for feature in self.feature_names_:
            binner = self._get_binner(feature)
            if binner is None:
                continue

            # Transform raw data to index mapping, then to points
            # Optbinning metric="indices" gives us the row index of the bin table
            bin_indices = binner.transform(X[feature], metric="indices")
            
            # Extract the points array for this feature, aligning with the indices
            feature_points = self.scorecard_[self.scorecard_["Feature"] == feature]["Points"].values
            
            # Map indices to actual points and accumulate
            # Handle out-of-bounds indices (optbinning assigns -1 for errors sometimes)
            points_mapped = np.where(
                (bin_indices >= 0) & (bin_indices < len(feature_points)), 
                feature_points[bin_indices], 
                0
            )
            
            total_scores += points_mapped

        df["Score"] = total_scores
        return df

    # ---------------------------------------------------------------
    # BINNER ACCESS
    # ---------------------------------------------------------------

    def _get_binner(self, feature: str) -> Any:
        """Safely extract the binner from the WoE Transformer."""
        if hasattr(self.woe_, "binners_"):
            return self.woe_.binners_.get(feature)
        if hasattr(self.woe_, "get_binner"):
            return self.woe_.get_binner(feature)
        return None

    # ---------------------------------------------------------------
    # SUMMARY & EXPORT
    # ---------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        self._check_is_fitted()
        return self.feature_summary_.copy()

    def export(self, filepath: Union[str, Path]) -> None:
        self._check_is_fitted()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            self.scorecard_.to_excel(
                writer,
                sheet_name="Scorecard_Detail",
                index=False,
            )
            self.feature_summary_.to_excel(
                writer,
                sheet_name="Feature_Summary",
                index=False,
            )

        logger.info("Scorecard exported to %s", filepath)

    # ---------------------------------------------------------------
    # UTILS
    # ---------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Internal check to ensure fit has been called."""
        if self.scorecard_ is None:
            raise RuntimeError("ScorecardBuilder has not been fitted.")

    def __repr__(self) -> str:
        if self.scorecard_ is not None:
            return (
                f"ScorecardBuilder("
                f"features={len(self.feature_names_)}, "
                f"bins={len(self.scorecard_)})"
            )
        return "ScorecardBuilder(not_fitted)"