"""
iv_calculator.py

Standalone Information Value (IV) calculator for credit scorecard
development.

This module calculates Information Value (IV) using fitted
OptimalBinner objects or directly from WOETransformer outputs.

Author
------
Credit Risk Platform
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Information Value Interpretation
# ---------------------------------------------------------------------

IV_RULES: List[Tuple[float, str]] = [
    (0.02, "Not Predictive"),
    (0.10, "Weak"),
    (0.30, "Medium"),
    (0.50, "Strong"),
    (float("inf"), "Suspicious"),
]


# ---------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------

@dataclass
class IVResult:
    """Information Value summary for a single feature."""
    feature: str
    iv: float
    strength: str
    bins: int
    converged: bool


# ---------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------

class IVCalculator:
    """
    Calculate Information Value (IV) for fitted OptimalBinner objects.

    Examples
    --------
    calculator = IVCalculator()
    calculator.fit(woe.binners_)
    calculator.summary()
    calculator.select_features(min_iv=0.02)
    """

    def __init__(self):
        self.results_: List[IVResult] = []
        self.summary_: Optional[pd.DataFrame] = None

    ####################################################################
    # FIT
    ####################################################################

    def fit(self, binners: Dict[str, Any]) -> "IVCalculator":
        """
        Calculate IV for every fitted binner.

        Parameters
        ----------
        binners : dict
            Dictionary mapping feature names to OptimalBinner objects.

        Returns
        -------
        self
        """
        logger.info("Calculating Information Value...")
        
        rows = []
        self.results_ = []

        for feature, binner in binners.items():
            diag = binner.diagnostics
            
            result = IVResult(
                feature=feature,
                iv=diag.iv,
                strength=self._strength(diag.iv),
                bins=diag.n_bins,
                converged=diag.converged,
            )
            
            self.results_.append(result)
            rows.append({
                "Feature": result.feature,
                "IV": result.iv,
                "Strength": result.strength,
                "Bins": result.bins,
                "Converged": result.converged,
            })

        self.summary_ = (
            pd.DataFrame(rows)
            .sort_values("IV", ascending=False)
            .reset_index(drop=True)
        )

        logger.info("Calculated IV for %d features.", len(rows))
        return self

    ####################################################################
    # SUMMARY
    ####################################################################

    def summary(self) -> pd.DataFrame:
        """Return IV summary."""
        self._check_is_fitted()
        return self.summary_.copy()

    ####################################################################
    # FEATURE SELECTION & FILTERING
    ####################################################################

    def select_features(self, min_iv: float = 0.02, max_iv: float = 0.50) -> List[str]:
        """
        Select features using Information Value.

        Parameters
        ----------
        min_iv : float
        max_iv : float

        Returns
        -------
        list of str
        """
        self._check_is_fitted()
        selected = self.summary_.loc[
            (self.summary_["IV"] >= min_iv) & (self.summary_["IV"] <= max_iv),
            "Feature"
        ]
        return selected.tolist()

    def filter_summary(self, min_iv: float = 0.02, max_iv: float = 0.50) -> pd.DataFrame:
        """Return filtered IV table."""
        self._check_is_fitted()
        return (
            self.summary_
            .loc[(self.summary_["IV"] >= min_iv) & (self.summary_["IV"] <= max_iv)]
            .reset_index(drop=True)
        )

    def top(self, n: int = 20) -> pd.DataFrame:
        """Return top IV variables."""
        self._check_is_fitted()
        return self.summary_.head(n)

    def weak_features(self, threshold: float = 0.02) -> List[str]:
        """Return weak predictors."""
        self._check_is_fitted()
        return self.summary_.loc[self.summary_["IV"] < threshold, "Feature"].tolist()

    def suspicious_features(self, threshold: float = 0.50) -> List[str]:
        """
        Return suspicious predictors.
        High IV values often indicate target leakage.
        """
        self._check_is_fitted()
        return self.summary_.loc[self.summary_["IV"] > threshold, "Feature"].tolist()

    ####################################################################
    # PLOT
    ####################################################################

    def plot(self, top_n: int = 20):
        """
        Plot Information Value. Requires matplotlib.
        """
        self._check_is_fitted()
        
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Matplotlib is required for plotting. Install it using `pip install matplotlib`."
            ) from e

        df = self.summary_.head(top_n)

        plt.figure(figsize=(10, 6))
        plt.barh(df["Feature"], df["IV"])
        plt.gca().invert_yaxis()
        plt.xlabel("Information Value")
        plt.title("Feature Information Value")
        plt.tight_layout()
        plt.show()

    ####################################################################
    # EXPORT
    ####################################################################

    def export(self, filepath: Union[str, Path], index: bool = False) -> Path:
        """
        Export the IV summary to a CSV or Excel file.

        The output format is determined from the file extension.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Output file path.
        index : bool, default=False
            Whether to write the DataFrame index.

        Returns
        -------
        pathlib.Path
        """
        self._check_is_fitted()
        filepath = Path(filepath)

        # Create parent directories if they don't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        suffix = filepath.suffix.lower()

        if suffix == ".csv":
            self.summary_.to_csv(filepath, index=index)
        elif suffix == ".xlsx":
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                self.summary_.to_excel(writer, sheet_name="Information Value", index=index)
        else:
            raise ValueError("Unsupported file format. Use '.csv' or '.xlsx'.")

        logger.info("IV summary exported to %s", filepath)
        return filepath

    ####################################################################
    # PRIVATE HELPERS
    ####################################################################

    @staticmethod
    def _strength(iv: float) -> str:
        """Map a numeric IV to a categorical strength label."""
        return next(
            (label for threshold, label in IV_RULES if iv < threshold), "Unknown"
        )

    def _check_is_fitted(self):
        """Internal check to verify calculator has been fitted."""
        if self.summary_ is None:
            raise RuntimeError("Calculator has not been fitted.")