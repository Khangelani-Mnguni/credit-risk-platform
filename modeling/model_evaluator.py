"""
model_evaluator.py

Comprehensive evaluation toolkit for Credit Risk models.

Provides:
- ROC-AUC
- KS Statistic
- Gini
- Precision
- Recall
- Specificity
- Accuracy
- F1
- PR-AUC
- Log Loss
- Brier Score
- Confusion Matrix
- Gains/Decile Table

Designed to evaluate any binary classification model
used within the Credit Risk Platform.

Author
------
Credit Risk Platform
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation thresholds."""
    threshold: float = 0.50


class ModelEvaluator:
    """
    Production-grade binary classifier evaluator tailored for Credit Risk.
    """

    def __init__(self, config: EvaluationConfig = EvaluationConfig()):
        self.config = config
        self.metrics_: Dict[str, float] = {}

    ####################################################################
    # MAIN ENTRY POINT
    ####################################################################

    def evaluate(
        self,
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
    ) -> "ModelEvaluator":
        """
        Calculate all performance metrics.

        Parameters
        ----------
        y_true : array-like
            Ground truth binary labels (0 or 1).
        y_prob : array-like
            Predicted probabilities of the positive class (1).

        Returns
        -------
        self
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have the same length.")

        y_pred = (y_prob >= self.config.threshold).astype(int)

        # Confusion Matrix Elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Core Classification Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Probabilistic Metrics
        auc = roc_auc_score(y_true, y_prob)
        gini = 2 * auc - 1
        pr_auc = average_precision_score(y_true, y_prob)
        ll = log_loss(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)

        # Kolmogorov-Smirnov (KS) Statistic
        ks = ks_2samp(
            y_prob[y_true == 0],
            y_prob[y_true == 1],
        ).statistic

        self.metrics_ = {
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "Specificity": float(specificity),
            "F1": float(f1),
            "ROC_AUC": float(auc),
            "Gini": float(gini),
            "KS": float(ks),
            "PR_AUC": float(pr_auc),
            "LogLoss": float(ll),
            "BrierScore": float(brier),
            "Threshold": float(self.config.threshold),
            "TP": float(tp),
            "FP": float(fp),
            "TN": float(tn),
            "FN": float(fn),
        }

        logger.info("Model evaluation completed. Gini: %.4f | KS: %.4f", gini, ks)

        return self

    ####################################################################
    # SUMMARY
    ####################################################################

    def summary(self) -> pd.DataFrame:
        """Return all calculated metrics as a DataFrame."""
        self._check_is_evaluated()
        
        return pd.DataFrame({
            "Metric": list(self.metrics_.keys()),
            "Value": list(self.metrics_.values()),
        })

    ####################################################################
    # METRICS DICTIONARY
    ####################################################################

    def metrics(self) -> Dict[str, float]:
        """Return all calculated metrics as a dictionary."""
        self._check_is_evaluated()
        return self.metrics_

    ####################################################################
    # CONFUSION MATRIX
    ####################################################################

    def confusion(self) -> pd.DataFrame:
        """Return the confusion matrix as a formatted DataFrame."""
        self._check_is_evaluated()
        
        return pd.DataFrame(
            [
                [self.metrics_["TN"], self.metrics_["FP"]],
                [self.metrics_["FN"], self.metrics_["TP"]],
            ],
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        )

    ####################################################################
    # DECILE TABLE (LIFT / GAINS)
    ####################################################################

    def decile_table(
        self,
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """
        Generate a decile gains table for credit risk reporting.
        Automatically ranks from highest risk (Decile 1) to lowest risk.
        """
        df = pd.DataFrame({
            "target": np.asarray(y_true),
            "probability": np.asarray(y_prob),
        })

        # Rank method 'first' ensures exactly 10 bins even with heavily tied probabilities
        df["decile"] = pd.qcut(
            df["probability"].rank(method="first"), 
            10, 
            labels=False
        )

        # Reverse so Decile 1 represents the highest probability of default
        max_dec = df["decile"].max()
        df["decile"] = (max_dec - df["decile"] + 1).astype(int)

        # Aggregate metrics per decile
        report = (
            df.groupby("decile")
            .agg(
                observations=("target", "count"),
                defaults=("target", "sum"),
                avg_probability=("probability", "mean"),
            )
            .sort_index(ascending=True)
        )

        report["non_defaults"] = report["observations"] - report["defaults"]
        report["default_rate"] = report["defaults"] / report["observations"]
        
        # Cumulative metrics for Lift and KS Curves
        report["cum_observations"] = report["observations"].cumsum()
        report["cum_defaults"] = report["defaults"].cumsum()
        report["cum_non_defaults"] = report["non_defaults"].cumsum()
        
        total_defaults = report["defaults"].sum()
        total_non_defaults = report["non_defaults"].sum()
        
        report["cum_default_capture_rate"] = report["cum_defaults"] / total_defaults
        report["cum_non_default_capture_rate"] = report["cum_non_defaults"] / total_non_defaults
        
        # KS per decile
        report["ks_stat"] = np.abs(report["cum_default_capture_rate"] - report["cum_non_default_capture_rate"])

        return report.reset_index()

    ####################################################################
    # EXPORT
    ####################################################################

    def export_excel(self, filepath: Union[str, Path]) -> None:
        """Export the evaluation summary and confusion matrix to Excel."""
        self._check_is_evaluated()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            self.summary().to_excel(
                writer,
                sheet_name="Metrics",
                index=False,
            )
            self.confusion().to_excel(
                writer,
                sheet_name="ConfusionMatrix",
            )

        logger.info("Evaluation report exported to %s", filepath)

    ####################################################################
    # UTILS
    ####################################################################

    def _check_is_evaluated(self) -> None:
        """Internal check to ensure evaluate() has been called."""
        if not self.metrics_:
            raise RuntimeError(
                "ModelEvaluator has not calculated metrics yet. "
                "Call evaluate() first."
            )