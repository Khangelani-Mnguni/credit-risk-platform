"""
optimal_binner.py

Wrapper around optbinning.OptimalBinning providing a consistent API
for credit scorecard development.

Supports:

- Numerical variables
- Categorical variables
- Weight of Evidence transformation
- Event rate transformation
- Bin index transformation
- Binning tables
- Plotting
- Serialization

Author
------
Credit Risk Platform
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional, List

from matplotlib.table import table

import joblib
import pandas as pd
from optbinning import OptimalBinning

logger = logging.getLogger(__name__)

@dataclass
class BinStatistics:
    """
    Stores statistics for a fitted OptimalBinning model.
    """

    feature: str

    bins: List

    counts: List[int]

    goods: List[int]

    bads: List[int]

    event_rate: List[float]

    woe: List[float]

    iv: float
    
@dataclass
class BinningDiagnostics:
    """
    Stores optimisation diagnostics.
    """

    feature: str

    dtype: str

    status: str

    iv: float

    n_bins: int

    monotonic_trend: str

    converged: bool

class OptimalBinner:
    """
    Wrapper around optbinning.OptimalBinning.

    Parameters
    ----------
    name : str

        Feature name.

    dtype : {"numerical","categorical"}

    max_n_bins : int, default=6

    min_bin_size : float, default=0.05

    monotonic_trend : str, default="auto"

    special_codes : optional
    """
    @property
    def statistics(self) -> BinStatistics:
      """
      Return all fitted bin statistics.
      """

      table = self.get_binning_table()

      rows = table.iloc[:-1]

      return BinStatistics(

        feature=self.name,

        bins=rows["Bin"].tolist(),

        counts=rows["Count"].tolist(),

        goods=rows["Non-event"].tolist(),

        bads=rows["Event"].tolist(),

        event_rate=rows["Event rate"].tolist(),

        woe=rows["WoE"].tolist(),

        iv=self.iv,

    )  
      
    @property
    def diagnostics(self) -> BinningDiagnostics:
      """
      Return optimisation diagnostics.
      """

      table = self.get_binning_table()

      return BinningDiagnostics(

        feature=self.name,

        dtype=self.dtype,

        status=self.status,

        iv=self.iv,

        n_bins=len(table) - 1,

        monotonic_trend=self.monotonic_trend,

        converged=self.status == "OPTIMAL",

      )  
      
    @property
    def woe_values(self):

        return self.statistics.woe
    
    @property
    def event_rates(self):

        return self.statistics.event_rate
    
    @property
    def goods(self):

        return self.statistics.goods
    
    @property
    def goods(self):

        return self.statistics.bads

    @property
    def bads(self):

        return self.statistics.counts

    @property
    def bads(self):

        return self.statistics.bins

    @staticmethod
    def diagnostics_report(
        binners: dict,
    ) -> pd.DataFrame:
        """
        Create a diagnostics report for multiple fitted binners.

        Parameters
        ----------
        binners : dict
            Dictionary mapping feature names to fitted OptimalBinner instances.

        Returns
        -------
        pandas.DataFrame
            Diagnostics summary.
        """

        rows = []

        for binner in binners.values():

            d = binner.diagnostics

            rows.append(
                {
                    "Feature": d.feature,
                    "Type": d.dtype,
                    "IV": d.iv,
                    "Bins": d.n_bins,
                    "Status": d.status,
                    "Converged": d.converged,
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values("IV", ascending=False)
            .reset_index(drop=True)
        )

    def __init__(
        self,
        name: str,
        dtype: Literal["numerical", "categorical"],
        max_n_bins: int = 6,
        min_bin_size: float = 0.05,
        monotonic_trend: str = "auto",
        special_codes=None,
    ):

        self.name = name
        self.dtype = dtype
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.monotonic_trend = monotonic_trend
        self.special_codes = special_codes

    ####################################################################
    # FIT
    ####################################################################

    def fit(
        self,
        X: pd.Series,
        y: pd.Series,
    ) -> "OptimalBinner":
        """
        Fit optimal binning.

        Parameters
        ----------
        X : pandas.Series

        y : pandas.Series

        Returns
        -------
        self
        """

        logger.info("Fitting optimal binning for %s", self.name)

        self.model_ = OptimalBinning(
            name=self.name,
            dtype=self.dtype,
            max_n_bins=self.max_n_bins,
            min_bin_size=self.min_bin_size,
            monotonic_trend=self.monotonic_trend,
            special_codes=self.special_codes,
        )

        self.model_.fit(X, y)

        self.table_ = self.model_.binning_table.build()

        self.iv_ = float(
            self.table_.loc["Totals", "IV"]
        )

        logger.info(
            "%s fitted successfully (IV=%.4f)",
            self.name,
            self.iv_,
        )

        return self

    ####################################################################
    # TRANSFORM
    ####################################################################

    def transform(
        self,
        X: pd.Series,
        metric: Literal[
            "woe",
            "event_rate",
            "indices",
            "bins"
        ] = "woe",
    ) -> pd.Series:
        """
        Transform values.

        Parameters
        ----------
        X : pandas.Series

        metric : str

        Returns
        -------
        pandas.Series
        """

        if not hasattr(self, "model_"):
            raise RuntimeError(
                "Binner has not been fitted."
            )

        values = self.model_.transform(
            X,
            metric=metric,
        )

        return pd.Series(
            values,
            index=X.index,
            name=X.name,
        )

    ####################################################################
    # BINNING TABLE
    ####################################################################

    def get_binning_table(
        self,
    ) -> pd.DataFrame:
        """
        Return binning table.
        """

        if not hasattr(self, "table_"):
            raise RuntimeError(
                "Model has not been fitted."
            )

        return self.table_.copy()

    ####################################################################
    # IV
    ####################################################################

    @property
    def iv(self) -> float:
        """
        Information Value.
        """

        if not hasattr(self, "iv_"):
            raise RuntimeError(
                "Model has not been fitted."
            )

        return self.iv_

    ####################################################################
    # SPLITS
    ####################################################################

    @property
    def splits(self):
        """
        Return numerical cut points.

        Returns
        -------
        ndarray
        """

        if not hasattr(self, "model_"):
            raise RuntimeError(
                "Model has not been fitted."
            )

        return self.model_.splits

    ####################################################################
    # STATUS
    ####################################################################

    @property
    def status(self):
        """
        Return optimisation status.
        """

        if not hasattr(self, "model_"):
            raise RuntimeError(
                "Model has not been fitted."
            )

        return self.model_.status

    ####################################################################
    # PLOT
    ####################################################################

    def plot(
        self,
        metric: str = "woe",
        figsize=(8, 5),
    ):
        """
        Plot optimal binning.

        Parameters
        ----------
        metric : str

        figsize : tuple
        """

        if not hasattr(self, "model_"):
            raise RuntimeError(
                "Model has not been fitted."
            )

        self.model_.binning_table.plot(
            metric=metric,
            figsize=figsize,
        )

    ####################################################################
    # SAVE
    ####################################################################

    def save(
        self,
        filepath: str | Path,
    ):
        """
        Save fitted binner.
        """

        if not hasattr(self, "model_"):
            raise RuntimeError(
                "Model has not been fitted."
            )

        joblib.dump(
            self,
            filepath,
        )

        logger.info(
            "Saved binner to %s",
            filepath,
        )

    ####################################################################
    # LOAD
    ####################################################################

    @staticmethod
    def load(
        filepath: str | Path,
    ) -> "OptimalBinner":
        """
        Load fitted binner.
        """

        logger.info(
            "Loading binner from %s",
            filepath,
        )

        return joblib.load(filepath)

    ####################################################################
    # SUMMARY
    ####################################################################

    def summary(self) -> pd.DataFrame:
        """
        Return model summary.
        """

        if not hasattr(self, "model_"):
            raise RuntimeError(
                "Model has not been fitted."
            )

        return pd.DataFrame(
            {
                "feature": [self.name],
                "dtype": [self.dtype],
                "iv": [self.iv_],
                "status": [self.status],
                "bins": [
                    len(self.table_) - 1
                ],
            }
        )