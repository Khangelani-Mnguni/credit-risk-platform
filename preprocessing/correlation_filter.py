"""
correlation_filter.py

Graph-based correlation filter for WoE-transformed features.

Improvements over naive pairwise filtering:
- Uses connected components (correlation clusters)
- Deterministic selection
- IV-based feature retention within clusters
- Fully reproducible for governance

Author: Credit Risk Platform
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Removes highly correlated variables using graph clustering.

    Parameters
    ----------
    threshold : float, default=0.7
        Absolute correlation threshold.
    iv_source : object, required
        IVCalculator or WOETransformer providing IV table.
        Must have either a `.summary()` method or `.iv_table_` attribute.

    Attributes
    ----------
    corr_matrix_ : pandas.DataFrame
        Absolute correlation matrix of the fitted features.
    selected_features_ : list of str
        Features retained after filtering.
    removed_features_ : list of str
        Features dropped due to high correlation.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        iv_source: Optional[Any] = None,
    ):
        self.threshold = threshold
        self.iv_source = iv_source

    # ---------------------------------------------------------------
    # FIT
    # ---------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CorrelationFilter":
        """
        Build correlation graph and select features based on IV.
        """
        self._check_input(X)

        logger.info("Calculating absolute correlation matrix...")
        self.corr_matrix_ = X.corr().abs()

        iv_table = self._get_iv_table()
        iv_lookup = iv_table.set_index("Feature")["IV"].to_dict()

        logger.info("Building correlation graph (threshold >= %s)...", self.threshold)
        adjacency = self._build_graph(self.corr_matrix_)
        clusters = self._get_connected_components(adjacency, list(X.columns))

        selected: List[str] = []
        removed: Set[str] = set()

        for cluster in clusters:
            if len(cluster) == 1:
                selected.append(cluster[0])
                continue

            # Sort alphabetically first to guarantee deterministic tie-breaking 
            # if two features have the exact same IV.
            cluster_sorted = sorted(cluster)
            
            # Select highest IV feature in the cluster
            best_feature = max(
                cluster_sorted,
                key=lambda f: iv_lookup.get(f, 0.0),
            )

            selected.append(best_feature)

            for f in cluster:
                if f != best_feature:
                    removed.add(f)

        self.selected_features_ = selected
        self.removed_features_ = sorted(list(removed))

        logger.info(
            "Correlation filter: kept %d features, removed %d features.",
            len(self.selected_features_),
            len(self.removed_features_),
        )

        return self

    # ---------------------------------------------------------------
    # TRANSFORM
    # ---------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the DataFrame to retain only uncorrelated features.
        """
        self._check_is_fitted()
        
        if (missing := [col for col in self.selected_features_ if col not in X.columns]):
            raise ValueError(f"Selected features missing from input: {missing}")

        return X.loc[:, self.selected_features_].copy()

    # ---------------------------------------------------------------
    # GRAPH BUILDING
    # ---------------------------------------------------------------

    def _build_graph(self, corr: pd.DataFrame) -> Dict[str, List[str]]:
        """Construct adjacency list for features above correlation threshold."""
        graph = {col: [] for col in corr.columns}
        cols = corr.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr.iloc[i, j] >= self.threshold:
                    f1, f2 = cols[i], cols[j]
                    graph[f1].append(f2)
                    graph[f2].append(f1)

        return graph

    # ---------------------------------------------------------------
    # CONNECTED COMPONENTS
    # ---------------------------------------------------------------

    def _get_connected_components(
        self,
        graph: Dict[str, List[str]],
        nodes: List[str],
    ) -> List[List[str]]:
        """
        Find connected components using an iterative Depth-First Search.
        Iterative approach prevents RecursionError on large graphs.
        """
        visited = set()
        components = []

        def iterative_dfs(start_node: str, component: List[str]):
            stack = [start_node]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    
                    for neigh in graph[node]:
                        if neigh not in visited:
                            stack.append(neigh)

        for node in nodes:
            if node not in visited:
                comp = []
                iterative_dfs(node, comp)
                components.append(comp)

        return components

    # ---------------------------------------------------------------
    # IV SOURCE
    # ---------------------------------------------------------------

    def _get_iv_table(self) -> pd.DataFrame:
        """Extract IV table from the provided iv_source."""
        if self.iv_source is None:
            raise ValueError("iv_source is required to determine which correlated features to keep.")

        if hasattr(self.iv_source, "summary"):
            return self.iv_source.summary()

        if hasattr(self.iv_source, "iv_table_"):
            return self.iv_source.iv_table_

        raise ValueError("Invalid iv_source format. Must have a 'summary()' method or 'iv_table_' attribute.")

    # ---------------------------------------------------------------
    # UTILS
    # ---------------------------------------------------------------

    def _check_input(self, X: pd.DataFrame) -> None:
        """Ensure input is a DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

    def _check_is_fitted(self) -> None:
        """Ensure the filter has been fitted before transforming."""
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("CorrelationFilter has not been fitted.")

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Return a summary table of features indicating their selection status.
        """
        self._check_is_fitted()

        iv_table = self._get_iv_table().copy()

        iv_table["Selected"] = iv_table["Feature"].isin(self.selected_features_)
        iv_table["Removed_Correlation"] = iv_table["Feature"].isin(self.removed_features_)

        return iv_table.sort_values("IV", ascending=False).reset_index(drop=True)