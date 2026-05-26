"""Leak-safe preprocessing & feature engineering pipeline.

The single rule we enforce here: every stateful transformer must be fitted
strictly on the training fold, never on the full dataset. Wrapping everything
into a sklearn Pipeline guarantees that — when used inside KFold.split() —
the validation fold sees only transforms learned on its complementary train
fold. This removes the residual leakage from the baseline notebook where
imputer/scaler/PCA/KMeans were fit on the entire train before CV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop columns that are highly correlated with an already-kept column."""

    def __init__(self, threshold: float = 0.98) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> CorrelationFilter:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop: set[str] = set()
        for col in upper.columns:
            if col in to_drop:
                continue
            high = upper.index[upper[col] > self.threshold]
            to_drop.update(high)
        self.cols_to_drop_ = sorted(to_drop)
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.cols_to_drop_, errors="ignore")

    def get_feature_names_out(self, input_features=None):
        kept = [c for c in self.feature_names_in_ if c not in self.cols_to_drop_]
        return np.array(kept)


class UnsupervisedFeatureExpander(BaseEstimator, TransformerMixin):
    """Append PCA components, KMeans cluster id, and per-cluster distances.

    This is the same idea as the baseline notebook, but learned inside the CV
    fold rather than on the whole training set.
    """

    def __init__(self, n_pca: int = 40, n_clusters: int = 12, random_state: int = 42) -> None:
        self.n_pca = n_pca
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> UnsupervisedFeatureExpander:
        n_pca = min(self.n_pca, X.shape[1], X.shape[0] - 1)
        self.pca_ = PCA(n_components=n_pca, random_state=self.random_state).fit(X)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20,
        ).fit(X)
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        idx = X.index
        pca_arr = self.pca_.transform(X)
        pca_df = pd.DataFrame(
            pca_arr,
            columns=[f"pca_{i}" for i in range(pca_arr.shape[1])],
            index=idx,
        )
        cluster = pd.Series(self.kmeans_.predict(X), index=idx, name="cluster")
        dist_arr = self.kmeans_.transform(X)
        dist_df = pd.DataFrame(
            dist_arr,
            columns=[f"cluster_dist_{i}" for i in range(dist_arr.shape[1])],
            index=idx,
        )
        return pd.concat([X.reset_index(drop=False).set_index(idx), pca_df, cluster, dist_df], axis=1).loc[
            :, lambda df: ~df.columns.duplicated()
        ].drop(columns="index", errors="ignore")


def build_preprocessor(
    *,
    corr_threshold: float = 0.98,
    n_pca: int = 40,
    n_clusters: int = 12,
    random_state: int = 42,
) -> Pipeline:
    """Full leak-safe preprocessing pipeline.

    Steps:
        median impute -> drop constant cols -> standardize -> drop highly
        correlated -> append PCA + KMeans features.
    """
    impute = SimpleImputer(strategy="median").set_output(transform="pandas")
    variance = VarianceThreshold(threshold=0.0).set_output(transform="pandas")
    scale = StandardScaler().set_output(transform="pandas")
    corr = CorrelationFilter(threshold=corr_threshold)
    expand = UnsupervisedFeatureExpander(
        n_pca=n_pca, n_clusters=n_clusters, random_state=random_state
    )
    return Pipeline(
        steps=[
            ("impute", impute),
            ("variance", variance),
            ("scale", scale),
            ("corr", corr),
            ("expand", expand),
        ]
    )
