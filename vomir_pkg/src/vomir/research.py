"""
vomir/research.py — Unsupervised structure discovery in agamotto filter data.

Loads all filter CSVs for a given timeframe, standardizes features, and runs:
  - PCA (explained variance + loadings)
  - KMeans clustering in PCA space
  - UMAP (if umap-learn installed) or t-SNE fallback (2D projection)
  - Cluster analysis: per-cluster mean_ret, std_ret, Sharpe, count, top features
"""

import logging
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Columns that are never features
_META_COLS = {"symbol", "timestamp", "regime", "position", "year", "month"}
_TARGET_COLS = {
    "ret", "ret_raw", "return", "return_long", "return_short",
    "return_long_raw", "return_short_raw",
}
_EXCLUDE_COLS = _META_COLS | _TARGET_COLS

# TF → experiment dir suffix per universe (under gauntlet/)
_TF_DIRS = {
    "liquid": {
        "15m": "pred_agamotto.base.15m_1",
        "1h": "pred_agamotto.base.1h_1",
        "4h": "pred_agamotto.base.4h_1",
        "1d": "pred_agamotto.base.1d_1",
    },
    "altcoins": {
        "15m": "pred_agamotto.base.15m_2",
        "1h": "pred_agamotto.base.1h_2",
        "4h": "pred_agamotto.base.4h_2",
        "1d": "pred_agamotto.base.1d_2",
    },
}

VALID_TFS = list(_TF_DIRS["liquid"].keys())
VALID_UNIVERSES = list(_TF_DIRS.keys())


@dataclass
class PCAResult:
    explained_variance_ratio: np.ndarray   # shape (n_components,)
    cumulative_variance: np.ndarray         # shape (n_components,)
    loadings: pd.DataFrame                  # (n_features, n_components)
    components: np.ndarray                  # (n_samples, n_components) — transformed X


@dataclass
class ClusterResult:
    labels: np.ndarray                      # (n_samples,) int cluster ids
    n_clusters: int
    inertia: float


@dataclass
class UMAPResult:
    coords: np.ndarray                      # (n_samples, 2)
    method: str                             # "umap" or "tsne"


@dataclass
class FeatureMatrix:
    X: np.ndarray          # (n_samples, n_features) — standardized
    y: np.ndarray          # (n_samples,) — ret
    meta: pd.DataFrame     # timestamp / symbol / regime / position columns
    feature_names: list = field(default_factory=list)


class VomirResearch:
    """
    Unsupervised structure discovery for a single agamotto timeframe.

    Parameters
    ----------
    tf : str
        Timeframe — one of "15m", "1h", "4h", "1d".
    data_dir : str
        Root of the marvel repo. The filter CSVs are expected at
        ``{data_dir}/gauntlet/{tf_dir}/filter/filter_*.csv``.
    """

    def __init__(self, tf: str, data_dir: str, universe: str = "liquid"):
        if tf not in VALID_TFS:
            raise ValueError(f"tf must be one of {VALID_TFS}, got {tf!r}")
        if universe not in VALID_UNIVERSES:
            raise ValueError(f"universe must be one of {VALID_UNIVERSES}, got {universe!r}")
        self.tf = tf
        self.universe = universe
        self.data_dir = data_dir
        self._filter_dir = os.path.join(
            data_dir, "gauntlet", _TF_DIRS[universe][tf], "filter"
        )
        self._df: Optional[pd.DataFrame] = None
        self._fm: Optional[FeatureMatrix] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load and stack all filter CSVs for this TF.

        Deduplicates by (timestamp, symbol) — keeps first occurrence.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all rows from all filter files.
        """
        pattern = os.path.join(self._filter_dir, "filter_*.parquet")
        files = sorted(glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No filter parquets found in {self._filter_dir}"
            )
        logger.info("Loading %d filter files for tf=%s", len(files), self.tf)

        chunks = []
        for path in files:
            try:
                df = pd.read_parquet(path)
                # Add regime from filename if column missing
                if "regime" not in df.columns:
                    fname = os.path.splitext(os.path.basename(path))[0]
                    df["regime"] = fname.replace("filter_", "")
                chunks.append(df)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path, exc)

        combined = pd.concat(chunks, ignore_index=True)
        before = len(combined)
        combined = combined.drop_duplicates(subset=["timestamp", "symbol"])
        logger.info(
            "Loaded %d rows (%d after dedup) from %d files",
            before, len(combined), len(files),
        )
        self._df = combined
        return combined

    # ------------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------------

    def get_feature_matrix(self) -> FeatureMatrix:
        """Build a standardized feature matrix from the loaded data.

        Returns
        -------
        FeatureMatrix
            Named tuple with X (standardized ndarray), y (ret ndarray),
            meta (DataFrame with timestamp/symbol/regime/position),
            and feature_names (list of str).
        """
        if self._df is None:
            self.load_data()
        df = self._df

        # Identify feature columns
        feature_cols = [
            c for c in df.columns
            if c not in _EXCLUDE_COLS
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]
        logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols[:10])

        X_raw = df[feature_cols].values.astype(np.float64)
        y = df["ret"].values.astype(np.float64)

        meta_cols = [c for c in ["timestamp", "symbol", "regime", "position"] if c in df.columns]
        meta = df[meta_cols].reset_index(drop=True)

        # Median imputation then standardize
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X_raw)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        fm = FeatureMatrix(X=X_scaled, y=y, meta=meta, feature_names=feature_cols)
        self._fm = fm
        return fm

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    def run_pca(self, n_components: int = 20) -> PCAResult:
        """Fit PCA on the feature matrix.

        Parameters
        ----------
        n_components : int
            Number of principal components to retain.

        Returns
        -------
        PCAResult
        """
        if self._fm is None:
            self.get_feature_matrix()
        fm = self._fm
        n_components = min(n_components, fm.X.shape[1], fm.X.shape[0])

        pca = PCA(n_components=n_components, random_state=42)
        components = pca.fit_transform(fm.X)

        evr = pca.explained_variance_ratio_
        cumvar = np.cumsum(evr)

        loadings = pd.DataFrame(
            pca.components_.T,
            index=fm.feature_names,
            columns=[f"PC{i+1}" for i in range(n_components)],
        )

        logger.info(
            "PCA: %d components explain %.1f%% variance",
            n_components, cumvar[-1] * 100,
        )
        result = PCAResult(
            explained_variance_ratio=evr,
            cumulative_variance=cumvar,
            loadings=loadings,
            components=components,
        )
        self._pca_result = result
        return result

    # ------------------------------------------------------------------
    # KMeans clustering
    # ------------------------------------------------------------------

    def run_clustering(self, n_clusters: int = 8, pca_components: int = 10) -> ClusterResult:
        """KMeans clustering in PCA-reduced space.

        Parameters
        ----------
        n_clusters : int
            Number of KMeans clusters.
        pca_components : int
            Number of PCA components to use as input to KMeans.

        Returns
        -------
        ClusterResult
        """
        if not hasattr(self, "_pca_result"):
            self.run_pca(n_components=max(pca_components, 20))

        pca_result = self._pca_result
        n_use = min(pca_components, pca_result.components.shape[1])
        X_reduced = pca_result.components[:, :n_use]

        logger.info(
            "KMeans: n_clusters=%d on top-%d PCA components (%d samples)",
            n_clusters, n_use, X_reduced.shape[0],
        )
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_reduced)

        result = ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            inertia=float(km.inertia_),
        )
        self._cluster_result = result
        return result

    # ------------------------------------------------------------------
    # UMAP / t-SNE
    # ------------------------------------------------------------------

    def run_umap(
        self,
        n_neighbors: int = 15,
        pca_components: int = 10,
        max_samples: int = 50_000,
    ) -> UMAPResult:
        """2D dimensionality reduction for visualization.

        Tries ``umap-learn`` first; falls back to sklearn ``TSNE``.

        Parameters
        ----------
        n_neighbors : int
            UMAP n_neighbors (ignored for t-SNE fallback).
        pca_components : int
            PCA components used as input (reduces cost of UMAP/t-SNE).
        max_samples : int
            Subsample rows if data exceeds this size (t-SNE is O(n²)).

        Returns
        -------
        UMAPResult
        """
        if not hasattr(self, "_pca_result"):
            self.run_pca(n_components=max(pca_components, 20))

        pca_result = self._pca_result
        n_use = min(pca_components, pca_result.components.shape[1])
        X_reduced = pca_result.components[:, :n_use]

        # Subsample for tractability
        n = X_reduced.shape[0]
        if n > max_samples:
            logger.info(
                "Subsampling %d → %d rows for 2D projection", n, max_samples
            )
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_samples, replace=False)
            X_proj_input = X_reduced[idx]
        else:
            idx = None
            X_proj_input = X_reduced

        try:
            import umap as umap_lib  # noqa: F401
            reducer = umap_lib.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                random_state=42,
            )
            coords = reducer.fit_transform(X_proj_input)
            method = "umap"
            logger.info("UMAP projection complete (%d samples)", len(coords))
        except ImportError:
            from sklearn.manifold import TSNE
            logger.info("umap-learn not available — using t-SNE")
            tsne = TSNE(
                n_components=2,
                perplexity=min(30, X_proj_input.shape[0] - 1),
                random_state=42,
                max_iter=500,
            )
            coords = tsne.fit_transform(X_proj_input)
            method = "tsne"
            logger.info("t-SNE projection complete (%d samples)", len(coords))

        result = UMAPResult(coords=coords, method=method)
        self._umap_result = result
        self._umap_idx = idx  # None if no subsampling
        return result

    # ------------------------------------------------------------------
    # Cluster analysis
    # ------------------------------------------------------------------

    def analyze_clusters(
        self,
        labels: np.ndarray,
        y: np.ndarray,
        top_n_features: int = 5,
    ) -> pd.DataFrame:
        """Per-cluster return statistics and top differentiating features.

        Parameters
        ----------
        labels : np.ndarray
            Cluster label per sample (shape n_samples).
        y : np.ndarray
            True return per sample (shape n_samples).
        top_n_features : int
            How many top features to report per cluster.

        Returns
        -------
        pd.DataFrame
            One row per cluster with columns: cluster, count, mean_ret,
            std_ret, sharpe, top_features.
        """
        if self._fm is None:
            raise RuntimeError("Must call get_feature_matrix() before analyze_clusters()")

        fm = self._fm
        X = fm.X
        feature_names = fm.feature_names

        rows = []
        cluster_ids = np.unique(labels)
        for cid in cluster_ids:
            mask = labels == cid
            y_c = y[mask]
            count = int(mask.sum())
            mean_ret = float(np.mean(y_c))
            std_ret = float(np.std(y_c))
            sharpe = float(mean_ret / std_ret) if std_ret > 0 else 0.0

            # Top features: highest absolute mean deviation from global mean
            X_c = X[mask]
            feature_means = np.mean(X_c, axis=0)
            top_idx = np.argsort(np.abs(feature_means))[::-1][:top_n_features]
            top_features = ",".join(
                f"{feature_names[i]}({feature_means[i]:+.2f})" for i in top_idx
            )

            rows.append({
                "cluster": int(cid),
                "count": count,
                "mean_ret": round(mean_ret, 6),
                "std_ret": round(std_ret, 6),
                "sharpe": round(sharpe, 4),
                "top_features": top_features,
            })

        return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def run_all(
        self,
        n_components: int = 20,
        n_clusters: int = 8,
        pca_for_cluster: int = 10,
        n_neighbors: int = 15,
    ) -> dict:
        """Run the full unsupervised pipeline and return a summary dict.

        Parameters
        ----------
        n_components : int
            PCA components to fit.
        n_clusters : int
            KMeans clusters.
        pca_for_cluster : int
            PCA components used as input to KMeans / UMAP.
        n_neighbors : int
            UMAP n_neighbors.

        Returns
        -------
        dict with keys: pca, cluster, umap, cluster_summary, fm
        """
        logger.info("=== VomirResearch.run_all() tf=%s ===", self.tf)

        fm = self.get_feature_matrix()
        pca = self.run_pca(n_components=n_components)
        cluster = self.run_clustering(n_clusters=n_clusters, pca_components=pca_for_cluster)
        umap_result = self.run_umap(n_neighbors=n_neighbors, pca_components=pca_for_cluster)
        summary = self.analyze_clusters(cluster.labels, fm.y)

        logger.info(
            "Cluster summary:\n%s",
            summary[["cluster", "count", "mean_ret", "sharpe"]].to_string(index=False),
        )

        return {
            "fm": fm,
            "pca": pca,
            "cluster": cluster,
            "umap": umap_result,
            "cluster_summary": summary,
        }
