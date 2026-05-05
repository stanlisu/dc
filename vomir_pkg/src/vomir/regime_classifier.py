"""
vomir/regime_classifier.py — Per-regime-group cluster-then-classify pipeline.

Extends VomirClassifier with a 4-group regime partitioning:
  baseline (0), high_volume (1), low_volume (2), vol_breakout (3)

Per-window pipeline:
  For each regime_group:
    subset_train → impute → StandardScaler → PCA(n_comp[g]) → KMeans(k[g]) → cluster_id

  Recombine all groups:
    X_aug = [global_scaled_features | cluster_id | regime_group_id (0-3)]

  Train LGBMClassifier(3-class, balanced) on augmented X.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from vomir.classifier import (
    VomirClassifier,
    ClassifierWindow,
    WindowResult,
    CLASS_SHORT,
    CLASS_NEUTRAL,
    CLASS_LONG,
    _sharpe,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime group definitions — order determines group_id (0-3)
# ---------------------------------------------------------------------------

REGIME_GROUP_NAMES = ["baseline", "high_volume", "low_volume", "vol_breakout"]
REGIME_GROUP_ID: Dict[str, int] = {name: i for i, name in enumerate(REGIME_GROUP_NAMES)}

# Min rows required in a group to run PCA+KMeans (else cluster_id=0)
_MIN_ROWS_FOR_CLUSTERING = 100


def _regime_to_group(regime: str) -> int:
    """Map a regime string to its group_id (0-3).

    Uses prefix matching: 'high_volume_and_macd_bullish_long' → 1.
    Unrecognised regimes fall back to 'baseline' (0).
    """
    for name in REGIME_GROUP_NAMES:
        if regime.startswith(name):
            return REGIME_GROUP_ID[name]
    logger.warning("Unknown regime prefix: %r — assigning to baseline (0)", regime)
    return REGIME_GROUP_ID["baseline"]


# ---------------------------------------------------------------------------
# VomirRegimeClassifier
# ---------------------------------------------------------------------------

class VomirRegimeClassifier(VomirClassifier):
    """
    Rolling per-regime-group cluster-then-classify pipeline.

    Extends VomirClassifier.  The parent class handles data loading, label
    creation, window building, rolling orchestration, and result saving.
    We override only:
      - fit_predict_window  (per-group PCA+KMeans, augmented LGBM)
      - _save_model         (persist 4 PCA+KMeans pairs per group)
      - load_model_artifacts (load 4 pairs)

    Additional parameters
    --------------------
    k_per_group : list[int]
        KMeans k for each regime group [baseline, high_volume, low_volume, vol_breakout].
    pca_per_group : list[int]
        PCA components for each group.
    """

    def __init__(
        self,
        tf: str,
        data_dir: str,
        universe: str = "liquid",
        label_threshold: float = 0.0,
        weights_dir: Optional[str] = None,
        k_per_group: Optional[List[int]] = None,
        pca_per_group: Optional[List[int]] = None,
        # kept for API compat with parent, but not used
        n_clusters: int = 8,
        pca_components: int = 10,
    ):
        super().__init__(
            tf=tf,
            data_dir=data_dir,
            universe=universe,
            n_clusters=n_clusters,
            pca_components=pca_components,
            label_threshold=label_threshold,
            weights_dir=weights_dir,
        )
        n_groups = len(REGIME_GROUP_NAMES)
        self.k_per_group: List[int] = k_per_group if k_per_group else [6, 6, 4, 6]
        self.pca_per_group: List[int] = pca_per_group if pca_per_group else [10, 10, 8, 10]
        if len(self.k_per_group) != n_groups:
            raise ValueError(f"k_per_group must have {n_groups} elements")
        if len(self.pca_per_group) != n_groups:
            raise ValueError(f"pca_per_group must have {n_groups} elements")

    # ------------------------------------------------------------------
    # Per-group clustering helper
    # ------------------------------------------------------------------

    def _fit_group_pipeline(
        self,
        X_scaled: np.ndarray,
        group_mask: np.ndarray,
        group_id: int,
    ) -> Tuple[Optional[PCA], Optional[KMeans], np.ndarray]:
        """Fit PCA+KMeans for one group on training rows.

        Returns (pca, kmeans, cluster_ids) for rows where group_mask is True.
        cluster_ids has the same length as X_scaled.sum(group_mask).

        If the group has fewer than _MIN_ROWS_FOR_CLUSTERING rows, skip
        clustering and return (None, None, zeros).
        """
        idx = np.where(group_mask)[0]
        if len(idx) < _MIN_ROWS_FOR_CLUSTERING:
            logger.info(
                "Group %s (%d) has only %d rows — skipping clustering, cluster_id=0",
                REGIME_GROUP_NAMES[group_id], group_id, len(idx),
            )
            return None, None, np.zeros(len(idx), dtype=np.float64)

        X_g = X_scaled[idx]
        n_comp = min(self.pca_per_group[group_id], X_g.shape[1], X_g.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X_g)

        k = min(self.k_per_group[group_id], len(idx))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(X_pca).astype(np.float64)

        return pca, km, cluster_ids

    def _transform_group(
        self,
        X_scaled: np.ndarray,
        group_mask: np.ndarray,
        pca: Optional[PCA],
        km: Optional[KMeans],
    ) -> np.ndarray:
        """Apply fitted PCA+KMeans to new rows.

        Returns cluster_ids for rows where group_mask is True.
        """
        idx = np.where(group_mask)[0]
        if len(idx) == 0:
            return np.empty(0, dtype=np.float64)
        if pca is None or km is None:
            return np.zeros(len(idx), dtype=np.float64)
        X_g = X_scaled[idx]
        X_pca = pca.transform(X_g)
        return km.predict(X_pca).astype(np.float64)

    # ------------------------------------------------------------------
    # Main override: fit + predict a single window
    # ------------------------------------------------------------------

    def fit_predict_window(
        self, window: ClassifierWindow, save: bool = True, no_test: bool = False
    ) -> WindowResult:
        """Per-regime-group cluster augmentation, then global LGBM classifier."""
        df = self._load_data()
        feature_cols = self._get_feature_cols(df)

        # ---- split train / test ----
        train_mask = pd.Series(False, index=df.index)
        for y, m in window.train_months:
            train_mask |= (df["year"] == y) & (df["month"] == m)
        ty, tm = window.test_month
        test_mask = (df["year"] == ty) & (df["month"] == tm)

        train_df = df[train_mask].copy()
        if no_test:
            extra = df[test_mask]
            if not extra.empty:
                train_df = pd.concat([train_df, extra], ignore_index=True)
            test_df = pd.DataFrame()
        else:
            test_df = df[test_mask].copy()

        logger.info(
            "Window %d — train %s → test %s | train=%d test=%d",
            window.window_id, window.train_label, window.test_label,
            len(train_df), len(test_df),
        )

        X_train_raw = train_df[feature_cols].values.astype(np.float64)
        X_test_raw = (
            test_df[feature_cols].values.astype(np.float64)
            if not no_test else np.empty((0, len(feature_cols)))
        )

        y_train = self._create_labels(train_df)
        y_test = self._create_labels(test_df) if not no_test else np.empty(0, dtype=int)

        n_long = int((y_train == CLASS_LONG).sum())
        n_short = int((y_train == CLASS_SHORT).sum())
        n_neutral = int((y_train == CLASS_NEUTRAL).sum())
        logger.info(
            "Window %d — train labels: LONG=%d SHORT=%d NEUTRAL=%d",
            window.window_id, n_long, n_short, n_neutral,
        )

        # ---- global impute + scale ----
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp = imputer.transform(X_test_raw) if not no_test else X_test_raw

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_imp)
        X_test_sc = scaler.transform(X_test_imp) if not no_test else X_test_imp

        # ---- per-group PCA + KMeans ----
        train_regime = (
            train_df["regime"].values if "regime" in train_df.columns
            else np.array(["baseline"] * len(train_df))
        )
        test_regime = (
            test_df["regime"].values if (not no_test and "regime" in test_df.columns)
            else np.array(["baseline"] * len(test_df))
        )

        train_group_ids = np.array([_regime_to_group(r) for r in train_regime])
        test_group_ids = np.array([_regime_to_group(r) for r in test_regime]) if not no_test else np.empty(0, dtype=int)

        # Allocate cluster_id columns
        train_cluster_ids = np.zeros(len(train_df), dtype=np.float64)
        test_cluster_ids = np.zeros(len(test_df) if not no_test else 0, dtype=np.float64)

        fitted_pcas: List[Optional[PCA]] = []
        fitted_kms: List[Optional[KMeans]] = []

        for gid in range(len(REGIME_GROUP_NAMES)):
            tr_mask = train_group_ids == gid
            pca_g, km_g, cids_tr = self._fit_group_pipeline(X_train_sc, tr_mask, gid)
            fitted_pcas.append(pca_g)
            fitted_kms.append(km_g)
            train_cluster_ids[tr_mask] = cids_tr

            if not no_test and len(test_df) > 0:
                te_mask = test_group_ids == gid
                cids_te = self._transform_group(X_test_sc, te_mask, pca_g, km_g)
                test_cluster_ids[te_mask] = cids_te

        # ---- augment: [scaled_features | cluster_id | regime_group_id] ----
        X_train_aug = np.hstack([
            X_train_sc,
            train_cluster_ids.reshape(-1, 1),
            train_group_ids.reshape(-1, 1).astype(np.float64),
        ])
        if not no_test and len(test_df) > 0:
            X_test_aug = np.hstack([
                X_test_sc,
                test_cluster_ids.reshape(-1, 1),
                test_group_ids.reshape(-1, 1).astype(np.float64),
            ])
        else:
            X_test_aug = np.empty((0, X_train_aug.shape[1]))

        # ---- train LGBM ----
        classes_present = set(np.unique(y_train))
        y_pred = np.full(len(test_df) if not no_test else 0, CLASS_NEUTRAL, dtype=int)
        y_prob = np.zeros((len(test_df) if not no_test else 0, 3))
        if not no_test:
            y_prob[:, CLASS_NEUTRAL] = 1.0
        clf = None

        if {CLASS_LONG, CLASS_SHORT, CLASS_NEUTRAL}.issubset(classes_present):
            clf = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )
            clf.fit(X_train_aug, y_train)

            if no_test:
                weights_path = None
                if save:
                    weights_path = self._save_model(
                        window=window,
                        imputer=imputer, scaler=scaler,
                        fitted_pcas=fitted_pcas, fitted_kms=fitted_kms,
                        clf=clf, feature_cols=feature_cols,
                    )
                    logger.info(
                        "Production dump complete — n_train=%d  weights: %s",
                        len(train_df), weights_path,
                    )
                return WindowResult(
                    window=window,
                    predictions=pd.DataFrame(),
                    sharpe_long=0.0, sharpe_short=0.0, sharpe_combined=0.0,
                    sharpe_baseline_long=0.0, sharpe_baseline_short=0.0,
                    n_train=len(train_df), n_test=0,
                    n_long_label=n_long, n_short_label=n_short, n_neutral_label=n_neutral,
                    weights_path=weights_path,
                )

            raw_proba = clf.predict_proba(X_test_aug)
            class_order = list(clf.classes_)
            prob_reordered = np.zeros((len(test_df), 3))
            for cls_id in (CLASS_SHORT, CLASS_NEUTRAL, CLASS_LONG):
                if cls_id in class_order:
                    prob_reordered[:, cls_id] = raw_proba[:, class_order.index(cls_id)]
            y_prob = prob_reordered
            y_pred = np.argmax(y_prob, axis=1)
        else:
            logger.warning(
                "Window %d — not all 3 classes present %s. Predictions default to NEUTRAL.",
                window.window_id, classes_present,
            )

        # Production dump edge case (not all 3 classes)
        if no_test:
            weights_path = None
            if save and clf is not None:
                weights_path = self._save_model(
                    window=window,
                    imputer=imputer, scaler=scaler,
                    fitted_pcas=fitted_pcas, fitted_kms=fitted_kms,
                    clf=clf, feature_cols=feature_cols,
                )
                logger.info(
                    "Production dump complete — n_train=%d  weights: %s",
                    len(train_df), weights_path,
                )
            return WindowResult(
                window=window,
                predictions=pd.DataFrame(),
                sharpe_long=0.0, sharpe_short=0.0, sharpe_combined=0.0,
                sharpe_baseline_long=0.0, sharpe_baseline_short=0.0,
                n_train=len(train_df), n_test=0,
                n_long_label=n_long, n_short_label=n_short, n_neutral_label=n_neutral,
                weights_path=None,
            )

        # ---- evaluate ----
        meta_cols = [c for c in ["timestamp", "symbol"] if c in test_df.columns]
        pred_df = test_df[meta_cols + ["return_long_raw", "return_short_raw"]].copy().reset_index(drop=True)
        pred_df["y_true"] = y_test
        pred_df["y_pred"] = y_pred
        pred_df["y_prob_long"] = y_prob[:, CLASS_LONG]
        pred_df["y_prob_short"] = y_prob[:, CLASS_SHORT]

        long_ret = pred_df.loc[pred_df["y_pred"] == CLASS_LONG, "return_long_raw"].values
        short_ret = -pred_df.loc[pred_df["y_pred"] == CLASS_SHORT, "return_short_raw"].values
        all_signal_ret = np.concatenate([long_ret, short_ret])

        sharpe_long = _sharpe(long_ret, self._annualise)
        sharpe_short = _sharpe(short_ret, self._annualise)
        sharpe_combined = _sharpe(all_signal_ret, self._annualise)
        sharpe_baseline_long = _sharpe(test_df["return_long_raw"].values, self._annualise)
        sharpe_baseline_short = _sharpe(-test_df["return_short_raw"].values, self._annualise)

        logger.info(
            "Window %d — Sharpe: L=%.3f S=%.3f Comb=%.3f (baseline L=%.3f S=%.3f) n_signal=%d",
            window.window_id, sharpe_long, sharpe_short, sharpe_combined,
            sharpe_baseline_long, sharpe_baseline_short,
            len(long_ret) + len(short_ret),
        )

        weights_path = None
        if save and clf is not None:
            weights_path = self._save_model(
                window=window,
                imputer=imputer, scaler=scaler,
                fitted_pcas=fitted_pcas, fitted_kms=fitted_kms,
                clf=clf, feature_cols=feature_cols,
            )

        return WindowResult(
            window=window,
            predictions=pred_df,
            sharpe_long=round(sharpe_long, 4),
            sharpe_short=round(sharpe_short, 4),
            sharpe_combined=round(sharpe_combined, 4),
            sharpe_baseline_long=round(sharpe_baseline_long, 4),
            sharpe_baseline_short=round(sharpe_baseline_short, 4),
            n_train=len(train_df),
            n_test=len(test_df),
            n_long_label=n_long,
            n_short_label=n_short,
            n_neutral_label=n_neutral,
            weights_path=weights_path,
        )

    # ------------------------------------------------------------------
    # Model persistence overrides
    # ------------------------------------------------------------------

    def _save_model(
        self,
        window: ClassifierWindow,
        imputer: SimpleImputer,
        scaler: StandardScaler,
        fitted_pcas: List[Optional[PCA]],
        fitted_kms: List[Optional[KMeans]],
        clf: LGBMClassifier,
        feature_cols: List[str],
        # accept but ignore parent kwargs so signature is compatible
        **_kwargs,
    ) -> str:
        out_dir = os.path.join(self.weights_dir, window.window_name)
        os.makedirs(out_dir, exist_ok=True)

        joblib.dump(imputer, os.path.join(out_dir, "imputer.pkl"))
        joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
        joblib.dump(clf, os.path.join(out_dir, "lgbm_model.pkl"))

        for gid, gname in enumerate(REGIME_GROUP_NAMES):
            pca_g = fitted_pcas[gid]
            km_g = fitted_kms[gid]
            if pca_g is not None:
                joblib.dump(pca_g, os.path.join(out_dir, f"pca_{gname}.pkl"))
            if km_g is not None:
                joblib.dump(km_g, os.path.join(out_dir, f"kmeans_{gname}.pkl"))

        meta = {
            "feature_cols": feature_cols,
            "k_per_group": self.k_per_group,
            "pca_per_group": self.pca_per_group,
            "label_threshold": self.label_threshold,
            "tf": self.tf,
            "universe": self.universe,
            "train_period": window.train_label,
            "test_month": window.test_label,
            "class_map": {"SHORT": CLASS_SHORT, "NEUTRAL": CLASS_NEUTRAL, "LONG": CLASS_LONG},
            "regime_groups": REGIME_GROUP_NAMES,
            "classifier_type": "VomirRegimeClassifier",
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved regime model artifacts to %s", out_dir)
        return out_dir

    @staticmethod
    def load_model_artifacts(weights_path: str) -> Dict:
        """Load saved artifacts from a weights directory.

        Returns dict with keys: imputer, scaler, pca_{group}, kmeans_{group},
        clf, meta.  PCA/KMeans keys are None when a group had too few rows.
        """
        if not os.path.isdir(weights_path):
            raise FileNotFoundError(f"Vomir weights dir not found: {weights_path}")
        with open(os.path.join(weights_path, "meta.json")) as f:
            meta = json.load(f)

        artifacts: Dict = {
            "imputer": joblib.load(os.path.join(weights_path, "imputer.pkl")),
            "scaler": joblib.load(os.path.join(weights_path, "scaler.pkl")),
            "clf": joblib.load(os.path.join(weights_path, "lgbm_model.pkl")),
            "meta": meta,
        }
        for gname in REGIME_GROUP_NAMES:
            pca_path = os.path.join(weights_path, f"pca_{gname}.pkl")
            km_path = os.path.join(weights_path, f"kmeans_{gname}.pkl")
            artifacts[f"pca_{gname}"] = joblib.load(pca_path) if os.path.exists(pca_path) else None
            artifacts[f"kmeans_{gname}"] = joblib.load(km_path) if os.path.exists(km_path) else None

        return artifacts
