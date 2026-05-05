"""
vomir/classifier.py — Rolling cluster-then-classify pipeline (v2).

Single global LGBMClassifier with cluster_id as a feature.
Ternary labels: LONG (2), NEUTRAL (1), SHORT (0).
Saves model artifacts per window for live inference by VomirTrading.

Label convention:
    return_long_raw  > label_threshold  → LONG  (price went up, long profitable)
    return_short_raw < -label_threshold → SHORT (price went down, short profitable)
    Conflict (both qualify)             → higher |magnitude| wins
    Neither                             → NEUTRAL
"""

import json
import logging
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CLASS_SHORT = 0
CLASS_NEUTRAL = 1
CLASS_LONG = 2

_META_COLS = {"symbol", "timestamp", "regime", "position", "year", "month"}
_TARGET_COLS = {
    "ret", "ret_raw", "return", "return_long", "return_short",
    "return_long_raw", "return_short_raw",
    "btc_return_lag1", "btc_return_lag2",
}
_EXCLUDE_COLS = _META_COLS | _TARGET_COLS

# TF → agamotto experiment dir per universe (filter parquets live here)
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

_ANNUALISE = {
    "15m": np.sqrt(96 * 252),
    "1h": np.sqrt(24 * 252),
    "4h": np.sqrt(6 * 252),
    "1d": np.sqrt(252),
}

VALID_TFS = list(_TF_DIRS["liquid"].keys())
VALID_UNIVERSES = list(_TF_DIRS.keys())


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ClassifierWindow:
    window_id: int
    train_months: List[Tuple[int, int]]
    test_month: Tuple[int, int]
    production: bool = False  # True = no-test production dump; window_name uses test_month

    @property
    def train_label(self) -> str:
        sy, sm = self.train_months[0]
        ey, em = self.train_months[-1]
        return f"{int(sy)}-{int(sm):02d} → {int(ey)}-{int(em):02d}"

    @property
    def test_label(self) -> str:
        y, m = self.test_month
        return f"{int(y)}-{int(m):02d}"

    @property
    def window_name(self) -> str:
        """Directory name.

        Production dump (production=True): window_{test_month} — matches deploy month.
        Rolling eval (production=False):   window_{last_train_month}.
        """
        y, m = self.test_month if self.production else self.train_months[-1]
        return f"window_{int(y)}_{int(m):02d}"


@dataclass
class WindowResult:
    window: ClassifierWindow
    predictions: pd.DataFrame        # symbol, timestamp, y_true, y_pred, y_prob_long, y_prob_short, return_long_raw, return_short_raw
    sharpe_long: float               # Sharpe of LONG-predicted bars using return_long_raw
    sharpe_short: float              # Sharpe of SHORT-predicted bars using -return_short_raw
    sharpe_combined: float           # Sharpe across all signal bars
    sharpe_baseline_long: float      # Sharpe if always long
    sharpe_baseline_short: float     # Sharpe if always short
    n_train: int
    n_test: int
    n_long_label: int
    n_short_label: int
    n_neutral_label: int
    weights_path: Optional[str] = None


@dataclass
class RollingResult:
    window_results: List[WindowResult]
    summary: pd.DataFrame


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class VomirClassifier:
    """
    Rolling cluster-then-classify pipeline.

    Parameters
    ----------
    tf : str
        Timeframe — one of "15m", "1h", "4h", "1d".
    data_dir : str
        Marvel repo root. Filter parquets at
        {data_dir}/gauntlet/{tf_dir}/filter/filter_*.parquet.
    universe : str
        "liquid" (_1) or "altcoins" (_2).
    n_clusters : int
        KMeans clusters (default 8).
    pca_components : int
        PCA components fed into KMeans (default 10).
    label_threshold : float
        Minimum |return_raw| to be labelled LONG or SHORT (default 0.0).
    weights_dir : str, optional
        Root for saved model artifacts (default: {data_dir}/vomir/weights/).
    """

    def __init__(
        self,
        tf: str,
        data_dir: str,
        universe: str = "liquid",
        n_clusters: int = 8,
        pca_components: int = 10,
        label_threshold: float = 0.0,
        weights_dir: Optional[str] = None,
    ):
        if tf not in VALID_TFS:
            raise ValueError(f"tf must be one of {VALID_TFS}, got {tf!r}")
        if universe not in VALID_UNIVERSES:
            raise ValueError(f"universe must be one of {VALID_UNIVERSES}, got {universe!r}")
        self.tf = tf
        self.universe = universe
        self.data_dir = data_dir
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.label_threshold = label_threshold
        self._annualise = _ANNUALISE[tf]
        self._filter_dir = os.path.join(
            data_dir, "gauntlet", _TF_DIRS[universe][tf], "filter"
        )
        self.weights_dir = weights_dir or os.path.join(data_dir, "vomir", "weights")
        self._df: Optional[pd.DataFrame] = None
        self._feature_cols: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        files = sorted(glob(os.path.join(self._filter_dir, "filter_*.parquet")))
        if not files:
            raise FileNotFoundError(f"No filter parquets found in {self._filter_dir}")

        logger.info("Loading %d filter files for tf=%s", len(files), self.tf)
        chunks = []
        for path in files:
            try:
                df = pd.read_parquet(path)
                if "regime" not in df.columns:
                    fname = os.path.splitext(os.path.basename(path))[0]
                    df["regime"] = fname.replace("filter_", "")
                chunks.append(df)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path, exc)

        combined = pd.concat(chunks, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp", "symbol"])

        if "year" not in combined.columns or "month" not in combined.columns:
            ts = pd.to_datetime(combined["timestamp"], errors="coerce", utc=True)
            combined["year"] = ts.dt.year
            combined["month"] = ts.dt.month
        combined["year"] = combined["year"].astype(int)
        combined["month"] = combined["month"].astype(int)

        self._df = combined
        logger.info("Loaded %d rows after dedup", len(combined))
        return combined

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        if self._feature_cols is not None:
            return self._feature_cols
        cols = [
            c for c in df.columns
            if c not in _EXCLUDE_COLS
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]
        self._feature_cols = cols
        return cols

    # ------------------------------------------------------------------
    # Label creation
    # ------------------------------------------------------------------

    def _create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Ternary labels: LONG=2, NEUTRAL=1, SHORT=0.

        return_long_raw  > label_threshold  → LONG  (positive = price rose = long profit)
        return_short_raw < -label_threshold → SHORT (negative = price fell = short profit)
        Conflict                            → higher |magnitude| wins
        """
        long_raw = df["return_long_raw"].values.astype(np.float64) if "return_long_raw" in df.columns else np.zeros(len(df))
        short_raw = df["return_short_raw"].values.astype(np.float64) if "return_short_raw" in df.columns else np.zeros(len(df))

        is_long = long_raw > self.label_threshold
        is_short = short_raw < -self.label_threshold

        labels = np.full(len(df), CLASS_NEUTRAL, dtype=int)
        labels[is_long] = CLASS_LONG
        labels[is_short] = CLASS_SHORT

        # Resolve conflicts: take higher absolute magnitude
        conflict = is_long & is_short
        if conflict.any():
            long_mag = np.abs(long_raw[conflict])
            short_mag = np.abs(short_raw[conflict])
            labels[conflict] = np.where(long_mag >= short_mag, CLASS_LONG, CLASS_SHORT)

        return labels

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def build_windows(self, window_size: int = 12) -> List[ClassifierWindow]:
        """Build rolling monthly windows (window_size-1 train + 1 test month)."""
        df = self._load_data()
        months = sorted(df.groupby(["year", "month"]).groups.keys())
        if len(months) < window_size:
            logger.warning("Only %d months available, need %d", len(months), window_size)
            return []
        windows = []
        for i in range(len(months) - window_size + 1):
            train_months = list(months[i: i + window_size - 1])
            test_month = months[i + window_size - 1]
            windows.append(ClassifierWindow(
                window_id=len(windows) + 1,
                train_months=train_months,
                test_month=test_month,
            ))
        logger.info("Built %d windows (window_size=%d)", len(windows), window_size)
        return windows

    # ------------------------------------------------------------------
    # Single window: fit + predict + save
    # ------------------------------------------------------------------

    def fit_predict_window(
        self, window: ClassifierWindow, save: bool = True, no_test: bool = False
    ) -> WindowResult:
        """Fit and evaluate a single rolling window.

        Architecture
        ------------
        StandardScaler → PCA → KMeans → [scaled_features | cluster_id] → LGBMClassifier(3 class)

        Parameters
        ----------
        no_test : bool
            Production dump mode. Train on all of window.train_months (no holdout).
            Skips Sharpe evaluation; returns zero metrics.
        """
        df = self._load_data()
        feature_cols = self._get_feature_cols(df)

        # Split by year+month
        train_mask = pd.Series(False, index=df.index)
        for y, m in window.train_months:
            train_mask |= (df["year"] == y) & (df["month"] == m)
        ty, tm = window.test_month
        test_mask = (df["year"] == ty) & (df["month"] == tm)

        train_df = df[train_mask].copy()
        # In no_test mode, fold any available test data into training
        if no_test:
            extra = df[test_mask]
            if not extra.empty:
                train_df = pd.concat([train_df, extra], ignore_index=True)
            test_df = pd.DataFrame()  # empty — no evaluation
        else:
            test_df = df[test_mask].copy()

        logger.info(
            "Window %d — train %s → test %s | train=%d test=%d",
            window.window_id, window.train_label, window.test_label,
            len(train_df), len(test_df),
        )

        X_train_raw = train_df[feature_cols].values.astype(np.float64)
        X_test_raw = test_df[feature_cols].values.astype(np.float64) if not no_test else np.empty((0, len(feature_cols)))

        y_train = self._create_labels(train_df)
        y_test = self._create_labels(test_df) if not no_test else np.empty(0, dtype=int)

        n_long = int((y_train == CLASS_LONG).sum())
        n_short = int((y_train == CLASS_SHORT).sum())
        n_neutral = int((y_train == CLASS_NEUTRAL).sum())
        logger.info(
            "Window %d — train labels: LONG=%d SHORT=%d NEUTRAL=%d",
            window.window_id, n_long, n_short, n_neutral,
        )

        # Impute (median) → scale
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp = imputer.transform(X_test_raw) if not no_test else X_test_raw

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_imp)
        X_test_sc = scaler.transform(X_test_imp) if not no_test else X_test_imp

        # PCA on training data
        n_comp = min(self.pca_components, X_train_sc.shape[1], X_train_sc.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        X_train_pca = pca.fit_transform(X_train_sc)

        # KMeans — assign cluster_id
        n_clusters = min(self.n_clusters, len(train_df))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        train_cluster_ids = km.fit_predict(X_train_pca).reshape(-1, 1).astype(np.float64)
        test_cluster_ids = (
            km.predict(pca.transform(X_test_sc)).reshape(-1, 1).astype(np.float64)
            if not no_test else np.empty((0, 1))
        )

        # Augment: [scaled_features | cluster_id]
        X_train_aug = np.hstack([X_train_sc, train_cluster_ids])
        X_test_aug = np.hstack([X_test_sc, test_cluster_ids]) if not no_test else np.empty((0, X_train_aug.shape[1]))

        # Train single global LGBMClassifier
        classes_present = set(np.unique(y_train))
        y_pred = np.full(len(test_df), CLASS_NEUTRAL, dtype=int)
        y_prob = np.zeros((len(test_df), 3))
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

            # Production dump: save and return immediately after training
            if no_test:
                weights_path = None
                if save:
                    weights_path = self._save_model(
                        window=window,
                        imputer=imputer, scaler=scaler, pca=pca, kmeans=km, clf=clf,
                        feature_cols=feature_cols, n_clusters=n_clusters, n_comp=n_comp,
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

            raw_proba = clf.predict_proba(X_test_aug)  # shape (n, n_classes)
            class_order = list(clf.classes_)

            # Reorder columns to match [SHORT=0, NEUTRAL=1, LONG=2]
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

        # Production dump with no training data (edge case: not all 3 classes)
        if no_test:
            weights_path = None
            if save and clf is not None:
                weights_path = self._save_model(
                    window=window,
                    imputer=imputer, scaler=scaler, pca=pca, kmeans=km, clf=clf,
                    feature_cols=feature_cols, n_clusters=n_clusters, n_comp=n_comp,
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

        # Build predictions DataFrame
        meta_cols = [c for c in ["timestamp", "symbol"] if c in test_df.columns]
        pred_df = test_df[meta_cols + ["return_long_raw", "return_short_raw"]].copy().reset_index(drop=True)
        pred_df["y_true"] = y_test
        pred_df["y_pred"] = y_pred
        pred_df["y_prob_long"] = y_prob[:, CLASS_LONG]
        pred_df["y_prob_short"] = y_prob[:, CLASS_SHORT]

        # Metrics: SHORT PnL = -return_short_raw (negate because short profits when price falls)
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
                imputer=imputer, scaler=scaler, pca=pca, kmeans=km, clf=clf,
                feature_cols=feature_cols, n_clusters=n_clusters, n_comp=n_comp,
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
    # Model persistence
    # ------------------------------------------------------------------

    def _save_model(
        self,
        window: ClassifierWindow,
        imputer: SimpleImputer,
        scaler: StandardScaler,
        pca: PCA,
        kmeans: KMeans,
        clf: LGBMClassifier,
        feature_cols: List[str],
        n_clusters: int,
        n_comp: int,
    ) -> str:
        out_dir = os.path.join(self.weights_dir, window.window_name)
        os.makedirs(out_dir, exist_ok=True)

        joblib.dump(imputer, os.path.join(out_dir, "imputer.pkl"))
        joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
        joblib.dump(pca, os.path.join(out_dir, "pca.pkl"))
        joblib.dump(kmeans, os.path.join(out_dir, "kmeans.pkl"))
        joblib.dump(clf, os.path.join(out_dir, "lgbm_model.pkl"))

        meta = {
            "feature_cols": feature_cols,
            "n_clusters": n_clusters,
            "pca_components": n_comp,
            "label_threshold": self.label_threshold,
            "tf": self.tf,
            "universe": self.universe,
            "train_period": window.train_label,
            "test_month": window.test_label,
            "class_map": {"SHORT": CLASS_SHORT, "NEUTRAL": CLASS_NEUTRAL, "LONG": CLASS_LONG},
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved model artifacts to %s", out_dir)
        return out_dir

    @staticmethod
    def load_model_artifacts(weights_path: str) -> Dict:
        """Load saved artifacts from a weights directory.

        Returns dict with keys: imputer, scaler, pca, kmeans, clf, meta.
        """
        if not os.path.isdir(weights_path):
            raise FileNotFoundError(f"Vomir weights dir not found: {weights_path}")
        with open(os.path.join(weights_path, "meta.json")) as f:
            meta = json.load(f)
        return {
            "imputer": joblib.load(os.path.join(weights_path, "imputer.pkl")),
            "scaler": joblib.load(os.path.join(weights_path, "scaler.pkl")),
            "pca": joblib.load(os.path.join(weights_path, "pca.pkl")),
            "kmeans": joblib.load(os.path.join(weights_path, "kmeans.pkl")),
            "clf": joblib.load(os.path.join(weights_path, "lgbm_model.pkl")),
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # Rolling run
    # ------------------------------------------------------------------

    def run_rolling(self, window_size: int = 12, save: bool = True) -> RollingResult:
        """Fit-predict across all rolling windows and optionally save artifacts."""
        windows = self.build_windows(window_size=window_size)
        if not windows:
            raise RuntimeError(
                f"No windows for tf={self.tf} with window_size={window_size}. "
                "Check that filter parquets exist and span enough months."
            )

        results = [self.fit_predict_window(w, save=save) for w in windows]

        summary_rows = []
        for wr in results:
            summary_rows.append({
                "window_id": wr.window.window_id,
                "train_period": wr.window.train_label,
                "test_month": wr.window.test_label,
                "window_name": wr.window.window_name,
                "n_train": wr.n_train,
                "n_test": wr.n_test,
                "n_long": wr.n_long_label,
                "n_short": wr.n_short_label,
                "n_neutral": wr.n_neutral_label,
                "sharpe_long": wr.sharpe_long,
                "sharpe_short": wr.sharpe_short,
                "sharpe_combined": wr.sharpe_combined,
                "sharpe_baseline_long": wr.sharpe_baseline_long,
                "sharpe_baseline_short": wr.sharpe_baseline_short,
                "weights_path": wr.weights_path or "",
            })

        summary = pd.DataFrame(summary_rows)
        return RollingResult(window_results=results, summary=summary)

    def save_results(self, result: RollingResult, output_dir: str) -> None:
        """Save rolling summary and all predictions to CSV."""
        out = os.path.join(output_dir, self.universe, self.tf)
        os.makedirs(out, exist_ok=True)

        all_pred = pd.concat(
            [wr.predictions for wr in result.window_results], ignore_index=True
        )
        pred_path = os.path.join(out, "classifier_predictions.csv")
        all_pred.to_csv(pred_path, index=False)
        logger.info("Saved %s (%d rows)", pred_path, len(all_pred))

        summary_path = os.path.join(out, "classifier_rolling_summary.csv")
        result.summary.to_csv(summary_path, index=False)
        logger.info("Saved %s", summary_path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray, annualise: float) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns))
    return float(np.mean(returns) / std * annualise) if std > 0 else 0.0
