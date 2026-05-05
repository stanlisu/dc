"""Aether research: pooled cross-TF training on top of OrbResearch."""
from __future__ import annotations

import logging
import os
import sys
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from orb.research import OrbResearch

logger = logging.getLogger(__name__)


class AetherResearch(OrbResearch):
    """Pooled cross-TF research: one model per TF×position, trained on all rows."""

    def __init__(self, config: Dict[str, object], home_root: str) -> None:
        super().__init__(config, home_root)

    def create(self) -> str:
        """Write vertical_features.csv only — no per-regime filter CSVs.

        Overrides AgamottoResearch.create() to skip the regime-stack-driven
        filter CSV writing step. Aether's regime stack is applied only at
        inference (and threshold optimization), not at research time.
        """
        if self.features is None:
            self.engineer_features()

        # Drop last row (NaN targets) — same as parent
        if self.features is not None and not self.features.empty:
            self.features = self.features.iloc[:-1]

        self.verticalize()

        version = self.config.get("VERSION")
        if not version:
            raise ValueError("VERSION missing from config")

        if "OUTPUT_DIR" in self.config:
            out_dir = self.config["OUTPUT_DIR"]
        else:
            out_dir = os.path.join("gauntlet", f"pred_{version}")

        if not os.path.isabs(out_dir):
            out_dir = os.path.join(self.home_root, out_dir)

        os.makedirs(out_dir, exist_ok=True)

        if (hasattr(self, "vertical_features")
                and self.vertical_features is not None
                and not self.vertical_features.empty):
            v_out_path = os.path.join(out_dir, "vertical_features.csv")
            self.vertical_features.to_csv(v_out_path, index=False)
            logger.info(
                f"Wrote vertical_features.csv "
                f"({len(self.vertical_features)} rows) to {v_out_path}")

        return out_dir

    def train_pooled_models(self, out_dir: str, period: str) -> None:
        """Train one model per {base_tf}_{position} on ALL vertical_features rows.

        Args:
            out_dir: Experiment directory (e.g. gauntlet/pred_aether1h_1).
            period:  Weight window name (e.g. "window_2026_03").
        """
        # Lazy import to avoid circular imports and keep aether/ self-contained
        _gauntlet = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gauntlet")
        if _gauntlet not in sys.path:
            sys.path.insert(0, _gauntlet)
        from rolling_predict_returns import (
            select_feature_columns,
            prepare_xy,
            train_models as _train_models,
        )

        vf = getattr(self, "vertical_features", None)
        if vf is None or vf.empty:
            vf_path = os.path.join(out_dir, "vertical_features.csv")
            if not os.path.exists(vf_path):
                raise FileNotFoundError(
                    f"vertical_features.csv not found at {vf_path}. "
                    "Run create() first.")
            vf = pd.read_csv(vf_path, parse_dates=["timestamp"])

        feature_cols = select_feature_columns(vf.columns.tolist())
        sweep_models = self.config.get(
            "SWEEP_MODELS", ["LightGBM", "XGBoost", "Ridge", "HistGBR"])
        weights_root = Path(out_dir) / "weights" / period

        for position, target_col in [("long", "return_long"), ("short", "return_short")]:
            if target_col not in vf.columns:
                logger.warning(
                    f"{target_col} not in vertical_features — skipping {position}")
                continue

            df_clean = vf.dropna(subset=[target_col]).copy()
            if len(df_clean) < 100:
                logger.warning(
                    f"Only {len(df_clean)} rows with valid {target_col} — skipping")
                continue

            # Chronological 80/20 split
            df_sorted = df_clean.sort_values("timestamp")
            split = int(len(df_sorted) * 0.8)
            train_df = df_sorted.iloc[:split].copy()
            test_df = df_sorted.iloc[split:].copy()

            # Drop feature columns that are entirely NaN in the training set
            # (all-NaN cols cause RobustScaler to emit NaN centers → Ridge fails)
            non_null_cols = [
                c for c in feature_cols
                if not train_df[c].replace([np.inf, -np.inf], np.nan).isna().all()
            ]
            if len(non_null_cols) < len(feature_cols):
                logger.info(
                    f"Dropped {len(feature_cols) - len(non_null_cols)} all-NaN "
                    f"feature cols for {position}")
            active_feature_cols = non_null_cols

            logger.info(
                f"Training {position}: {len(train_df)} train rows, "
                f"{len(test_df)} test rows, {len(active_feature_cols)} features")

            results = _train_models(
                train_df=train_df,
                test_df=test_df,
                feature_cols=active_feature_cols,
                imputation="median",
                target_col=target_col,
                target_models=sweep_models,
            )

            if not results:
                logger.warning(
                    f"No models trained for {position} — skipping")
                continue

            # Pick best model by test R2 (skip error entries)
            valid = {
                n: r for n, r in results.items()
                if "error" not in r and not np.isnan(r.get("test_r2", float("nan")))
            }
            if not valid:
                logger.warning(
                    f"All models failed for {position} — skipping")
                continue

            best_name = max(valid, key=lambda n: valid[n]["test_r2"])
            best = valid[best_name]
            logger.info(
                f"Best model for {position}: {best_name} "
                f"(test_r2={best['test_r2']:.4f})")

            key = f"{self.base_tf}_{position}"
            pos_dir = weights_root / key
            pos_dir.mkdir(parents=True, exist_ok=True)

            low = best_name.lower()
            joblib.dump(best["model"],  pos_dir / f"{low}_model.pkl")
            joblib.dump(best["scaler"], pos_dir / f"{low}_scaler.pkl")
            joblib.dump(
                {"feature_columns": active_feature_cols, "model_name": best_name},
                pos_dir / f"{low}_meta.pkl",
            )
            logger.info(f"Saved {best_name} artifacts to {pos_dir}")

        # Generate full predictions for all rows using the best trained models
        self._write_pooled_predictions(out_dir, period, vf, feature_cols)

    def _write_pooled_predictions(
        self,
        out_dir: str,
        period: str,
        vf: "pd.DataFrame",
        feature_cols: list,
    ) -> None:
        """Run forward pass on all rows; write predictions_long/short.csv."""
        weights_root = Path(out_dir) / "weights" / period

        for position in ["long", "short"]:
            key = f"{self.base_tf}_{position}"
            pos_dir = weights_root / key
            if not pos_dir.exists():
                logger.warning(
                    f"No weights for {key} — skipping prediction export")
                continue

            # Find the pkl files
            model_files = list(pos_dir.glob("*_model.pkl"))
            if not model_files:
                logger.warning(f"No model .pkl files found in {pos_dir}")
                continue
            model_path = model_files[0]
            low = model_path.stem.replace("_model", "")
            scaler_path = pos_dir / f"{low}_scaler.pkl"
            meta_path   = pos_dir / f"{low}_meta.pkl"

            model  = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            meta   = joblib.load(meta_path)
            f_cols = meta["feature_columns"]

            df = vf.copy()
            X = df[f_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X_scaled = scaler.transform(X)
            df["y_pred"] = model.predict(X_scaled)

            out_cols = ["timestamp", "symbol", "y_pred"]
            pred_path = Path(out_dir) / f"predictions_{position}.csv"
            df[out_cols].to_csv(pred_path, index=False)
            logger.info(
                f"Wrote {len(df)} rows to {pred_path}")
