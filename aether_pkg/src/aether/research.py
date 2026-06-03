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
        """Rolling pooled training: one model per {base_tf}_{position} per window.

        For each rolling window (train ``window_size - 1`` months, test 1 month)
        a model is trained on the pooled (all-symbol) rows of the train months
        and evaluated on the test month. Predictions are emitted ONLY for the
        test month of each window, so ``predictions_{long,short}.csv`` are fully
        out-of-sample (no leakage). The LAST window's best model is saved as the
        production weights under ``weights/<period>/``.

        Args:
            out_dir: Experiment directory (e.g. gauntlet/pred_aether1h_1).
            period:  Weight window name for the production (last-window) weights.
        """
        # Lazy import to avoid circular imports and keep aether/ self-contained
        _gauntlet = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gauntlet")
        if _gauntlet not in sys.path:
            sys.path.insert(0, _gauntlet)
        from rolling_predict_returns import (
            select_feature_columns,
            train_models as _train_models,
            build_windows,
            split_by_month,
        )

        vf = getattr(self, "vertical_features", None)
        if vf is None or vf.empty:
            vf_path = os.path.join(out_dir, "vertical_features.csv")
            if not os.path.exists(vf_path):
                raise FileNotFoundError(
                    f"vertical_features.csv not found at {vf_path}. "
                    "Run create() first.")
            vf = pd.read_csv(vf_path, parse_dates=["timestamp"])

        for col in ("year", "month"):
            if col not in vf.columns:
                raise KeyError(
                    f"vertical_features missing '{col}' column required for "
                    "rolling windows (expected from verticalize()).")

        feature_cols = select_feature_columns(vf.columns.tolist())
        sweep_models = self.config.get(
            "SWEEP_MODELS", ["LightGBM", "XGBoost", "Ridge", "HistGBR"])
        # Window length is required and explicit — no silent fallback. train =
        # window_size - 1 months, test = 1 month (e.g. 12 => train 11 / test 1).
        if "WINDOW_SIZE" not in self.config:
            raise KeyError("WINDOW_SIZE missing from config (required for rolling)")
        window_size = int(self.config["WINDOW_SIZE"])
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

            months = sorted(set(zip(df_clean["year"].astype(int),
                                    df_clean["month"].astype(int))))
            windows = build_windows(months, window_size)
            if not windows:
                # Fail fast rather than silently degrading to a single split —
                # a one-shot 80/20 over all history is exactly the leakage bug
                # this method replaces.
                raise ValueError(
                    f"{position}: only {len(months)} month(s) of data, need "
                    f">= {window_size} for rolling windows (WINDOW_SIZE).")

            months_map = split_by_month(df_clean)
            oos_parts: list[pd.DataFrame] = []
            last_artifact = None  # (best, best_name, active_feature_cols) for prod weights

            for window in windows:
                train_parts = [months_map[m] for m in window.train_months
                               if m in months_map]
                test_df = months_map.get(window.test_month)
                if not train_parts or test_df is None or test_df.empty:
                    logger.warning(
                        f"{position} {window.test_label}: missing train/test "
                        "rows — skipping window")
                    continue
                train_df = pd.concat(train_parts, ignore_index=True)

                # Drop feature cols entirely NaN in THIS window's train set
                # (all-NaN cols make RobustScaler emit NaN centers → Ridge fails).
                active_feature_cols = [
                    c for c in feature_cols
                    if not train_df[c].replace([np.inf, -np.inf], np.nan).isna().all()
                ]

                results = _train_models(
                    train_df=train_df,
                    test_df=test_df.copy(),
                    feature_cols=active_feature_cols,
                    imputation="median",
                    target_col=target_col,
                    target_models=sweep_models,
                )
                if not results:
                    continue

                valid = {
                    n: r for n, r in results.items()
                    if "error" not in r
                    and not np.isnan(r.get("test_r2", float("nan")))
                }
                if not valid:
                    logger.warning(
                        f"{position} {window.test_label}: all models failed")
                    continue

                best_name = max(valid, key=lambda n: valid[n]["test_r2"])
                best = valid[best_name]

                # Accumulate OOS predictions for this window's TEST month only.
                # train_models predicts on test rows surviving prepare_xy's
                # non-NaN-target mask, so align against the same subset.
                clean_test = test_df.dropna(subset=[target_col])
                if len(clean_test) != len(best["y_pred"]):
                    logger.warning(
                        f"{position} {window.test_label}: pred/test length "
                        f"mismatch ({len(best['y_pred'])} vs {len(clean_test)}) "
                        "— skipping window predictions")
                else:
                    oos_parts.append(pd.DataFrame({
                        "timestamp": clean_test["timestamp"].values,
                        "symbol": clean_test["symbol"].values,
                        "y_pred": best["y_pred"],
                    }))

                last_artifact = (best, best_name, active_feature_cols)

            # Production weights = LAST window's best model.
            if last_artifact is not None:
                best, best_name, active_feature_cols = last_artifact
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
                logger.info(
                    f"Saved {best_name} production artifacts to {pos_dir} "
                    f"(last window {windows[-1].test_label})")

            # Write OOS predictions (test months only — leakage-free).
            if oos_parts:
                preds = (pd.concat(oos_parts, ignore_index=True)
                         .sort_values(["timestamp", "symbol"]))
                pred_path = Path(out_dir) / f"predictions_{position}.csv"
                preds[["timestamp", "symbol", "y_pred"]].to_csv(pred_path, index=False)
                logger.info(
                    f"Wrote {len(preds)} OOS rows across {len(windows)} windows "
                    f"to {pred_path}")
            else:
                logger.warning(
                    f"{position}: no OOS predictions produced (all windows skipped)")
