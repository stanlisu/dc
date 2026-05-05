# Aether Algorithm Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the Aether algorithm — pooled cross-TF ML with regime stacks applied only at inference — following the design at `docs/aether/design_README.md`.

**Architecture:** `AetherResearch(OrbResearch)` inherits all cross-TF feature engineering unchanged. It overrides `create()` to skip per-regime filter CSVs, adds `train_pooled_models()` for pooled training (8 models), and `optimize_aether_thresholds.py` runs threshold optimization against pooled predictions. `AetherTrading` does 2 forward passes at inference regardless of regime stack size.

**Tech Stack:** Python, LightGBM, XGBoost, Ridge, HistGBR (via `rolling_predict_returns.train_models`), RobustScaler, pandas, joblib, pytest.

**PYTHONPATH:** All commands must be run as:
```bash
cd /home/ubuntu/sandbox/marvel
PYTHONPATH="agamotto_pkg/src:." python <script>
```

---

### Task 1: Package scaffold + inheritance test

**Files:**
- Create: `aether/__init__.py`
- Create: `aether/research.py`
- Create: `aether/trading.py`
- Create: `aether/tests/__init__.py`
- Create: `aether/tests/test_aether_research.py`

**Step 1: Write the failing tests**

```python
# aether/tests/test_aether_research.py
"""Tests for AetherResearch."""
from __future__ import annotations
import pytest
from aether.research import AetherResearch
from orb.research import OrbResearch


def test_aether_is_subclass_of_orb():
    assert issubclass(AetherResearch, OrbResearch)


def test_aether_instantiates_with_minimal_config():
    config = {
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "EXCHANGE": "BINANCE",
        "DATA": "liquid",
        "TIMEFRAMES": ["15m", "1h", "4h", "1d"],
        "BASE_TF": "1h",
        "TARGET_TF": "1h",
        "TIME_UNIT": "1h",
        "LADDER": 5,
        "FEE": 2.25,
        "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14,
        "VERSION": "aether1h_test",
    }
    obj = AetherResearch(config, "/tmp/fake_root")
    assert obj.base_tf == "1h"
    assert obj.target_tf == "1h"
    assert obj.timeframes == ["15m", "1h", "4h", "1d"]
```

**Step 2: Run to confirm FAIL**

```bash
PYTHONPATH="agamotto_pkg/src:." pytest aether/tests/test_aether_research.py -v
```
Expected: `ModuleNotFoundError: No module named 'aether'`

**Step 3: Write minimal implementation**

```python
# aether/__init__.py
"""Aether: pooled cross-TF ML with regime stacks applied only at inference."""
from .research import AetherResearch
from .trading import AetherTrading

__all__ = ["AetherResearch", "AetherTrading"]
```

```python
# aether/research.py
"""Aether research: pooled cross-TF training on top of OrbResearch."""
from __future__ import annotations

import logging
import os
from typing import Dict

from orb.research import OrbResearch

logger = logging.getLogger(__name__)


class AetherResearch(OrbResearch):
    """Pooled cross-TF research: one model per TF×position, trained on all rows."""

    def __init__(self, config: Dict[str, object], home_root: str) -> None:
        super().__init__(config, home_root)
```

```python
# aether/trading.py
"""Aether trading: pooled cross-TF live inference."""
from __future__ import annotations

from typing import Dict, Optional

from .research import AetherResearch

import logging
logger = logging.getLogger(__name__)


class AetherTrading(AetherResearch):
    """Live pooled cross-TF inference."""

    def __init__(
        self,
        config: Dict[str, object],
        home_root: str,
        period: Optional[str] = None,
        skip_load: bool = False,
    ) -> None:
        super().__init__(config, home_root)
```

```python
# aether/tests/__init__.py
```

**Step 4: Run tests to confirm PASS**

```bash
PYTHONPATH="agamotto_pkg/src:." pytest aether/tests/test_aether_research.py -v
```
Expected: 2 tests PASS.

**Step 5: Commit**

```bash
git add aether/
git commit -m "feat: scaffold aether package — AetherResearch(OrbResearch), AetherTrading stubs

[dev]"
```

---

### Task 2: `AetherResearch.create()` — write vertical_features only

ORB's `AgamottoResearch.create()` (at `agamotto_pkg/src/agamotto/research.py:457`) writes per-regime filter CSVs by loading `REGIME_STACK_PATH` and calling `filter_signals()` per regime. Aether skips that — it only needs `vertical_features.csv`.

**Files:**
- Modify: `aether/research.py`
- Modify: `aether/tests/test_aether_research.py`

**Step 1: Write the failing test**

```python
# Add to aether/tests/test_aether_research.py
import numpy as np
import pandas as pd
from agamotto import AgamottoResearch


def _build_aether_with_raw(config=None):
    """Build AetherResearch with pre-populated raw data — no disk I/O."""
    cfg = config or {
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "EXCHANGE": "BINANCE",
        "DATA": "liquid",
        "TIMEFRAMES": ["15m", "1h"],
        "BASE_TF": "15m",
        "TARGET_TF": "1h",
        "TIME_UNIT": "1h",
        "LADDER": 1,
        "FEE": 0,
        "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14,
        "VERSION": "aether15m_test",
    }
    aether = AetherResearch(cfg, "/tmp/fake_root")

    np.random.seed(42)
    n_15m = 200
    for tf in cfg["TIMEFRAMES"]:
        inst = AgamottoResearch({**cfg, "TIME_UNIT": tf}, "/tmp/fake_root")
        freq_map = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
        n = {"15m": n_15m, "1h": n_15m // 4, "4h": n_15m // 16, "1d": max(2, n_15m // 96)}
        idx = pd.date_range("2025-01-01", periods=n[tf], freq=freq_map[tf])
        close = 100.0 + np.cumsum(np.random.randn(n[tf]) * 0.5)
        inst.raw = pd.DataFrame({
            "BTCUSDT_open": close - 0.1,
            "BTCUSDT_high": close + 0.5,
            "BTCUSDT_low": close - 0.5,
            "BTCUSDT_close": close,
            "BTCUSDT_volume": np.random.randint(100, 1000, n[tf]).astype(float),
        }, index=idx)
        aether._tf_instances[tf] = inst

    aether.raw = aether._tf_instances[cfg["BASE_TF"]].raw
    return aether


def test_create_writes_vertical_features(tmp_path):
    aether = _build_aether_with_raw()
    aether.config["VERSION"] = "aether15m_test"
    aether.config["OUTPUT_DIR"] = str(tmp_path)
    aether.engineer_features()
    out_dir = aether.create()
    assert (tmp_path / "vertical_features.csv").exists()


def test_create_does_not_write_filter_csvs(tmp_path):
    aether = _build_aether_with_raw()
    aether.config["VERSION"] = "aether15m_test"
    aether.config["OUTPUT_DIR"] = str(tmp_path)
    aether.engineer_features()
    aether.create()
    filter_dir = tmp_path / "filter"
    assert not filter_dir.exists() or len(list(filter_dir.iterdir())) == 0


def test_create_does_not_require_regime_stack_path(tmp_path):
    aether = _build_aether_with_raw()
    aether.config["VERSION"] = "aether15m_test"
    aether.config["OUTPUT_DIR"] = str(tmp_path)
    # No REGIME_STACK_PATH — should not raise
    aether.config.pop("REGIME_STACK_PATH", None)
    aether.engineer_features()
    aether.create()  # must not raise KeyError/ValueError
```

**Step 2: Run to confirm FAIL**

```bash
PYTHONPATH="agamotto_pkg/src:." pytest aether/tests/test_aether_research.py::test_create_writes_vertical_features -v
```
Expected: FAIL (create() delegates to parent which raises ValueError about REGIME_STACK_PATH)

**Step 3: Implement `AetherResearch.create()`**

Add to `aether/research.py` inside the `AetherResearch` class:

```python
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
```

**Step 4: Run tests to confirm PASS**

```bash
PYTHONPATH="agamotto_pkg/src:." pytest aether/tests/test_aether_research.py -v
```
Expected: all 5 tests PASS.

**Step 5: Commit**

```bash
git add aether/research.py aether/tests/test_aether_research.py
git commit -m "feat: AetherResearch.create() — writes vertical_features.csv, skips filter CSVs

[dev]"
```

---

### Task 3: Pooled model training — `AetherResearch.train_pooled_models()`

Trains one model per `{base_tf}_{position}` on ALL rows (no regime filtering).
Reuses `train_models()` and `select_feature_columns()` from `gauntlet/rolling_predict_returns.py`.
Saves artifacts exactly like ORB: `{model_name.lower()}_model.pkl`, `_scaler.pkl`, `_meta.pkl`.
Writes `predictions_long.csv` and `predictions_short.csv` for all rows.

**Files:**
- Modify: `aether/research.py`
- Modify: `aether/tests/test_aether_research.py`

**Step 1: Write the failing tests**

```python
# Add to aether/tests/test_aether_research.py
import os


def test_train_pooled_models_creates_weight_files(tmp_path):
    """Pooled training must write model artifacts for both long and short."""
    aether = _build_aether_with_raw()
    aether.config["VERSION"] = "aether15m_test"
    aether.config["OUTPUT_DIR"] = str(tmp_path)
    aether.config["SWEEP_MODELS"] = ["Ridge"]  # Ridge only — fast for tests
    aether.engineer_features()
    aether.create()
    period = "window_test"
    aether.train_pooled_models(str(tmp_path), period)

    weights_dir = tmp_path / "weights" / period
    assert (weights_dir / "15m_long" / "ridge_model.pkl").exists()
    assert (weights_dir / "15m_long" / "ridge_scaler.pkl").exists()
    assert (weights_dir / "15m_long" / "ridge_meta.pkl").exists()
    assert (weights_dir / "15m_short" / "ridge_model.pkl").exists()


def test_train_pooled_models_writes_predictions_csvs(tmp_path):
    """Predictions CSVs must exist and have dt, symbol, y_pred columns."""
    aether = _build_aether_with_raw()
    aether.config["VERSION"] = "aether15m_test"
    aether.config["OUTPUT_DIR"] = str(tmp_path)
    aether.config["SWEEP_MODELS"] = ["Ridge"]
    aether.engineer_features()
    aether.create()
    aether.train_pooled_models(str(tmp_path), "window_test")

    pred_long = pd.read_csv(tmp_path / "predictions_long.csv")
    pred_short = pd.read_csv(tmp_path / "predictions_short.csv")
    for df in [pred_long, pred_short]:
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "y_pred" in df.columns
        assert len(df) > 0
```

**Step 2: Run to confirm FAIL**

```bash
PYTHONPATH="agamotto_pkg/src:." pytest aether/tests/test_aether_research.py::test_train_pooled_models_creates_weight_files -v
```
Expected: `AttributeError: 'AetherResearch' object has no attribute 'train_pooled_models'`

**Step 3: Implement `train_pooled_models()`**

Add these imports to the top of `aether/research.py`:

```python
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
```

Add this method inside `AetherResearch`:

```python
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

            logger.info(
                f"Training {position}: {len(train_df)} train rows, "
                f"{len(test_df)} test rows, {len(feature_cols)} features")

            results = _train_models(
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
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
                {"feature_columns": feature_cols, "model_name": best_name},
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
```

**Step 4: Run tests to confirm PASS**

```bash
PYTHONPATH="agamotto_pkg/src:." pytest aether/tests/test_aether_research.py -v
```
Expected: all 7 tests PASS.

**Step 5: Commit**

```bash
git add aether/research.py aether/tests/test_aether_research.py
git commit -m "feat: AetherResearch.train_pooled_models() — pooled training + prediction export

[dev]"
```

---

### Task 4: `gauntlet/run_aether_research.py` — pipeline wrapper

Mirrors `gauntlet/run_orb_research.py`. Generates regime stack, runs research, trains pooled models.

**Files:**
- Create: `gauntlet/run_aether_research.py`

**Step 1: No test needed (thin wrapper) — write directly**

```python
#!/usr/bin/env python3
"""Aether research pipeline wrapper: regime stack → features → pooled models."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def _write_aether_regime_stack(out_dir: Path) -> Path:
    """Write regime_stack.csv using the same cross-TF regimes as ORB.

    Aether's regime_stack.csv has only (regime, position) — no model column.
    The model column is added later by optimize_aether_thresholds.py once
    the best pooled model has been selected.
    """
    import sys
    gauntlet_dir = str(Path(__file__).parent)
    if gauntlet_dir not in sys.path:
        sys.path.insert(0, gauntlet_dir)
    from generate_orb_regimes import generate_regimes

    out_dir.mkdir(parents=True, exist_ok=True)
    regimes = generate_regimes()  # list of {regime, position}

    csv_path = out_dir / "regime_stack.csv"
    fields = ["regime", "position"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in regimes:
            writer.writerow({"regime": r["regime"], "position": r["position"]})

    n = len(regimes)
    print(f"Wrote {n} Aether regimes to {csv_path}")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aether cross-TF research pipeline")
    parser.add_argument(
        "-c", required=True,
        help="Path to setting JSON (gauntlet/pred_aether.../setting.json)")
    parser.add_argument(
        "--output-dir",
        help="Optional output directory (default: setting parent)")
    return parser.parse_args()


def main() -> None:
    import sys
    repo_root = str(Path(__file__).parent.parent)
    agamotto_src = os.path.join(repo_root, "agamotto_pkg", "src")
    for p in [agamotto_src, repo_root]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from aether.research import AetherResearch

    args = parse_args()
    setting_path = Path(args.c).resolve()
    with setting_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    if args.output_dir:
        config["OUTPUT_DIR"] = str(Path(args.output_dir).resolve())
    elif "OUTPUT_DIR" not in config:
        config["OUTPUT_DIR"] = str(setting_path.parent)

    out_dir = Path(config["OUTPUT_DIR"])

    # Step 1: Generate regime stack
    _write_aether_regime_stack(out_dir)
    # Point config at the regime stack — not used by create() but needed
    # by optimize_aether_thresholds.py which reads it directly from out_dir
    config["REGIME_STACK_PATH"] = str(out_dir / "regime_stack.csv")

    home_root = repo_root + "/"

    # Step 2: Load data, engineer features, write vertical_features.csv
    research = AetherResearch(config, home_root)
    research.load()
    research.engineer_features()
    out_dir_str = research.create()
    print(f"Aether research output directory: {out_dir_str}")

    # Step 3: Train pooled models, write predictions_long/short.csv
    period = config.get("WEIGHTS_PERIOD", "window_latest")
    research.train_pooled_models(out_dir_str, period)
    print(f"Pooled models written to {out_dir_str}/weights/{period}/")


if __name__ == "__main__":
    main()
```

**Step 2: Verify script is importable (quick syntax check)**

```bash
PYTHONPATH="agamotto_pkg/src:." python -c "import gauntlet.run_aether_research"
```
Expected: no output (no errors).

**Step 3: Commit**

```bash
git add gauntlet/run_aether_research.py
git commit -m "feat: gauntlet/run_aether_research.py — regime stack + features + pooled training

[dev]"
```

---

### Task 5: `gauntlet/optimize_aether_thresholds.py`

Computes per-regime thresholds from pooled predictions. Reuses `_optimize_single()` from `optimize_thresholds.py`.

**Files:**
- Create: `gauntlet/optimize_aether_thresholds.py`

**Step 1: Write directly (integration script, no unit tests)**

```python
#!/usr/bin/env python3
"""Optimize per-regime thresholds for Aether from pooled model predictions.

For each (regime, position) in regime_stack.csv:
  1. Load pooled predictions (predictions_long.csv / predictions_short.csv)
  2. Apply the regime condition filter to vertical_features
  3. Join with pooled predictions
  4. Run threshold grid sweep (same logic as optimize_thresholds.py)
  5. Write optimal_regime_stack.csv with optimal_threshold + holdout Sharpe/TPM
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _setup_sys_path():
    script_dir = str(Path(__file__).parent)
    repo_root = str(Path(__file__).parent.parent)
    agamotto_src = os.path.join(repo_root, "agamotto_pkg", "src")
    for p in [agamotto_src, repo_root, script_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize Aether per-regime thresholds from pooled predictions")
    parser.add_argument(
        "-c", "--config", required=True,
        help="Path to experiment directory (e.g. gauntlet/pred_aether1h_1)")
    parser.add_argument(
        "--step", type=float, default=0.001,
        help="Threshold grid step (default: 0.001)")
    parser.add_argument(
        "--max-thresh", type=float, default=0.01,
        help="Max threshold magnitude (default: 0.01)")
    parser.add_argument(
        "--method", default="baseline",
        choices=["baseline", "walk-forward", "kfold-cv", "multi-objective"],
        help="Threshold selection method (default: baseline)")
    return parser.parse_args()


def main() -> None:
    _setup_sys_path()

    from aether.research import AetherResearch
    from optimize_thresholds import _optimize_single

    args = parse_args()
    exp_dir = Path(args.config).resolve()
    setting_path = exp_dir / "setting.json"
    with setting_path.open() as f:
        config = json.load(f)

    # ------------------------------------------------------------------ Load
    vf_path = exp_dir / "vertical_features.csv"
    if not vf_path.exists():
        raise FileNotFoundError(
            f"vertical_features.csv not found at {vf_path}. "
            "Run run_aether_research.py first.")

    logger.info(f"Loading vertical_features from {vf_path}")
    vf = pd.read_csv(vf_path, parse_dates=["timestamp"])

    # Load pooled predictions
    preds = {}
    for position in ["long", "short"]:
        p = exp_dir / f"predictions_{position}.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"predictions_{position}.csv not found at {p}. "
                "Run run_aether_research.py first.")
        preds[position] = pd.read_csv(p, parse_dates=["timestamp"])
        logger.info(
            f"Loaded {len(preds[position])} rows from predictions_{position}.csv")

    # Load regime stack
    stack_path = exp_dir / "regime_stack.csv"
    if not stack_path.exists():
        raise FileNotFoundError(
            f"regime_stack.csv not found at {stack_path}. "
            "Run run_aether_research.py first.")

    with stack_path.open() as f:
        regime_rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(regime_rows)} regime entries from {stack_path}")

    # ---------------------------------------- Build AetherResearch for filtering
    repo_root = str(Path(__file__).parent.parent) + "/"
    config["OUTPUT_DIR"] = str(exp_dir)
    research = AetherResearch(config, repo_root)
    # Inject pre-loaded vertical_features so filter_signals works without disk I/O
    research.vertical_features = vf

    # ---------------------------------------- Merge returns into predictions
    # vf has return_long, return_short, return_long_raw, return_short_raw
    ret_cols = ["timestamp", "symbol",
                "return_long", "return_short",
                "return_long_raw", "return_short_raw"]
    available = [c for c in ret_cols if c in vf.columns]
    returns_df = vf[available].copy()

    fee_rate = float(config.get("FEE", 2.25)) / 10000.0
    round_trip_fee = fee_rate * 2.0

    # ---------------------------------------- Optimize per regime
    results = []
    for entry in regime_rows:
        regime_name = entry["regime"]
        position = entry["position"]

        logger.info(f"Optimizing {regime_name} ({position})...")

        try:
            sig = research.filter_signals(entry, save=False)
        except Exception as e:
            logger.warning(
                f"filter_signals failed for {regime_name}/{position}: {e}")
            continue

        if sig is None or sig.empty:
            logger.debug(f"Empty signal for {regime_name}/{position} — skipping")
            continue

        # Join filtered signals with pooled predictions
        pred_df = preds[position].copy()
        sig_keys = sig[["timestamp", "symbol"]].copy()
        merged = pred_df.merge(sig_keys, on=["timestamp", "symbol"], how="inner")
        if merged.empty:
            continue

        # Add return columns
        merged = merged.merge(returns_df, on=["timestamp", "symbol"], how="left")
        target_col = "return_long" if position == "long" else "return_short"
        raw_col = "return_long_raw" if position == "long" else "return_short_raw"
        if target_col not in merged.columns:
            continue

        merged = merged.rename(columns={
            target_col: "y_true",
            raw_col: "y_true_raw",
        })
        merged["position"] = position
        merged["model"] = "pooled"

        # Use the same _optimize_single function as optimize_thresholds.py
        all_res, top = _optimize_single(
            regime_name=regime_name,
            model_name="pooled",
            regime_preds=merged,
            round_trip_fee=round_trip_fee,
            max_thresh=args.max_thresh,
            step=args.step,
            base_dir_name=config.get("VERSION", "aether"),
            method=args.method,
        )

        if top is None:
            logger.debug(
                f"No valid threshold found for {regime_name}/{position}")
            continue

        results.append(top)
        logger.info(
            f"  threshold={top['optimal_threshold']:.4f}  "
            f"sharpe={top['sharpe']:.3f}  tpm={top['trades_per_month']:.0f}")

    if not results:
        logger.warning("No results produced — check regime stack and predictions")
        return

    # ---------------------------------------- Write optimal_regime_stack.csv
    out_path = exp_dir / "optimal_regime_stack.csv"
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
    logger.info(
        f"Wrote {len(out_df)} entries to {out_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Quick syntax check**

```bash
PYTHONPATH="agamotto_pkg/src:." python -c "import gauntlet.optimize_aether_thresholds"
```
Expected: no errors.

**Step 3: Commit**

```bash
git add gauntlet/optimize_aether_thresholds.py
git commit -m "feat: gauntlet/optimize_aether_thresholds.py — per-regime thresholds from pooled preds

[dev]"
```

---

### Task 6: `AetherTrading` — pooled live inference

Loads 8 model artifacts at startup. `make_decision()` does 2 forward passes then applies regime masks.

**Files:**
- Modify: `aether/trading.py`
- Modify: `aether/tests/test_aether_research.py` (add trading tests)

**Step 1: Write the failing tests**

```python
# Add to aether/tests/test_aether_research.py
from unittest.mock import MagicMock, patch
import joblib
import numpy as np
import pandas as pd


def _mock_models_on_disk(tmp_path, base_tf="15m"):
    """Write fake model artifacts to tmp_path/weights/window_test/{tf}_{pos}/."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler
    from gauntlet.rolling_predict_returns import select_feature_columns

    # We need feature_cols — build a minimal AetherResearch to get them
    cfg = {
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "EXCHANGE": "BINANCE", "DATA": "liquid",
        "TIMEFRAMES": ["15m", "1h"], "BASE_TF": base_tf,
        "TARGET_TF": "1h", "TIME_UNIT": "1h",
        "LADDER": 1, "FEE": 0,
        "MA_PERIODS": [7, 25, 99], "STATS_WINDOW": 14,
        "VERSION": "aether_test",
    }
    aether = _build_aether_with_raw(cfg)
    aether.engineer_features()
    aether.create.__wrapped__ = None  # skip parent create
    aether.verticalize()
    f_cols = select_feature_columns(aether.vertical_features.columns.tolist())

    period = "window_test"
    for pos in ["long", "short"]:
        key = f"{base_tf}_{pos}"
        pos_dir = tmp_path / "weights" / period / key
        pos_dir.mkdir(parents=True, exist_ok=True)
        X_dummy = np.random.randn(50, len(f_cols))
        y_dummy = np.random.randn(50)
        scaler = RobustScaler().fit(X_dummy)
        model = Ridge().fit(scaler.transform(X_dummy), y_dummy)
        joblib.dump(model,  pos_dir / "ridge_model.pkl")
        joblib.dump(scaler, pos_dir / "ridge_scaler.pkl")
        joblib.dump({"feature_columns": f_cols}, pos_dir / "ridge_meta.pkl")
    return period, f_cols


def test_aether_trading_loads_models(tmp_path):
    """AetherTrading must load both long and short models on init."""
    from aether.trading import AetherTrading

    period, f_cols = _mock_models_on_disk(tmp_path, base_tf="15m")
    # Write regime stack CSV
    rstack = tmp_path / "regime_stack.csv"
    rstack.write_text("regime,position,optimal_threshold\n15m_baseline,long,0.005\n")

    config = {
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "EXCHANGE": "BINANCE", "DATA": "liquid",
        "TIMEFRAMES": ["15m", "1h"], "BASE_TF": "15m",
        "TARGET_TF": "1h", "TIME_UNIT": "15m",
        "LADDER": 1, "FEE": 0,
        "MA_PERIODS": [7, 25, 99], "STATS_WINDOW": 14,
        "VERSION": "aether_test",
        "WEIGHTS_PERIOD": period,
        "WEIGHTS_PATH": str(tmp_path / "weights" / period),
        "REGIME_STACK_PATH": str(rstack),
        "CAPITAL": 100,
        "SIZES": [0.001],
    }
    trader = AetherTrading(config, str(tmp_path) + "/", skip_load=True)
    assert "long" in trader.models
    assert "short" in trader.models
    assert len(trader.regime_stack) == 1
    assert trader.regime_stack[0]["regime"] == "15m_baseline"
    assert trader.regime_stack[0]["threshold"] == 0.005
```

**Step 2: Run to confirm FAIL**

```bash
PYTHONPATH="agamotto_pkg/src:." pytest aether/tests/test_aether_research.py::test_aether_trading_loads_models -v
```
Expected: FAIL (AetherTrading.__init__ is a stub)

**Step 3: Implement `AetherTrading`**

Replace `aether/trading.py` content:

```python
"""Aether trading: pooled cross-TF live inference."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from agamotto.utils import (
    _symbol_to_native,
    _timeframe_to_seconds,
)

from .research import AetherResearch

logger = logging.getLogger(__name__)


class AetherTrading(AetherResearch):
    """Live pooled cross-TF inference.

    Key difference vs OrbTrading:
    - Loads ONE model per position (long/short) instead of one per regime
    - make_decision() does 2 forward passes then applies regime masks
    """

    def __init__(
        self,
        config: Dict[str, object],
        home_root: str,
        period: Optional[str] = None,
        skip_load: bool = False,
    ) -> None:
        super().__init__(config, home_root)
        self.period = self.config.get("WEIGHTS_PERIOD") or period
        self.trading_mode = self.config.get("TRADING_MODE", "both")
        self.models: Dict[str, dict] = {}
        self.regime_stack: List[dict] = []
        self.decisions: Dict[str, list] = {}
        self._data_fresh = False

        # Load regime stack from CSV
        stack_path = self.config.get("REGIME_STACK_PATH")
        if not stack_path:
            raise ValueError("REGIME_STACK_PATH not provided in configuration.")
        if not os.path.isabs(stack_path):
            stack_path = os.path.join(home_root, stack_path)
        if not os.path.exists(stack_path):
            raise FileNotFoundError(f"REGIME_STACK_PATH not found: {stack_path}")
        self._load_regime_stack(stack_path)

        # Load pooled model artifacts
        self._load_models()

        # Calculate position sizes (reuse OrbTrading logic via parent)
        from orb.trading import OrbTrading
        OrbTrading._calculate_sizes(self)

        if not skip_load:
            try:
                self.load_data(limit=700)
            except Exception as e:
                logger.warning(
                    f"Failed to load initial data in __init__: {e}")

        logger.info(
            f"AetherTrading initiated — period: {self.period}, "
            f"base_tf: {self.base_tf}, "
            f"regimes: {len(self.regime_stack)}, "
            f"models: {list(self.models.keys())}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_regime_stack(self, stack_path: str) -> None:
        """Load regime stack CSV into self.regime_stack.

        CSV columns: regime, position, optimal_threshold
        No model weights per regime — threshold is the only regime artifact.
        """
        import csv as _csv
        with open(stack_path, newline="") as f:
            rows = list(_csv.DictReader(f))

        self.regime_stack = []
        for row in rows:
            regime = row.get("regime", "")
            if not regime or regime.startswith("__"):
                continue
            position = row.get("position", "long")
            threshold = float(row.get("optimal_threshold", 0.0))
            self.regime_stack.append({
                "regime": regime,
                "position": position,
                "threshold": threshold,
            })
        logger.info(
            f"Loaded {len(self.regime_stack)} regime entries from {stack_path}")

    def _load_models(self) -> None:
        """Load one model artifact per position (long + short)."""
        weights_path = self.config.get("WEIGHTS_PATH")
        if not weights_path:
            raise ValueError("WEIGHTS_PATH not provided in configuration.")
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(self.home_root, weights_path)

        for position in ["long", "short"]:
            key = f"{self.base_tf}_{position}"
            pos_dir = os.path.join(weights_path, key)
            if not os.path.isdir(pos_dir):
                raise FileNotFoundError(
                    f"Model directory not found for {key}: {pos_dir}")

            model_files = [
                f for f in os.listdir(pos_dir) if f.endswith("_model.pkl")]
            if not model_files:
                raise FileNotFoundError(
                    f"No *_model.pkl files found in {pos_dir}")

            low = model_files[0].replace("_model.pkl", "")
            model  = joblib.load(os.path.join(pos_dir, f"{low}_model.pkl"))
            scaler = joblib.load(os.path.join(pos_dir, f"{low}_scaler.pkl"))
            meta   = joblib.load(os.path.join(pos_dir, f"{low}_meta.pkl"))

            self.models[position] = {
                "model": model,
                "scaler": scaler,
                "feature_columns": meta["feature_columns"],
            }
            logger.info(
                f"Loaded {low} model for {key} "
                f"({len(meta['feature_columns'])} features)")

    # ------------------------------------------------------------------
    # Data loading — delegate entirely to OrbTrading
    # ------------------------------------------------------------------

    def load_data(self, limit: int = 700) -> None:
        """Delegate to OrbTrading.load_data() — identical multi-TF fetch."""
        from orb.trading import OrbTrading
        OrbTrading.load_data(self, limit=limit)

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def make_decision(self, label: str = "both") -> Dict[str, list]:
        self.decisions = {
            sym: [0.0, 0.0] for sym in self.config.get("SYMBOLS", [])}

        if not getattr(self, "_data_fresh", True):
            logger.warning(
                "Data not fresh — returning CLOSE (all zeros) for all symbols.")
            return self.decisions

        if self.features is None:
            self.engineer_features()
        self.verticalize()

        if self.vertical_features is None or self.vertical_features.empty:
            logger.error("vertical_features is empty — cannot make decision")
            return self.decisions

        # ---- Step 1: one forward pass per position ----
        target_ts = self.vertical_features["timestamp"].max()
        latest = self.vertical_features[
            self.vertical_features["timestamp"] == target_ts].copy()

        y_pred: Dict[str, Dict[str, float]] = {"long": {}, "short": {}}
        for position in ["long", "short"]:
            if position not in self.models:
                continue
            artifact = self.models[position]
            f_cols = artifact["feature_columns"]
            missing = [c for c in f_cols if c not in latest.columns]
            if missing:
                logger.warning(
                    f"Missing {len(missing)} features for {position} model")
                fill = pd.DataFrame(
                    0.0, index=latest.index,
                    columns=[c for c in f_cols if c not in latest.columns])
                latest = pd.concat([latest, fill], axis=1)
            X = latest[f_cols].replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)
            X_scaled = artifact["scaler"].transform(X)
            preds = artifact["model"].predict(X_scaled)
            for i, row in enumerate(latest.itertuples(index=False)):
                sym = getattr(row, "symbol", None)
                if sym:
                    y_pred[position][sym] = float(preds[i])

        # ---- Step 2: regime voting ----
        long_votes: Dict[str, int] = {
            s: 0 for s in self.config.get("SYMBOLS", [])}
        short_votes: Dict[str, int] = {
            s: 0 for s in self.config.get("SYMBOLS", [])}

        for regime in self.regime_stack:
            try:
                sig = self.filter_signals(regime, save=False)
            except Exception as e:
                logger.error(
                    f"filter_signals error for {regime['regime']}: {e}")
                continue
            if sig is None or sig.empty:
                continue

            active_syms = set(
                sig[sig["timestamp"] == target_ts]["symbol"].tolist())
            threshold = regime["threshold"]
            position = regime["position"]

            for sym in self.config.get("SYMBOLS", []):
                if sym not in active_syms:
                    continue
                pred = y_pred.get(position, {}).get(sym)
                if pred is None:
                    continue
                if position == "long" and pred > threshold:
                    long_votes[sym] += 1
                elif position == "short" and pred < threshold:
                    short_votes[sym] += 1

        # ---- Step 3: net position per symbol ----
        from agamotto.utils import _symbol_to_native
        from decimal import Decimal
        capital = self.config.get("CAPITAL", 100)
        lot_sizes = self.config.get("LOT_SIZES", {})
        sizes = self.config.get("SIZES", [])
        symbols = self.config.get("SYMBOLS", [])

        # Latest closes from raw
        latest_closes: Dict[str, float] = {}
        if hasattr(self, "raw") and not self.raw.empty:
            for sym in symbols:
                native = _symbol_to_native(sym)
                col = f"{native}_close"
                if col in self.raw.columns:
                    val = (self.raw[col].iloc[-2]
                           if len(self.raw) >= 2 else self.raw[col].iloc[-1])
                    latest_closes[sym] = float(val) if not pd.isna(val) else 0.0

        for sym in symbols:
            net = long_votes.get(sym, 0) - short_votes.get(sym, 0)
            close = latest_closes.get(sym, 0.0)
            lot_info = lot_sizes.get(sym, {})
            step = lot_info.get("step_size", 0)
            if close > 0 and step > 0:
                qty = capital / close
                prec = max(
                    0, int(round(-Decimal(str(step)).log10()))) if step < 1 else 0
                qty = round(round(qty / step) * step, prec)
                base_size = qty
            else:
                idx = symbols.index(sym)
                base_size = sizes[idx] if idx < len(sizes) else 0.0
            price = close
            self.decisions[sym] = [price, float(base_size) * net]

            if net != 0:
                logger.info(
                    f"Decision for {sym}: long_votes={long_votes[sym]} "
                    f"short_votes={short_votes[sym]} → net={net}")

        return self.decisions

    def clean(self) -> Dict[str, list]:
        symbols = self.config.get("SYMBOLS", [])
        self.decisions = {sym: [0.0, 0.0] for sym in symbols}
        return self.decisions
```

**Step 4: Run tests**

```bash
PYTHONPATH="agamotto_pkg/src:." pytest aether/tests/ -v
```
Expected: all tests PASS.

**Step 5: Commit**

```bash
git add aether/trading.py aether/tests/test_aether_research.py
git commit -m "feat: AetherTrading — pooled inference, 2 forward passes + regime voting

[dev]"
```

---

### Task 7: `tesseract/tesseract.py` — add "aether" branch

Tesseract uses `strategy == "orb"` in 3 spots. Aether needs identical treatment (same multi-TF data loading and feature engineering).

**Files:**
- Modify: `tesseract/tesseract.py`

**Step 1: Find the 3 locations**

```bash
grep -n 'strategy == "orb"' tesseract/tesseract.py
```
Expected output (3 lines): lines ~210, ~331, ~365 and ~380.

**Step 2: Edit location 1 — timeframes detection (line ~210)**

Find:
```python
    if strategy == "orb":
        timeframes = config.get("TIMEFRAMES", ["15m", "1h", "4h", "1d"])
    else:
        timeframes = [timeframe]
```

Replace with:
```python
    if strategy in ("orb", "aether"):
        timeframes = config.get("TIMEFRAMES", ["15m", "1h", "4h", "1d"])
    else:
        timeframes = [timeframe]
```

**Step 3: Edit location 2 — strategy class selection (line ~331)**

Find:
```python
    if strategy == "orb":
        from orb.research import OrbResearch
        agamotto = OrbResearch(setting, project_root + "/")
        logger.info("Using OrbResearch (cross-TF strategy)")
```

Replace with:
```python
    if strategy in ("orb", "aether"):
        from orb.research import OrbResearch
        agamotto = OrbResearch(setting, project_root + "/")
        logger.info(f"Using OrbResearch for strategy={strategy} (cross-TF)")
```

**Step 4: Edit location 3 — TF instance trimming (lines ~365, ~380)**

Find:
```python
    if strategy == "orb" and hasattr(agamotto, '_tf_instances'):
```
(appears twice, for trim and for bar-drop)

Replace each with:
```python
    if strategy in ("orb", "aether") and hasattr(agamotto, '_tf_instances'):
```

**Step 5: Also update `daily_validate.py` (same pattern)**

```bash
grep -n 'strategy == "orb"' tesseract/daily_validate.py
```

Apply the same `strategy in ("orb", "aether")` change to any matching lines in `daily_validate.py`.

**Step 6: Run existing tesseract tests**

```bash
PYTHONPATH="agamotto_pkg/src:." pytest tesseract/tests/ -v
```
Expected: all existing tests still PASS.

**Step 7: Commit**

```bash
git add tesseract/tesseract.py tesseract/daily_validate.py
git commit -m "feat: tesseract — add aether strategy branch alongside orb

[dev]"
```

---

### Task 8: `filter_regime_stacks.py` — add pred_aether* glob

`filter_regime_stacks.py` hardcodes glob patterns for known algorithm prefixes.

**Files:**
- Modify: `gauntlet/filter_regime_stacks.py`

**Step 1: Find the glob list**

```bash
grep -n "pred_orb\|pred_agamotto\|pred_vomir" gauntlet/filter_regime_stacks.py | head -10
```

**Step 2: Add pred_aether* to the glob list**

Find the block:
```python
    csv_files = (
        glob.glob(f"{project}/pred_agamotto*/optimal_regime_stack.csv")
        + glob.glob(f"{project}/pred_orb*/optimal_regime_stack.csv")
        + glob.glob(f"{project}/pred_vomir*/optimal_regime_stack.csv")
        + glob.glob(f"{project}/pred_mjolnir*/optimal_regime_stack.csv")
        + glob.glob(f"{project}/pred_valkyrie*/optimal_regime_stack.csv")
    )
```

Replace with:
```python
    csv_files = (
        glob.glob(f"{project}/pred_agamotto*/optimal_regime_stack.csv")
        + glob.glob(f"{project}/pred_orb*/optimal_regime_stack.csv")
        + glob.glob(f"{project}/pred_aether*/optimal_regime_stack.csv")
        + glob.glob(f"{project}/pred_vomir*/optimal_regime_stack.csv")
        + glob.glob(f"{project}/pred_mjolnir*/optimal_regime_stack.csv")
        + glob.glob(f"{project}/pred_valkyrie*/optimal_regime_stack.csv")
    )
```

**Step 3: Run linting**

```bash
flake8 gauntlet/filter_regime_stacks.py --count --select=E9,F63,F7,F82 --show-source --statistics
```
Expected: 0 errors.

**Step 4: Commit**

```bash
git add gauntlet/filter_regime_stacks.py
git commit -m "feat: filter_regime_stacks.py — add pred_aether* to glob patterns

[dev]"
```

---

### Task 9: Experiment configs + pipeline script

**Files:**
- Create: `gauntlet/pred_aether15m_1/setting.json`
- Create: `gauntlet/pred_aether1h_1/setting.json`
- Create: `gauntlet/pred_aether4h_1/setting.json`
- Create: `gauntlet/pred_aether1d_1/setting.json`
- Create: `gauntlet/run_aether1_pipeline.sh`

**Step 1: Copy and adapt liquid ORB symbols list**

```bash
# Use pred_orb1h_1 as a template for symbols, API keys, etc.
cat gauntlet/pred_orb1h_1/setting.json
```

**Step 2: Create the four setting.json files**

For each TF, create `gauntlet/pred_aether{TF}_1/setting.json` with these differences from the ORB template:
- `"VERSION"`: `"aether{TF}_1"` (e.g., `"aether1h_1"`)
- `"STRATEGY"`: `"aether"`
- `"BASE_TF"`: the TF (e.g., `"1h"`)
- `"TARGET_TF"`: same as BASE_TF
- `"TIME_UNIT"`: same as BASE_TF
- `"LADDER"`: same as corresponding ORB config
- `"WEIGHTS_PERIOD"`: `"window_2026_03"` (current month)
- `"WEIGHTS_PATH"`: `"gauntlet/pred_aether{TF}_1/weights/window_2026_03"`
- `"REGIME_STACK_PATH"`: `"gauntlet/pred_aether{TF}_1/filtered_optimal_regime_stack.csv"`
- `"OUTPUT_DIR"`: `/mnt/tardis-data-archive/marvel-research/gauntlet/pred_aether{TF}_1`

All API keys, SYMBOLS, CAPITAL, LEVERAGE, STOPLOSS, etc. — copy from the corresponding ORB config (`pred_orb{TF}_1`).

**Step 3: Create the pipeline script**

```bash
#!/usr/bin/env bash
# gauntlet/run_aether1_pipeline.sh
# Full Aether _1 pipeline: research → optimize thresholds → filter

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/py313/bin/python}"
export PYTHONPATH="${REPO}/agamotto_pkg/src:${REPO}"

# Resource check
echo "=== Resource check ==="
free -h
df -h /home/ubuntu/
pgrep -f "run_aether" && { echo "ERROR: pipeline already running — abort"; exit 1; } || true
echo ""

EXPS=(
  "pred_aether15m_1|0.001|0.01"
  "pred_aether1h_1|0.001|0.01"
  "pred_aether4h_1|0.01|0.1"
  "pred_aether1d_1|0.01|0.1"
)

# Step 1: Research — features + pooled model training (sequential!)
echo "=== Step 1: Research ==="
for entry in "${EXPS[@]}"; do
  NAME="${entry%%|*}"
  DIR="${REPO}/gauntlet/${NAME}"
  echo "--- Research: ${NAME} ---"
  cd "${REPO}"
  "${PYTHON}" gauntlet/run_aether_research.py -c "${DIR}/setting.json"
done

# Step 2: Optimize thresholds — per-regime from pooled predictions
echo "=== Step 2: Optimize Thresholds ==="
for entry in "${EXPS[@]}"; do
  NAME="${entry%%|*}"
  DIR="${REPO}/gauntlet/${NAME}"
  STEP=$(echo "$entry" | cut -d'|' -f2)
  MAXTH=$(echo "$entry" | cut -d'|' -f3)
  echo "--- Optimize: ${NAME} (step=${STEP}, max=${MAXTH}) ---"
  cd "${REPO}"
  "${PYTHON}" gauntlet/optimize_aether_thresholds.py \
    -c "${DIR}" --step "${STEP}" --max-thresh "${MAXTH}"
done

# Step 3: Filter regime stacks (reuses existing filter script)
echo "=== Step 3: Filter Regime Stacks ==="
cd "${REPO}"
"${PYTHON}" gauntlet/filter_regime_stacks.py \
  --project gauntlet \
  --top-n 120 \
  --min-sharpe 0.5 \
  --min-tpm 50 \
  --min-both-thresh 0.001 \
  --name-filter "pred_aether"

echo ""
echo "=== Aether _1 pipeline complete ==="
```

**Step 4: Make executable**

```bash
chmod +x gauntlet/run_aether1_pipeline.sh
```

**Step 5: Lint new files**

```bash
flake8 gauntlet/run_aether_research.py gauntlet/optimize_aether_thresholds.py \
  --count --select=E9,F63,F7,F82 --show-source --statistics
```

**Step 6: Run full test suite**

```bash
PYTHONPATH="agamotto_pkg/src:." pytest agamotto_pkg/tests/ ltp/tests/ optimus/tests/ orb/tests/ aether/tests/ -v
```
Expected: all PASS.

**Step 7: Commit**

```bash
git add gauntlet/pred_aether15m_1/ gauntlet/pred_aether1h_1/ \
        gauntlet/pred_aether4h_1/ gauntlet/pred_aether1d_1/ \
        gauntlet/run_aether1_pipeline.sh
git commit -m "feat: aether experiment configs (15m/1h/4h/1d) + run_aether1_pipeline.sh

[dev]"
```

---

## Verification Checklist

Before claiming complete, verify all of the following:

```bash
# 1. All tests pass
PYTHONPATH="agamotto_pkg/src:." pytest aether/tests/ orb/tests/ -v

# 2. No new lint errors
flake8 aether/ gauntlet/run_aether_research.py gauntlet/optimize_aether_thresholds.py \
  --count --select=E9,F63,F7,F82 --show-source --statistics

# 3. Package imports cleanly
PYTHONPATH="agamotto_pkg/src:." python -c "
from aether import AetherResearch, AetherTrading
from orb.research import OrbResearch
assert issubclass(AetherResearch, OrbResearch)
print('OK — AetherResearch is a subclass of OrbResearch')
"

# 4. Pipeline script is syntactically valid
bash -n gauntlet/run_aether1_pipeline.sh && echo "OK"
```

---

## Files Created / Modified Summary

| File | Action |
|---|---|
| `aether/__init__.py` | New |
| `aether/research.py` | New — `AetherResearch(OrbResearch)` |
| `aether/trading.py` | New — `AetherTrading(AetherResearch)` |
| `aether/tests/__init__.py` | New |
| `aether/tests/test_aether_research.py` | New |
| `gauntlet/run_aether_research.py` | New |
| `gauntlet/optimize_aether_thresholds.py` | New |
| `gauntlet/run_aether1_pipeline.sh` | New |
| `gauntlet/pred_aether{15m,1h,4h,1d}_1/setting.json` | New (4 files) |
| `gauntlet/filter_regime_stacks.py` | Edit — add pred_aether* glob |
| `tesseract/tesseract.py` | Edit — 3 locations: `"orb"` → `("orb", "aether")` |
| `tesseract/daily_validate.py` | Edit — same pattern as tesseract.py |
