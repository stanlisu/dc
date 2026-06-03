# agamotto_pkg/tests/test_validate_pipeline_output.py
import json
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# `gauntlet` lives in the marvel repo, not dc — skip when unavailable (dc CI).
pytest.importorskip("gauntlet")
from gauntlet.validate_pipeline_output import (
    validate_regime_stack,
    validate_filtered_stack,
    validate_setting,
    ValidationError,
)


def _setting(tmp_path, models=None):
    s = {
        "VERSION": "test_1",
        "PROJECT": "gauntlet",
        "SWEEP_MODELS": models or ["LightGBM", "Ridge", "ElasticNet"],
        "WEIGHTS_PERIOD": "window_2026_03",
        "REGIME_STACK_PATH": str(tmp_path / "regime_stack.csv"),
    }
    p = tmp_path / "setting.json"
    p.write_text(json.dumps(s))
    return s


def _regime_stack(tmp_path, rows=None):
    # Real schema: regime,model — direction encoded as _long/_short suffix in regime name
    rows = rows or [
        {"regime": "baseline_long", "model": "LightGBM"},
        {"regime": "baseline_short", "model": "Ridge"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "regime_stack.csv", index=False)
    return df


def _filtered_stack(tmp_path, rows=None):
    rows = rows or [{"regime": "baseline_long", "model": "LightGBM", "position": "long",
                     "optimal_threshold": 0.002, "sharpe": 1.5, "selection_sharpe": 1.2,
                     "trades_per_month": 80, "avg_pnl_per_trade_bps": 45.0,
                     "n_trades": 120, "win_rate": 0.52, "directory": "pred_test_1",
                     "method": "baseline"}]
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "filtered_optimal_regime_stack.csv", index=False)
    return df


# --- regime_stack.csv ---

def test_regime_stack_valid(tmp_path):
    cfg = _setting(tmp_path)
    _regime_stack(tmp_path)
    validate_regime_stack(tmp_path / "regime_stack.csv", cfg)  # no raise


def test_regime_stack_bad_position(tmp_path):
    """Regime names without _long/_short suffix should fail."""
    cfg = _setting(tmp_path)
    _regime_stack(tmp_path, [{"regime": "baseline", "model": "LightGBM"}])
    with pytest.raises(ValidationError, match="_long"):
        validate_regime_stack(tmp_path / "regime_stack.csv", cfg)


def test_regime_stack_long_suffix_in_regime(tmp_path):
    """_long suffix in regime name is the correct convention — must pass."""
    cfg = _setting(tmp_path)
    _regime_stack(tmp_path, [{"regime": "vol_breakout_long", "model": "LightGBM"}])
    validate_regime_stack(tmp_path / "regime_stack.csv", cfg)  # no raise


def test_regime_stack_unknown_model(tmp_path):
    cfg = _setting(tmp_path)
    _regime_stack(tmp_path, [{"regime": "baseline_long", "model": "RandomForest"}])
    with pytest.raises(ValidationError, match="model"):
        validate_regime_stack(tmp_path / "regime_stack.csv", cfg)


# --- filtered_optimal_regime_stack.csv ---

def test_filtered_stack_valid(tmp_path):
    cfg = _setting(tmp_path)
    _filtered_stack(tmp_path)
    validate_filtered_stack(tmp_path / "filtered_optimal_regime_stack.csv", cfg)  # no raise


def test_filtered_stack_missing_column(tmp_path):
    cfg = _setting(tmp_path)
    df = pd.DataFrame([{"regime": "baseline_long", "model": "LightGBM"}])
    df.to_csv(tmp_path / "filtered_optimal_regime_stack.csv", index=False)
    with pytest.raises(ValidationError, match="missing columns"):
        validate_filtered_stack(tmp_path / "filtered_optimal_regime_stack.csv", cfg)


def test_filtered_stack_nan_sharpe(tmp_path):
    cfg = _setting(tmp_path)
    rows = [{"regime": "baseline_long", "model": "LightGBM", "position": "long",
             "optimal_threshold": 0.002, "sharpe": None, "selection_sharpe": 1.2,
             "trades_per_month": 80, "avg_pnl_per_trade_bps": 45.0,
             "n_trades": 120, "win_rate": 0.52, "directory": "pred_test_1", "method": "baseline"}]
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "filtered_optimal_regime_stack.csv", index=False)
    with pytest.raises(ValidationError, match="NaN"):
        validate_filtered_stack(tmp_path / "filtered_optimal_regime_stack.csv", cfg)


# --- setting.json ---

def test_setting_valid(tmp_path):
    cfg = _setting(tmp_path)
    _regime_stack(tmp_path)
    validate_setting(tmp_path / "setting.json")  # no raise


def test_setting_missing_key(tmp_path):
    s = {"VERSION": "test_1"}  # missing required keys
    (tmp_path / "setting.json").write_text(json.dumps(s))
    with pytest.raises(ValidationError, match="missing"):
        validate_setting(tmp_path / "setting.json")
