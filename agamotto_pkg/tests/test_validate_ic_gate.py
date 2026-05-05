# agamotto_pkg/tests/test_validate_ic_gate.py
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from gauntlet.validate_ic_gate import compute_ic_summary, load_preds, IC_GATE


def _make_preds(n=2000, ic=0.05, seed=42):
    """Make synthetic preds parquet rows with known IC."""
    rng = np.random.default_rng(seed)
    y_true = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    y_pred = ic * y_true + np.sqrt(1 - ic**2) * noise
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="1h"),
        "symbol": "BINANCE_PERP_BTC_USDT",
        "position": "long",
        "regime": "baseline",
        "model": "LightGBM",
        "y_true": y_true,
        "y_true_raw": y_true,
        "y_pred": y_pred,
        "ret_1bar": y_true,
        "window_id": np.repeat(np.arange(10), n // 10),
    })


def test_ic_passes_gate(tmp_path):
    df = _make_preds(ic=0.05)
    summary = compute_ic_summary(df)
    assert summary["mean_ic"].iloc[0] > IC_GATE


def test_ic_fails_gate(tmp_path):
    df = _make_preds(ic=0.001)
    summary = compute_ic_summary(df)
    assert summary["mean_ic"].iloc[0] < IC_GATE


def test_load_preds_reads_parquet(tmp_path):
    df = _make_preds()
    f = tmp_path / "preds_window_29.parquet"
    df.to_parquet(f, index=False)
    loaded = load_preds(tmp_path)
    assert len(loaded) == len(df)
    assert "y_pred" in loaded.columns


def test_empty_preds_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_preds(tmp_path)
