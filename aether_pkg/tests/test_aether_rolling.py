"""Tests for AetherResearch.train_pooled_models rolling OOS rewrite.

Verifies the leakage fix: predictions_{long,short}.csv must contain ONLY
out-of-sample test-month rows (train months excluded, warmup gap present),
and production weights come from the last rolling window.

Cross-repo: train_pooled_models lazily imports `rolling_predict_returns`
(lives in marvel/gauntlet). Skip when it's not on PYTHONPATH (dc-only run) —
run from marvel/ with PYTHONPATH=/home/stan/sandbox/dc/aether_pkg/src:.:
    PYTHONPATH=/home/stan/sandbox/dc/aether_pkg/src:/home/stan/sandbox/dc/orb_pkg/src:\
/home/stan/sandbox/dc/agamotto_pkg/src:. pytest aether_pkg/tests/test_aether_rolling.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

# gauntlet/rolling_predict_returns lives in marvel/ — skip if unreachable.
pytest.importorskip(
    "rolling_predict_returns",
    reason="rolling_predict_returns not on PYTHONPATH (run from marvel/ with "
           "gauntlet on the path)",
)
from aether.research import AetherResearch  # noqa: E402

BASE_TF = "1h"
N_MONTHS = 18
WINDOW_SIZE = 12  # train 11 / test 1
SYMBOLS = ["BINANCE_PERP_BTC_USDT", "BINANCE_PERP_ETH_USDT", "BINANCE_PERP_SOL_USDT"]
ROWS_PER_SYMBOL_MONTH = 40
FEATURES = [f"{BASE_TF}_mom", f"{BASE_TF}_vol_z", f"{BASE_TF}_rsi"]


def _minimal_config(window_size=WINDOW_SIZE):
    return {
        "SYMBOLS": SYMBOLS,
        "EXCHANGE": "BINANCE", "DATA": "liquid",
        "TIMEFRAMES": ["15m", "1h", "4h", "1d"],
        "BASE_TF": BASE_TF, "TARGET_TF": BASE_TF, "TIME_UNIT": BASE_TF,
        "LADDER": 1, "FEE": 0, "MA_PERIODS": [7, 25, 99], "STATS_WINDOW": 14,
        "VERSION": "aether1h_rolltest",
        "SWEEP_MODELS": ["Ridge"],   # Ridge only — fast, deterministic enough
        "WINDOW_SIZE": window_size,
    }


def _make_vf(n_months=N_MONTHS, seed=7):
    """Synthetic pooled vertical_features: months × symbols × rows."""
    rng = np.random.default_rng(seed)
    parts = []
    start = pd.Timestamp("2024-01-01")
    months = [(start + pd.DateOffset(months=i)) for i in range(n_months)]
    for mstart in months:
        for sym in SYMBOLS:
            ts = pd.date_range(mstart, periods=ROWS_PER_SYMBOL_MONTH, freq="6h")
            f = {c: rng.standard_normal(len(ts)) for c in FEATURES}
            ret_long = 0.5 * f[FEATURES[0]] - 0.3 * f[FEATURES[1]] \
                + 0.1 * rng.standard_normal(len(ts))
            parts.append(pd.DataFrame({
                "timestamp": ts,
                "year": ts.year, "month": ts.month,
                "symbol": sym.split("_")[-2] + "USDT",
                **f,
                "return_long": ret_long,
                "return_short": -ret_long + 0.05 * rng.standard_normal(len(ts)),
            }))
    return pd.concat(parts, ignore_index=True)


def _build(vf, window_size=WINDOW_SIZE):
    aether = AetherResearch(_minimal_config(window_size), "/tmp/fake_root")
    aether.vertical_features = vf
    return aether


def _expected_months(vf):
    return sorted(set(zip(vf["year"].astype(int), vf["month"].astype(int))))


def test_predictions_are_oos_only(tmp_path):
    vf = _make_vf()
    aether = _build(vf)
    aether.train_pooled_models(str(tmp_path), "window_test")

    months = _expected_months(vf)
    oos_expected = set(months[WINDOW_SIZE - 1:])           # months[11:] → 7 test months
    train_first = set(months[:WINDOW_SIZE - 1])            # window 0 train months

    for pos in ["long", "short"]:
        p = tmp_path / f"predictions_{pos}.csv"
        assert p.exists(), f"predictions_{pos}.csv missing"
        df = pd.read_csv(p, parse_dates=["timestamp"])
        assert list(df.columns) == ["timestamp", "symbol", "y_pred"]
        assert len(df) > 0

        got_months = set(zip(df["timestamp"].dt.year, df["timestamp"].dt.month))
        # Exactly the OOS test months — no train months, warmup excluded.
        assert got_months == oos_expected, (pos, got_months, oos_expected)
        # No leakage: nothing from window-0 training span leaked in.
        assert got_months.isdisjoint(train_first)
        # Earliest OOS row strictly later than earliest vf row (warmup gap).
        assert df["timestamp"].min() > vf["timestamp"].min()


def test_production_weights_from_last_window(tmp_path):
    vf = _make_vf()
    aether = _build(vf)
    aether.train_pooled_models(str(tmp_path), "window_test")

    for pos in ["long", "short"]:
        wdir = tmp_path / "weights" / "window_test" / f"{BASE_TF}_{pos}"
        assert wdir.exists(), f"missing weights dir {wdir}"
        assert list(wdir.glob("*_model.pkl")), f"no model pkl in {wdir}"
        assert list(wdir.glob("*_scaler.pkl"))
        assert list(wdir.glob("*_meta.pkl"))


def test_fail_fast_when_too_few_months(tmp_path):
    vf = _make_vf(n_months=WINDOW_SIZE - 1)   # 11 months < window_size 12
    aether = _build(vf)
    with pytest.raises(ValueError):
        aether.train_pooled_models(str(tmp_path), "window_test")


def test_missing_window_size_raises(tmp_path):
    vf = _make_vf()
    aether = AetherResearch(_minimal_config(), "/tmp/fake_root")
    aether.vertical_features = vf
    del aether.config["WINDOW_SIZE"]
    with pytest.raises(KeyError):
        aether.train_pooled_models(str(tmp_path), "window_test")
