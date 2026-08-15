"""Tests for OrbTrading initialization, make_decision, predict, and clean."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from orb.trading import OrbTrading


def _make_regime(regime_id, position, threshold, prediction_value):
    """Create a mock regime dict with a model returning a fixed prediction."""
    model = MagicMock()
    model.predict.return_value = [prediction_value]
    scaler = MagicMock()
    scaler.transform.return_value = [[0.1, 0.2]]
    return {
        "id": regime_id,
        "regime": "test_regime",
        "position": position,
        "model_name": "mock",
        "threshold": threshold,
        "artifact": {
            "model": model,
            "scaler": scaler,
            "metadata": {"feature_columns": ["feat_a", "feat_b"]},
        },
        "filter": None,
        "config": {},
    }


@pytest.fixture
def base_config():
    return {
        "TIME_UNIT": "15m",
        "TIMEFRAME_SECONDS": 900,
        "TIMEFRAMES": ["15m"],
        "BASE_TF": "15m",
        "TARGET_TF": "1h",
        "SIZES": [0.01],
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "CAPITAL": 1000,
        "TRADING_MODE": "both",
        "LONG_PRED_THRESHOLD": 0.0,
        "SHORT_PRED_THRESHOLD": 0.0,
        "REGIME_STACK_PATH": "/tmp/fake_regime_stack.csv",
        "LOT_SIZES": {
            "BINANCE_PERP_BTC_USDT": {"step_size": 0.001, "min_notional": 5.0}
        },
    }


@pytest.fixture
def orb(base_config):
    """Create OrbTrading with all external deps mocked."""
    def fake_load_regime_stack(self_inner):
        self_inner.regime_stack = []
        self_inner.models = {"long": {}, "short": {}}

    with patch.object(OrbTrading, "_load_regime_stack",
                      fake_load_regime_stack), \
         patch.object(OrbTrading, "_calculate_sizes"), \
         patch.object(OrbTrading, "load_data"):
        inst = OrbTrading(
            config=base_config, home_root="/tmp",
            period="window_test", skip_load=True)

    inst.engineer_features = MagicMock()
    inst.verticalize = MagicMock()

    settled_ts = pd.Timestamp("2025-01-01 00:00:00")
    inst.vertical_features = pd.DataFrame({
        "timestamp": [settled_ts],
        "symbol": ["BINANCE_PERP_BTC_USDT"],
        "feat_a": [1.0],
        "feat_b": [2.0],
        "position": ["long"],
    })
    inst.features = inst.vertical_features.copy()
    inst.raw = pd.DataFrame(
        {"BTCUSDT_close": [49000.0, 50000.0]},
        index=pd.to_datetime(["2024-12-30", "2024-12-31"]),
    )
    inst._data_fresh = True
    return inst


def _regime_aware_filter(inst):
    def mock_filter(regime, save=False):
        vf = inst.vertical_features.copy()
        vf["position"] = regime["position"]
        return vf
    return mock_filter


# -------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------


def test_initialization(orb):
    assert orb.config["CAPITAL"] == 1000
    assert orb.trading_mode == "both"
    assert "BINANCE_PERP_BTC_USDT" in orb.config["SYMBOLS"]
    assert isinstance(orb.regime_stack, list)


# -------------------------------------------------------------------
# make_decision
# -------------------------------------------------------------------


def test_make_decision_long(orb):
    """One long regime above threshold => positive qty."""
    regime = _make_regime("r_long", "long", 0.005, 0.01)
    orb.regime_stack = [regime]
    orb.filter_signals = _regime_aware_filter(orb)
    decisions = orb.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0, f"Expected positive qty for long, got {qty}"


def test_make_decision_short(orb):
    """One short regime below threshold => negative qty."""
    regime = _make_regime("r_short", "short", -0.005, -0.01)
    orb.regime_stack = [regime]
    orb.filter_signals = _regime_aware_filter(orb)
    decisions = orb.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty < 0, f"Expected negative qty for short, got {qty}"


def test_clean_returns_zeros(orb):
    decisions = orb.clean()
    assert decisions == {"BINANCE_PERP_BTC_USDT": [0.0, 0.0]}


# -------------------------------------------------------------------
# predict
# -------------------------------------------------------------------


def test_predict_returns_expected_columns(orb):
    """predict() should add prediction, strategy_id, regime, opt_threshold."""
    regime = _make_regime("r_test", "long", 0.005, 0.01)
    sig = orb.vertical_features.copy()
    sig["position"] = "long"
    result = orb.predict(sig, regime)
    assert not result.empty
    for col in ["prediction", "strategy_id", "regime", "opt_threshold"]:
        assert col in result.columns, f"Missing column: {col}"
    assert float(result["prediction"].iloc[0]) == 0.01
    assert result["strategy_id"].iloc[0] == "r_test"


def test_predict_empty_signals(orb):
    """Empty filtered_signals => empty DataFrame."""
    regime = _make_regime("r_test", "long", 0.005, 0.01)
    result = orb.predict(pd.DataFrame(), regime)
    assert result.empty


def test_data_not_fresh_returns_zeros(orb):
    orb._data_fresh = False
    decisions = orb.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0
