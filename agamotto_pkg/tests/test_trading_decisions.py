"""
Tests for AgamottoTrading.make_decision threshold logic and NaN feature warning.
"""

import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agamotto.trading import AgamottoTrading


def _make_regime(regime_id, position, threshold, prediction_value):
    """Create a mock regime with a model that returns a fixed prediction."""
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
def trading_instance():
    """Create a minimal AgamottoTrading with mocked init dependencies."""
    config = {
        "TIME_UNIT": "1d",
        "TIMEFRAME_SECONDS": 86400,
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

    def fake_load_regime_stack(self_inner):
        self_inner.regime_stack = []
        self_inner.models = {"long": {}, "short": {}}

    with patch.object(AgamottoTrading, "_load_regime_stack", fake_load_regime_stack), \
         patch("agamotto.trading.AgamottoTrading._calculate_sizes"), \
         patch("agamotto.trading.AgamottoTrading.load_data"), \
         patch("agamotto.trading.fetch_futures_klines"), \
         patch("agamotto.trading.klines_to_dataframe"):

        inst = AgamottoTrading(config=config, home_root="/tmp", period="window_test")

    inst.engineer_features = MagicMock()

    # Build a vertical_features DataFrame with settled timestamp
    settled_ts = pd.Timestamp("2025-01-01 00:00:00")
    vf = pd.DataFrame({
        "timestamp": [settled_ts],
        "symbol": ["BINANCE_PERP_BTC_USDT"],
        "feat_a": [1.0],
        "feat_b": [2.0],
        "position": ["long"],
    })
    inst.vertical_features = vf
    inst.verticalize = MagicMock()

    # Raw data for close price lookup
    inst.raw = pd.DataFrame(
        {"BTCUSDT_close": [49000.0, 50000.0]},
        index=pd.to_datetime(["2024-12-30", "2024-12-31"]),
    )

    inst._data_fresh = True
    return inst


# ---------------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------------

def _regime_aware_filter(inst):
    """Return a filter_signals mock that sets position from the regime."""
    def mock_filter(regime, save=False):
        vf = inst.vertical_features.copy()
        vf["position"] = regime["position"]
        return vf
    return mock_filter


def test_long_signal_above_threshold(trading_instance):
    """Prediction > buy threshold => positive qty (long signal)."""
    regime = _make_regime("r_long", "long", 0.005, 0.01)
    trading_instance.regime_stack = [regime]
    trading_instance.filter_signals = _regime_aware_filter(trading_instance)

    decisions = trading_instance.make_decision()
    price, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0, f"Expected positive qty for long signal, got {qty}"


def test_short_signal_below_threshold(trading_instance):
    """Prediction < sell threshold => negative qty (short signal)."""
    regime = _make_regime("r_short", "short", -0.005, -0.01)
    trading_instance.regime_stack = [regime]
    trading_instance.filter_signals = _regime_aware_filter(trading_instance)

    decisions = trading_instance.make_decision()
    price, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty < 0, f"Expected negative qty for short signal, got {qty}"


def test_no_signal_between_thresholds(trading_instance):
    """Prediction between thresholds => qty = 0 (no signal)."""
    # Long regime: threshold 0.05, prediction 0.01 (below threshold, no signal)
    regime_long = _make_regime("r_long", "long", 0.05, 0.01)
    # Short regime: threshold -0.05, prediction -0.01 (above threshold, no signal)
    regime_short = _make_regime("r_short", "short", -0.05, -0.01)
    trading_instance.regime_stack = [regime_long, regime_short]

    def mock_filter(regime, save=False):
        vf = trading_instance.vertical_features.copy()
        vf["position"] = regime["position"]
        return vf

    trading_instance.filter_signals = mock_filter

    decisions = trading_instance.make_decision()
    price, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0, f"Expected zero qty between thresholds, got {qty}"


def test_multiple_regimes_net_count(trading_instance):
    """Net count across regimes determines final signal direction and size."""
    # 2 long regimes triggering, 1 short regime triggering
    # net_count = 2 - 1 = 1 => positive qty, 1x base_size
    r_long1 = _make_regime("r_long1", "long", 0.005, 0.01)
    r_long2 = _make_regime("r_long2", "long", 0.005, 0.02)
    r_short = _make_regime("r_short", "short", -0.005, -0.01)

    trading_instance.regime_stack = [r_long1, r_long2, r_short]
    trading_instance.filter_signals = _regime_aware_filter(trading_instance)

    decisions = trading_instance.make_decision()
    price, qty = decisions["BINANCE_PERP_BTC_USDT"]
    # net_count = 2 long - 1 short = 1 => positive
    assert qty > 0, f"Expected positive net qty, got {qty}"

    # Now test net_count = 0 (balanced)
    r_long_only = _make_regime("r_long_only", "long", 0.005, 0.01)
    r_short_only = _make_regime("r_short_only", "short", -0.005, -0.01)
    trading_instance.regime_stack = [r_long_only, r_short_only]
    trading_instance.filter_signals = _regime_aware_filter(trading_instance)

    decisions = trading_instance.make_decision()
    price, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0, f"Expected zero qty for balanced signals, got {qty}"


# ---------------------------------------------------------------------------
# NaN feature warning
# ---------------------------------------------------------------------------

def test_nan_features_warning_logged(trading_instance, caplog):
    """When features contain NaN, a warning should be logged before fillna."""
    regime = _make_regime("r_nan", "long", 0.005, 0.01)
    trading_instance.regime_stack = [regime]

    # Add NaN to one feature column
    vf = trading_instance.vertical_features.copy()
    vf.loc[vf.index[0], "feat_a"] = np.nan
    trading_instance.vertical_features = vf
    trading_instance.filter_signals = _regime_aware_filter(trading_instance)

    with caplog.at_level(logging.WARNING, logger="agamotto.trading"):
        trading_instance.make_decision()

    # Check that the NaN warning was emitted
    nan_warnings = [r for r in caplog.records if "NaN" in r.message and "feat_a" in r.message]
    assert len(nan_warnings) > 0, (
        f"Expected NaN warning mentioning feat_a, got: {[r.message for r in caplog.records]}"
    )


def test_nan_features_still_produces_prediction(trading_instance):
    """NaN features are filled with 0.0 and prediction still proceeds."""
    regime = _make_regime("r_nan2", "long", 0.005, 0.01)
    trading_instance.regime_stack = [regime]

    vf = trading_instance.vertical_features.copy()
    vf.loc[vf.index[0], "feat_b"] = np.nan
    trading_instance.vertical_features = vf
    trading_instance.filter_signals = _regime_aware_filter(trading_instance)

    decisions = trading_instance.make_decision()
    price, qty = decisions["BINANCE_PERP_BTC_USDT"]
    # Should still produce a long signal despite NaN
    assert qty > 0, f"Expected positive qty after NaN fill, got {qty}"
