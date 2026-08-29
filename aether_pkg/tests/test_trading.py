"""Tests for AetherTrading initialization, make_decision, and clean."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from aether.trading import AetherTrading


def _mock_model(prediction_value):
    model = MagicMock()
    model.predict.return_value = np.array([prediction_value])
    scaler = MagicMock()
    scaler.transform.side_effect = lambda X: X
    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": ["feat_a", "feat_b"],
    }


def _make_aether(config_overrides=None, symbols=None):
    if symbols is None:
        symbols = ["BINANCE_PERP_BTC_USDT"]

    config = {
        "TIME_UNIT": "15m",
        "TIMEFRAME_SECONDS": 900,
        "TIMEFRAMES": ["15m"],
        "BASE_TF": "15m",
        "TARGET_TF": "1h",
        "SIZES": [0.01] * len(symbols),
        "SYMBOLS": symbols,
        "CAPITAL": 1000,
        "TRADING_MODE": "both",
        "REGIME_STACK_PATH": "/tmp/fake_regime_stack.csv",
        "WEIGHTS_PATH": "/tmp/fake_weights",
        "LOT_SIZES": {
            s: {"step_size": 0.001, "min_notional": 5.0} for s in symbols
        },
    }
    if config_overrides:
        config.update(config_overrides)

    with patch.object(AetherTrading, "_load_regime_stack"), \
         patch.object(AetherTrading, "_load_models"), \
         patch("orb.trading.OrbTrading._calculate_sizes"), \
         patch.object(AetherTrading, "load_data"), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.isabs", return_value=True):
        inst = AetherTrading(
            config=config, home_root="/tmp",
            period="window_test", skip_load=True)

    inst.engineer_features = MagicMock()
    inst.verticalize = MagicMock()

    # Default empty regime stack and models — tests override
    inst.regime_stack = []
    inst.models = {}

    settled_ts = pd.Timestamp("2025-01-01 00:00:00")
    vf_rows = []
    raw_data = {}
    for sym in symbols:
        native = sym.replace("BINANCE_PERP_", "").replace("_", "")
        raw_data[f"{native}_close"] = [49000.0, 50000.0]
        vf_rows.append({
            "timestamp": settled_ts,
            "symbol": sym,
            "feat_a": 1.0,
            "feat_b": 2.0,
        })

    inst.vertical_features = pd.DataFrame(vf_rows)
    inst.features = inst.vertical_features.copy()
    # `raw` must END at the row the forward pass targets — aether's `load_data`
    # delegates to `OrbTrading.load_data`, and `_align_timeframes` builds
    # `features` on the BASE_TF raw index, so `vertical_features["timestamp"]`
    # can never run past `raw.index.max()`. The old index ("2024-12-30",
    # "2024-12-31") ran a day SHORT of `settled_ts`, a state the real
    # `_fetch_and_prepare_data` cannot produce; it went unnoticed only because
    # the positional `iloc[-2]` never looked at the index at all.
    inst.raw = pd.DataFrame(
        raw_data,
        index=pd.DatetimeIndex(
            [settled_ts - pd.Timedelta(minutes=15), settled_ts]),
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


def test_initialization():
    inst = _make_aether()
    assert inst.config["CAPITAL"] == 1000
    # No TRADING_MODE read. It was a banned magic-default that nothing ever
    # consumed -- not even a log line. Direction comes from the regime stack's
    # `position`; TRADING_MODE is the executor's execution style.
    assert not hasattr(inst, "trading_mode")
    assert isinstance(inst.regime_stack, list)
    assert isinstance(inst.models, dict)


# -------------------------------------------------------------------
# make_decision
# -------------------------------------------------------------------


def test_make_decision_long():
    """Long model pred > threshold, one long regime => positive qty."""
    inst = _make_aether()
    inst.models = {"long": _mock_model(0.05), "short": _mock_model(0.0)}
    inst.regime_stack = [
        {"regime": "r1", "position": "long", "threshold": 0.01},
    ]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0, f"Expected positive qty for long, got {qty}"


def test_make_decision_short():
    """Short model pred < threshold, one short regime => negative qty."""
    inst = _make_aether()
    inst.models = {"long": _mock_model(0.0), "short": _mock_model(-0.05)}
    inst.regime_stack = [
        {"regime": "r1", "position": "short", "threshold": -0.01},
    ]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty < 0, f"Expected negative qty for short, got {qty}"


def test_clean_returns_zeros():
    inst = _make_aether()
    decisions = inst.clean()
    assert decisions == {"BINANCE_PERP_BTC_USDT": [0.0, 0.0]}


def test_data_not_fresh_returns_zeros():
    inst = _make_aether()
    inst._data_fresh = False
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0


def test_empty_vertical_features_returns_zeros():
    inst = _make_aether()
    inst.vertical_features = pd.DataFrame()
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0
