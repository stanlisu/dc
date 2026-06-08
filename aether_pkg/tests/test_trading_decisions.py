"""Tests for AetherTrading decision logic: regime voting, missing features,
inf/NaN handling, multi-symbol, filter errors, and pred None skips."""

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
    inst.raw = pd.DataFrame(
        raw_data,
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
# Single regime voting
# -------------------------------------------------------------------


def test_single_regime_long_vote():
    inst = _make_aether()
    inst.models = {"long": _mock_model(0.05), "short": _mock_model(0.0)}
    inst.regime_stack = [
        {"regime": "r1", "position": "long", "threshold": 0.01},
    ]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0, f"Long vote should produce positive qty, got {qty}"


def test_single_regime_short_vote():
    inst = _make_aether()
    inst.models = {"long": _mock_model(0.0), "short": _mock_model(-0.05)}
    inst.regime_stack = [
        {"regime": "r1", "position": "short", "threshold": -0.01},
    ]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty < 0, f"Short vote should produce negative qty, got {qty}"


# -------------------------------------------------------------------
# Multi-regime voting
# -------------------------------------------------------------------


def test_multi_regime_voting_net_positive():
    """3 long + 1 short => net = 2 => positive qty."""
    inst = _make_aether()
    inst.models = {"long": _mock_model(0.05), "short": _mock_model(-0.05)}
    inst.regime_stack = [
        {"regime": "r1", "position": "long", "threshold": 0.01},
        {"regime": "r2", "position": "long", "threshold": 0.01},
        {"regime": "r3", "position": "long", "threshold": 0.01},
        {"regime": "r4", "position": "short", "threshold": -0.01},
    ]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0


def test_multi_regime_voting_balanced():
    """2 long + 2 short => net = 0 => qty = 0."""
    inst = _make_aether()
    inst.models = {"long": _mock_model(0.05), "short": _mock_model(-0.05)}
    inst.regime_stack = [
        {"regime": "r1", "position": "long", "threshold": 0.01},
        {"regime": "r2", "position": "long", "threshold": 0.01},
        {"regime": "r3", "position": "short", "threshold": -0.01},
        {"regime": "r4", "position": "short", "threshold": -0.01},
    ]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0


# -------------------------------------------------------------------
# Filter excludes symbol
# -------------------------------------------------------------------


def test_regime_filter_excludes_symbol():
    """filter_signals returns empty => vote skipped, qty = 0."""
    inst = _make_aether()
    inst.models = {"long": _mock_model(0.05)}
    inst.regime_stack = [
        {"regime": "r1", "position": "long", "threshold": 0.01},
    ]
    inst.filter_signals = MagicMock(return_value=pd.DataFrame())
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0


# -------------------------------------------------------------------
# Missing features filled with zero
# -------------------------------------------------------------------


def test_missing_features_filled_with_zero():
    """Model expects feat_c not present => filled with 0.0 (line 184-186)."""
    inst = _make_aether()
    inst.models = {
        "long": {
            "model": MagicMock(predict=MagicMock(return_value=np.array([0.05]))),
            "scaler": MagicMock(transform=MagicMock(side_effect=lambda X: X)),
            "feature_columns": ["feat_a", "feat_b", "feat_c"],
        }
    }
    inst.regime_stack = [
        {"regime": "r1", "position": "long", "threshold": 0.01},
    ]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0, "Missing feature filled with 0, prediction should proceed"


# -------------------------------------------------------------------
# inf values replaced
# -------------------------------------------------------------------


def test_inf_values_replaced():
    """inf/-inf => NaN => 0 via replace+fillna chain (line 188-189)."""
    inst = _make_aether()
    inst.models = {"long": _mock_model(0.05)}
    inst.regime_stack = [
        {"regime": "r1", "position": "long", "threshold": 0.01},
    ]
    inst.vertical_features.loc[0, "feat_a"] = np.inf
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0, "inf should be replaced, prediction should proceed"


# -------------------------------------------------------------------
# Multi-symbol independence
# -------------------------------------------------------------------


def test_multi_symbol_independent_voting():
    syms = ["BINANCE_PERP_BTC_USDT", "BINANCE_PERP_ETH_USDT"]
    inst = _make_aether(symbols=syms)

    # Models must return predictions for ALL rows in `latest` (2 symbols)
    long_model = MagicMock()
    long_model.predict.return_value = np.array([0.05, 0.05])
    long_scaler = MagicMock()
    long_scaler.transform.side_effect = lambda X: X
    short_model = MagicMock()
    short_model.predict.return_value = np.array([-0.05, -0.05])
    short_scaler = MagicMock()
    short_scaler.transform.side_effect = lambda X: X
    inst.models = {
        "long": {"model": long_model, "scaler": long_scaler,
                 "feature_columns": ["feat_a", "feat_b"]},
        "short": {"model": short_model, "scaler": short_scaler,
                  "feature_columns": ["feat_a", "feat_b"]},
    }

    settled_ts = pd.Timestamp("2025-01-01 00:00:00")

    def selective_filter(regime, save=False):
        """Only BTC passes long filter; only ETH passes short filter."""
        if regime["position"] == "long":
            return pd.DataFrame({
                "timestamp": [settled_ts],
                "symbol": ["BINANCE_PERP_BTC_USDT"],
                "feat_a": [1.0], "feat_b": [2.0],
            })
        else:
            return pd.DataFrame({
                "timestamp": [settled_ts],
                "symbol": ["BINANCE_PERP_ETH_USDT"],
                "feat_a": [1.0], "feat_b": [2.0],
            })

    inst.regime_stack = [
        {"regime": "r1", "position": "long", "threshold": 0.01},
        {"regime": "r2", "position": "short", "threshold": -0.01},
    ]
    inst.filter_signals = selective_filter
    decisions = inst.make_decision()
    _, qty_btc = decisions["BINANCE_PERP_BTC_USDT"]
    _, qty_eth = decisions["BINANCE_PERP_ETH_USDT"]
    assert qty_btc > 0, f"BTC should have long vote, got {qty_btc}"
    assert qty_eth < 0, f"ETH should have short vote, got {qty_eth}"


# -------------------------------------------------------------------
# No model for position => skipped
# -------------------------------------------------------------------


def test_no_model_for_position_skipped():
    """Missing position key in models => no predictions for that side."""
    inst = _make_aether()
    # Only long model loaded, no short
    inst.models = {"long": _mock_model(0.05)}
    inst.regime_stack = [
        {"regime": "r1", "position": "short", "threshold": -0.01},
    ]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0


# -------------------------------------------------------------------
# filter_signals exception => regime skipped
# -------------------------------------------------------------------


def test_filter_error_continues():
    """filter_signals raises => regime skipped, not crash."""
    inst = _make_aether()
    inst.models = {"long": _mock_model(0.05)}
    inst.regime_stack = [
        {"regime": "r1", "position": "long", "threshold": 0.01},
    ]
    inst.filter_signals = MagicMock(side_effect=RuntimeError("boom"))
    # Should not raise
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0


# -------------------------------------------------------------------
# pred None => vote not counted
# -------------------------------------------------------------------


def test_pred_none_skipped():
    """If symbol has no prediction for a position => vote not counted."""
    inst = _make_aether()
    # Model predicts value, but filter returns a symbol not in SYMBOLS
    inst.models = {"long": _mock_model(0.05)}
    inst.regime_stack = [
        {"regime": "r1", "position": "long", "threshold": 0.01},
    ]
    # Filter returns rows with unknown symbol
    settled_ts = pd.Timestamp("2025-01-01 00:00:00")
    inst.filter_signals = MagicMock(return_value=pd.DataFrame({
        "timestamp": [settled_ts],
        "symbol": ["UNKNOWN_SYMBOL"],
        "feat_a": [1.0], "feat_b": [2.0],
    }))
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0
