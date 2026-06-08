"""Tests for OrbTrading decision logic: regime voting, retain threshold,
NaN/inf handling, multi-symbol, and edge cases."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from orb.trading import OrbTrading


def _make_regime(regime_id, position, threshold, prediction_value):
    model = MagicMock()
    model.predict.return_value = [prediction_value]
    scaler = MagicMock()
    scaler.transform.return_value = [[0.1, 0.2]]
    return {
        "id": regime_id,
        "regime": f"regime_{regime_id}",
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


def _make_orb(config_overrides=None, symbols=None):
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
        "LONG_PRED_THRESHOLD": 0.0,
        "SHORT_PRED_THRESHOLD": 0.0,
        "REGIME_STACK_PATH": "/tmp/fake_regime_stack.csv",
        "LOT_SIZES": {
            s: {"step_size": 0.001, "min_notional": 5.0} for s in symbols
        },
    }
    if config_overrides:
        config.update(config_overrides)

    def fake_load_regime_stack(self_inner):
        self_inner.regime_stack = []
        self_inner.models = {"long": {}, "short": {}}

    with patch.object(OrbTrading, "_load_regime_stack",
                      fake_load_regime_stack), \
         patch.object(OrbTrading, "_calculate_sizes"), \
         patch.object(OrbTrading, "load_data"):
        inst = OrbTrading(
            config=config, home_root="/tmp",
            period="window_test", skip_load=True)

    inst.engineer_features = MagicMock()
    inst.verticalize = MagicMock()

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
            "position": "long",
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
# Signal direction tests
# -------------------------------------------------------------------


def test_long_signal_above_threshold():
    inst = _make_orb()
    r = _make_regime("r1", "long", 0.005, 0.01)
    inst.regime_stack = [r]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0


def test_short_signal_below_threshold():
    inst = _make_orb()
    r = _make_regime("r1", "short", -0.005, -0.01)
    inst.regime_stack = [r]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty < 0


def test_no_signal_between_thresholds():
    inst = _make_orb()
    r_long = _make_regime("r1", "long", 0.05, 0.01)   # pred < thresh
    r_short = _make_regime("r2", "short", -0.05, -0.01)  # pred > thresh
    inst.regime_stack = [r_long, r_short]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0


# -------------------------------------------------------------------
# Multi-regime net count
# -------------------------------------------------------------------


def test_multiple_regimes_net_positive():
    """2 long + 1 short => net = 1 => positive qty."""
    inst = _make_orb()
    r1 = _make_regime("l1", "long", 0.005, 0.01)
    r2 = _make_regime("l2", "long", 0.005, 0.02)
    r3 = _make_regime("s1", "short", -0.005, -0.01)
    inst.regime_stack = [r1, r2, r3]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0


def test_multiple_regimes_net_negative():
    """1 long + 2 short => net = -1 => negative qty."""
    inst = _make_orb()
    r1 = _make_regime("l1", "long", 0.005, 0.01)
    r2 = _make_regime("s1", "short", -0.005, -0.01)
    r3 = _make_regime("s2", "short", -0.005, -0.02)
    inst.regime_stack = [r1, r2, r3]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty < 0


def test_multiple_regimes_balanced():
    """1 long + 1 short => net = 0 => qty = 0."""
    inst = _make_orb()
    r1 = _make_regime("l1", "long", 0.005, 0.01)
    r2 = _make_regime("s1", "short", -0.005, -0.01)
    inst.regime_stack = [r1, r2]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0


# -------------------------------------------------------------------
# Empty regime stack
# -------------------------------------------------------------------


def test_empty_regime_stack_returns_zeros():
    inst = _make_orb()
    inst.regime_stack = []
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0


# -------------------------------------------------------------------
# Retain threshold
# -------------------------------------------------------------------


def _make_open_trade(position_qty):
    """Create a mock trade object with status=OPEN."""
    trade = MagicMock()
    trade.status.value = "OPEN"
    trade.position_qty = position_qty
    return trade


def test_retain_threshold_closes_long():
    """Open long position, net_score < retain => close (qty=0)."""
    inst = _make_orb(config_overrides={"RETAIN_THRESHOLD": 0.5})
    # Two short regimes trigger => net_score is strongly negative
    r1 = _make_regime("s1", "short", -0.005, -0.01)
    r2 = _make_regime("s2", "short", -0.005, -0.02)
    inst.regime_stack = [r1, r2]
    inst.filter_signals = _regime_aware_filter(inst)
    inst.venom = {"BINANCE_PERP_BTC_USDT": _make_open_trade(0.01)}
    decisions = inst.make_decision()
    price, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert price == 0.0 and qty == 0.0


def test_retain_threshold_closes_short():
    """Open short, net_score > -retain => close."""
    inst = _make_orb(config_overrides={"RETAIN_THRESHOLD": 0.5})
    # Two long regimes trigger => net_score is positive
    r1 = _make_regime("l1", "long", 0.005, 0.01)
    r2 = _make_regime("l2", "long", 0.005, 0.02)
    inst.regime_stack = [r1, r2]
    inst.filter_signals = _regime_aware_filter(inst)
    inst.venom = {"BINANCE_PERP_BTC_USDT": _make_open_trade(-0.01)}
    decisions = inst.make_decision()
    price, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert price == 0.0 and qty == 0.0


def test_retain_threshold_keeps_position():
    """net_score exceeds retain for long => position kept."""
    inst = _make_orb(config_overrides={"RETAIN_THRESHOLD": 0.001})
    # Strong long signal => net_score well above retain
    r1 = _make_regime("l1", "long", 0.005, 0.5)
    r2 = _make_regime("l2", "long", 0.005, 0.6)
    inst.regime_stack = [r1, r2]
    inst.filter_signals = _regime_aware_filter(inst)
    inst.venom = {"BINANCE_PERP_BTC_USDT": _make_open_trade(0.01)}
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0, f"Position should be kept, got qty={qty}"


def test_retain_threshold_not_set():
    """No RETAIN_THRESHOLD config => retain logic skipped entirely."""
    inst = _make_orb()
    assert "RETAIN_THRESHOLD" not in inst.config
    r = _make_regime("l1", "long", 0.005, 0.01)
    inst.regime_stack = [r]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0


# -------------------------------------------------------------------
# NaN / inf handling
# -------------------------------------------------------------------


def test_nan_features_filled():
    """NaN features => fillna(0.0) inside predict."""
    inst = _make_orb()
    inst.vertical_features.loc[0, "feat_a"] = np.nan
    r = _make_regime("r1", "long", 0.005, 0.01)
    inst.regime_stack = [r]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0, "NaN should be filled, prediction should proceed"


def test_inf_features_cleaned():
    """inf/-inf features don't crash predict (handled by fillna chain)."""
    inst = _make_orb()
    inst.vertical_features.loc[0, "feat_b"] = np.inf
    r = _make_regime("r1", "long", 0.005, 0.01)
    inst.regime_stack = [r]
    inst.filter_signals = _regime_aware_filter(inst)
    # Should not raise
    decisions = inst.make_decision()
    assert "BINANCE_PERP_BTC_USDT" in decisions


# -------------------------------------------------------------------
# Multi-symbol
# -------------------------------------------------------------------


def test_multi_symbol_decisions():
    """Independent per-symbol outcomes."""
    syms = ["BINANCE_PERP_BTC_USDT", "BINANCE_PERP_ETH_USDT"]
    inst = _make_orb(symbols=syms)

    # BTC: long regime triggers. ETH: short regime triggers.
    model_long = MagicMock()
    model_long.predict.return_value = [0.01]
    model_short = MagicMock()
    model_short.predict.return_value = [-0.01]
    scaler = MagicMock()
    scaler.transform.side_effect = lambda X: X

    r_long = {
        "id": "l1", "regime": "r_l1", "position": "long",
        "model_name": "mock", "threshold": 0.005,
        "artifact": {"model": model_long, "scaler": scaler,
                     "metadata": {"feature_columns": ["feat_a", "feat_b"]}},
        "filter": None, "config": {},
    }
    r_short = {
        "id": "s1", "regime": "r_s1", "position": "short",
        "model_name": "mock", "threshold": -0.005,
        "artifact": {"model": model_short, "scaler": scaler,
                     "metadata": {"feature_columns": ["feat_a", "feat_b"]}},
        "filter": None, "config": {},
    }
    inst.regime_stack = [r_long, r_short]
    inst.filter_signals = _regime_aware_filter(inst)
    decisions = inst.make_decision()

    # Both symbols see both regimes: 1 long - 1 short = 0
    _, qty_btc = decisions["BINANCE_PERP_BTC_USDT"]
    _, qty_eth = decisions["BINANCE_PERP_ETH_USDT"]
    assert qty_btc == 0.0
    assert qty_eth == 0.0
