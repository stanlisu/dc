"""Tests for MjolnirTrading: initialization, add_bar, predict basics."""

import pytest
import pandas as pd
import numpy as np
from collections import deque
from unittest.mock import MagicMock, patch

from mjolnir.trading import MjolnirTrading


def _base_config():
    return {
        "TIME_UNIT": "5s",
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "TARGET_HORIZON_BARS": 1,
        "FEE": 2.0,
        "OUTPUT_DIR": "/tmp/mjolnir_test",
        "MIN_SIGNAL_COUNT": 1,
        "REVERSE": 1,
        "REGIME_STACK_PATH": "/tmp/regime_stack.csv",
    }


def _make_bar(close=50000.0, volume=1.0):
    return {
        "open": close - 10,
        "high": close + 10,
        "low": close - 20,
        "close": close,
        "volume": volume,
        "quote_volume": close * volume,
        "number_of_trades": 10,
        "taker_buy_base_volume": volume * 0.5,
        "taker_buy_quote_volume": close * volume * 0.5,
    }


def _fill_buffer(inst, symbol, n_bars, base_close=50000.0):
    """Seed a symbol's buffer with n_bars bars."""
    base_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    for i in range(n_bars):
        bar = _make_bar(close=base_close + i)
        ts = base_ts + pd.Timedelta(seconds=5 * i)
        inst.add_bar(symbol, bar, ts)


@pytest.fixture
def mj():
    """Create MjolnirTrading with mocked regime stack and model loading."""
    config = _base_config()

    with patch.object(MjolnirTrading, "_load_regime_stack",
                      return_value=[]), \
         patch.object(MjolnirTrading, "_load_models"):
        inst = MjolnirTrading(config=config, home_root="/tmp")

    inst._regime_stack = []
    inst._models = {}
    inst._scalers = {}
    inst._metas = {}
    inst._research = MagicMock()
    return inst


# -------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------


def test_initialization(mj):
    assert mj.config["MIN_SIGNAL_COUNT"] == 1
    assert isinstance(mj._buffers, dict)
    assert isinstance(mj._regime_stack, list)
    assert isinstance(mj._models, dict)
    assert mj.BUFFER_MAXLEN == 1000
    assert mj.MIN_WARMUP_BARS == 120


# -------------------------------------------------------------------
# add_bar
# -------------------------------------------------------------------


def test_add_bar_creates_buffer(mj):
    """First bar for a symbol creates a deque."""
    sym = "BINANCE_PERP_BTC_USDT"
    assert sym not in mj._buffers
    bar = _make_bar()
    ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    mj.add_bar(sym, bar, ts)
    assert sym in mj._buffers
    assert len(mj._buffers[sym]) == 1


def test_add_bar_respects_maxlen(mj):
    """BUFFER_MAXLEN + 1 bars => oldest evicted."""
    sym = "BINANCE_PERP_BTC_USDT"
    _fill_buffer(mj, sym, mj.BUFFER_MAXLEN + 1)
    assert len(mj._buffers[sym]) == mj.BUFFER_MAXLEN


# -------------------------------------------------------------------
# predict: edge cases
# -------------------------------------------------------------------


def test_predict_no_buffer_returns_none(mj):
    """Unknown symbol => None."""
    result = mj.predict("UNKNOWN_SYMBOL")
    assert result is None


def test_predict_warmup_insufficient_returns_none(mj):
    """50 bars < MIN_WARMUP_BARS (120) => None."""
    sym = "BINANCE_PERP_BTC_USDT"
    _fill_buffer(mj, sym, 50)
    result = mj.predict(sym)
    assert result is None


def test_predict_feature_computation_error(mj):
    """Feature engine raises => None, not crash."""
    sym = "BINANCE_PERP_BTC_USDT"
    _fill_buffer(mj, sym, 200)
    mj._feat_engine = MagicMock()
    mj._feat_engine.compute.side_effect = RuntimeError("compute failed")
    result = mj.predict(sym)
    assert result is None


def test_predict_needs_two_rows(mj):
    """feats with < 2 rows => None (needs iloc[-2])."""
    sym = "BINANCE_PERP_BTC_USDT"
    _fill_buffer(mj, sym, 200)
    mj._feat_engine = MagicMock()
    # Return only 1-row feats DataFrame
    mj._feat_engine.compute.return_value = pd.DataFrame(
        {"feat_a": [1.0]}, index=pd.DatetimeIndex(
            [pd.Timestamp("2025-01-01", tz="UTC")]))
    result = mj.predict(sym)
    assert result is None


# -------------------------------------------------------------------
# predict: signal generation
# -------------------------------------------------------------------


def _setup_predict(mj, symbol, regime_entries, model_predictions):
    """Set up mocked regime stack and models for predict tests."""
    _fill_buffer(mj, symbol, 200)

    # Build feats with enough rows
    base_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    idx = pd.DatetimeIndex([
        base_ts + pd.Timedelta(seconds=5 * i) for i in range(200)
    ])
    feat_cols = ["feat_a", "feat_b"]
    feats = pd.DataFrame(
        np.random.randn(200, 2), columns=feat_cols, index=idx)
    mj._feat_engine = MagicMock()
    mj._feat_engine.compute.return_value = feats
    mj._feat_engine.add_btc_cross_features = MagicMock(
        side_effect=lambda f, _: f)

    mj._regime_stack = regime_entries
    for entry in regime_entries:
        regime_name = entry.get("regime", "")
        position = entry.get("position", "long")
        model_name = entry.get("model", "LightGBM").lower()
        dir_key = f"{regime_name}_{position}"
        model_key = f"{dir_key}_{model_name}"
        pred_val = model_predictions.get(model_key, 0.0)
        model = MagicMock()
        model.predict.return_value = np.array([pred_val])
        scaler = MagicMock()
        scaler.transform.side_effect = lambda X: X
        mj._models[model_key] = model
        mj._scalers[model_key] = scaler
        mj._metas[model_key] = {"feature_columns": feat_cols}

    mj._research._apply_filter_mask = MagicMock(
        side_effect=lambda df, *a, **k: pd.Series(True, index=df.index))


def test_predict_long_signal(mj):
    """pred > threshold, mask passes => ('long', threshold)."""
    sym = "BINANCE_PERP_BTC_USDT"
    entry = {"regime": "r1", "position": "long",
             "model": "LightGBM", "threshold": 0.001}
    _setup_predict(mj, sym, [entry], {"r1_long_lightgbm": 0.01})
    result = mj.predict(sym)
    assert result is not None
    assert result.side == "long"
    assert result.threshold == 0.001


def test_predict_short_signal(mj):
    """pred < threshold => ('short', threshold)."""
    sym = "BINANCE_PERP_BTC_USDT"
    entry = {"regime": "r1", "position": "short",
             "model": "LightGBM", "threshold": -0.001}
    _setup_predict(mj, sym, [entry], {"r1_short_lightgbm": -0.01})
    result = mj.predict(sym)
    assert result is not None
    assert result.side == "short"
    assert result.threshold == -0.001


def test_predict_no_signal(mj):
    """pred between thresholds => None."""
    sym = "BINANCE_PERP_BTC_USDT"
    entry = {"regime": "r1", "position": "long",
             "model": "LightGBM", "threshold": 0.05}
    _setup_predict(mj, sym, [entry], {"r1_long_lightgbm": 0.001})
    result = mj.predict(sym)
    assert result is None
