"""Tests for MjolnirTrading decision logic: MIN_SIGNAL_COUNT gate,
multi-regime voting, threshold selection, filter masks, and NaN/inf."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from mjolnir.trading import MjolnirTrading, PredictDiag
from mjolnir.core.research import MjolnirResearch


def _base_config(**overrides):
    cfg = {
        "TIME_UNIT": "5s",
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "TARGET_HORIZON_BARS": 1,
        "FEE": 2.0,
        "OUTPUT_DIR": "/tmp/mjolnir_test",
        "MIN_SIGNAL_COUNT": 1,
        "REVERSE": 1,
        "REGIME_STACK_PATH": "/tmp/regime_stack.csv",
    }
    cfg.update(overrides)
    return cfg


def _make_mj(config_overrides=None):
    config = _base_config(**(config_overrides or {}))
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


def _fill_buffer(inst, symbol, n_bars):
    base_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    for i in range(n_bars):
        bar = {
            "open": 49990.0 + i, "high": 50010.0 + i,
            "low": 49980.0 + i, "close": 50000.0 + i,
            "volume": 1.0, "quote_volume": 50000.0,
            "number_of_trades": 10,
            "taker_buy_base_volume": 0.5,
            "taker_buy_quote_volume": 25000.0,
        }
        inst.add_bar(symbol, bar, base_ts + pd.Timedelta(seconds=5 * i))


def _setup_predict(inst, symbol, regime_entries, model_predictions,
                   feat_cols=None, mask_map=None):
    """Set up regime stack, models, and feature engine for predict tests."""
    if feat_cols is None:
        feat_cols = ["feat_a", "feat_b"]
    _fill_buffer(inst, symbol, 200)

    base_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    idx = pd.DatetimeIndex([
        base_ts + pd.Timedelta(seconds=5 * i) for i in range(200)
    ])
    feats = pd.DataFrame(
        np.random.randn(200, len(feat_cols)),
        columns=feat_cols, index=idx)
    inst._feat_engine = MagicMock()
    inst._feat_engine.compute.return_value = feats
    inst._feat_engine.add_btc_cross_features = MagicMock(
        side_effect=lambda f, _: f)

    inst._regime_stack = regime_entries
    for entry in regime_entries:
        regime_name = entry.get("regime", "")
        position = entry.get("position", "long")
        model_name = entry.get("model", "LightGBM").lower()
        dir_key = f"{regime_name}_{position}"
        model_key = f"{dir_key}_{model_name}"
        if model_key in model_predictions:
            pred_val = model_predictions[model_key]
            model = MagicMock()
            model.predict.return_value = np.array([pred_val])
            scaler = MagicMock()
            scaler.transform.side_effect = lambda X: X
            inst._models[model_key] = model
            inst._scalers[model_key] = scaler
            inst._metas[model_key] = {"feature_columns": feat_cols}

    # Real _apply_filter_mask returns a per-row mask aligned to the input
    # index; mocks must mirror that (predict() reads mask.iloc[-2]).
    if mask_map is None:
        inst._research._apply_filter_mask = MagicMock(
            side_effect=lambda df, *a, **k: pd.Series(True, index=df.index))
    else:
        def _mask_fn(df, regime_name, position):
            key = f"{regime_name}_{position}"
            return pd.Series(mask_map.get(key, True), index=df.index)
        inst._research._apply_filter_mask = MagicMock(
            side_effect=_mask_fn)


# -------------------------------------------------------------------
# MIN_SIGNAL_COUNT gate
# -------------------------------------------------------------------


def test_min_signal_count_gate_1():
    """MIN_SIGNAL_COUNT=1, 1 regime fires => signal."""
    inst = _make_mj()
    sym = "BINANCE_PERP_BTC_USDT"
    entry = {"regime": "r1", "position": "long",
             "model": "LightGBM", "threshold": 0.001}
    _setup_predict(inst, sym, [entry], {"r1_long_lightgbm": 0.01})
    result = inst.predict(sym)
    assert result is not None
    assert result.side == "long"


def test_min_signal_count_gate_2_fails():
    """MIN_SIGNAL_COUNT=2, only 1 fires => None."""
    inst = _make_mj(config_overrides={"MIN_SIGNAL_COUNT": 2})
    sym = "BINANCE_PERP_BTC_USDT"
    entry = {"regime": "r1", "position": "long",
             "model": "LightGBM", "threshold": 0.001}
    _setup_predict(inst, sym, [entry], {"r1_long_lightgbm": 0.01})
    result = inst.predict(sym)
    assert result is None


def test_min_signal_count_gate_2_passes():
    """MIN_SIGNAL_COUNT=2, 2 fire => signal."""
    inst = _make_mj(config_overrides={"MIN_SIGNAL_COUNT": 2})
    sym = "BINANCE_PERP_BTC_USDT"
    entries = [
        {"regime": "r1", "position": "long",
         "model": "LightGBM", "threshold": 0.001},
        {"regime": "r2", "position": "long",
         "model": "LightGBM", "threshold": 0.001},
    ]
    preds = {
        "r1_long_lightgbm": 0.01,
        "r2_long_lightgbm": 0.01,
    }
    _setup_predict(inst, sym, entries, preds)
    result = inst.predict(sym)
    assert result is not None
    assert result.side == "long"


def test_missing_min_signal_count_raises():
    """MIN_SIGNAL_COUNT absent => KeyError (Bug 7)."""
    cfg = _base_config()
    del cfg["MIN_SIGNAL_COUNT"]
    with patch.object(MjolnirTrading, "_load_regime_stack",
                      return_value=[]), \
         patch.object(MjolnirTrading, "_load_models"):
        inst = MjolnirTrading(config=cfg, home_root="/tmp")
    inst._regime_stack = []
    inst._models = {}
    inst._scalers = {}
    inst._metas = {}
    inst._research = MagicMock()

    sym = "BINANCE_PERP_BTC_USDT"
    _fill_buffer(inst, sym, 200)
    # Set up feats to get past early returns
    base_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    idx = pd.DatetimeIndex([
        base_ts + pd.Timedelta(seconds=5 * i) for i in range(200)])
    feats = pd.DataFrame(np.random.randn(200, 2),
                         columns=["fa", "fb"], index=idx)
    inst._feat_engine = MagicMock()
    inst._feat_engine.compute.return_value = feats
    inst._feat_engine.add_btc_cross_features = MagicMock(
        side_effect=lambda f, _: f)

    with pytest.raises(KeyError, match="MIN_SIGNAL_COUNT"):
        inst.predict(sym)


# -------------------------------------------------------------------
# Multi-regime voting
# -------------------------------------------------------------------


def test_long_beats_short():
    """3 long + 1 short => 'long'."""
    inst = _make_mj()
    sym = "BINANCE_PERP_BTC_USDT"
    entries = [
        {"regime": "r1", "position": "long",
         "model": "LightGBM", "threshold": 0.001},
        {"regime": "r2", "position": "long",
         "model": "LightGBM", "threshold": 0.001},
        {"regime": "r3", "position": "long",
         "model": "LightGBM", "threshold": 0.001},
        {"regime": "r4", "position": "short",
         "model": "LightGBM", "threshold": -0.001},
    ]
    preds = {
        "r1_long_lightgbm": 0.01,
        "r2_long_lightgbm": 0.01,
        "r3_long_lightgbm": 0.01,
        "r4_short_lightgbm": -0.01,
    }
    _setup_predict(inst, sym, entries, preds)
    result = inst.predict(sym)
    assert result is not None
    assert result.side == "long"


def test_short_beats_long():
    """1 long + 3 short => 'short'."""
    inst = _make_mj()
    sym = "BINANCE_PERP_BTC_USDT"
    entries = [
        {"regime": "r1", "position": "long",
         "model": "LightGBM", "threshold": 0.001},
        {"regime": "r2", "position": "short",
         "model": "LightGBM", "threshold": -0.001},
        {"regime": "r3", "position": "short",
         "model": "LightGBM", "threshold": -0.001},
        {"regime": "r4", "position": "short",
         "model": "LightGBM", "threshold": -0.001},
    ]
    preds = {
        "r1_long_lightgbm": 0.01,
        "r2_short_lightgbm": -0.01,
        "r3_short_lightgbm": -0.01,
        "r4_short_lightgbm": -0.01,
    }
    _setup_predict(inst, sym, entries, preds)
    result = inst.predict(sym)
    assert result is not None
    assert result.side == "short"


def test_equal_counts_returns_none():
    """2 long + 2 short => None."""
    inst = _make_mj()
    sym = "BINANCE_PERP_BTC_USDT"
    entries = [
        {"regime": "r1", "position": "long",
         "model": "LightGBM", "threshold": 0.001},
        {"regime": "r2", "position": "long",
         "model": "LightGBM", "threshold": 0.001},
        {"regime": "r3", "position": "short",
         "model": "LightGBM", "threshold": -0.001},
        {"regime": "r4", "position": "short",
         "model": "LightGBM", "threshold": -0.001},
    ]
    preds = {
        "r1_long_lightgbm": 0.01,
        "r2_long_lightgbm": 0.01,
        "r3_short_lightgbm": -0.01,
        "r4_short_lightgbm": -0.01,
    }
    _setup_predict(inst, sym, entries, preds)
    result = inst.predict(sym)
    assert result is None


# -------------------------------------------------------------------
# Best threshold selection
# -------------------------------------------------------------------


def test_best_long_thresh_is_minimum():
    """Multiple long regimes: best_long_thresh = min of thresholds."""
    inst = _make_mj(config_overrides={"MIN_SIGNAL_COUNT": 1})
    sym = "BINANCE_PERP_BTC_USDT"
    entries = [
        {"regime": "r1", "position": "long",
         "model": "LightGBM", "threshold": 0.02},
        {"regime": "r2", "position": "long",
         "model": "LightGBM", "threshold": 0.01},
    ]
    preds = {
        "r1_long_lightgbm": 0.05,
        "r2_long_lightgbm": 0.05,
    }
    _setup_predict(inst, sym, entries, preds)
    result = inst.predict(sym)
    assert result is not None
    assert result.side == "long"
    assert result.threshold == 0.01  # min(0.02, 0.01)


def test_best_short_thresh_is_maximum():
    """Multiple short regimes: best_short_thresh = max of thresholds."""
    inst = _make_mj(config_overrides={"MIN_SIGNAL_COUNT": 1})
    sym = "BINANCE_PERP_BTC_USDT"
    entries = [
        {"regime": "r1", "position": "short",
         "model": "LightGBM", "threshold": -0.02},
        {"regime": "r2", "position": "short",
         "model": "LightGBM", "threshold": -0.01},
    ]
    preds = {
        "r1_short_lightgbm": -0.05,
        "r2_short_lightgbm": -0.05,
    }
    _setup_predict(inst, sym, entries, preds)
    result = inst.predict(sym)
    assert result is not None
    assert result.side == "short"
    assert result.threshold == -0.01  # max(-0.02, -0.01)


# -------------------------------------------------------------------
# Filter mask
# -------------------------------------------------------------------


def test_regime_filter_excludes():
    """mask=[False] => regime skipped, no signal."""
    inst = _make_mj()
    sym = "BINANCE_PERP_BTC_USDT"
    entry = {"regime": "r1", "position": "long",
             "model": "LightGBM", "threshold": 0.001}
    _setup_predict(inst, sym, [entry], {"r1_long_lightgbm": 0.01},
                   mask_map={"r1_long": False})
    result = inst.predict(sym)
    assert result is None


# -------------------------------------------------------------------
# Quantile-based regime filter — window vs single-row (2026-06-15 bug)
# -------------------------------------------------------------------


def test_quantile_regime_filter_uses_window_not_single_row():
    """Quantile-based regime filters (wide_spread, tight_spread,
    high/low_liquidation_pressure) must be evaluated over the feature WINDOW,
    not a single row.

    Regression for the 2026-06-15 all-short bug: live ``predict()`` passed
    ``feats.iloc[[-2]]`` (one row) to ``_apply_filter_mask``. ``wide_spread`` is
    ``relative_spread > relative_spread.quantile(0.5)`` — on one row,
    ``quantile(0.5) == that value`` so ``x > x`` is ALWAYS False, permanently
    disabling every quantile regime live (the backtest passes the full panel,
    so it worked there). With the only-long entry being ``wide_spread_long``,
    this pinned ``long_count`` at 0 and forced an all-short book.

    Here the -2 (last complete) bar's relative_spread is ABOVE the window
    median, so a correct window evaluation activates wide_spread_long → 'long'.
    Uses the REAL filter (not a mock) — the bug only shows with real quantiles.
    """
    inst = _make_mj()
    sym = "BINANCE_PERP_BTC_USDT"
    # Real filter logic: _apply_filter_mask/_named_filter use self only for
    # recursion, so a bare instance (no __init__) is sufficient.
    inst._research = MjolnirResearch.__new__(MjolnirResearch)

    _fill_buffer(inst, sym, 200)
    base_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    idx = pd.DatetimeIndex(
        [base_ts + pd.Timedelta(seconds=5 * i) for i in range(200)])
    # relative_spread spans 0..1 across the window; the -2 row sits near the top
    # (above the window median ~0.5) but EQUAL to itself on a single row.
    rs = np.linspace(0.0, 1.0, 200)
    rs[-2] = 0.99
    feats = pd.DataFrame(
        {"feat_a": np.zeros(200), "relative_spread": rs}, index=idx)
    inst._feat_engine = MagicMock()
    inst._feat_engine.compute.return_value = feats
    inst._feat_engine.add_btc_cross_features = MagicMock(
        side_effect=lambda f, _: f)

    entry = {"regime": "wide_spread_long", "position": "long",
             "model": "LightGBM", "threshold": 0.001}
    inst._regime_stack = [entry]
    mkey = "wide_spread_long_long_lightgbm"
    model = MagicMock()
    model.predict.return_value = np.array([0.01])   # bullish > threshold
    scaler = MagicMock()
    scaler.transform.side_effect = lambda X: X
    inst._models[mkey] = model
    inst._scalers[mkey] = scaler
    inst._metas[mkey] = {"feature_columns": ["feat_a", "relative_spread"]}

    result = inst.predict(sym)
    # Single-row (buggy): wide_spread always False → regime skipped → None.
    # Window (fixed): -2 row above window median → active → long fires.
    assert result is not None, (
        "wide_spread regime never activated — quantile evaluated on a single "
        "row is always False (the 2026-06-15 all-short bug)")
    assert result.side == "long"
    assert result.regime == "wide_spread_long"


# -------------------------------------------------------------------
# Model key missing
# -------------------------------------------------------------------


def test_model_key_missing_skipped():
    """model_key not in _models => regime skipped."""
    inst = _make_mj()
    sym = "BINANCE_PERP_BTC_USDT"
    entry = {"regime": "r1", "position": "long",
             "model": "LightGBM", "threshold": 0.001}
    # Don't add model to _models
    _fill_buffer(inst, sym, 200)
    base_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    idx = pd.DatetimeIndex([
        base_ts + pd.Timedelta(seconds=5 * i) for i in range(200)])
    feats = pd.DataFrame(np.random.randn(200, 2),
                         columns=["feat_a", "feat_b"], index=idx)
    inst._feat_engine = MagicMock()
    inst._feat_engine.compute.return_value = feats
    inst._feat_engine.add_btc_cross_features = MagicMock(
        side_effect=lambda f, _: f)
    inst._regime_stack = [entry]
    inst._research._apply_filter_mask = MagicMock(
        side_effect=lambda df, *a, **k: pd.Series(True, index=df.index))
    result = inst.predict(sym)
    assert result is None


# -------------------------------------------------------------------
# NaN/inf cleaning
# -------------------------------------------------------------------


def test_nan_inf_features_cleaned():
    """inf/NaN in features => replaced with 0 before predict."""
    inst = _make_mj()
    sym = "BINANCE_PERP_BTC_USDT"
    _fill_buffer(inst, sym, 200)

    base_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    idx = pd.DatetimeIndex([
        base_ts + pd.Timedelta(seconds=5 * i) for i in range(200)])
    feat_data = np.random.randn(200, 2)
    feat_data[-2, 0] = np.nan
    feat_data[-2, 1] = np.inf
    feats = pd.DataFrame(feat_data, columns=["feat_a", "feat_b"], index=idx)

    inst._feat_engine = MagicMock()
    inst._feat_engine.compute.return_value = feats
    inst._feat_engine.add_btc_cross_features = MagicMock(
        side_effect=lambda f, _: f)

    entry = {"regime": "r1", "position": "long",
             "model": "LightGBM", "threshold": 0.001}
    model = MagicMock()
    model.predict.return_value = np.array([0.01])
    scaler = MagicMock()
    scaler.transform.side_effect = lambda X: X
    inst._regime_stack = [entry]
    inst._models["r1_long_lightgbm"] = model
    inst._scalers["r1_long_lightgbm"] = scaler
    inst._metas["r1_long_lightgbm"] = {"feature_columns": ["feat_a", "feat_b"]}
    inst._research._apply_filter_mask = MagicMock(
        side_effect=lambda df, *a, **k: pd.Series(True, index=df.index))

    result = inst.predict(sym)
    assert result is not None
    assert result.side == "long"
    # Verify nan_to_num was applied: model should have received clean values
    call_args = model.predict.call_args[0][0]
    assert not np.any(np.isnan(call_args))
    assert not np.any(np.isinf(call_args))


# -------------------------------------------------------------------
# Strongest-signal diagnostics
# -------------------------------------------------------------------


def test_strongest_signal_diagnostics():
    """y_pred/regime/model_name report the model with the largest margin
    over threshold."""
    inst = _make_mj(config_overrides={"MIN_SIGNAL_COUNT": 1})
    sym = "BINANCE_PERP_BTC_USDT"
    entries = [
        {"regime": "r1", "position": "long",
         "model": "LightGBM", "threshold": 0.001},
        {"regime": "r2", "position": "long",
         "model": "LightGBM", "threshold": 0.002},
    ]
    # r1: pred=0.01, margin=0.01-0.001=0.009
    # r2: pred=0.05, margin=0.05-0.002=0.048  ← strongest
    preds = {
        "r1_long_lightgbm": 0.01,
        "r2_long_lightgbm": 0.05,
    }
    _setup_predict(inst, sym, entries, preds)
    result = inst.predict(sym)
    assert result is not None
    assert isinstance(result, PredictDiag)
    assert result.side == "long"
    assert result.threshold == 0.001  # min(0.001, 0.002)
    assert result.y_pred == 0.05
    assert result.y_pred_thresh == 0.002
    assert result.regime == "r2"
    assert result.model_name == "r2_long_lightgbm"


def test_strongest_signal_diagnostics_short():
    """Short side: strongest model is the one with largest |pred - threshold|."""
    inst = _make_mj(config_overrides={"MIN_SIGNAL_COUNT": 1})
    sym = "BINANCE_PERP_BTC_USDT"
    entries = [
        {"regime": "r1", "position": "short",
         "model": "LightGBM", "threshold": -0.001},
        {"regime": "r2", "position": "short",
         "model": "LightGBM", "threshold": -0.002},
    ]
    # r1: pred=-0.01, margin=(-0.001)-(-0.01)=0.009
    # r2: pred=-0.05, margin=(-0.002)-(-0.05)=0.048  ← strongest
    preds = {
        "r1_short_lightgbm": -0.01,
        "r2_short_lightgbm": -0.05,
    }
    _setup_predict(inst, sym, entries, preds)
    result = inst.predict(sym)
    assert result is not None
    assert isinstance(result, PredictDiag)
    assert result.side == "short"
    assert result.threshold == -0.001  # max(-0.001, -0.002)
    assert result.y_pred == -0.05
    assert result.y_pred_thresh == -0.002
    assert result.regime == "r2"
    assert result.model_name == "r2_short_lightgbm"
