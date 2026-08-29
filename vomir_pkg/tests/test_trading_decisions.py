"""Tests for VomirTrading decision edge cases: conflicts, trading modes,
thresholds, missing data, multi-symbol, NaN handling, and size precision."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from vomir.trading import VomirTrading
from vomir.classifier import CLASS_LONG, CLASS_SHORT, CLASS_NEUTRAL


def _mock_vomir_artifacts(feature_cols=None):
    if feature_cols is None:
        feature_cols = ["feat_a", "feat_b"]
    imputer = MagicMock()
    imputer.transform.side_effect = lambda X: X
    scaler = MagicMock()
    scaler.transform.side_effect = lambda X: X
    pca = MagicMock()
    pca.transform.side_effect = lambda X: X[:, :2] if X.shape[1] > 2 else X
    kmeans = MagicMock()
    kmeans.predict.side_effect = lambda X: np.zeros(X.shape[0], dtype=int)
    clf = MagicMock()
    clf.classes_ = np.array([CLASS_SHORT, CLASS_NEUTRAL, CLASS_LONG])
    clf.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])
    return {
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "clf": clf,
        "meta": {
            "feature_cols": feature_cols,
            "n_clusters": 4,
            "pca_components": 2,
        },
    }


def _fake_research_init(self, config, home_root):
    self.config = config
    self.home_root = home_root.rstrip("/")
    self.raw = None
    self.features = None
    self.filtered_signals = None
    self.filtered_signals_long = None
    self.filtered_signals_short = None


def _make_vomir(config_overrides=None, symbols=None, feature_cols=None):
    """Factory: build a VomirTrading with controlled config and mocked deps."""
    if symbols is None:
        symbols = ["BINANCE_PERP_BTC_USDT"]
    if feature_cols is None:
        feature_cols = ["feat_a", "feat_b"]

    config = {
        "TIME_UNIT": "1d",
        "TIMEFRAME_SECONDS": 86400,
        "SIZES": [0.01] * len(symbols),
        "SYMBOLS": symbols,
        "CAPITAL": 1000,
        "TRADING_MODE": "both",
        "VOMIR_WEIGHTS_PATH": "/tmp/fake_vomir_weights",
        "VOMIR_LONG_THRESHOLD": 0.4,
        "VOMIR_SHORT_THRESHOLD": 0.4,
        "LOT_SIZES": {
            s: {"step_size": 0.001, "min_notional": 5.0} for s in symbols
        },
    }
    if config_overrides:
        config.update(config_overrides)

    arts = _mock_vomir_artifacts(feature_cols=feature_cols)
    with patch("vomir.trading.AgamottoResearch.__init__",
               _fake_research_init), \
         patch("vomir.trading.VomirClassifier.load_model_artifacts",
               return_value=arts), \
         patch.object(VomirTrading, "_calculate_sizes"), \
         patch.object(VomirTrading, "load_data"):
        inst = VomirTrading(
            config=config, home_root="/tmp", skip_load=True)

    # Build per-symbol raw + vertical_features
    settled_ts = pd.Timestamp("2025-01-01 00:00:00")
    raw_data = {}
    vf_rows = []
    for sym in symbols:
        native = sym.replace("BINANCE_PERP_", "").replace("_", "")
        raw_data[f"{native}_close"] = [49000.0, 50000.0]
        row = {"timestamp": settled_ts, "symbol": sym}
        for fc in feature_cols:
            row[fc] = 1.0
        vf_rows.append(row)

    # `raw` must END at the row the classifier scores — see the same note in
    # test_trading.py. The old index stopped a day short of `settled_ts`, which
    # `_process_combined` cannot produce.
    inst.raw = pd.DataFrame(
        raw_data,
        index=pd.DatetimeIndex(
            [settled_ts - pd.Timedelta(days=1), settled_ts]),
    )
    inst.vertical_features = pd.DataFrame(vf_rows)
    inst._data_fresh = True
    return inst


# -------------------------------------------------------------------
# Conflict: both P(LONG) and P(SHORT) exceed threshold
# -------------------------------------------------------------------


def test_conflict_both_exceed_long_wins():
    """P(LONG)=0.7 > P(SHORT)=0.6, both > 0.4 => LONG wins (line 192)."""
    inst = _make_vomir()
    inst._vomir_clf.predict_proba.return_value = np.array([[0.6, 0.0, 0.7]])
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0, f"Expected LONG (positive qty), got {qty}"


def test_conflict_both_exceed_short_stronger_neutral():
    """P(LONG)=0.5, P(SHORT)=0.6, both > 0.4 but p_short > p_long.
    Code line 190-192: if-branch entered (p_long > threshold) but inner
    guard (p_short <= threshold OR p_long >= p_short) is False, so LONG
    doesn't fire. Since outer if was True, elif is skipped => qty=0."""
    inst = _make_vomir()
    inst._vomir_clf.predict_proba.return_value = np.array([[0.6, 0.0, 0.5]])
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0, (
        f"Both exceed threshold but short stronger: "
        f"expected neutral (0.0), got {qty}"
    )


# -------------------------------------------------------------------
# Direction comes from the CLASSIFIER, never from TRADING_MODE
# -------------------------------------------------------------------
#
# These two tests used to pin the opposite contract
# (test_long_only_mode_blocks_short / test_short_only_mode_blocks_long): vomir
# gated each leg on `self.trading_mode in ("both", "long_only")`. That gate is
# gone -- direction is the classifier's answer, exactly as it is for every
# other algo, where it comes from the regime stack's `position` column.
#
# WHY IT HAD TO GO. `TRADING_MODE` is a COLLIDING key: knull's BaseExecutor
# reads it as an EXECUTION STYLE and hard-validates it against
# ("MarketMaker", "LadderMaker", "Anchor", "Taker", "Dimer", "EntryTaker"),
# while vomir read the same key as a DIRECTION mode with a silent "both"
# default and matched neither vocabulary against the other. Measured on the
# full (p_short, p_long) decision surface -- 1323 single-symbol cells, 21x21
# grid at three threshold pairs -- before the removal:
#
#     TRADING_MODE          LONG cells   SHORT cells   FLAT cells
#     "both"                       612           248          463
#     key absent (default)         612           248          463   (identical)
#     "long_only"                  612             0          711
#     "short_only"                   0           588          735
#     "MarketMaker"                  0             0         1323   <-- silent
#
# An execution style in that key made vomir trade NOTHING, with no error, on
# every one of 1323 cells. The vocabulary that produced that is deleted rather
# than validated: a key vomir does not read cannot be fed the wrong word.


@pytest.mark.parametrize("mode", [
    "both", "long_only", "short_only", "long", "short",
    "MarketMaker", "LadderMaker", "nonsense",
])
def test_short_fires_whatever_trading_mode_says(mode):
    """P(SHORT)=0.9 => SHORT, for EVERY value of the colliding key."""
    inst = _make_vomir(config_overrides={"TRADING_MODE": mode})
    inst._vomir_clf.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty < 0, (
        f"TRADING_MODE={mode!r} must not gate direction; the classifier said "
        f"SHORT (P=0.9) and got qty={qty}")


@pytest.mark.parametrize("mode", [
    "both", "long_only", "short_only", "long", "short",
    "MarketMaker", "LadderMaker", "nonsense",
])
def test_long_fires_whatever_trading_mode_says(mode):
    """P(LONG)=0.9 => LONG, for EVERY value of the colliding key."""
    inst = _make_vomir(config_overrides={"TRADING_MODE": mode})
    inst._vomir_clf.predict_proba.return_value = np.array([[0.05, 0.05, 0.9]])
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0, (
        f"TRADING_MODE={mode!r} must not gate direction; the classifier said "
        f"LONG (P=0.9) and got qty={qty}")


def test_both_legs_fire_with_no_trading_mode_key_at_all():
    """The key is not read, so its ABSENCE is not a missing-config failure."""
    cfg = {"TIME_UNIT": "1d", "TIMEFRAME_SECONDS": 86400, "SIZES": [0.01],
           "SYMBOLS": ["BINANCE_PERP_BTC_USDT"], "CAPITAL": 1000,
           "VOMIR_WEIGHTS_PATH": "/tmp/fake_vomir_weights",
           "VOMIR_LONG_THRESHOLD": 0.4, "VOMIR_SHORT_THRESHOLD": 0.4,
           "LOT_SIZES": {"BINANCE_PERP_BTC_USDT": {"step_size": 0.001,
                                                   "min_notional": 5.0}}}
    inst = _make_vomir()
    inst.config = dict(cfg)          # no TRADING_MODE anywhere
    inst._vomir_clf.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])
    assert inst.make_decision()["BINANCE_PERP_BTC_USDT"][1] < 0
    inst._vomir_clf.predict_proba.return_value = np.array([[0.05, 0.05, 0.9]])
    assert inst.make_decision()["BINANCE_PERP_BTC_USDT"][1] > 0


def test_vomir_does_not_read_trading_mode():
    """No `trading_mode` attribute: the read itself is gone, not just its use.

    An attribute that still exists is an attribute a future edit can gate on
    again. Pins the absence, so reinstating the read fails here too.
    """
    inst = _make_vomir(config_overrides={"TRADING_MODE": "long_only"})
    assert not hasattr(inst, "trading_mode"), (
        "vomir must not carry a direction mode read off the colliding "
        "TRADING_MODE key")


# -------------------------------------------------------------------
# Threshold boundary
# -------------------------------------------------------------------


def test_threshold_boundary_not_triggered():
    """P = threshold exactly => NOT triggered (strict > comparison)."""
    inst = _make_vomir(config_overrides={
        "VOMIR_LONG_THRESHOLD": 0.5,
        "VOMIR_SHORT_THRESHOLD": 0.5,
    })
    inst._vomir_clf.predict_proba.return_value = np.array([[0.5, 0.0, 0.5]])
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0, f"P == threshold should NOT trigger, got qty={qty}"


# -------------------------------------------------------------------
# Early-return paths
# -------------------------------------------------------------------


def test_missing_feature_cols_returns_zeros():
    """Missing feature columns => early return zeros."""
    inst = _make_vomir(feature_cols=["feat_a", "feat_b", "feat_missing"])
    # vertical_features only has feat_a and feat_b, not feat_missing
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0


def test_data_not_fresh_returns_zeros():
    inst = _make_vomir()
    inst._data_fresh = False
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0


def test_empty_vertical_features_returns_zeros():
    inst = _make_vomir()
    inst.vertical_features = pd.DataFrame()
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0


# -------------------------------------------------------------------
# Multi-symbol
# -------------------------------------------------------------------


def test_multi_symbol_independent_decisions():
    """Two symbols get independent decisions based on their probas."""
    syms = ["BINANCE_PERP_BTC_USDT", "BINANCE_PERP_ETH_USDT"]
    inst = _make_vomir(symbols=syms)
    # Row 0 (BTC) => LONG, Row 1 (ETH) => SHORT
    inst._vomir_clf.predict_proba.return_value = np.array([
        [0.05, 0.05, 0.9],   # BTC => LONG
        [0.9, 0.05, 0.05],   # ETH => SHORT
    ])
    decisions = inst.make_decision()
    _, qty_btc = decisions["BINANCE_PERP_BTC_USDT"]
    _, qty_eth = decisions["BINANCE_PERP_ETH_USDT"]
    assert qty_btc > 0, f"BTC should be LONG, got {qty_btc}"
    assert qty_eth < 0, f"ETH should be SHORT, got {qty_eth}"


# -------------------------------------------------------------------
# NaN handling
# -------------------------------------------------------------------


def test_nan_features_replaced_with_zero():
    """NaN features => np.nan_to_num fills with 0, prediction still works."""
    inst = _make_vomir()
    inst.vertical_features.loc[0, "feat_a"] = np.nan
    inst._vomir_clf.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])
    decisions = inst.make_decision()
    _, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0, f"NaN should be filled; expected LONG, got {qty}"


# -------------------------------------------------------------------
# Size precision
# -------------------------------------------------------------------


def test_compute_size_step_precision():
    """_compute_size rounds qty to correct step_size precision."""
    inst = _make_vomir()
    lot_sizes = {"SYM": {"step_size": 0.01, "min_notional": 0.0}}
    size = inst._compute_size("SYM", 50000.0, 1000.0, lot_sizes)
    # 1000/50000 = 0.02, step=0.01, round(0.02/0.01)*0.01 = 0.02
    assert size == 0.02
    # Check sub-step precision: price makes qty not a clean multiple
    size2 = inst._compute_size("SYM", 33333.0, 1000.0, lot_sizes)
    assert size2 == round(round(1000 / 33333.0 / 0.01) * 0.01, 2)
