"""Tests for VomirTrading initialization, make_decision, and clean."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from vomir.trading import VomirTrading
from vomir.classifier import CLASS_LONG, CLASS_SHORT, CLASS_NEUTRAL


def _mock_vomir_artifacts():
    """Build a dict mimicking VomirClassifier.load_model_artifacts output."""
    imputer = MagicMock()
    imputer.transform.return_value = np.array([[0.1, 0.2]])
    scaler = MagicMock()
    scaler.transform.return_value = np.array([[0.3, 0.4]])
    pca = MagicMock()
    pca.transform.return_value = np.array([[0.5, 0.6]])
    kmeans = MagicMock()
    kmeans.predict.return_value = np.array([0])
    clf = MagicMock()
    clf.classes_ = np.array([CLASS_SHORT, CLASS_NEUTRAL, CLASS_LONG])
    # Default: neutral — override per test
    clf.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])
    return {
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "clf": clf,
        "meta": {
            "feature_cols": ["feat_a", "feat_b"],
            "n_clusters": 4,
            "pca_components": 2,
        },
    }


@pytest.fixture
def base_config():
    return {
        "TIME_UNIT": "1d",
        "TIMEFRAME_SECONDS": 86400,
        "SIZES": [0.01],
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "CAPITAL": 1000,
        "TRADING_MODE": "both",
        "VOMIR_WEIGHTS_PATH": "/tmp/fake_vomir_weights",
        "VOMIR_LONG_THRESHOLD": 0.4,
        "VOMIR_SHORT_THRESHOLD": 0.4,
        "LOT_SIZES": {
            "BINANCE_PERP_BTC_USDT": {"step_size": 0.001, "min_notional": 5.0}
        },
    }


def _fake_research_init(self, config, home_root):
    """Minimal AgamottoResearch.__init__ replacement."""
    self.config = config
    self.home_root = home_root.rstrip("/")
    self.raw = None
    self.features = None
    self.filtered_signals = None
    self.filtered_signals_long = None
    self.filtered_signals_short = None


@pytest.fixture
def vomir(base_config):
    """Create VomirTrading with all external deps mocked."""
    arts = _mock_vomir_artifacts()
    with patch("vomir.trading.AgamottoResearch.__init__",
               _fake_research_init), \
         patch("vomir.trading.VomirClassifier.load_model_artifacts",
               return_value=arts), \
         patch.object(VomirTrading, "_calculate_sizes"), \
         patch.object(VomirTrading, "load_data"):
        inst = VomirTrading(
            config=base_config, home_root="/tmp", skip_load=True)

    settled_ts = pd.Timestamp("2025-01-01 00:00:00")
    # `raw` must END at the row the classifier scores.
    # `AgamottoTrading._process_combined` assigns `self.raw` and then re-runs
    # `engineer_features()` + `verticalize()` in the same breath, and
    # `engineer_features` preserves `raw.index` row-for-row — so
    # `vertical_features["timestamp"].max()` can never run past `raw.index.max()`.
    # The old index ("2024-12-30", "2024-12-31") stopped a day SHORT of
    # `settled_ts`, a state `_process_combined` cannot produce; it went unnoticed
    # because the positional `iloc[-1]` never looked at the index at all.
    inst.raw = pd.DataFrame(
        {"BTCUSDT_close": [49000.0, 50000.0]},
        index=pd.DatetimeIndex(
            [settled_ts - pd.Timedelta(days=1), settled_ts]),
    )
    inst.vertical_features = pd.DataFrame({
        "timestamp": [settled_ts],
        "symbol": ["BINANCE_PERP_BTC_USDT"],
        "feat_a": [1.0],
        "feat_b": [2.0],
    })
    inst._data_fresh = True
    return inst


# -------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------


def test_initialization(vomir):
    assert vomir.long_pred_threshold == 0.4
    assert vomir.short_pred_threshold == 0.4
    assert vomir.trading_mode == "both"
    assert vomir._vomir_clf is not None
    assert vomir._vomir_feature_cols == ["feat_a", "feat_b"]


def test_init_missing_weights_path():
    config = {
        "TIME_UNIT": "1d",
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "CAPITAL": 1000,
    }
    with patch("vomir.trading.AgamottoResearch.__init__",
               return_value=None):
        with pytest.raises(ValueError, match="VOMIR_WEIGHTS_PATH"):
            inst = VomirTrading.__new__(VomirTrading)
            inst.config = config
            inst.home_root = "/tmp"
            inst._load_vomir_model()


# -------------------------------------------------------------------
# make_decision
# -------------------------------------------------------------------


def test_make_decision_long(vomir):
    """P(LONG) = 0.8 > 0.4 threshold => positive qty."""
    vomir._vomir_clf.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])
    decisions = vomir.make_decision()
    price, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0, f"Expected positive qty for LONG signal, got {qty}"


def test_make_decision_short(vomir):
    """P(SHORT) = 0.8 > 0.4 threshold => negative qty."""
    vomir._vomir_clf.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])
    decisions = vomir.make_decision()
    price, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty < 0, f"Expected negative qty for SHORT signal, got {qty}"


def test_make_decision_neutral(vomir):
    """Both P(LONG) and P(SHORT) < 0.4 => qty = 0."""
    vomir._vomir_clf.predict_proba.return_value = np.array([[0.2, 0.6, 0.2]])
    decisions = vomir.make_decision()
    price, qty = decisions["BINANCE_PERP_BTC_USDT"]
    assert qty == 0.0, f"Expected zero qty for neutral, got {qty}"


# -------------------------------------------------------------------
# clean
# -------------------------------------------------------------------


def test_clean_returns_zeros(vomir):
    decisions = vomir.clean()
    assert decisions == {"BINANCE_PERP_BTC_USDT": [0.0, 0.0]}
