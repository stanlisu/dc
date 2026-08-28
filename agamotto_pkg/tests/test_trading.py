
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path (prefer local source)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agamotto.trading import AgamottoTrading

@pytest.fixture
def mock_config():
    return {
        "TIME_UNIT": "1d",
        "TIMEFRAME_SECONDS": 86400, # 1 day
        "SIZES": [0.01],
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "CAPITAL": 1000,
        "LADDER": 1.0,
        "WEIGHTS_PATH": "/tmp/mock_weights",
        "WEIGHTS_PERIOD": "window_2026_01",
        "TRADING_MODE": "both",
        "LONG_PRED_THRESHOLD": 0.0,
        "SHORT_PRED_THRESHOLD": 0.0,
        "REGIME_STACK_PATH": "/tmp/fake_regime_stack.csv",
    }

@pytest.fixture
def mock_agamotto(mock_config):
    def fake_load_regime_stack(self_inner):
        self_inner.regime_stack = []
        self_inner.models = {"long": {}, "short": {}}

    with patch.object(AgamottoTrading, "_load_regime_stack", fake_load_regime_stack), \
         patch("agamotto.trading.AgamottoTrading._calculate_sizes"), \
         patch("agamotto.trading.AgamottoTrading.load_data"), \
         patch("agamotto.trading.fetch_futures_klines"), \
         patch("agamotto.trading.klines_to_dataframe"):

        agamotto = AgamottoTrading(config=mock_config, home_root="/tmp", period="window_test")
        
        # Setup mocks for models
        mock_model_artifact = {
            "model": MagicMock(),
            "scaler": MagicMock(),
            "metadata": {"feature_columns": ["open", "close"]}
        }
        mock_model_artifact["model"].predict.return_value = [0.1]
        mock_model_artifact["scaler"].transform.return_value = [[0.1, 0.2]]
        
        agamotto.models = {
            "long": {"mock_model": mock_model_artifact},
            "short": {"mock_model": mock_model_artifact}
        }
        
        # Setup filtered signals with PAST timestamp so it's considered a completed candle
        past_ts = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=1, hours=1)
        
        df = pd.DataFrame({
            "timestamp": [past_ts], 
            "symbol": ["BINANCE_PERP_BTC_USDT"],
            "open": [100.0],
            "close": [100.0],
            "position": ["long"],
            "prediction": [0.1], # High enough to pass threshold
            "opt_threshold": [0.01]
        }, index=[past_ts])
        
        agamotto.features = df # Should be a DataFrame
        agamotto.filtered_signals = df
        
        # Mock methods to avoid Side Effects
        agamotto.engineer_features = MagicMock()
        agamotto.verticalize = MagicMock()
        agamotto.filter_signals = MagicMock()

        return agamotto

def test_initialization(mock_agamotto):
    assert mock_agamotto.config["CAPITAL"] == 1000
    assert "BINANCE_PERP_BTC_USDT" in mock_agamotto.config["SYMBOLS"]


def test_no_trading_mode_attribute(mock_agamotto):
    """agamotto does not read TRADING_MODE.

    It used to hold `self.trading_mode = config.get("TRADING_MODE", "both")` --
    a banned magic-default read whose ONLY use was a boot log line. Direction
    comes from the regime stack's `position` column; TRADING_MODE now means
    execution style and nothing else (knull/execution_style.py), so the read
    is not just dead, it is dead in the WRONG vocabulary: the value logged as
    "Trading Mode" was whatever the executor's style happened to be, which is
    exactly the confusion that made a direction gate on the same key silently
    flatten vomir. Pinned as ABSENT so it cannot come back as a gate.
    """
    assert not hasattr(mock_agamotto, "trading_mode")

def test_make_decision_long(mock_agamotto):
    """Long decision via regime stack."""
    settled_ts = pd.Timestamp("2025-01-01 00:00:00")

    # Set up vertical_features and raw for make_decision
    mock_agamotto.vertical_features = pd.DataFrame({
        "timestamp": [settled_ts],
        "symbol": ["BINANCE_PERP_BTC_USDT"],
        "open": [50000.0],
        "close": [50000.0],
    })
    mock_agamotto.raw = pd.DataFrame(
        {"BTCUSDT_close": [49000.0, 50000.0]},
        # raw's LAST row IS the row predict() targets — `_process_combined`
        # drops the in-flight candle before assigning `self.raw`, so `raw.index[-1]`
        # is the settled bar. The old fixture ended a day BEFORE
        # vertical_features' timestamp, a state the loader cannot produce; it
        # survived only because iloc[-2] never looked at the index.
        index=pd.to_datetime(["2024-12-31", "2025-01-01"]),
    )
    mock_agamotto.config["LOT_SIZES"] = {
        "BINANCE_PERP_BTC_USDT": {"step_size": 0.001, "min_notional": 5.0}
    }
    mock_agamotto._data_fresh = True

    # Create a long regime that triggers
    model = MagicMock()
    model.predict.return_value = [0.05]
    scaler = MagicMock()
    scaler.transform.return_value = [[0.1, 0.2]]
    mock_agamotto.regime_stack = [{
        "id": "test_long",
        "regime": "baseline",
        "position": "long",
        "model_name": "mock",
        "threshold": 0.01,
        "artifact": {"model": model, "scaler": scaler,
                     "metadata": {"feature_columns": ["open", "close"]}},
        "filter": None,
        "config": {},
    }]

    def mock_filter(regime, save=False):
        vf = mock_agamotto.vertical_features.copy()
        vf["position"] = regime["position"]
        return vf

    mock_agamotto.filter_signals = mock_filter
    mock_agamotto.make_decision(label="long")

    assert "BINANCE_PERP_BTC_USDT" in mock_agamotto.decisions
    price, qty = mock_agamotto.decisions["BINANCE_PERP_BTC_USDT"]
    assert qty > 0


def test_make_decision_short(mock_agamotto):
    """Short decision via regime stack."""
    settled_ts = pd.Timestamp("2025-01-01 00:00:00")

    mock_agamotto.vertical_features = pd.DataFrame({
        "timestamp": [settled_ts],
        "symbol": ["BINANCE_PERP_BTC_USDT"],
        "open": [50000.0],
        "close": [50000.0],
    })
    mock_agamotto.raw = pd.DataFrame(
        {"BTCUSDT_close": [49000.0, 50000.0]},
        # raw's LAST row IS the row predict() targets — `_process_combined`
        # drops the in-flight candle before assigning `self.raw`, so `raw.index[-1]`
        # is the settled bar. The old fixture ended a day BEFORE
        # vertical_features' timestamp, a state the loader cannot produce; it
        # survived only because iloc[-2] never looked at the index.
        index=pd.to_datetime(["2024-12-31", "2025-01-01"]),
    )
    mock_agamotto.config["LOT_SIZES"] = {
        "BINANCE_PERP_BTC_USDT": {"step_size": 0.001, "min_notional": 5.0}
    }
    mock_agamotto._data_fresh = True

    model = MagicMock()
    model.predict.return_value = [-0.05]
    scaler = MagicMock()
    scaler.transform.return_value = [[0.1, 0.2]]
    mock_agamotto.regime_stack = [{
        "id": "test_short",
        "regime": "baseline",
        "position": "short",
        "model_name": "mock",
        "threshold": -0.01,
        "artifact": {"model": model, "scaler": scaler,
                     "metadata": {"feature_columns": ["open", "close"]}},
        "filter": None,
        "config": {},
    }]

    def mock_filter(regime, save=False):
        vf = mock_agamotto.vertical_features.copy()
        vf["position"] = regime["position"]
        return vf

    mock_agamotto.filter_signals = mock_filter
    mock_agamotto.make_decision(label="short")

    assert "BINANCE_PERP_BTC_USDT" in mock_agamotto.decisions
    price, qty = mock_agamotto.decisions["BINANCE_PERP_BTC_USDT"]
    assert qty < 0
