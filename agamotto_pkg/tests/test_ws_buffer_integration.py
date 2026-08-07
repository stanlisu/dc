"""
Tests for WS buffer integration in AgamottoTrading._fetch_and_prepare_data().

Covers:
  - When _kline_buffer is set and ready, data is read from buffer (no REST)
  - When _kline_buffer is not ready, falls back to REST
  - When _kline_buffer is None, uses REST (backward compat)
  - Buffer data goes through same post-processing as REST data
"""

import sys
import os

MARVEL_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, MARVEL_ROOT)
sys.path.insert(0, os.path.join(MARVEL_ROOT, "agamotto_pkg", "src"))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from agamotto.trading import AgamottoTrading
# `symbiote` lives in the marvel repo, not dc — skip when unavailable (dc CI).
pytest.importorskip("symbiote")
from symbiote.kline_buffer import KlineBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kline_df(symbol_native: str, tf: str, n_bars: int = 100):
    """Build DataFrame matching klines_to_dataframe output format.
    Uses {symbol}_{tf}_{col} naming (pre-rename in _fetch_and_prepare_data).
    """
    tf_seconds = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}[tf]
    # End at a recent aligned time so freshness check passes
    end = pd.Timestamp.utcnow().floor(f"{tf_seconds}s")
    start = end - pd.Timedelta(seconds=tf_seconds * n_bars)
    index = pd.date_range(start, periods=n_bars,
                          freq=pd.Timedelta(seconds=tf_seconds))

    cols = ["open", "high", "low", "close", "volume",
            "quote_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume"]

    data = {}
    for col in cols:
        values = np.random.rand(n_bars).astype(np.float64) + 1.0
        data[f"{symbol_native}_{tf}_{col}"] = values

    return pd.DataFrame(data, index=index)


def _make_trading_instance(config_overrides=None):
    """Create AgamottoTrading with mocked init."""
    config = {
        "TIME_UNIT": "15m",
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT", "BINANCE_PERP_ETH_USDT"],
        "CAPITAL": 10000,
        "LEVERAGE": 5,
        "SIZES": [0.001, 0.01],
        "WEIGHTS_PATH": "/tmp/mock_weights",
        "WEIGHTS_PERIOD": "window_2026_01",
        "TRADING_MODE": "both",
        "LONG_PRED_THRESHOLD": 0.0,
        "SHORT_PRED_THRESHOLD": 0.0,
        "REGIME_STACK_PATH": "/tmp/fake_regime_stack.csv",
    }
    if config_overrides:
        config.update(config_overrides)

    def fake_load_regime_stack(self_inner):
        self_inner.regime_stack = []

    with patch.object(AgamottoTrading, "_load_regime_stack",
                      fake_load_regime_stack), \
         patch("agamotto.trading.AgamottoTrading._calculate_sizes"), \
         patch("agamotto.trading.AgamottoTrading.load_data"), \
         patch("agamotto.trading.fetch_futures_klines"), \
         patch("agamotto.trading.klines_to_dataframe"):
        inst = AgamottoTrading(
            config=config, home_root="/tmp", period="window_test")

    return inst


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWsBufferPath:
    def test_buffer_ready_skips_rest(self):
        """When buffer has all symbols, REST is not called."""
        inst = _make_trading_instance()
        buf = KlineBuffer()

        for sym in ["BTCUSDT", "ETHUSDT"]:
            df = _make_kline_df(sym, "15m", n_bars=100)
            buf.initialize(sym, "15m", df)

        inst._kline_buffer = buf

        with patch("agamotto.trading.fetch_futures_klines") as mock_rest, \
             patch.object(inst, "engineer_features"), \
             patch.object(inst, "verticalize"):
            inst._fetch_and_prepare_data(limit=100)
            mock_rest.assert_not_called()

        assert inst.raw is not None
        assert len(inst.raw) > 0

    def test_buffer_not_ready_falls_back_to_rest(self):
        """When buffer is missing a symbol, falls through to REST."""
        inst = _make_trading_instance()
        buf = KlineBuffer()

        # Only initialize one symbol
        buf.initialize("BTCUSDT", "15m",
                       _make_kline_df("BTCUSDT", "15m", n_bars=100))

        inst._kline_buffer = buf

        mock_df = _make_kline_df("BTCUSDT", "15m", n_bars=100)

        with patch("agamotto.trading.fetch_futures_klines") as mock_rest, \
             patch("agamotto.trading.klines_to_dataframe",
                   return_value=mock_df), \
             patch.object(inst, "engineer_features"), \
             patch.object(inst, "verticalize"):
            mock_rest.return_value = [[0] * 12]
            inst._fetch_and_prepare_data(limit=100)
            assert mock_rest.call_count > 0

    def test_no_buffer_uses_rest(self):
        """When _kline_buffer is None, pure REST path (backward compat)."""
        inst = _make_trading_instance()
        inst._kline_buffer = None

        mock_df = _make_kline_df("BTCUSDT", "15m", n_bars=100)

        with patch("agamotto.trading.fetch_futures_klines") as mock_rest, \
             patch("agamotto.trading.klines_to_dataframe",
                   return_value=mock_df), \
             patch.object(inst, "engineer_features"), \
             patch.object(inst, "verticalize"):
            mock_rest.return_value = [[0] * 12]
            inst._fetch_and_prepare_data(limit=100)
            assert mock_rest.call_count > 0

    def test_buffer_data_drops_incomplete_bar(self):
        """WS buffer path should still drop the last bar (incomplete)."""
        inst = _make_trading_instance()
        buf = KlineBuffer()

        n_bars = 50
        for sym in ["BTCUSDT", "ETHUSDT"]:
            df = _make_kline_df(sym, "15m", n_bars=n_bars)
            buf.initialize(sym, "15m", df)

        inst._kline_buffer = buf

        with patch.object(inst, "engineer_features"), \
             patch.object(inst, "verticalize"):
            inst._fetch_and_prepare_data(limit=100)

        assert len(inst.raw) == n_bars - 1

    def test_buffer_data_is_tz_naive(self):
        """Output from buffer path should be tz-naive like REST path."""
        inst = _make_trading_instance()
        buf = KlineBuffer()

        for sym in ["BTCUSDT", "ETHUSDT"]:
            df = _make_kline_df(sym, "15m", n_bars=50)
            buf.initialize(sym, "15m", df)

        inst._kline_buffer = buf

        with patch.object(inst, "engineer_features"), \
             patch.object(inst, "verticalize"):
            inst._fetch_and_prepare_data(limit=100)

        assert inst.raw.index.tz is None

    def test_not_ready_fallback_names_the_missing_symbols(self, caplog):
        """A REST fallback here costs ~10s inside the decision window and the
        caller who wired up a buffer believes the cycle is cheap. CLAUDE.md
        bans the silent version: the log must say WHY and WHICH symbols.
        """
        inst = _make_trading_instance()
        buf = KlineBuffer()
        buf.initialize("BTCUSDT", "15m",
                       _make_kline_df("BTCUSDT", "15m", n_bars=100))
        inst._kline_buffer = buf

        mock_df = _make_kline_df("BTCUSDT", "15m", n_bars=100)
        with patch("agamotto.trading.fetch_futures_klines") as mock_rest, \
             patch("agamotto.trading.klines_to_dataframe",
                   return_value=mock_df), \
             patch.object(inst, "engineer_features"), \
             patch.object(inst, "verticalize"), \
             caplog.at_level("WARNING"):
            mock_rest.return_value = [[0] * 12]
            inst._fetch_and_prepare_data(limit=100)

        assert "NOT READY" in caplog.text
        assert "FALLING BACK TO REST" in caplog.text
        assert "ETHUSDT" in caplog.text          # the symbol that was absent

    def test_incomplete_read_fallback_names_the_empty_symbols(self, caplog):
        """``is_ready`` passes (both keys present) but one entry is empty."""
        inst = _make_trading_instance()
        buf = KlineBuffer()
        buf.initialize("BTCUSDT", "15m",
                       _make_kline_df("BTCUSDT", "15m", n_bars=100))
        buf.initialize("ETHUSDT", "15m", pd.DataFrame())
        inst._kline_buffer = buf
        assert buf.is_ready(["BTCUSDT", "ETHUSDT"], "15m") is True

        mock_df = _make_kline_df("BTCUSDT", "15m", n_bars=100)
        with patch("agamotto.trading.fetch_futures_klines") as mock_rest, \
             patch("agamotto.trading.klines_to_dataframe",
                   return_value=mock_df), \
             patch.object(inst, "engineer_features"), \
             patch.object(inst, "verticalize"), \
             caplog.at_level("WARNING"):
            mock_rest.return_value = [[0] * 12]
            inst._fetch_and_prepare_data(limit=100)
            assert mock_rest.call_count > 0

        assert "INCOMPLETE" in caplog.text
        assert "ETHUSDT" in caplog.text

    def test_feature_tf_reads_correct_tf_from_buffer(self):
        """When FEATURE_TF is set, buffer should be read with that TF."""
        inst = _make_trading_instance(config_overrides={
            "TIME_UNIT": "1h",
            "FEATURE_TF": "15m",
        })
        buf = KlineBuffer()

        for sym in ["BTCUSDT", "ETHUSDT"]:
            df = _make_kline_df(sym, "15m", n_bars=100)
            buf.initialize(sym, "15m", df)

        inst._kline_buffer = buf

        with patch("agamotto.trading.fetch_futures_klines") as mock_rest, \
             patch.object(inst, "engineer_features"), \
             patch.object(inst, "verticalize"):
            inst._fetch_and_prepare_data(limit=100)
            mock_rest.assert_not_called()

        assert inst.raw is not None
