"""
Tests for parse_ws_kline_to_row — converts Binance WS kline message
to a single-row DataFrame matching klines_to_dataframe() output exactly.
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

from agamotto.lib_binance import (
    parse_ws_kline_to_row,
    klines_to_dataframe,
)


# ---------------------------------------------------------------------------
# Sample WS kline data (matches Binance Futures kline stream format)
# ---------------------------------------------------------------------------

SAMPLE_WS_KLINE = {
    "t": 1704067200000,   # 2024-01-01 00:00:00 UTC (open time ms)
    "T": 1704070799999,   # close time ms
    "s": "BTCUSDT",
    "i": "15m",
    "o": "42000.50",       # open
    "h": "42100.75",       # high
    "l": "41950.25",       # low
    "c": "42050.00",       # close
    "v": "123.456",        # volume
    "n": 5000,             # number of trades
    "x": True,             # is closed
    "q": "5189012.34",     # quote volume
    "V": "61.728",         # taker buy base volume
    "Q": "2594506.17",     # taker buy quote volume
}

# Equivalent REST row (12-element list matching /fapi/v1/klines format)
EQUIVALENT_REST_ROW = [
    1704067200000,          # open_time_ms
    "42000.50",             # open
    "42100.75",             # high
    "41950.25",             # low
    "42050.00",             # close
    "123.456",              # volume
    1704070799999,          # close_time_ms
    "5189012.34",           # quote_volume
    5000,                   # number_of_trades
    "61.728",               # taker_buy_base_volume
    "2594506.17",           # taker_buy_quote_volume
    "0",                    # ignore
]


class TestParseWsKlineToRow:
    def test_returns_single_row_dataframe(self):
        result = parse_ws_kline_to_row(SAMPLE_WS_KLINE, "BTCUSDT", "15m")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_index_is_utc_naive_timestamp(self):
        result = parse_ws_kline_to_row(SAMPLE_WS_KLINE, "BTCUSDT", "15m")
        ts = result.index[0]
        assert isinstance(ts, pd.Timestamp)
        assert ts.tzinfo is None  # tz-naive
        assert ts == pd.Timestamp("2024-01-01 00:00:00")

    def test_column_names_match_rest_format(self):
        """Column names should be {symbol}_{interval}_{col}."""
        result = parse_ws_kline_to_row(SAMPLE_WS_KLINE, "BTCUSDT", "15m")
        expected_cols = [
            "BTCUSDT_15m_open", "BTCUSDT_15m_high", "BTCUSDT_15m_low",
            "BTCUSDT_15m_close", "BTCUSDT_15m_volume",
            "BTCUSDT_15m_quote_volume", "BTCUSDT_15m_number_of_trades",
            "BTCUSDT_15m_taker_buy_base_volume",
            "BTCUSDT_15m_taker_buy_quote_volume",
        ]
        assert list(result.columns) == expected_cols

    def test_values_are_float64(self):
        result = parse_ws_kline_to_row(SAMPLE_WS_KLINE, "BTCUSDT", "15m")
        for col in result.columns:
            assert result[col].dtype == np.float64

    def test_values_match_rest_pipeline(self):
        """WS-parsed row must be identical to REST-parsed row for same data."""
        ws_result = parse_ws_kline_to_row(
            SAMPLE_WS_KLINE, "BTCUSDT", "15m")
        rest_result = klines_to_dataframe(
            [EQUIVALENT_REST_ROW], "BTCUSDT", "15m")

        # Same index
        assert ws_result.index[0] == rest_result.index[0]
        # Same columns
        assert list(ws_result.columns) == list(rest_result.columns)
        # Same values
        for col in ws_result.columns:
            assert ws_result[col].iloc[0] == pytest.approx(
                rest_result[col].iloc[0])

    def test_different_symbols(self):
        """Works for any symbol."""
        kline = {**SAMPLE_WS_KLINE, "s": "ETHUSDT"}
        result = parse_ws_kline_to_row(kline, "ETHUSDT", "1h")
        assert all(c.startswith("ETHUSDT_1h_") for c in result.columns)

    def test_number_of_trades_is_float(self):
        """number_of_trades comes as int in WS, must be float64 like REST."""
        result = parse_ws_kline_to_row(SAMPLE_WS_KLINE, "BTCUSDT", "15m")
        col = "BTCUSDT_15m_number_of_trades"
        assert result[col].dtype == np.float64
        assert result[col].iloc[0] == 5000.0
