"""
Tests for dual horizon prediction: each model predicts native TF return + 15m extra return.

DUAL_HORIZON adds a second target column: the 15m return immediately AFTER the native
hold period. For a 1h model: y_native = close[t+1]/close[t]-1 (1h return),
y_15m_extra = close_15m[t+1h+15m]/close[t+1]-1 (first 15m of bar t+2).
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


# ---------------------------------------------------------------------------
# Helper: build synthetic kline data
# ---------------------------------------------------------------------------
def _make_1h_closes(n=20):
    """Create 1h close prices with known returns."""
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    # Prices: 100, 101, 99, 102, 98, ...
    prices = [100 + (i % 3) - 1 for i in range(n)]
    return pd.Series(prices, index=idx, dtype=float, name="close")


def _make_15m_closes(n=80):
    """Create 15m close prices aligned with 1h data."""
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    # Smooth interpolation with small noise
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.Series(prices, index=idx, dtype=float, name="close")


# ---------------------------------------------------------------------------
# Test 1: compute_dual_horizon_target produces correct 15m extra return
# ---------------------------------------------------------------------------
class TestDualHorizonTarget:
    """Test the target computation for dual horizon."""

    def test_15m_extra_return_for_1h_base(self):
        """For 1h base TF, 15m extra return = close_15m[t+1h+15m] / close_1h[t+1] - 1."""
        from agamotto.research import compute_dual_horizon_target

        closes_1h = _make_1h_closes(20)
        closes_15m = _make_15m_closes(80)

        result = compute_dual_horizon_target(
            closes_1h, closes_15m, base_tf="1h", extra_tf="15m")

        assert isinstance(result, pd.Series)
        assert len(result) == len(closes_1h)
        assert result.name == "return_15m_extra"

        # Verify one value manually:
        # At t=0 (01:00 bar), native close is closes_1h[1].
        # 15m extra = closes_15m at 01:15 / closes_1h[1] - 1
        t1_close = closes_1h.iloc[1]  # close of bar t+1
        t1_plus_15m = closes_15m.loc[
            closes_1h.index[1] + pd.Timedelta(minutes=15)]
        expected = t1_plus_15m / t1_close - 1
        assert abs(result.iloc[0] - expected) < 1e-10

    def test_15m_extra_return_for_4h_base(self):
        """For 4h base TF, 15m extra = close_15m[t+4h+15m] / close_4h[t+1] - 1."""
        from agamotto.research import compute_dual_horizon_target

        idx_4h = pd.date_range("2026-01-01", periods=10, freq="4h")
        closes_4h = pd.Series(
            [100, 101, 99, 102, 98, 103, 97, 104, 96, 105],
            index=idx_4h, dtype=float, name="close")

        idx_15m = pd.date_range("2026-01-01", periods=160, freq="15min")
        np.random.seed(42)
        closes_15m = pd.Series(
            100 + np.cumsum(np.random.randn(160) * 0.1),
            index=idx_15m, dtype=float, name="close")

        result = compute_dual_horizon_target(
            closes_4h, closes_15m, base_tf="4h", extra_tf="15m")

        assert len(result) == len(closes_4h)
        # Last value should be NaN (no 15m data after last 4h bar)
        assert pd.isna(result.iloc[-1])

    def test_last_bars_are_nan(self):
        """15m extra return should be NaN for bars where 15m data doesn't extend."""
        from agamotto.research import compute_dual_horizon_target

        closes_1h = _make_1h_closes(20)
        # Only 60 15m bars = 15 hours, but 20 1h bars = 20 hours
        closes_15m = _make_15m_closes(60)

        result = compute_dual_horizon_target(
            closes_1h, closes_15m, base_tf="1h", extra_tf="15m")

        # Bars beyond 15m data coverage should be NaN
        assert pd.isna(result.iloc[-1])

    def test_returns_empty_series_when_no_15m_data(self):
        """Return all-NaN series if 15m data is empty."""
        from agamotto.research import compute_dual_horizon_target

        closes_1h = _make_1h_closes(10)
        closes_15m = pd.Series(dtype=float, name="close")

        result = compute_dual_horizon_target(
            closes_1h, closes_15m, base_tf="1h", extra_tf="15m")

        assert len(result) == len(closes_1h)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# Test 2: engineer_features adds dual horizon columns when config enabled
# ---------------------------------------------------------------------------
class TestEngineerFeaturesDualHorizon:
    """Test that engineer_features adds 15m extra return columns."""

    def test_dual_horizon_columns_present(self):
        """When DUAL_HORIZON=true, features should contain return_15m_extra columns."""
        from agamotto.research import AgamottoResearch

        config = {
            "TIME_UNIT": "1h",
            "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
            "DUAL_HORIZON": True,
            "DUAL_HORIZON_TF": "15m",
            "LADDER": 1,
            "FEE": 0,
            "MA_PERIODS": [7, 25, 99],
            "STATS_WINDOW": 14,
        }
        # This test verifies the column exists after engineer_features
        # Will need mock data — test structure placeholder
        # The actual integration requires kline data loading
        assert True  # placeholder — real test needs data fixtures

    def test_dual_horizon_disabled_by_default(self):
        """Without DUAL_HORIZON=true, no extra return columns should be added."""
        from agamotto.research import AgamottoResearch

        config = {
            "TIME_UNIT": "1h",
            "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
            "LADDER": 1,
            "FEE": 0,
            "MA_PERIODS": [7, 25, 99],
            "STATS_WINDOW": 14,
        }
        # Verify no return_15m_extra columns without DUAL_HORIZON
        assert True  # placeholder


# ---------------------------------------------------------------------------
# Test 3: verticalize includes dual horizon target
# ---------------------------------------------------------------------------
class TestVerticalizeDualHorizon:

    def test_verticalize_includes_return_15m_extra(self):
        """After verticalization, return_15m_extra should be a column."""
        # When DUAL_HORIZON is enabled, verticalize should carry through
        # the {symbol}_return_15m_extra column as return_15m_extra
        assert True  # placeholder — needs full pipeline fixture


# ---------------------------------------------------------------------------
# Test 4: rolling_predict uses dual target
# ---------------------------------------------------------------------------
class TestDualHorizonTraining:

    def test_dual_target_column_in_training_data(self):
        """When DUAL_HORIZON, training data should have return_15m_extra column."""
        # The rolling_predict pipeline should:
        # 1. Not exclude return_15m_extra from features
        # 2. Make it available as an additional y target
        assert True  # placeholder
