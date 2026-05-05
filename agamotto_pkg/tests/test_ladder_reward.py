"""Tests for LADDER sizing-up reward in ORB and Mjolnir research engines.

These are RED-phase TDD tests — they will fail with ImportError until
gauntlet/paths.py (compute_ladder_multiplier) and the OrbResearch /
MjolnirResearch classes are updated with _compute_ladder_returns.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ensure agamotto_pkg/src is on the path (matches PYTHONPATH=agamotto_pkg/src:.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# This import will fail until gauntlet/paths.py exports compute_ladder_multiplier.
from gauntlet.paths import compute_ladder_multiplier  # noqa: E402


# ---------------------------------------------------------------------------
# Tests for compute_ladder_multiplier() pure function
# ---------------------------------------------------------------------------

class TestLadderMultiplierPureFunction:

    def test_ladder_multiplier_no_dip(self):
        """When low_next == close (no dip), multiplier is 1x for every row."""
        close = pd.Series([100.0, 200.0])
        low_next = close.copy()  # no dip at all
        result = compute_ladder_multiplier(close, low_next, ladder=2)
        expected = pd.Series([1, 1])
        pd.testing.assert_series_equal(result.reset_index(drop=True),
                                       expected.reset_index(drop=True),
                                       check_dtype=False)

    def test_ladder_multiplier_one_bps_dip(self):
        """When low_next = close * (1 - 0.0001) — exactly 1bp dip — with LADDER=2,
        multiplier should be 1 + 1 = 2x (one extra layer triggered)."""
        close = pd.Series([100.0])
        low_next = pd.Series([99.99])  # 1bp dip
        result = compute_ladder_multiplier(close, low_next, ladder=2)
        assert result.iloc[0] == 2  # 1 base + 1 layer

    def test_ladder_multiplier_capped(self):
        """When dip is 10bp but LADDER=2, multiplier is capped at 1 + 2 = 3x."""
        close = pd.Series([100.0])
        low_next = pd.Series([99.90])  # 10bp dip
        result = compute_ladder_multiplier(close, low_next, ladder=2)
        assert result.iloc[0] == 3  # 1 base + 2 layers (cap)

    def test_ladder_multiplier_returns_series(self):
        """Return type is always a pd.Series."""
        close = pd.Series([100.0, 100.0])
        low_next = pd.Series([99.95, 100.0])
        result = compute_ladder_multiplier(close, low_next, ladder=3)
        assert isinstance(result, pd.Series)

    def test_ladder_multiplier_zero_ladder(self):
        """LADDER=0 means no rungs — multiplier is always 1x."""
        close = pd.Series([100.0])
        low_next = pd.Series([99.0])  # large dip, but LADDER=0
        result = compute_ladder_multiplier(close, low_next, ladder=0)
        assert result.iloc[0] == 1

    def test_ladder_multiplier_two_bps_dip(self):
        """2bp dip with LADDER=5 => 1 + 2 = 3x multiplier."""
        close = pd.Series([100.0])
        low_next = pd.Series([99.98])  # 2bp dip
        result = compute_ladder_multiplier(close, low_next, ladder=5)
        assert result.iloc[0] == 3  # 1 base + 2 layers


# ---------------------------------------------------------------------------
# Tests for OrbResearch._compute_ladder_returns() method
# ---------------------------------------------------------------------------

class TestOrbLadderColumns:

    def test_orb_ladder_columns_created(self):
        """OrbResearch._compute_ladder_returns(df) produces the expected columns.

        Will fail with ImportError until orb research module is updated,
        or AttributeError if the method doesn't exist yet.
        """
        # Late import so the file-level ImportError on compute_ladder_multiplier
        # is hit first; this import will also fail until orb research exists.
        from orb.research import OrbResearch  # noqa: F401

        cfg = {"LADDER": 2, "FEE": 2.0}
        orb = OrbResearch.__new__(OrbResearch)
        orb.config = cfg

        # Build a minimal OHLCV DataFrame (3 bars)
        df = pd.DataFrame({
            "open":  [100.0, 101.0, 102.0],
            "high":  [105.0, 106.0, 107.0],
            "low":   [98.0,  99.0,  100.0],
            "close": [103.0, 104.0, 105.0],
        })

        result = orb._compute_ladder_returns(
            df,
            close_col="close",
            low_col="low",
            high_col="high",
        )

        for col in ("return_long", "return_short", "return_long_raw", "return_short_raw"):
            assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Tests for MjolnirResearch._compute_ladder_returns() method
# ---------------------------------------------------------------------------

class TestMjolnirLadderColumns:

    def test_mjolnir_ladder_columns_created(self):
        """MjolnirResearch._compute_ladder_returns(df) produces the expected columns.

        Will fail with ImportError until mjolnir research module is updated,
        or AttributeError if the method doesn't exist yet.
        """
        from agamotto.research_mjolnir import MjolnirResearch  # noqa: F401

        cfg = {"LADDER": 2, "FEE": 2.0}
        mjolnir = MjolnirResearch.__new__(MjolnirResearch)
        mjolnir.config = cfg

        # Build a minimal OHLCV DataFrame (3 bars)
        df = pd.DataFrame({
            "open":  [100.0, 101.0, 102.0],
            "high":  [105.0, 106.0, 107.0],
            "low":   [98.0,  99.0,  100.0],
            "close": [103.0, 104.0, 105.0],
        })

        result = mjolnir._compute_ladder_returns(
            df,
            close_col="close",
            low_col="low",
            high_col="high",
        )

        for col in ("return_long", "return_short", "return_long_raw", "return_short_raw"):
            assert col in result.columns, f"Missing column: {col}"
