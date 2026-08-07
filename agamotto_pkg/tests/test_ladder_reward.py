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

# gauntlet lives in marvel/ — skip when not on PYTHONPATH (e.g. dc-only run).
compute_ladder_multiplier = pytest.importorskip(
    "gauntlet.paths",
    reason="gauntlet package not on PYTHONPATH (run from marvel/ with "
           "PYTHONPATH=agamotto_pkg/src:.)").compute_ladder_multiplier


# ---------------------------------------------------------------------------
# Tests for compute_ladder_multiplier() pure function
# ---------------------------------------------------------------------------

class TestLadderMultiplierPureFunction:

    # NOTE 2026-08-06: `step_bps` is now REQUIRED (was hardcoded 0.0001), and the
    # cap is LADDER TOTAL rungs (was LADDER+1). LADDER counts every rung including
    # the entry one — `knull/ladder.py:157` runs levels 1..ladder_max inclusive —
    # so only LADDER-1 rungs are reachable by dipping. Canonical tests for this
    # function live with the function, in marvel/tests/test_ladder_multiplier.py.

    def test_ladder_multiplier_no_dip(self):
        """When low_next == close (no dip), multiplier is 1x for every row."""
        close = pd.Series([100.0, 200.0])
        low_next = close.copy()  # no dip at all
        result = compute_ladder_multiplier(close, low_next, ladder=2, step_bps=1.0)
        expected = pd.Series([1, 1])
        pd.testing.assert_series_equal(result.reset_index(drop=True),
                                       expected.reset_index(drop=True),
                                       check_dtype=False)

    def test_ladder_multiplier_one_bps_dip(self):
        """When low_next = close * (1 - 0.0001) — exactly 1bp dip — with LADDER=2,
        multiplier should be 1 + 1 = 2x (one extra layer triggered)."""
        close = pd.Series([100.0])
        low_next = pd.Series([99.99])  # 1bp dip
        result = compute_ladder_multiplier(close, low_next, ladder=2, step_bps=1.0)
        assert result.iloc[0] == 2  # 1 base + 1 layer

    def test_ladder_multiplier_capped(self):
        """When dip is 10bp but LADDER=2, multiplier is capped at LADDER = 2x.

        Was 3x: the old cap allowed LADDER+1 rungs, one deeper than the executor
        can fill.
        """
        close = pd.Series([100.0])
        low_next = pd.Series([99.90])  # 10bp dip
        result = compute_ladder_multiplier(close, low_next, ladder=2, step_bps=1.0)
        assert result.iloc[0] == 2  # base rung + 1 reachable extra = LADDER

    def test_ladder_multiplier_returns_series(self):
        """Return type is always a pd.Series."""
        close = pd.Series([100.0, 100.0])
        low_next = pd.Series([99.95, 100.0])
        result = compute_ladder_multiplier(close, low_next, ladder=3, step_bps=1.0)
        assert isinstance(result, pd.Series)

    def test_ladder_multiplier_zero_ladder(self):
        """LADDER=0 means no extra rungs — multiplier is always 1x (entry only)."""
        close = pd.Series([100.0])
        low_next = pd.Series([99.0])  # large dip, but LADDER=0
        result = compute_ladder_multiplier(close, low_next, ladder=0, step_bps=1.0)
        assert result.iloc[0] == 1

    def test_ladder_multiplier_two_bps_dip(self):
        """2bp dip with LADDER=5 => 1 + 2 = 3x multiplier."""
        close = pd.Series([100.0])
        low_next = pd.Series([99.98])  # 2bp dip
        result = compute_ladder_multiplier(close, low_next, ladder=5, step_bps=1.0)
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
        pytest.importorskip("orb.research",
                            reason="orb package not installed")
        from orb.research import OrbResearch  # noqa: F401

        cfg = {"LADDER": 2, "LADDER_BPS": 1.0, "FEE": 2.0}
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
        pytest.importorskip("mjolnir.core",
                            reason="mjolnir package not installed")
        from agamotto.research_mjolnir import MjolnirResearch  # noqa: F401

        # LADDER_FILL_MODE is a REQUIRED key (no default) — "ladder" is the
        # value this config silently received before the key became required.
        cfg = {"LADDER": 2, "FEE": 2.0, "LADDER_FILL_MODE": "ladder"}
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


class TestMjolnirLadderVariableHorizon:
    """Verify that the horizon_bars parameter widens the ladder fill window.

    Native-mode experiments (TIME_UNIT == bar resolution) pass
    horizon_bars=1 and get the original 1-bar shift behaviour. Boundary-
    aligned experiments (e.g. mjolnir.base.30s_1 = 5s bars predicting
    30s closes) pass horizon_bars = TIME_UNIT_seconds / bar_tf_seconds
    so the low/high lookahead spans the full prediction horizon —
    otherwise the regime-stack target collapses to a frictionless
    close-to-close return that the maker-first executor cannot trade.
    """

    def _new_mjolnir(self, ladder=2, fee=0.0):
        from agamotto.research_mjolnir import MjolnirResearch  # noqa: F401
        # LADDER_FILL_MODE is a REQUIRED key (no default) — "ladder" is the
        # value this config silently received before the key became required.
        cfg = {"LADDER": ladder, "FEE": fee, "LADDER_FILL_MODE": "ladder"}
        mj = MjolnirResearch.__new__(MjolnirResearch)
        mj.config = cfg
        return mj

    def test_horizon_1_default_unchanged(self):
        """horizon_bars=1 (default) matches the legacy 1-bar shift."""
        mj = self._new_mjolnir()
        df = pd.DataFrame({
            "close": [100.0, 100.0, 100.0, 100.0],
            "low":   [ 99.9,  99.5,  99.99,  100.0],
            "high":  [100.1, 100.5, 100.01,  100.0],
        })
        explicit = mj._compute_ladder_returns(
            df, "close", "low", "high", horizon_bars=1)
        default = mj._compute_ladder_returns(df, "close", "low", "high")
        pd.testing.assert_frame_equal(default, explicit)

    def test_horizon_widens_fill_window(self):
        """horizon_bars=6 sees a deeper dip occurring 5 bars ahead than
        horizon_bars=1 sees by only inspecting the immediate next bar."""
        mj = self._new_mjolnir(ladder=2)
        # Dip of 4 bp lives at bar index 5; only horizon>=5 sees it from t=0.
        df = pd.DataFrame({
            "close": [100.0] * 6 + [100.5],
            "low":   [100.0, 100.0, 100.0, 100.0, 100.0,  99.96, 100.4],
            "high":  [100.0, 100.0, 100.0, 100.0, 100.0, 100.04, 100.6],
        })
        r1 = mj._compute_ladder_returns(df, "close", "low", "high", horizon_bars=1)
        r6 = mj._compute_ladder_returns(df, "close", "low", "high", horizon_bars=6)
        # horizon=1 from t=0 sees low[1]=100.0 → no dip → 1 layer
        # → return_long_raw[0] = price_return * 1 = 0 (close[1]/close[0] - 1)
        assert r1["return_long_raw"].iloc[0] == 0.0
        # horizon=6 from t=0 sees min(low[1..6])=99.96 → 4 bp dip → 2 layers (capped)
        # → total_long_layers = 1 + 2 = 3
        # → price_return = close[6]/close[0] - 1 = 100.5/100 - 1 = 0.005
        # → return_long_raw[0] = 0.005 * 3 = 0.015
        assert abs(r6["return_long_raw"].iloc[0] - 0.015) < 1e-9, (
            f"expected 0.015, got {r6['return_long_raw'].iloc[0]}")

    def test_horizon_zero_rejected(self):
        """horizon_bars < 1 must raise — silent fallback is banned by CLAUDE.md."""
        mj = self._new_mjolnir()
        df = pd.DataFrame({
            "close": [100.0, 100.0],
            "low":   [100.0, 100.0],
            "high":  [100.0, 100.0],
        })
        with pytest.raises(ValueError, match="horizon_bars must be >= 1"):
            mj._compute_ladder_returns(
                df, "close", "low", "high", horizon_bars=0)
