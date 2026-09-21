"""TARGET_RETURN_MODE — the ladder target's forward-return convention.

`close_to_close` is the historical behaviour and MUST stay byte-identical when the
key is absent: every committed kline setting predates it, so an absent key that
changed the target would silently reprice every stored artifact in the repo.
"""
import numpy as np
import pandas as pd
import pytest

from agamotto.mm_target import (
    TARGET_RETURN_CLOSE_TO_CLOSE,
    TARGET_RETURN_TYPICAL_OVER_OPEN,
    target_return_mode,
)


class TestResolver:
    def test_absent_key_is_close_to_close(self):
        """The DEPRECATED back-compat branch. Every existing arm depends on it."""
        assert target_return_mode({}) == TARGET_RETURN_CLOSE_TO_CLOSE

    @pytest.mark.parametrize("mode", [TARGET_RETURN_CLOSE_TO_CLOSE, TARGET_RETURN_TYPICAL_OVER_OPEN])
    def test_explicit_modes_round_trip(self, mode):
        assert target_return_mode({"TARGET_RETURN_MODE": mode}) == mode

    @pytest.mark.parametrize("bad", ["typical", "hlc3", "", "CLOSE_TO_CLOSE", 0, True])
    def test_unknown_value_raises(self, bad):
        """The chain ends in a fail-fast — a typo must not fall through to a default."""
        with pytest.raises(ValueError, match="TARGET_RETURN_MODE"):
            target_return_mode({"TARGET_RETURN_MODE": bad})


def _typical_over_open(open_s, high_s, low_s, close_s):
    """The formula under test, written out independently of the implementation."""
    open_next = open_s.shift(-1)
    typical_next = ((high_s + low_s + close_s) / 3.0).shift(-1)
    return typical_next / open_next.replace(0, np.nan) - 1.0


class TestFormula:
    """The arithmetic, pinned against hand-computed values."""

    def setup_method(self):
        # bar 0: O=100 H=110 L=90  C=105
        # bar 1: O=106 H=120 L=100 C=112   -> typical = (120+100+112)/3 = 110.6667
        # bar 2: O=112 H=118 L=108 C=115   -> typical = (118+108+115)/3 = 113.6667
        self.open_s = pd.Series([100.0, 106.0, 112.0])
        self.high_s = pd.Series([110.0, 120.0, 118.0])
        self.low_s = pd.Series([90.0, 100.0, 108.0])
        self.close_s = pd.Series([105.0, 112.0, 115.0])

    def test_bar0_is_next_bars_typical_over_next_bars_open(self):
        got = _typical_over_open(self.open_s, self.high_s, self.low_s, self.close_s)
        expected0 = ((120.0 + 100.0 + 112.0) / 3.0) / 106.0 - 1.0
        assert got.iloc[0] == pytest.approx(expected0)
        assert got.iloc[0] == pytest.approx(0.0440252, abs=1e-6)

    def test_bar1_uses_bar2(self):
        got = _typical_over_open(self.open_s, self.high_s, self.low_s, self.close_s)
        expected1 = ((118.0 + 108.0 + 115.0) / 3.0) / 112.0 - 1.0
        assert got.iloc[1] == pytest.approx(expected1)

    def test_last_bar_has_no_forward_window(self):
        got = _typical_over_open(self.open_s, self.high_s, self.low_s, self.close_s)
        assert pd.isna(got.iloc[-1])

    def test_differs_from_close_to_close(self):
        """If the two modes agreed, the key would be pointless. On a bar whose
        open sits away from the prior close they must diverge."""
        c2c = self.close_s.pct_change(fill_method=None).shift(-1)
        tvo = _typical_over_open(self.open_s, self.high_s, self.low_s, self.close_s)
        assert c2c.iloc[0] != pytest.approx(tvo.iloc[0])

    def test_zero_open_is_nan_not_inf(self):
        """A zero open must not produce ±inf, which would survive into the target
        and blow up every downstream mean/std silently."""
        open_s = pd.Series([100.0, 0.0, 112.0])
        got = _typical_over_open(open_s, self.high_s, self.low_s, self.close_s)
        assert pd.isna(got.iloc[0])
        assert np.isfinite(got.dropna()).all()

    def test_flat_bar_gives_zero(self):
        """O == H == L == C is a genuinely flat bar and must read as zero return,
        not NaN — it is a real observation, not missing data."""
        flat = pd.Series([50.0, 50.0])
        got = _typical_over_open(flat, flat, flat, flat)
        assert got.iloc[0] == pytest.approx(0.0)


class TestDualHorizonGuard:
    def test_mixing_conventions_is_refused(self):
        """DUAL_HORIZON's 2-bar target is close-to-close by construction; the
        agreement gate between two different conventions would be meaningless."""
        from agamotto.research import AgamottoResearch

        cfg = {
            "DUAL_HORIZON": True,
            "TARGET_RETURN_MODE": TARGET_RETURN_TYPICAL_OVER_OPEN,
            "FEE": 0.0,
            "LADDER": 1,
            "LADDER_BPS": 1.0,
            "TIME_UNIT": "15m",
        }
        research = AgamottoResearch(cfg, "/tmp")
        research.raw = pd.DataFrame(
            {"BTCUSDT_open": [1.0, 2.0], "BTCUSDT_high": [1.0, 2.0],
             "BTCUSDT_low": [1.0, 2.0], "BTCUSDT_close": [1.0, 2.0]},
            index=pd.to_datetime(["2026-08-01 00:00", "2026-08-01 00:15"]))
        with pytest.raises(ValueError, match="DUAL_HORIZON"):
            research.engineer_features()
