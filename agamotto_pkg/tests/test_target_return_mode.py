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
    TARGET_RETURN_TYPICAL_OVER_CLOSE,
    TARGET_RETURN_TYPICAL_OVER_OPEN,
    target_return_mode,
)
from agamotto.ladder import compute_ladder_return
from agamotto.research import AgamottoResearch


class TestResolver:
    def test_absent_key_is_close_to_close(self):
        """The DEPRECATED back-compat branch. Every existing arm depends on it."""
        assert target_return_mode({}) == TARGET_RETURN_CLOSE_TO_CLOSE

    @pytest.mark.parametrize("mode", [TARGET_RETURN_CLOSE_TO_CLOSE, TARGET_RETURN_TYPICAL_OVER_OPEN,
                                      TARGET_RETURN_TYPICAL_OVER_CLOSE])
    def test_explicit_modes_round_trip(self, mode):
        assert target_return_mode({"TARGET_RETURN_MODE": mode}) == mode

    @pytest.mark.parametrize("bad", ["typical", "hlc3", "", "CLOSE_TO_CLOSE", "TYPICAL_OVER_CLOSE",
                                     0, True])
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
    @pytest.mark.parametrize("mode", [TARGET_RETURN_TYPICAL_OVER_OPEN, TARGET_RETURN_TYPICAL_OVER_CLOSE])
    def test_mixing_conventions_is_refused(self, mode):
        """DUAL_HORIZON's 2-bar target is close-to-close by construction; the
        agreement gate between two different conventions would be meaningless.
        Every non-close-to-close mode is refused, not just the first one added."""
        cfg = {
            "DUAL_HORIZON": True,
            "TARGET_RETURN_MODE": mode,
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
        with pytest.raises(ValueError, match=f"DUAL_HORIZON with TARGET_RETURN_MODE='{mode}'"):
            research.engineer_features()


# ---------------------------------------------------------------------------
# typical_over_close — driven through the REAL `engineer_features` path, not a
# local copy of the formula: ((H+L+C)/3)[t+1] / close[t] - 1.
# ---------------------------------------------------------------------------
STEP_BPS = 10.0               # LADDER_BPS -> rung 2 needs a 0.10% adverse move
STEP = STEP_BPS * 1e-4


def _toc_cfg(**over):
    cfg = {
        "EXCHANGE": "BINANCE", "DATA": "liquid", "TIME_UNIT": "15m",
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "TARGET_RETURN_MODE": TARGET_RETURN_TYPICAL_OVER_CLOSE,
        "LADDER": 2, "LADDER_BPS": STEP_BPS, "FEE": 0.0,
        "MA_PERIODS": [7, 25, 99], "STATS_WINDOW": 14,
    }
    cfg.update(over)
    return cfg


def _toc_raw():
    """Hand-picked first bars, then a quiet random walk so every rolling
    feature has enough history to run.

    bar 0: C=100
    bar 1: H=100.30 L=99.95  C=100.20 -> typ=100.15; low1 is 5bp under C0 (<10bp step)
                                        -> long 1 rung; high1 30bp over -> short 2 rungs
    bar 2: H=100.25 L=99.80  C=100.00 -> typ=100.0166..; low2 is 39.9bp under C1
                                        -> long 2 rungs
    """
    rng = np.random.default_rng(11)
    n = 40
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    high = close + 0.1
    low = close - 0.1
    close[:3] = [100.0, 100.20, 100.00]
    high[:3] = [100.05, 100.30, 100.25]
    low[:3] = [99.90, 99.95, 99.80]
    idx = pd.date_range("2026-08-01", periods=n, freq="15min")
    return pd.DataFrame({
        "BTCUSDT_open": close - 0.01,
        "BTCUSDT_high": high,
        "BTCUSDT_low": low,
        "BTCUSDT_close": close,
        "BTCUSDT_volume": np.full(n, 10.0),
    }, index=idx)


def _engineer(raw, **over):
    r = AgamottoResearch(_toc_cfg(**over), "/tmp/unused")
    r.raw = raw
    r.engineer_features()
    return r.features


class TestTypicalOverCloseEngineered:
    def test_row0_is_next_typical_over_this_close(self):
        f = _engineer(_toc_raw())
        expected = ((100.30 + 99.95 + 100.20) / 3.0) / 100.0 - 1.0
        assert f["BTCUSDT_return"].iloc[0] == pytest.approx(expected, abs=1e-15)
        assert f["BTCUSDT_return"].iloc[0] == pytest.approx(0.0015, abs=1e-12)
        # 1 long rung (5bp dip < 10bp step): the long label IS the anchor return.
        assert f["BTCUSDT_return_long_raw"].iloc[0] == pytest.approx(expected, abs=1e-15)

    def test_differs_from_typical_over_open_and_close_to_close(self):
        raw = _toc_raw()
        toc = _engineer(raw.copy())["BTCUSDT_return"].iloc[0]
        tvo = _engineer(raw.copy(), TARGET_RETURN_MODE=TARGET_RETURN_TYPICAL_OVER_OPEN)["BTCUSDT_return"].iloc[0]
        c2c = _engineer(raw.copy(), TARGET_RETURN_MODE=TARGET_RETURN_CLOSE_TO_CLOSE)["BTCUSDT_return"].iloc[0]
        assert toc != pytest.approx(tvo, abs=1e-12)
        assert toc != pytest.approx(c2c, abs=1e-12)

    def test_last_bar_has_no_forward_window(self):
        f = _engineer(_toc_raw())
        for col in ("BTCUSDT_return", "BTCUSDT_return_long_raw", "BTCUSDT_return_short_raw",
                    "BTCUSDT_return_long", "BTCUSDT_return_short"):
            assert pd.isna(f[col].iloc[-1]), col
            assert f[col].iloc[:-1].notna().all(), col

    @pytest.mark.parametrize("drop", ["BTCUSDT_high", "BTCUSDT_low"])
    def test_missing_high_or_low_raises(self, drop):
        """`sdf.get(col, close)` would substitute close and quietly make this
        close-to-close; the branch must refuse instead."""
        raw = _toc_raw().drop(columns=[drop])
        with pytest.raises(ValueError, match="typical_over_close"):
            _engineer(raw)

    def test_zero_close_is_nan_not_inf(self):
        raw = _toc_raw()
        raw.iloc[5, raw.columns.get_loc("BTCUSDT_close")] = 0.0
        f = _engineer(raw)
        assert pd.isna(f["BTCUSDT_return"].iloc[5])
        for col in ("BTCUSDT_return", "BTCUSDT_return_long_raw", "BTCUSDT_return_short_raw"):
            assert not np.isinf(f[col].to_numpy(dtype=float)).any(), col

    def test_ladder2_long_rung_anchored_on_this_close(self):
        """Row 1: bar 2's low sits 39.9bp under close[1] -> rung 2 fills at
        close[1]*(1-step); both rungs mark out at bar 2's typical price."""
        f = _engineer(_toc_raw())
        c1 = 100.20
        typ2 = (100.25 + 99.80 + 100.00) / 3.0
        rung1 = typ2 / c1 - 1.0
        rung2 = typ2 / (c1 * (1.0 - STEP)) - 1.0
        assert f["BTCUSDT_return_long_raw"].iloc[1] == pytest.approx(rung1 + rung2, abs=1e-15)

    def test_ladder2_short_rung_anchored_on_this_close(self):
        """Row 0: bar 1's high sits 30bp over close[0] -> the short's rung 2 fills
        at close[0]*(1+step)."""
        f = _engineer(_toc_raw())
        typ1 = (100.30 + 99.95 + 100.20) / 3.0
        rung1 = typ1 / 100.0 - 1.0
        rung2 = typ1 / (100.0 * (1.0 + STEP)) - 1.0
        assert f["BTCUSDT_return_short_raw"].iloc[0] == pytest.approx(rung1 + rung2, abs=1e-15)

    def test_fee_is_charged_per_rung(self):
        fee_bps = 2.0
        f = _engineer(_toc_raw(), FEE=fee_bps)
        fee_cost = 2.0 * fee_bps / 10000.0
        raw_long = f["BTCUSDT_return_long_raw"].iloc[1]     # 2 long rungs
        assert f["BTCUSDT_return_long"].iloc[1] == pytest.approx(raw_long - 2 * fee_cost, abs=1e-15)


# ---------------------------------------------------------------------------
# Paths that build close-to-close BY CONSTRUCTION must refuse any other mode,
# rather than silently ignore the key.
# ---------------------------------------------------------------------------
_NON_C2C = [TARGET_RETURN_TYPICAL_OVER_OPEN, TARGET_RETURN_TYPICAL_OVER_CLOSE]


def _ohlc():
    return pd.DataFrame({"open": [100.0, 100.0], "high": [100.1, 100.2],
                         "low": [99.9, 99.8], "close": [100.0, 100.1]})


class TestComputeLadderReturnsRefusesNonC2C:
    @pytest.mark.parametrize("mode", _NON_C2C)
    def test_non_c2c_mode_raises(self, mode):
        r = AgamottoResearch.__new__(AgamottoResearch)
        r.config = {"LADDER": 2, "LADDER_BPS": 1.0, "FEE": 0.0, "TARGET_RETURN_MODE": mode}
        with pytest.raises(ValueError, match=f"TARGET_RETURN_MODE='{mode}'"):
            r._compute_ladder_returns(_ohlc(), "close", "low", "high")

    @pytest.mark.parametrize("extra", [{}, {"TARGET_RETURN_MODE": TARGET_RETURN_CLOSE_TO_CLOSE}])
    def test_absent_or_c2c_still_works(self, extra):
        r = AgamottoResearch.__new__(AgamottoResearch)
        r.config = {"LADDER": 2, "LADDER_BPS": 1.0, "FEE": 0.0, **extra}
        out = r._compute_ladder_returns(_ohlc(), "close", "low", "high")
        assert out["return_long_raw"].iloc[0] == pytest.approx(
            compute_ladder_return(pd.Series([0.001]), pd.Series([2]), 1.0, "long").iloc[0])


def _orb(mode, target_tf):
    pytest.importorskip("orb.research", reason="orb package not installed")
    from orb.research import OrbResearch

    orb = OrbResearch.__new__(OrbResearch)
    orb.config = {"SYMBOLS": ["BINANCE_PERP_BTC_USDT"], "LADDER": 2, "LADDER_BPS": 1.0,
                  "FEE": 0.0, "TARGET_RETURN_MODE": mode}
    orb.timeframes = sorted({"15m", target_tf})
    orb.base_tf = "15m"
    orb.target_tf = target_tf
    orb.features = pd.DataFrame({
        "15m_BTCUSDT_close": [100.0],
        f"{target_tf}_BTCUSDT_exit_close": [99.9],
        f"{target_tf}_BTCUSDT_exit_low": [99.9],
        f"{target_tf}_BTCUSDT_exit_high": [100.0],
        "year": [2026], "month": [1],
    }, index=pd.date_range("2026-01-01", periods=1, freq="15min"))
    return orb


class TestOrbCrossTfRefusesNonC2C:
    @pytest.mark.parametrize("mode", _NON_C2C)
    def test_cross_tf_non_c2c_raises(self, mode):
        with pytest.raises(ValueError, match=f"TARGET_RETURN_MODE='{mode}'"):
            _orb(mode, "1h").verticalize()

    def test_cross_tf_c2c_still_builds_the_label(self):
        orb = _orb(TARGET_RETURN_CLOSE_TO_CLOSE, "1h")
        orb.verticalize()
        assert orb.vertical_features["return"].iloc[0] == pytest.approx(99.9 / 100.0 - 1.0)

    @pytest.mark.parametrize("mode", _NON_C2C)
    def test_same_tf_is_not_refused(self, mode):
        """Same-TF orb takes its labels from the per-TF AgamottoResearch, which
        honours the mode — the guard must not fire there (the three committed
        orb base+stock 1m arms run typical_over_open same-TF)."""
        _orb(mode, "15m").verticalize()
