"""Regression tests for AgamottoResearch._apply_filter_mask dispatch order.

2026-08-02: the shared `required_base = ["close", "mvg1", "mvg2"]` guard sat
ABOVE every own-column price filter (high_vol, low_vol, strong_candle, rsi_*,
macd_*, stoch_bullish, cci_reversal, adx_trend, *_volume, vol_breakout,
mom_positive, buy_pressure, mfi_*, bop_*, roc_*), so on frames lacking the
price columns they logged a warning and fired on EVERY bar even when their own
column was present — high_vol AND low_vol simultaneously all-True, the banned
baseline-shaped always-on pattern (marvel CLAUDE.md "baseline is REMOVED
forever" / no-silent-fallback).

Own-column filters now dispatch BEFORE the price-column check, and the
genuinely price-dependent filters (AgamottoResearch.MVG_DEPENDENT_FILTERS plus
the `combined_*` composites) RAISE on missing close/mvg1/mvg2 instead of
returning all-True. Mirrors the mjolnir fix (b5ea04a,
mjolnir/core/regime_filters.py::named_filter) — see
mjolnir_pkg/tests/test_regime_filters.py.
"""

import pandas as pd
import pytest

from agamotto.research import AgamottoResearch

MVG_DEPENDENT = AgamottoResearch.MVG_DEPENDENT_FILTERS
COMBINED = (
    "combined_union", "combined_intersection",
    "combined_union_plus", "combined_union_alt",
)

# Every own-column atom, with the column it reads. These must resolve without
# close/mvg1/mvg2 present. Doubles as a drift check: a new atom added to the
# if-chain without a price-column dependency belongs here.
OWN_COLUMN_ATOMS = {
    "high_vol": "price_range_pct",
    "low_vol": "price_range_pct",
    "strong_candle": "open_close_pct",
    "rsi_oversold": "rsi",
    "rsi_overbought": "rsi",
    "macd_bullish": "macdhist",
    "macd_bearish": "macdhist",
    "stoch_bullish": "stoch_k",
    "cci_reversal": "cci",
    "adx_trend": "adx",
    "mom_positive": "mom",
    "low_volume": "vol_ratio",
    "high_volume": "vol_ratio",
    "vol_breakout": "vol_ratio",
    "buy_pressure": "buy_pressure",
    "mfi_oversold": "mfi",
    "mfi_overbought": "mfi",
    "bop_bullish": "bop",
    "bop_bearish": "bop",
    "roc_positive": "roc",
    "roc_negative": "roc",
}


@pytest.fixture
def research():
    """__init__ only stores config/home_root — no I/O, no TA-Lib."""
    return AgamottoResearch({}, "/tmp")


def _bare_frame() -> pd.DataFrame:
    """Every own-column feature present, none of close/mvg1/mvg2.

    Shaped like a warmup/degenerate frame: the own-column filters must still
    discriminate on it rather than returning all-True.
    """
    return pd.DataFrame({
        # Rolling-700 (expanding, min_periods=1) median: [.1, .3, .5, .35, .5]
        # — chosen so BOTH high_vol and low_vol come out mixed, not all-True.
        "price_range_pct": [0.1, 0.5, 1.0, 0.2, 5.0],
        "open_close_pct": [-0.01, -0.001, 0.0, 0.001, 0.01],
        "rsi": [10.0, 25.0, 50.0, 75.0, 90.0],
        "macdhist": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "stoch_k": [10.0, 60.0, 30.0, 80.0, 50.0],
        "stoch_d": [20.0, 50.0, 40.0, 70.0, 50.0],
        "cci": [-150.0, -50.0, 0.0, 50.0, 150.0],
        "adx": [10.0, 20.0, 25.0, 30.0, 40.0],
        "mom": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "vol_ratio": [0.5, 0.9, 1.5, 2.5, 3.0],
        "buy_pressure": [0.2, 0.4, 0.5, 0.6, 0.8],
        "mfi": [10.0, 25.0, 50.0, 75.0, 90.0],
        "bop": [-0.5, -0.2, 0.0, 0.2, 0.5],
        "roc": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "sar": [1.0, 1.0, 1.0, 1.0, 1.0],
        "bb_lower": [1.0, 1.0, 1.0, 1.0, 1.0],
        "bb_upper": [9.0, 9.0, 9.0, 9.0, 9.0],
    })


def _full_frame() -> pd.DataFrame:
    """Control: identical own-column values WITH the price columns present."""
    return _bare_frame().assign(
        close=[3.0, 3.0, 3.0, 3.0, 3.0],
        mvg1=[1.0, 2.0, 3.0, 4.0, 5.0],
        mvg2=[2.0, 2.0, 2.0, 4.0, 4.0],
        mvg3=[1.5, 1.5, 1.5, 4.5, 4.5],
    )


def _positions(name: str) -> list:
    return AgamottoResearch.allowed_positions(name)


class TestOwnColumnFiltersNotGatedOnPriceColumns:
    """2026-08-02 regression: own-column atoms must read their own column even
    when close/mvg1/mvg2 are absent — previously they fired on every bar."""

    def test_high_low_vol_use_own_column_without_price_columns(self, research):
        bare, full = _bare_frame(), _full_frame()
        for name in ("high_vol", "low_vol"):
            for position in ("long", "short"):
                got = research._apply_filter_mask(bare, name, position)
                want = research._apply_filter_mask(full, name, position)
                assert got.tolist() == want.tolist(), (name, position)
                assert not got.all(), f"{name}/{position} fired on every bar"

    def test_high_vol_and_low_vol_never_both_all_true(self, research):
        """The observed failure: both sides of the same split all-True."""
        bare = _bare_frame()
        hv = research._apply_filter_mask(bare, "high_vol", "long")
        lv = research._apply_filter_mask(bare, "low_vol", "long")
        assert not (hv.all() and lv.all())

    @pytest.mark.parametrize("name", sorted(OWN_COLUMN_ATOMS))
    def test_own_column_mask_matches_full_frame_control(self, research, name):
        """Bare-frame mask must equal the full-frame mask, atom by atom."""
        bare, full = _bare_frame(), _full_frame()
        for position in _positions(name):
            got = research._apply_filter_mask(bare, name, position)
            want = research._apply_filter_mask(full, name, position)
            assert isinstance(got, pd.Series), (name, position)
            assert got.tolist() == want.tolist(), (name, position)

    @pytest.mark.parametrize("name,position,expected", [
        ("high_vol", "long", [False, True, True, False, True]),
        ("low_vol", "long", [False, False, False, True, False]),
        ("strong_candle", "long", [False, False, False, False, True]),
        ("strong_candle", "short", [True, False, False, False, False]),
        ("rsi_oversold", "long", [True, True, False, False, False]),
        ("rsi_overbought", "short", [False, False, False, True, True]),
        ("macd_bullish", "long", [False, False, False, True, True]),
        ("macd_bearish", "short", [True, True, False, False, False]),
        ("stoch_bullish", "long", [False, True, False, True, False]),
        ("cci_reversal", "long", [False, False, False, False, True]),
        ("cci_reversal", "short", [True, False, False, False, False]),
        ("adx_trend", "long", [False, False, False, True, True]),
        ("mom_positive", "long", [False, False, False, True, True]),
        ("mom_positive", "short", [True, True, False, False, False]),
        ("low_volume", "long", [True, True, False, False, False]),
        ("high_volume", "long", [False, False, True, True, True]),
        ("vol_breakout", "long", [False, False, False, True, True]),
        ("buy_pressure", "long", [False, False, False, True, True]),
        ("buy_pressure", "short", [True, True, False, False, False]),
        ("mfi_oversold", "long", [True, True, False, False, False]),
        ("mfi_overbought", "short", [False, False, False, True, True]),
        ("bop_bullish", "long", [False, False, False, True, True]),
        ("bop_bearish", "short", [True, True, False, False, False]),
        ("roc_positive", "long", [False, False, False, True, True]),
        ("roc_negative", "short", [True, True, False, False, False]),
    ])
    def test_own_column_filter_selective_without_price_columns(
            self, research, name, position, expected):
        mask = research._apply_filter_mask(_bare_frame(), name, position)
        assert mask.tolist() == expected


class TestMvgGuardFailsLoud:
    """2026-08-02: missing close/mvg1/mvg2 must raise, never all-True."""

    @pytest.mark.parametrize("name", sorted(MVG_DEPENDENT))
    def test_mvg_dependent_filter_missing_price_columns_raises(
            self, research, name):
        for position in _positions(name):
            with pytest.raises(ValueError, match="requires price columns"):
                research._apply_filter_mask(_bare_frame(), name, position)

    @pytest.mark.parametrize("name", COMBINED)
    def test_combined_composites_missing_price_columns_raise(
            self, research, name):
        with pytest.raises(ValueError, match="requires price columns"):
            research._apply_filter_mask(_bare_frame(), name, "short")

    def test_error_names_the_missing_columns(self, research):
        df = pd.DataFrame({"mvg1": [1.0, 2.0], "close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="mvg2"):
            research._apply_filter_mask(df, "trend_aligned", "long")

    def test_partial_price_columns_still_raise(self, research):
        """close present, mvg1/mvg2 absent — bb_rebound/sar_aligned only read
        close, but the guard is shared with the rest of the group (mirrors
        mjolnir) and must fail loud rather than silently pass."""
        df = _bare_frame().assign(close=[3.0] * 5)
        for name in ("bb_rebound", "sar_aligned"):
            with pytest.raises(ValueError, match="requires price columns"):
                research._apply_filter_mask(df, name, "long")

    @pytest.mark.parametrize("name", sorted(MVG_DEPENDENT))
    def test_full_price_frame_still_dispatches(self, research, name):
        full = _full_frame()
        for position in _positions(name):
            mask = research._apply_filter_mask(full, name, position)
            assert isinstance(mask, pd.Series), (name, position)
            assert len(mask) == len(full)

    def test_trend_aligned_math_unchanged(self, research):
        df = pd.DataFrame({
            "mvg1": [1.0, 2.0], "mvg2": [2.0, 1.0], "close": [3.0, 3.0]})
        assert research._apply_filter_mask(
            df, "trend_aligned", "long").tolist() == [False, True]
        assert research._apply_filter_mask(
            df, "trend_aligned", "short").tolist() == [False, False]


class TestRollingQuantileMathUnchanged:
    """high_vol/low_vol use a rolling 700-bar median, not a full-frame one.
    Pinned so the dispatch move can't quietly swap the math."""

    def test_rolling_700_median_used_when_q50_column_absent(self, research):
        vals = [1.0, 2.0, 3.0, 4.0, 100.0]
        df = pd.DataFrame({"price_range_pct": vals})
        want = pd.Series(vals) > pd.Series(vals).rolling(
            700, min_periods=1).quantile(0.5)
        got = research._apply_filter_mask(df, "high_vol", "long")
        assert got.tolist() == want.tolist()

    def test_precomputed_q50_column_takes_priority(self, research):
        df = pd.DataFrame({
            "price_range_pct": [1.0, 2.0, 3.0],
            "price_range_pct_q50": [5.0, 5.0, 0.0],
        })
        assert research._apply_filter_mask(
            df, "high_vol", "long").tolist() == [False, False, True]
        assert research._apply_filter_mask(
            df, "low_vol", "long").tolist() == [True, True, False]


class TestNoDispatchDrift:
    """Every known atom must reach a branch: own-column atoms return a mask,
    price-dependent atoms raise the missing-price-columns error. Nothing may
    fall through to the 'Unknown filter name' raise."""

    def test_every_known_atom_dispatches_on_bare_frame(self, research):
        atoms = (
            set(OWN_COLUMN_ATOMS)
            | set(MVG_DEPENDENT)
            | set(COMBINED)
            | set(AgamottoResearch.LONG_ONLY_FILTERS)
            | set(AgamottoResearch.SHORT_ONLY_FILTERS)
            | set(AgamottoResearch._SWEEP_VOL_FILTERS)
            | set(AgamottoResearch._SWEEP_TECH_FILTERS)
        )
        bare = _bare_frame()
        for name in sorted(atoms):
            for position in _positions(name):
                if name in MVG_DEPENDENT or name.startswith("combined_"):
                    with pytest.raises(
                            ValueError, match="requires price columns"):
                        research._apply_filter_mask(bare, name, position)
                else:
                    mask = research._apply_filter_mask(bare, name, position)
                    assert isinstance(mask, pd.Series), (name, position)

    def test_every_known_atom_dispatches_on_full_frame(self, research):
        atoms = (
            set(OWN_COLUMN_ATOMS) | set(MVG_DEPENDENT)
            | set(AgamottoResearch._SWEEP_TECH_FILTERS)
        )
        full = _full_frame()
        for name in sorted(atoms):
            for position in _positions(name):
                mask = research._apply_filter_mask(full, name, position)
                assert isinstance(mask, pd.Series), (name, position)

    @pytest.mark.parametrize("name", COMBINED)
    def test_combined_composites_dispatch_on_full_frame(self, research, name):
        """`combined_*` is implemented on the short side only — pre-existing
        (pre-2026-08-02) and unchanged here; long still hits the unknown-name
        raise. Pinned so the dispatch move can't be blamed for it later."""
        full = _full_frame()
        mask = research._apply_filter_mask(full, name, "short")
        assert isinstance(mask, pd.Series) and len(mask) == len(full)
        with pytest.raises(ValueError, match="Unknown filter name"):
            research._apply_filter_mask(full, name, "long")

    def test_unknown_name_still_raises_unknown_filter(self, research):
        """The price-column check must not swallow the unknown-name path."""
        for df in (_bare_frame(), _full_frame()):
            with pytest.raises(ValueError, match="Unknown filter name"):
                research._apply_filter_mask(df, "definitely_not_a_filter",
                                            "long")


class TestBaseRegimeCompositesUnaffected:
    """BASE_REGIMES composites split on `_and_` and AND the atom masks; the
    dispatch move must not change how a composite resolves on a full frame."""

    def test_composite_equals_manual_and_of_atoms(self, research):
        full = _full_frame()
        composite = research._apply_filter_mask(
            full, "vol_breakout_and_strong_trend", "long")
        a = research._apply_filter_mask(full, "vol_breakout", "long")
        b = research._apply_filter_mask(full, "strong_trend", "long")
        assert composite.tolist() == (a & b).tolist()

    def test_composite_with_mvg_atom_raises_on_bare_frame(self, research):
        with pytest.raises(ValueError, match="requires price columns"):
            research._apply_filter_mask(
                _bare_frame(), "vol_breakout_and_strong_trend", "long")
