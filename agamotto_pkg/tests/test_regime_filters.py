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

2026-08-02 (separate, pre-existing): the three volume atoms resolved their
column via `df.get("quote_vol_ratio", df.get("vol_ratio", 1.0))`. On a frame
carrying neither column that collapsed to the scalar 1.0, so `low_volume` /
`high_volume` / `vol_breakout` returned a plain Python bool instead of a
per-row mask — the banned `cfg.get("KEY", X)`-shaped magic default. They now
raise via AgamottoResearch._volume_ratio. See TestVolumeRatioFailsLoud.
"""

import pandas as pd
import pytest

from agamotto.research import (
    AgamottoResearch,
    VOL_QUANTILE_FEATURES,
    VOL_Q_LEVELS,
    VOL_Q_WINDOW,
)
from agamotto.research_filters import (
    _VOL_QUANTILE_ATOMS,
    comprehensive_sweep_regimes,
    generate_regime_stack,
)

MVG_DEPENDENT = AgamottoResearch.MVG_DEPENDENT_FILTERS
COMBINED = (
    "combined_union", "combined_intersection",
    "combined_union_plus", "combined_union_alt",
)

# Every own-column atom, with the column(s) it reads. These must resolve without
# close/mvg1/mvg2 present. Doubles as a drift check: a new atom added to the
# if-chain without a price-column dependency belongs here.
#
# Values are `str | tuple[str, ...]`: the trailing vol-quantile atoms
# (2026-08-16) read TWO columns — `price_range_pct` and their own cutoff — and
# must raise when EITHER is absent, so a single-string value cannot express them.
OWN_COLUMN_ATOMS = {
    "high_vol": "price_range_pct",
    "low_vol": "price_range_pct",
    "high_vol_q80": ("price_range_pct", "price_range_pct_q80"),
    "high_vol_q90": ("price_range_pct", "price_range_pct_q90"),
    "high_vol_q95": ("price_range_pct", "price_range_pct_q95"),
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


def _own_columns(name: str) -> tuple:
    """Normalise an OWN_COLUMN_ATOMS value to a tuple of column names."""
    col = OWN_COLUMN_ATOMS[name]
    return (col,) if isinstance(col, str) else tuple(col)


# The atoms that resolve their column through AgamottoResearch._volume_ratio,
# i.e. accept EITHER quote_vol_ratio (preferred) or vol_ratio. Derived from
# OWN_COLUMN_ATOMS so the two can't drift apart.
VOLUME_ATOMS = tuple(
    sorted(n for n in OWN_COLUMN_ATOMS if _own_columns(n) == ("vol_ratio",)))


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
        # Trailing vol-quantile cutoffs (2026-08-16). Nested q80 <= q90 <= q95
        # row-wise, like the real trailing quantiles, and chosen so EVERY gate
        # comes out MIXED — never all-True (the banned baseline shape) and
        # never all-False (which would hide a broken comparison):
        #   q80 -> [T, F, T, T, T]   q90 -> [T, F, T, F, F]   q95 -> [T, F, F, F, F]
        "price_range_pct_q80": [0.05, 0.60, 0.90, 0.15, 4.0],
        "price_range_pct_q90": [0.08, 0.70, 0.95, 0.30, 6.0],
        "price_range_pct_q95": [0.09, 0.80, 1.50, 0.50, 7.0],
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
        ("high_vol_q80", "long", [True, False, True, True, True]),
        ("high_vol_q80", "short", [True, False, True, True, True]),
        ("high_vol_q90", "long", [True, False, True, False, False]),
        ("high_vol_q90", "short", [True, False, True, False, False]),
        ("high_vol_q95", "long", [True, False, False, False, False]),
        ("high_vol_q95", "short", [True, False, False, False, False]),
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


class TestVolumeRatioFailsLoud:
    """2026-08-02 (pre-existing, separate from the dispatch-order fix): with
    neither quote_vol_ratio nor vol_ratio present, the three volume atoms used
    `df.get("quote_vol_ratio", df.get("vol_ratio", 1.0))` and compared against
    the scalar 1.0, returning a plain bool rather than a per-row mask."""

    def test_volume_atoms_cover_the_expected_three(self):
        assert VOLUME_ATOMS == ("high_volume", "low_volume", "vol_breakout")

    @pytest.mark.parametrize("name", VOLUME_ATOMS)
    def test_missing_both_volume_columns_raises(self, research, name):
        df = _bare_frame().drop(columns=["vol_ratio"])
        for position in _positions(name):
            with pytest.raises(
                    ValueError, match="requires a volume-ratio column"):
                research._apply_filter_mask(df, name, position)

    @pytest.mark.parametrize("name", VOLUME_ATOMS)
    def test_reported_repro_frame_raises(self, research, name):
        """The exact frame from the 2026-08-02 report: price columns only, so
        the atom used to return the bare bool False for both positions."""
        df = pd.DataFrame({
            "close": [1.0, 2.0], "mvg1": [1.0, 1.0], "mvg2": [1.0, 1.0]})
        for position in _positions(name):
            with pytest.raises(
                    ValueError, match="requires a volume-ratio column"):
                research._apply_filter_mask(df, name, position)

    def test_error_names_both_accepted_columns(self, research):
        df = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="quote_vol_ratio"):
            research._apply_filter_mask(df, "vol_breakout", "long")
        with pytest.raises(ValueError, match="vol_ratio"):
            research._apply_filter_mask(df, "vol_breakout", "long")

    @pytest.mark.parametrize("name", VOLUME_ATOMS)
    def test_never_returns_a_scalar_bool(self, research, name):
        """Direct guard on the reported symptom: whatever comes back is a
        per-row Series, never a bool — raising counts as not returning one."""
        for df in (_bare_frame(), _full_frame(),
                   _bare_frame().drop(columns=["vol_ratio"])):
            for position in _positions(name):
                try:
                    got = research._apply_filter_mask(df, name, position)
                except ValueError:
                    continue
                assert isinstance(got, pd.Series), (name, position)
                assert len(got) == len(df)

    @pytest.mark.parametrize("name,expected", [
        ("low_volume", [True, True, False, False, False]),
        ("high_volume", [False, False, True, True, True]),
        ("vol_breakout", [False, False, False, True, True]),
    ])
    def test_vol_ratio_only_frame_still_dispatches(
            self, research, name, expected):
        """The common real shape: `volume` is a required load column so
        vol_ratio always exists, while `quote_volume` (hence quote_vol_ratio)
        is optional. Thresholds pinned: <1.0 / >1.0 / >2.0."""
        df = pd.DataFrame({"vol_ratio": [0.5, 0.9, 1.5, 2.5, 3.0]})
        assert research._apply_filter_mask(df, name, "long").tolist() == expected

    @pytest.mark.parametrize("name,expected", [
        ("low_volume", [True, True, False, False, False]),
        ("high_volume", [False, False, True, True, True]),
        ("vol_breakout", [False, False, False, True, True]),
    ])
    def test_quote_vol_ratio_only_frame_still_dispatches(
            self, research, name, expected):
        df = pd.DataFrame({"quote_vol_ratio": [0.5, 0.9, 1.5, 2.5, 3.0]})
        assert research._apply_filter_mask(df, name, "long").tolist() == expected

    @pytest.mark.parametrize("name", VOLUME_ATOMS)
    def test_quote_vol_ratio_takes_priority_over_vol_ratio(
            self, research, name):
        """Longstanding priority, unchanged by the helper extraction."""
        quote = [0.5, 0.9, 1.5, 2.5, 3.0]
        both = pd.DataFrame({
            "quote_vol_ratio": quote,
            "vol_ratio": [9.0, 9.0, 9.0, 0.1, 0.1],  # would flip every row
        })
        quote_only = pd.DataFrame({"quote_vol_ratio": quote})
        for position in _positions(name):
            got = research._apply_filter_mask(both, name, position)
            want = research._apply_filter_mask(quote_only, name, position)
            assert got.tolist() == want.tolist(), (name, position)

    def test_compound_and_propagates_the_raise(self, research):
        """Pre-fix, `False & Series` silently produced an all-False mask."""
        df = _bare_frame().drop(columns=["vol_ratio"])
        with pytest.raises(ValueError, match="requires a volume-ratio column"):
            research._apply_filter_mask(df, "vol_breakout_and_adx_trend", "long")

    def test_compound_or_propagates_the_raise(self, research):
        """Pre-fix, `False | Series` silently dropped the volume conjunct."""
        df = _bare_frame().drop(columns=["vol_ratio"])
        with pytest.raises(ValueError, match="requires a volume-ratio column"):
            research._apply_filter_mask(df, "vol_breakout_or_adx_trend", "long")

    def test_list_form_propagates_the_raise(self, research):
        df = _bare_frame().drop(columns=["vol_ratio"])
        with pytest.raises(ValueError, match="requires a volume-ratio column"):
            research._apply_filter_mask(
                df, ["high_volume", "&", "adx_trend"], "long")

    def test_unknown_name_unaffected_by_the_volume_guard(self, research):
        """A frame with no volume columns must still report unknown names as
        unknown, not as a missing-volume-column error."""
        df = _bare_frame().drop(columns=["vol_ratio"])
        with pytest.raises(ValueError, match="Unknown filter name"):
            research._apply_filter_mask(df, "definitely_not_a_filter", "long")


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


class TestOwnColumnMissingRaises:
    """2026-08-04 (dc #29 follow-up): an own-column atom whose source column
    is absent must RAISE, not return an all-True mask.

    An all-True mask there matches every row, i.e. `baseline` wearing another
    regime's name — the unconditional fires-on-every-bar regime CLAUDE.md
    removed forever on 2026-06-18. A renamed or unbuilt feature column
    silently reinstated it and inflated any screen that touched the regime.
    Mirrors mjolnir `core/regime_filters.py::_require_col` and stormbreaker
    `core/filters.py`.
    """

    @pytest.mark.parametrize("name", sorted(OWN_COLUMN_ATOMS))
    def test_absent_own_column_raises_rather_than_matching_every_row(
            self, research, name):
        # Start from the frame that has EVERY own column, then remove only
        # ONE of this atom's inputs at a time — so nothing else can explain
        # the raise, and a two-column atom is starved on each column in turn.
        for col in _own_columns(name):
            df = _bare_frame().drop(columns=[col])
            if name in VOLUME_ATOMS:
                # These resolve through _volume_ratio, which accepts either
                # quote_vol_ratio or vol_ratio — drop both to starve it.
                df = df.drop(columns=["quote_vol_ratio"], errors="ignore")
            for position in _positions(name):
                with pytest.raises(ValueError) as exc:
                    research._apply_filter_mask(df, name, position)
                msg = str(exc.value)
                assert name in msg, (name, position, col, msg)

    @pytest.mark.parametrize("name", sorted(OWN_COLUMN_ATOMS))
    def test_error_names_the_missing_column(self, research, name):
        if name in VOLUME_ATOMS:
            pytest.skip("_volume_ratio names the pair, covered by its own test")
        for col in _own_columns(name):
            df = _bare_frame().drop(columns=[col])
            for position in _positions(name):
                with pytest.raises(ValueError) as exc:
                    research._apply_filter_mask(df, name, position)
                assert col in str(exc.value), (name, position, str(exc.value))

    @pytest.mark.parametrize("name", sorted(OWN_COLUMN_ATOMS))
    def test_present_column_still_yields_a_mask(self, research, name):
        # Guard against over-correcting into "always raises".
        bare = _bare_frame()
        for position in _positions(name):
            mask = research._apply_filter_mask(bare, name, position)
            assert isinstance(mask, pd.Series), (name, position)
            assert len(mask) == len(bare), (name, position)

    def test_zero_column_frame_short_circuits_before_dispatch(self, research):
        # Rows but ZERO columns. pandas reports that frame as `.empty`, so
        # _apply_filter_mask's early `if df.empty` guard returns an empty
        # Series BEFORE any atom dispatch — the missing-column raise is never
        # reached. Pinned so the boundary is explicit rather than assumed.
        #
        # This is NOT the all-True baseline bug: the mask is length 0, not
        # all-True, so it cannot make a regime fire on every bar. It is a
        # separate pre-existing path that predates dc #29 and is deliberately
        # left alone by it.
        df = pd.DataFrame(index=range(5))
        assert df.empty  # the pandas property this whole test hinges on
        for name in sorted(OWN_COLUMN_ATOMS):
            for position in _positions(name):
                mask = research._apply_filter_mask(df, name, position)
                assert len(mask) == 0, (name, position)

    def test_composition_with_starved_atom_raises(self, research):
        # A compound must fail loud rather than composing an all-True conjunct.
        df = _bare_frame().drop(columns=["rsi"])
        with pytest.raises(ValueError, match="rsi"):
            research._apply_filter_mask(df, "rsi_oversold_and_adx_trend",
                                        "long")


class TestVolQuantileGate:
    """2026-08-16 — trailing vol-quantile ENTRY gates (Scope B).

    `high_vol` (r028) / `low_vol` (r038) split on the trailing MEDIAN, which is
    a 50/50 cut and appears in ZERO generated regime. These atoms cut at the
    trailing 80/90/95th percentile of `price_range_pct` instead, so the book
    only pays the fixed 4.48 bps round trip on bars whose forecast range can
    cover it. The predicate is position-INVARIANT (like high_vol): it says
    nothing about direction, only about whether the bar is worth entering.
    """

    ATOMS = ("high_vol_q80", "high_vol_q90", "high_vol_q95")

    def test_atom_map_is_exactly_the_three_gates(self):
        assert _VOL_QUANTILE_ATOMS == {
            "high_vol_q80": "price_range_pct_q80",
            "high_vol_q90": "price_range_pct_q90",
            "high_vol_q95": "price_range_pct_q95",
        }

    def test_feature_list_matches_levels(self):
        """The LITERAL feature list and the level tuple must agree.

        VOL_QUANTILE_FEATURES has to stay a literal list with a `_FEATURES`
        suffix: obfuscation/extract_inventory.py AST-scans list literals, so a
        comprehension over VOL_Q_LEVELS would be invisible to the extractor,
        get no feature code, and codec.encode_columns would then silently omit
        the columns — leaking the real names into the filter parquet.
        """
        assert len(VOL_QUANTILE_FEATURES) == len(VOL_Q_LEVELS)
        for name, level in zip(VOL_QUANTILE_FEATURES, VOL_Q_LEVELS):
            assert name == f"price_range_pct_q{int(round(level * 100))}"
        assert list(_VOL_QUANTILE_ATOMS.values()) == list(VOL_QUANTILE_FEATURES)

    def test_window_is_the_q50_window(self):
        assert VOL_Q_WINDOW == 700

    @pytest.mark.parametrize("name", ATOMS)
    def test_mask_is_position_invariant(self, research, name):
        bare = _bare_frame()
        long_mask = research._apply_filter_mask(bare, name, "long")
        short_mask = research._apply_filter_mask(bare, name, "short")
        assert long_mask.tolist() == short_mask.tolist()

    @pytest.mark.parametrize("name", ATOMS)
    def test_mask_is_mixed_never_all_true_or_all_false(self, research, name):
        mask = research._apply_filter_mask(_bare_frame(), name, "long")
        assert mask.any(), f"{name} fired on NO bar"
        assert not mask.all(), f"{name} fired on EVERY bar (baseline shape)"

    def test_gates_nest_by_level(self, research):
        """q95 ⊆ q90 ⊆ q80 — a higher cutoff can only fire on fewer bars."""
        bare = _bare_frame()
        q80, q90, q95 = (
            research._apply_filter_mask(bare, n, "long") for n in self.ATOMS)
        assert (q95 <= q90).all()
        assert (q90 <= q80).all()

    @pytest.mark.parametrize("name", ATOMS)
    def test_missing_cutoff_column_raises(self, research, name):
        """No rolling fallback: the cutoff is computed per SYMBOL on the wide
        frame, and a fallback here would run on the STACKED panel and cross
        symbol boundaries. Absent column must raise, never all-True."""
        col = _VOL_QUANTILE_ATOMS[name]
        df = _bare_frame().drop(columns=[col])
        for position in ("long", "short"):
            with pytest.raises(ValueError, match="requires column") as exc:
                research._apply_filter_mask(df, name, position)
            assert col in str(exc.value)

    @pytest.mark.parametrize("name", ATOMS)
    def test_missing_price_range_pct_raises(self, research, name):
        df = _bare_frame().drop(columns=["price_range_pct"])
        for position in ("long", "short"):
            with pytest.raises(ValueError, match="price_range_pct"):
                research._apply_filter_mask(df, name, position)

    @pytest.mark.parametrize("name", ATOMS)
    def test_codec_round_trip(self, name):
        """The extractor tripwire: an atom the AST scan never saw gets no code,
        so encode_regime raises and the atom can never reach a stack."""
        from agamotto._obf.codec import default
        c = default()
        assert c.decode_regime(c.encode_regime(name)) == name

    def test_atoms_are_in_the_generated_map(self):
        import json
        from pathlib import Path
        import agamotto
        map_path = Path(agamotto.__file__).parent / "_obf" / "map.json"
        regimes = json.loads(map_path.read_text())["regimes"]
        for name in self.ATOMS:
            assert name in regimes, name

    def test_sweep_carries_the_vol_gates(self):
        assert AgamottoResearch._SWEEP_VOL_FILTERS == [
            "low_volume", "high_volume", "vol_breakout",
            "high_vol", "high_vol_q80", "high_vol_q90", "high_vol_q95",
        ]

    def test_comprehensive_sweep_count(self):
        # 7 vol + 14 tech + 7*14 crosses
        assert len(comprehensive_sweep_regimes()) == 119

    def test_base_regimes_scope_b_counts(self):
        from agamotto.research_filters import BASE_REGIMES
        # 33 ungated + 33*3 volume-parented gated (Scope B)
        #  + 3 bare gates + 3*14 tech-crossed gates (Scope C) = 177
        assert len(BASE_REGIMES) == 177
        assert len(set(BASE_REGIMES)) == 177   # no duplicates
        assert len(generate_regime_stack()) == 299   # 224 (Scope B) + 75

    def test_every_gated_regime_is_a_parent_plus_one_gate(self):
        from agamotto.research_filters import (
            BASE_REGIMES, _BASE_REGIMES_UNGATED, _SWEEP_TECH_FILTERS)
        ungated = [r for r in BASE_REGIMES
                   if not any(r.endswith(f"_and_{a}") for a in self.ATOMS)
                   and r not in self.ATOMS]
        assert ungated == _BASE_REGIMES_UNGATED
        assert len(ungated) == 33
        # Scope B parents (volume-parented) then Scope C parents (tech atoms).
        for atom in self.ATOMS:
            gated = [r for r in BASE_REGIMES if r.endswith(f"_and_{atom}")]
            parents = [r[: -len(f"_and_{atom}")] for r in gated]
            assert parents == _BASE_REGIMES_UNGATED + _SWEEP_TECH_FILTERS

    def test_scope_c_bare_atoms_and_tech_crosses_present(self):
        """Scope C: the gate WITHOUT a volume parent.

        Every one of the 33 Scope-B parents is itself parented on
        high_volume / low_volume / vol_breakout. Volume and range co-move, so
        in conjunction the nominal-5% q95 gate fired on 14.71% of pooled bars.
        These 45 names carry no volume parent at all.
        """
        from agamotto.research_filters import BASE_REGIMES, _SWEEP_TECH_FILTERS
        for atom in self.ATOMS:
            assert atom in BASE_REGIMES, atom
            for tech in _SWEEP_TECH_FILTERS:
                assert f"{tech}_and_{atom}" in BASE_REGIMES
        # …and no volume atom leaks into a Scope C name.
        scope_c = list(self.ATOMS) + [
            f"{t}_and_{a}" for a in self.ATOMS for t in _SWEEP_TECH_FILTERS]
        assert len(scope_c) == 45
        for name in scope_c:
            for vol in ("low_volume", "high_volume", "vol_breakout"):
                assert vol not in name, name

    def test_bare_gate_is_position_invariant(self):
        from agamotto.research_filters import allowed_positions
        for atom in self.ATOMS:
            assert allowed_positions(atom) == ["long", "short"]

    @pytest.mark.parametrize("name", ATOMS)
    def test_tech_cross_equals_and_of_tech_and_gate(self, research, name):
        full = _full_frame()
        for position in ("long", "short"):
            composite = research._apply_filter_mask(
                full, f"adx_trend_and_{name}", position)
            tech = research._apply_filter_mask(full, "adx_trend", position)
            gate = research._apply_filter_mask(full, name, position)
            assert composite.tolist() == (tech & gate).tolist()

    def test_scope_c_codec_round_trip(self):
        """No map regeneration needed: the atoms already carry codes from
        _SWEEP_VOL_FILTERS, and the tech atoms from _SWEEP_TECH_FILTERS."""
        from agamotto._obf.codec import default
        from agamotto.research_filters import BASE_REGIMES
        c = default()
        for name in BASE_REGIMES:
            assert c.decode_regime(c.encode_regime(name)) == name

    def test_no_baseline_anywhere(self):
        from agamotto.research_filters import BASE_REGIMES
        for r in BASE_REGIMES:
            assert "baseline" not in r, r
        for r in comprehensive_sweep_regimes():
            assert "baseline" not in r, r

    def test_gated_composite_equals_and_of_parent_and_gate(self, research):
        full = _full_frame()
        composite = research._apply_filter_mask(
            full, "vol_breakout_and_strong_trend_and_high_vol_q80", "long")
        parent = research._apply_filter_mask(
            full, "vol_breakout_and_strong_trend", "long")
        gate = research._apply_filter_mask(full, "high_vol_q80", "long")
        assert composite.tolist() == (parent & gate).tolist()
        assert composite.sum() <= parent.sum()
