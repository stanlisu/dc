"""Regression tests for regime_filters fail-loud behavior.

2026-07-24: named_filter's price-column guard (mvg1/mvg2/close missing ->
all-True) used to swallow UNKNOWN filter names too: a stale stack referencing
a removed regime (e.g. oi_expansion) raised on real feature frames but
silently fired on every bar of a degenerate/warmup frame. Unknown names must
now raise regardless of df contents; the missing-column all-True guards apply
to KNOWN names only.

2026-08-01: the same shared guard sat ABOVE the own-column price filters
(high_vol, low_vol, rsi_*, macd_*, adx_trend, *_volume, mom_positive), so on
frames lacking mvg1/mvg2/close they fired on EVERY bar even when their own
column was present (high_vol AND low_vol simultaneously all-True — the banned
baseline-shaped always-on pattern). Own-column filters now dispatch before
the price-column check, and the MVG-dependent filters (trend_aligned,
strong_trend, ma_momentum, bb_rebound) RAISE on missing price columns
instead of returning all-True.
"""

import pandas as pd
import pytest

from mjolnir.core.regime_filters import KNOWN_FILTERS, apply_filter_mask, named_filter

# Filters that genuinely read mvg1/mvg2/close — missing price columns raise.
MVG_DEPENDENT = frozenset(
    {"trend_aligned", "strong_trend", "ma_momentum", "bb_rebound"})


def _degenerate_frame() -> pd.DataFrame:
    """Warmup-style frame with rows but none of mvg1/mvg2/close/mid_price."""
    return pd.DataFrame({"foo": [1.0, 2.0, 3.0]})


class TestUnknownNameRaises:
    def test_unknown_name_missing_mvg_columns_raises(self):
        # The original bug: unknown name + frame lacking price columns
        # returned an all-True mask instead of raising.
        with pytest.raises(ValueError, match="Unknown filter"):
            named_filter(_degenerate_frame(), "no_such_filter", "long")

    def test_retired_oi_regimes_raise_on_degenerate_frame(self):
        # oi_expansion/oi_contraction removed 2026-07-24 (dc PR #18); a stale
        # stack referencing them must fail loud even on a warmup frame.
        for name in ("oi_expansion", "oi_contraction"):
            with pytest.raises(ValueError, match="Unknown filter"):
                named_filter(_degenerate_frame(), name, "long")

    def test_unknown_name_empty_frame_raises(self):
        with pytest.raises(ValueError, match="Unknown filter"):
            named_filter(pd.DataFrame(), "no_such_filter", "long")

    def test_unknown_name_full_frame_raises(self):
        df = pd.DataFrame({"mvg1": [1.0], "mvg2": [1.0], "close": [1.0]})
        with pytest.raises(ValueError, match="Unknown filter"):
            named_filter(df, "no_such_filter", "long")

    def test_apply_filter_mask_unknown_name_degenerate_frame_raises(self):
        # The stack-consumption entry point must propagate the raise.
        with pytest.raises(ValueError, match="Unknown filter"):
            apply_filter_mask(_degenerate_frame(), "no_such_filter", "long")

    def test_unknown_conjunct_in_compound_name_raises(self):
        with pytest.raises(ValueError, match="Unknown filter"):
            apply_filter_mask(
                _degenerate_frame(), "wide_spread_and_oi_expansion", "long")


class TestKnownNameGuardsUnchanged:
    def test_microstructure_filter_missing_column_returns_all_true(self):
        mask = named_filter(_degenerate_frame(), "funding_positive", "long")
        assert mask.all() and len(mask) == 3

    def test_baseline_still_raises_dedicated_removal_error(self):
        with pytest.raises(ValueError, match="baseline regime removed"):
            named_filter(_degenerate_frame(), "baseline", "long")

    def test_every_known_filter_dispatches_on_degenerate_frame(self):
        # KNOWN_FILTERS and the branch chain must not drift: every listed name
        # (except baseline, which raises by design) must reach a branch —
        # own-column filters return a mask, MVG-dependent filters raise the
        # missing-price-columns error. Nothing may hit the
        # has-no-implementation drift raise.
        df = _degenerate_frame()
        for name in sorted(KNOWN_FILTERS - {"baseline"}):
            for position in ("long", "short"):
                if name in MVG_DEPENDENT:
                    with pytest.raises(
                            ValueError, match="requires price columns"):
                        named_filter(df, name, position)
                else:
                    mask = named_filter(df, name, position)
                    assert isinstance(mask, pd.Series), name


class TestMvgGuardFailsLoud:
    """2026-08-01: missing mvg1/mvg2/close must raise, never all-True."""

    def test_mvg_dependent_filter_missing_price_columns_raises(self):
        for name in sorted(MVG_DEPENDENT):
            with pytest.raises(ValueError, match="requires price columns"):
                named_filter(_degenerate_frame(), name, "long")

    def test_error_names_the_missing_columns(self):
        df = pd.DataFrame({"mvg1": [1.0, 2.0], "close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="mvg2"):
            named_filter(df, "trend_aligned", "long")

    def test_full_price_frame_still_dispatches(self):
        df = pd.DataFrame({
            "mvg1": [1.0, 2.0], "mvg2": [2.0, 1.0], "close": [3.0, 3.0]})
        mask = named_filter(df, "trend_aligned", "long")
        assert mask.tolist() == [False, True]


class TestOwnColumnFiltersNotGatedOnPriceColumns:
    """2026-08-01 regression: own-column filters must use their own column
    even when mvg1/mvg2/close are absent — previously they fired on every
    bar of such frames (high_vol AND low_vol simultaneously all-True)."""

    def test_high_low_vol_use_own_column_without_price_columns(self):
        vals = [0.1, 0.5, 1.0, 2.0, 5.0]
        bare = pd.DataFrame({"price_range_pct": vals})
        # Control: identical values WITH price columns present.
        full = bare.assign(mvg1=1.0, mvg2=1.0, close=1.0)
        for name in ("high_vol", "low_vol"):
            got = named_filter(bare, name, "long")
            want = named_filter(full, name, "long")
            assert got.tolist() == want.tolist(), name
            assert not got.all(), f"{name} fired on every bar"
        # And the two must never both be all-True (the observed failure).
        hv = named_filter(bare, "high_vol", "long")
        lv = named_filter(bare, "low_vol", "long")
        assert not (hv.all() and lv.all())

    @pytest.mark.parametrize("name,col,vals,expected", [
        ("rsi_oversold", "rsi", [10.0, 50.0, 90.0], [True, False, False]),
        ("rsi_overbought", "rsi", [10.0, 50.0, 90.0], [False, False, True]),
        ("macd_bullish", "macdhist", [-1.0, 1.0], [False, True]),
        ("adx_trend", "adx", [10.0, 30.0], [False, True]),
        ("high_volume", "vol_ratio", [0.5, 1.5], [False, True]),
        ("vol_breakout", "vol_ratio", [1.5, 2.5], [False, True]),
        ("low_volume", "vol_ratio", [0.5, 1.5], [True, False]),
        ("mom_positive", "mom", [-1.0, 1.0], [False, True]),
    ])
    def test_own_column_filter_selective_without_price_columns(
            self, name, col, vals, expected):
        df = pd.DataFrame({col: vals})
        mask = named_filter(df, name, "long")
        assert mask.tolist() == expected
