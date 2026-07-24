"""Regression tests for regime_filters fail-loud behavior.

2026-07-24: named_filter's price-column guard (mvg1/mvg2/close missing ->
all-True) used to swallow UNKNOWN filter names too: a stale stack referencing
a removed regime (e.g. oi_expansion) raised on real feature frames but
silently fired on every bar of a degenerate/warmup frame. Unknown names must
now raise regardless of df contents; the missing-column all-True guards apply
to KNOWN names only.
"""

import pandas as pd
import pytest

from mjolnir.core.regime_filters import KNOWN_FILTERS, apply_filter_mask, named_filter


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
    def test_price_filter_missing_mvg_columns_returns_all_true(self):
        mask = named_filter(_degenerate_frame(), "trend_aligned", "long")
        assert mask.all() and len(mask) == 3

    def test_microstructure_filter_missing_column_returns_all_true(self):
        mask = named_filter(_degenerate_frame(), "funding_positive", "long")
        assert mask.all() and len(mask) == 3

    def test_baseline_still_raises_dedicated_removal_error(self):
        with pytest.raises(ValueError, match="baseline regime removed"):
            named_filter(_degenerate_frame(), "baseline", "long")

    def test_every_known_filter_dispatches_on_degenerate_frame(self):
        # KNOWN_FILTERS and the branch chain must not drift: every listed name
        # (except baseline, which raises by design) must reach a branch and
        # return a mask — never the has-no-implementation drift raise.
        df = _degenerate_frame()
        for name in sorted(KNOWN_FILTERS - {"baseline"}):
            for position in ("long", "short"):
                mask = named_filter(df, name, position)
                assert isinstance(mask, pd.Series), name
