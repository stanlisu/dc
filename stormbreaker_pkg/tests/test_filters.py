"""Regression tests for Stormbreaker apply_filter fail-loud behavior.

Companion to mjolnir's 2026-07-24 named_filter hardening: apply_filter's
missing-column true_mask guards live inside known-name branches only, so an
unknown/retired name must fall through to the Unknown-filter raise regardless
of df contents — including an empty frame.
"""

import pandas as pd
import pytest

from stormbreaker.core.filters import _ALL_TICK_FILTERS, apply_filter


class TestUnknownNameRaises:
    def test_unknown_name_empty_frame_raises(self):
        with pytest.raises(ValueError, match="Unknown Stormbreaker filter"):
            apply_filter(pd.DataFrame(), "no_such_filter")

    def test_unknown_name_frame_without_columns_raises(self):
        df = pd.DataFrame({"foo": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Unknown Stormbreaker filter"):
            apply_filter(df, "no_such_filter")

    def test_retired_oi_expanding_raises(self):
        # oi_expanding removed 2026-07-24 with mjolnir's oi_velocity atoms.
        with pytest.raises(ValueError, match="Unknown Stormbreaker filter"):
            apply_filter(pd.DataFrame({"foo": [1.0]}), "oi_expanding")


class TestKnownNameGuardsUnchanged:
    def test_known_filter_missing_column_returns_all_true(self):
        df = pd.DataFrame({"foo": [1.0, 2.0]})
        mask = apply_filter(df, "buy_flow")
        assert mask.all() and len(mask) == 2

    def test_every_known_filter_dispatches_without_columns(self):
        df = pd.DataFrame({"foo": [1.0, 2.0]})
        for name in sorted(_ALL_TICK_FILTERS):
            mask = apply_filter(df, name)
            assert isinstance(mask, pd.Series), name
