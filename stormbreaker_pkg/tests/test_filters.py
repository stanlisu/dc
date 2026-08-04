"""Regression tests for Stormbreaker apply_filter fail-loud behavior.

Companion to mjolnir's 2026-07-24 named_filter hardening: an unknown/retired
name must fall through to the Unknown-filter raise regardless of df contents
— including an empty frame.

2026-08-04: the missing-column guards no longer return an all-True mask
either. That mask matched every row, i.e. a silent `baseline` regime, which
CLAUDE.md removed forever on 2026-06-18. It is especially easy to hit here:
`StormBreakerResearch._get_tf_view` hands back a frame with NO columns but
the SAME index when a cross-TF atom names a TF that has no `{tf}_` columns,
so one unbuilt context TF turned a whole regime unconditional across every
bar (marvel `stormbreaker/gauntlet/README.md`).
"""

import pandas as pd
import pytest

from stormbreaker.core.filters import (
    _ALL_TICK_FILTERS, _DEPTH_COLS, apply_filter)

# Every tick filter and the column(s) it reads. bid_heavy/ask_heavy accept a
# preference chain (any one suffices), so they are asserted separately.
FILTER_SOURCE_COLS = {
    "buy_flow": ("trade_imbalance",),
    "sell_flow": ("trade_imbalance",),
    "ofi_positive": ("ofi_agg",),
    "ofi_negative": ("ofi_agg",),
    "short_liq_spike": ("liq_burst_ratio", "liq_directional_imbalance"),
    "long_liq_spike": ("liq_burst_ratio", "liq_directional_imbalance"),
    "liq_spike": ("liq_burst_ratio",),
    "high_spread": ("relative_spread",),
    "low_spread": ("relative_spread",),
    "bid_heavy": _DEPTH_COLS,
    "ask_heavy": _DEPTH_COLS,
}

ALL_SOURCE_COLS = sorted({c for cols in FILTER_SOURCE_COLS.values()
                          for c in cols})


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


class TestMissingColumnRaises:
    """A known filter whose source column is absent must raise, not match
    every row."""

    def test_known_filter_missing_column_raises(self):
        df = pd.DataFrame({"foo": [1.0, 2.0]})
        with pytest.raises(ValueError) as exc:
            apply_filter(df, "buy_flow")
        msg = str(exc.value)
        assert "buy_flow" in msg and "trade_imbalance" in msg, msg

    def test_every_known_filter_raises_without_its_columns(self):
        df = pd.DataFrame({"foo": [1.0, 2.0]})
        for name in sorted(_ALL_TICK_FILTERS):
            with pytest.raises(ValueError) as exc:
                apply_filter(df, name)
            assert name in str(exc.value), name

    @pytest.mark.parametrize("name", sorted(FILTER_SOURCE_COLS))
    def test_absent_column_raises_rather_than_matching_every_row(self, name):
        # Frame carries every OTHER source column — only this filter's own
        # input is missing. The old behavior returned an all-True mask here.
        for missing in FILTER_SOURCE_COLS[name]:
            df = pd.DataFrame({
                c: [1.0, 2.0, 3.0] for c in ALL_SOURCE_COLS if c != missing
            })
            if name in ("bid_heavy", "ask_heavy"):
                # Preference chain: dropping ONE depth column is not enough,
                # the remaining ones still satisfy the filter.
                continue
            with pytest.raises(ValueError) as exc:
                apply_filter(df, name)
            msg = str(exc.value)
            assert name in msg and missing in msg, msg

    def test_depth_chain_raises_only_when_every_candidate_is_absent(self):
        # Any single depth column satisfies bid_heavy/ask_heavy...
        for present in _DEPTH_COLS:
            df = pd.DataFrame({present: [-1.0, 0.0, 1.0]})
            for name in ("bid_heavy", "ask_heavy"):
                mask = apply_filter(df, name)
                assert isinstance(mask, pd.Series) and len(mask) == 3
        # ...but none of them must raise, naming the whole chain.
        df = pd.DataFrame({"foo": [1.0, 2.0]})
        for name in ("bid_heavy", "ask_heavy"):
            with pytest.raises(ValueError) as exc:
                apply_filter(df, name)
            msg = str(exc.value)
            assert name in msg, msg
            assert all(c in msg for c in _DEPTH_COLS), msg

    def test_empty_column_frame_with_rows_raises(self):
        # The exact _get_tf_view shape: rows preserved, zero columns. This is
        # what silently produced a full-length all-True mask.
        df = pd.DataFrame(index=range(5))
        for name in sorted(_ALL_TICK_FILTERS):
            with pytest.raises(ValueError):
                apply_filter(df, name)

    def test_source_col_map_covers_every_known_filter(self):
        assert set(FILTER_SOURCE_COLS) == set(_ALL_TICK_FILTERS), (
            f"uncovered: {sorted(_ALL_TICK_FILTERS - set(FILTER_SOURCE_COLS))}")


class TestPresentColumnsStillDispatch:
    """Guard against over-correcting into 'always raises'."""

    def test_filters_still_produce_selective_masks(self):
        df = pd.DataFrame({
            "trade_imbalance": [-0.5, 0.0, 0.5],
            "ofi_agg": [-1.0, 0.0, 1.0],
            "liq_burst_ratio": [0.5, 1.0, 3.0],
            "liq_directional_imbalance": [-1.0, 0.0, 1.0],
            "relative_spread": [0.001, 0.002, 0.003],
            "depth_imbalance_L5": [-0.5, 0.0, 0.5],
        })
        for name in sorted(_ALL_TICK_FILTERS):
            mask = apply_filter(df, name)
            assert isinstance(mask, pd.Series), name
            assert len(mask) == 3, name

    def test_buy_flow_is_selective(self):
        df = pd.DataFrame({"trade_imbalance": [-0.5, 0.0, 0.5]})
        assert apply_filter(df, "buy_flow").tolist() == [False, False, True]
        assert apply_filter(df, "sell_flow").tolist() == [True, False, False]
