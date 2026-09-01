"""orb must CARRY agamotto's vol-quantile cutoffs, and must NOT model them.

agamotto's `high_vol_q80/q90/q95` atoms compare `price_range_pct` against a
per-symbol trailing cutoff (agamotto/research_filters.py:102-104). orb computes
those cutoffs already — engineer_features() delegates per TF to
AgamottoResearch — but before this fix neither carry list named them, so
verticalize() never emitted them and _remap_for_tf's `filter_cols` never exposed
the bare alias. _apply_filter_mask raised

    Filter 'high_vol_q80' requires column 'price_range_pct_q80'

on 144 of agamotto's 177 regimes (243 of 299 stack rows) — the same failure
SHAPE as the seven scale-free twins (see test_orb_verticalize_features.py):
computed by the engine, then dropped by a list that did not name them.
"""
import pytest

from agamotto.research import VOL_QUANTILE_FEATURES, VOL_Q_LEVELS
import orb.research as orb_research


def test_vol_quantiles_are_carried_for_filters():
    """They must reach the panel, via _RAW_COLUMNS."""
    missing = [f for f in VOL_QUANTILE_FEATURES
               if f not in orb_research._RAW_COLUMNS]
    assert not missing, (
        f"{missing} absent from orb's _RAW_COLUMNS — verticalize() steps 2 and 3 "
        f"iterate that list, so every high_vol_q* regime becomes unevaluable.")


def test_vol_quantiles_are_not_model_features():
    """They GATE ENTRY. _DERIVED_FEATURES would mint TF-prefixed ML copies.

    marvel gauntlet/rolling_predict_returns.py:1358 excludes all three by name
    and by TF-stripped/obfuscated alias precisely because they are cutoffs, not
    inputs. Their sibling price_range_pct_q50 IS a legitimate model feature and
    stays where it is.
    """
    leaked = [f for f in VOL_QUANTILE_FEATURES
              if f in orb_research._DERIVED_FEATURES]
    assert not leaked, (
        f"{leaked} leaked into _DERIVED_FEATURES — three near-collinear trailing "
        f"quantiles of price_range_pct would displace incumbents on |IC| and "
        f"break weight parity with every deployed leg.")
    assert "price_range_pct_q50" in orb_research._DERIVED_FEATURES


def test_filter_cols_exposes_the_bare_alias():
    """`filter_cols` (research.py:717) is what _remap_for_tf remaps through."""
    filter_cols = set(orb_research._RAW_COLUMNS) | set(orb_research._DERIVED_FEATURES)
    for feat in VOL_QUANTILE_FEATURES:
        assert feat in filter_cols, f"{feat} unreachable by _apply_filter_mask"


def test_carry_list_is_appended_from_canonical_not_retyped():
    """A second hardcoded copy drifting from agamotto's list is the defect.

    Order must follow VOL_Q_LEVELS, and the tail of _RAW_COLUMNS must BE the
    canonical list — asserting equality catches a re-typed literal that happens
    to hold the same names today but will not track a future level change.
    """
    assert len(VOL_QUANTILE_FEATURES) == len(VOL_Q_LEVELS)
    tail = orb_research._RAW_COLUMNS[-len(VOL_QUANTILE_FEATURES):]
    assert tail == list(VOL_QUANTILE_FEATURES), (
        f"tail {tail} != canonical {list(VOL_QUANTILE_FEATURES)}")


@pytest.mark.parametrize("feat", list(VOL_QUANTILE_FEATURES))
def test_tripwire_removing_one_makes_it_unreachable(monkeypatch, feat):
    """Guard the guard: prove the assertion above would actually fail."""
    pruned = [c for c in orb_research._RAW_COLUMNS if c != feat]
    monkeypatch.setattr(orb_research, "_RAW_COLUMNS", pruned)
    filter_cols = set(orb_research._RAW_COLUMNS) | set(orb_research._DERIVED_FEATURES)
    assert feat not in filter_cols
