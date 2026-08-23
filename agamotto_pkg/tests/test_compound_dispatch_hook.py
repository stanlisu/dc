"""`apply_filter_mask`'s composite split re-enters through the caller.

`apply_filter_mask` gained a `sub_filter_fn` recursion hook on 2026-08-23:
`_and_` / `_or_` / list legs are evaluated through it instead of through the
module function, so a SUBCLASS override stays in the loop for every leg. Before
that, `OrbResearch._apply_filter_mask` — the only place that points an atom at
its own timeframe — was skipped for every compound, and each leg silently read
the TARGET_TF columns (`orb_pkg/tests/test_cross_tf_compound_dispatch.py`).

Nothing about agamotto changes. It is single-TF, has no per-TF columns to point
at, and its hook is the same call with the same arguments, so the mask it
produces is identical to the hookless one. That invariance is what this file
pins — together with the standalone contract the non-instance callers depend
on (`sentinel_core/tests/regime_parity.py`, `agamotto_core/tests/*`, marvel's
`psylocke/`), which pass no hook and must keep the old semantics exactly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agamotto.research import AgamottoResearch
from agamotto.research_filters import apply_filter_mask


COMPOUND_NAMES = [
    "high_volume_and_macd_bullish",
    "vol_breakout_and_adx_trend",
    "low_volume_and_rsi_oversold",
    "high_volume_and_strong_trend_and_adx_trend",
    "macd_bearish_and_rsi_overbought",
    "high_vol_and_mom_positive",
    "low_vol_or_high_volume",
]


@pytest.fixture
def research():
    """__init__ only stores config/home_root — no I/O, no TA-Lib."""
    return AgamottoResearch({}, "/tmp")


def _frame() -> pd.DataFrame:
    """Own-column features plus the price columns the MVG atoms need."""
    return pd.DataFrame({
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
        "mfi": [10.0, 25.0, 50.0, 75.0, 90.0],
        "bop": [-0.5, -0.2, 0.0, 0.2, 0.5],
        "roc": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "sar": [1.0, 1.0, 1.0, 1.0, 1.0],
        "bb_lower": [1.0, 1.0, 1.0, 1.0, 1.0],
        "bb_upper": [9.0, 9.0, 9.0, 9.0, 9.0],
        "close": [3.0, 3.0, 3.0, 3.0, 3.0],
        "mvg1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "mvg2": [2.0, 2.0, 2.0, 4.0, 4.0],
        "mvg3": [1.5, 1.5, 1.5, 4.5, 4.5],
    })


def _standalone(df, name, position):
    """The hookless path — byte-for-byte what the non-instance callers get."""
    return apply_filter_mask(
        df, name, position, strict_filters=True,
        allowed_positions_fn=AgamottoResearch.allowed_positions)


@pytest.mark.parametrize("name", COMPOUND_NAMES)
@pytest.mark.parametrize("position", ["long", "short"])
def test_agamotto_compound_mask_is_identical_to_the_hookless_path(
        research, name, position):
    df = _frame()
    pd.testing.assert_series_equal(
        research._apply_filter_mask(df, name, position),
        _standalone(df, name, position), check_names=False)


@pytest.mark.parametrize("name", [
    "macd_bullish", "adx_trend", "high_vol", "low_vol", "strong_candle",
    "high_volume", "strong_trend", "above_all_mas", "sar_aligned",
])
@pytest.mark.parametrize("position", ["long", "short"])
def test_agamotto_atomic_mask_is_identical_to_the_hookless_path(
        research, name, position):
    df = _frame()
    pd.testing.assert_series_equal(
        research._apply_filter_mask(df, name, position),
        _standalone(df, name, position), check_names=False)


def test_agamotto_list_form_is_identical_to_the_hookless_path(research):
    df = _frame()
    for items in (["high_volume", "&", "macd_bullish"],
                  ["low_vol", "|", "adx_trend"],
                  ["high_volume", "&", "adx_trend", "&", "mom_positive"]):
        pd.testing.assert_series_equal(
            research._apply_filter_mask(df, items, "long"),
            _standalone(df, items, "long"), check_names=False)


def test_the_hook_actually_receives_every_leg(research, monkeypatch):
    """Mutation guard: without the hook wired up, this stays empty.

    The three assertions above all pass when `sub_filter_fn` is dropped
    entirely (agamotto's hook is a no-op by construction), so they cannot
    detect a regression on their own. This one can.
    """
    seen: list[str] = []
    original = AgamottoResearch._apply_filter_mask

    def spy(self, df, filter_name, position):
        if isinstance(filter_name, str):
            seen.append(filter_name)
        return original(self, df, filter_name, position)

    monkeypatch.setattr(AgamottoResearch, "_apply_filter_mask", spy)
    research._apply_filter_mask(
        _frame(), "high_volume_and_macd_bullish_and_adx_trend", "long")
    assert seen == ["high_volume_and_macd_bullish_and_adx_trend",
                    "high_volume", "macd_bullish", "adx_trend"]


def test_the_standalone_evaluator_still_splits_by_itself():
    """No hook => the module function recurses into itself, as it always did.

    `sentinel_core/tests/regime_parity.py` and marvel's `psylocke/` call this
    function directly with no instance to re-enter; that contract is unchanged.
    """
    df = _frame()
    got = _standalone(df, "high_volume_and_macd_bullish", "long")
    want = (_standalone(df, "high_volume", "long")
            & _standalone(df, "macd_bullish", "long"))
    pd.testing.assert_series_equal(got, want, check_names=False)


def test_an_explicit_hook_is_honoured_over_self_recursion():
    """The hook is what carries a subclass override into every leg."""
    df = _frame()
    calls: list[str] = []

    def hook(frame, name, position):
        calls.append(name)
        return pd.Series(False, index=frame.index)

    got = apply_filter_mask(
        df, "high_volume_and_macd_bullish", "long", strict_filters=True,
        allowed_positions_fn=AgamottoResearch.allowed_positions,
        sub_filter_fn=hook)
    assert calls == ["high_volume", "macd_bullish"]
    assert not got.any()
