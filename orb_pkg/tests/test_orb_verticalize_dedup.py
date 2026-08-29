"""orb must not put the same series in front of the model twice.

`verticalize()` step 1 emitted `{tf}_{feat}` for every TF in `TIMEFRAMES` —
which INCLUDES `TARGET_TF` — and step 3b emitted the bare `{feat}` from
`TARGET_TF` as well. With `TARGET_TF == BASE_TF` (all five shipped orb settings)
those are the SAME source column: `15m_rsi.equals(rsi)` is True on the built
panel. `select_feature_columns` excludes step 3's bare raw/MA copies by name
(`close`, `mvg1`, …) but has no rule for bare DERIVED names — agamotto's own
model features are exactly those names — so only the derived duplicates reached
the model. That asymmetry was the defect.

Measured (`marvel/gauntlet/orb_vs_agamotto_features_20260822.md` §1b): the |IC|
ranking places each pair adjacently, so the shipped `TOPN_ICS = 16` arm spent
its 16 slots on 9 DISTINCT features — `15m_r069_long`, `window_2026_07_31`.

THE UNPREFIXED COPY IS THE SURVIVOR, and that direction is load-bearing rather
than arbitrary: with the `{target_tf}_<derived>` block gone there is nothing for
`_remap_tf_columns` to remap on the target timeframe, so a `15m_<atom>` leg —
atomic or inside a compound — resolves by falling through to the bare column,
which holds exactly the TARGET_TF values it would have remapped. Dropping the
bare copy instead would make every TARGET_TF atom in the shipped stack raise.
The tests below pin BOTH halves of that: the duplicate is gone, and every
filter shape still evaluates to exactly what it evaluated to before.

(When this file was written the compound path could ONLY read bare columns —
`research_filters.apply_filter_mask` split `_and_` by calling ITSELF, so
`OrbResearch._apply_filter_mask` never saw a leg and every leg of a cross-TF
regime was evaluated on TARGET_TF. That was a second, separate defect; it was
fixed on 2026-08-23 with a recursion hook, and the cross-TF behaviour it
governs is pinned in `test_cross_tf_compound_dispatch.py`. The de-duplication
above is unaffected either way — the bare TARGET_TF copy is still the one to
keep.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("talib", reason="the derived feature block needs TA-Lib")

from agamotto import AgamottoResearch      # noqa: E402
from orb import research as orb_research   # noqa: E402
from orb.research import OrbResearch       # noqa: E402


NATIVE = "BTCUSDT"
SYMBOL = "BINANCE_PERP_BTC_USDT"
_ROWS_15M = 2000
_FREQ = {"15m": ("15min", 1), "1h": ("1h", 4)}


def _make_ohlcv(n: int, start: str, freq: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        f"{NATIVE}_open": close - 0.1,
        f"{NATIVE}_high": close + 0.5,
        f"{NATIVE}_low": close - 0.5,
        f"{NATIVE}_close": close,
        f"{NATIVE}_volume": rng.integers(100, 1000, n).astype(float),
    }, index=idx)


def _config(target_tf: str = "15m") -> dict:
    return {
        "SYMBOLS": [SYMBOL],
        "EXCHANGE": "BINANCE",
        "DATA": "liquid",
        "TIMEFRAMES": ["15m", "1h"],
        "BASE_TF": "15m",
        "TARGET_TF": target_tf,
        "TIME_UNIT": "15m",
        "LADDER": 1,
        "LADDER_BPS": 1.0,
        "FEE": 0,
        "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14,
    }


def _panel(cfg: dict | None = None) -> OrbResearch:
    """Engineer + verticalize a two-TF orb panel with no disk I/O."""
    cfg = cfg or _config()
    orb = OrbResearch(cfg, "/tmp/fake_root")
    for tf in cfg["TIMEFRAMES"]:
        freq, div = _FREQ[tf]
        inst = AgamottoResearch({**cfg, "TIME_UNIT": tf}, "/tmp/fake_root")
        inst.raw = _make_ohlcv(_ROWS_15M // div, "2025-01-01", freq)
        orb._tf_instances[tf] = inst
    orb.raw = orb._tf_instances[cfg["BASE_TF"]].raw
    orb.engineer_features()
    orb.verticalize()
    return orb


def _present(orb: OrbResearch, tf: str) -> list[str]:
    """Derived features this fixture's data actually produced for `tf`."""
    return [f for f in orb_research._DERIVED_FEATURES
            if f"{tf}_{NATIVE}_{f}" in orb.features.columns]


# ── the MODEL sees each series exactly once ──────────────────────────────────

def test_target_tf_derived_features_are_not_emitted_twice():
    orb = _panel()
    cols = set(orb.vertical_features.columns)
    dupes = sorted(f for f in _present(orb, "15m") if f in cols and f"15m_{f}" in cols)
    assert not dupes, (
        "TARGET_TF derived features are in the panel under BOTH names, so the "
        f"|IC| ranker will select each of these twice: {dupes}")


def test_the_bare_copy_is_the_survivor():
    orb = _panel()
    cols = set(orb.vertical_features.columns)
    missing = [f for f in _present(orb, "15m") if f not in cols]
    assert not missing, f"TARGET_TF derived block lost columns: {missing}"
    assert not [f for f in _present(orb, "15m") if f"15m_{f}" in cols]


def test_the_bare_copy_holds_the_target_tf_values():
    orb = _panel()
    for feat in ("rsi", "macdhist", "price_range_pct", "open_close_pct"):
        src = orb.features[f"15m_{NATIVE}_{feat}"].reset_index(drop=True)
        dst = orb.vertical_features[feat].reset_index(drop=True)
        pd.testing.assert_series_equal(src, dst, check_names=False)


def test_non_target_timeframes_keep_their_prefixed_block():
    orb = _panel()
    cols = set(orb.vertical_features.columns)
    missing = [f"1h_{f}" for f in _present(orb, "1h") if f"1h_{f}" not in cols]
    assert not missing, f"1h block lost columns: {missing}"


def test_the_1h_block_is_not_the_15m_values():
    """The dedup must not have collapsed the cross-TF block onto TARGET_TF."""
    orb = _panel()
    assert not orb.vertical_features["1h_rsi"].equals(orb.vertical_features["rsi"])


def test_bare_raw_and_ma_columns_are_untouched():
    """Step 3 stays: excluded from the model BY NAME, and read by other code."""
    orb = _panel()
    cols = set(orb.vertical_features.columns)
    for raw in ("close", "open", "high", "low", "volume", "mvg1", "mvg2", "mvg3"):
        assert raw in cols, f"bare raw column {raw} disappeared"
    for raw in ("15m_close", "1h_close", "1h_mvg1"):
        assert raw in cols, f"prefixed raw column {raw} disappeared"


def test_return_columns_are_untouched():
    orb = _panel()
    cols = set(orb.vertical_features.columns)
    for ret in ("return", "return_long", "return_short",
                "return_long_raw", "return_short_raw"):
        assert ret in cols


def test_target_tf_outside_timeframes_still_emits_nothing_twice():
    """TARGET_TF not in TIMEFRAMES: step 1 skips nothing, step 3b finds nothing.

    _align_timeframes only builds `{tf}_{native}_*` for tfs in TIMEFRAMES, so
    the bare block is empty here — the pre-fix code had the same behaviour and
    the skip must not change it.
    """
    orb = _panel(_config(target_tf="4h"))
    cols = set(orb.vertical_features.columns)
    assert "15m_rsi" in cols and "1h_rsi" in cols
    assert "rsi" not in cols


# ── every FILTER shape evaluates to exactly what it did before ───────────────

@pytest.mark.parametrize("filter_name", [
    "strong_candle", "low_vol", "high_vol", "rsi_oversold", "rsi_overbought",
    "macd_bullish", "mom_positive", "adx_trend", "bop_bullish",
])
def test_unprefixed_atomic_filter_still_resolves(filter_name):
    """Bare atoms read the bare columns, which step 3b still emits.

    Resolution is proved by NOT raising: `research_filters._require_col` fails
    loud on a missing source column rather than returning an all-True mask.
    """
    orb = _panel()
    mask = orb._apply_filter_mask(orb.vertical_features, filter_name, "long")
    assert isinstance(mask, pd.Series)
    assert len(mask) == len(orb.vertical_features)


@pytest.mark.parametrize("filter_name", [
    "rsi_oversold", "macd_bullish", "high_vol", "adx_trend",
])
def test_target_tf_prefixed_atom_falls_through_to_the_bare_column(filter_name):
    """`15m_x` == `x` — _remap_tf_columns finds no `15m_<derived>` and the bare
    column it leaves in place already holds the TARGET_TF values."""
    orb = _panel()
    vf = orb.vertical_features
    prefixed = orb._apply_filter_mask(vf, f"15m_{filter_name}", "long")
    bare = orb._apply_filter_mask(vf, filter_name, "long")
    pd.testing.assert_series_equal(prefixed, bare, check_names=False)


@pytest.mark.parametrize("filter_name", ["rsi_oversold", "macd_bullish"])
def test_cross_tf_atom_still_reads_its_own_timeframe(filter_name):
    """`1h_x` must remap the 1h block, not collapse onto the bare TARGET_TF."""
    orb = _panel()
    vf = orb.vertical_features
    one_h = orb._apply_filter_mask(vf, f"1h_{filter_name}", "long")
    bare = orb._apply_filter_mask(vf, filter_name, "long")
    assert not one_h.equals(bare)


def test_filter_signals_end_to_end_on_a_shipped_style_regime():
    """The whole path a research run takes: verticalize -> mask -> ret column."""
    orb = _panel()
    out = orb.filter_signals({"regime": "15m_macd_bullish", "position": "long"})
    assert not out.empty
    assert "ret" in out.columns
    assert (out["position"] == "long").all()


def test_compound_regime_evaluates_each_leg_on_its_own_timeframe():
    """160 of the 332 shipped regimes are compound; the TARGET_TF leg of one
    resolves through the BARE columns this commit kept.

    UPDATED 2026-08-23, and the flip is the point. This test used to assert the
    OPPOSITE — `both == leg_15m`, the 15m leg alone — because
    `research_filters.apply_filter_mask` split `_and_` by calling ITSELF, so
    `OrbResearch._apply_filter_mask` never saw a leg and both legs were
    evaluated on TARGET_TF. That is now fixed by the `sub_filter_fn` recursion
    hook, and the compound is the real cross-TF conjunction. What this test
    still pins for the DE-DUPLICATION is that the 15m leg resolves at all: it
    has no `15m_<derived>` block left to remap from and must fall through to
    the bare columns. Full cross-TF coverage lives in
    `test_cross_tf_compound_dispatch.py`.
    """
    orb = _panel()
    vf = orb.vertical_features
    both = orb._apply_filter_mask(vf, "15m_macd_bullish_and_1h_macd_bullish", "long")
    assert isinstance(both, pd.Series)
    assert len(both) == len(vf)
    leg_15m = orb._apply_filter_mask(vf, "15m_macd_bullish", "long")
    leg_1h = orb._apply_filter_mask(vf, "1h_macd_bullish", "long")
    pd.testing.assert_series_equal(
        both.astype(bool), (leg_15m & leg_1h).astype(bool), check_names=False)
    assert not both.astype(bool).equals(leg_15m.astype(bool))
    # The 15m leg is the bare block this commit kept, and it is a real
    # condition rather than an all-True stand-in for a missing column.
    pd.testing.assert_series_equal(
        leg_15m.astype(bool), (vf["macdhist"] > 0), check_names=False)
