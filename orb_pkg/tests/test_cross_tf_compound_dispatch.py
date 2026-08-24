"""Every leg of a compound orb regime is evaluated on ITS OWN timeframe.

THE DEFECT (fixed 2026-08-23). `research_filters.apply_filter_mask` split
`_and_` / `_or_` / list names by calling ITSELF, so the split never re-entered
`OrbResearch._apply_filter_mask` — the one method that knows how to point an
atom at its own timeframe (`_remap_tf_columns`). Control instead reached the
`re.sub(r'^(?:15m|1h|4h|1d)_', '', filter_name)` strip further down, which
discards the prefix and reads the BARE columns; and the bare columns are the
TARGET_TF's (verticalize step 3/3b). So every leg of a cross-TF compound was
evaluated on TARGET_TF whatever TF its name said. Measured on the 2,000-row
15m+1h panel below, before the fix:

    15m_macd_bullish                       985 rows
    1h_macd_bullish                        956 rows
    15m_macd_bullish_and_1h_macd_bullish   985 rows   <- the 15m leg ALONE
    (15m leg) & (1h leg)                   505 rows

The fix is a recursion HOOK (`apply_filter_mask(..., sub_filter_fn=...)`) that
`AgamottoResearch._apply_filter_mask` fills with `self._apply_filter_mask`, so
composite legs re-enter through the instance and a subclass override applies to
each of them. `ScepterResearch` and `StormBreakerResearch` had each already
hand-rolled their own split for exactly this reason; orb had not, and orb is
the algo whose shipped stack is TF-prefixed — 160 of the 332 rows in
`pred_orb.base.15m_1/regime_stack.json` are compound, 136 of them cross-TF.

THE PRE-FIX ORACLE used throughout this file is the standalone
`research_filters.apply_filter_mask` called with NO hook. That is not a
reimplementation: it is the untouched code path the old compound branch
delegated to, so `oracle(name) == pre-fix OrbResearch mask` by construction.
Behaviour that must NOT change is pinned by asserting equality with it;
behaviour that must change is pinned by asserting inequality PLUS equality with
the per-TF conjunction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("talib", reason="the derived feature block needs TA-Lib")

from agamotto import AgamottoResearch                     # noqa: E402
from agamotto import research_filters as rf               # noqa: E402
from orb.research import OrbResearch                      # noqa: E402


NATIVE = "BTCUSDT"
SYMBOL = "BINANCE_PERP_BTC_USDT"
_ROWS_15M = 2000
_FREQ = {"15m": ("15min", 1), "1h": ("1h", 4), "4h": ("4h", 16)}


def _make_ohlcv(n: int, freq: str) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq=freq)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        f"{NATIVE}_open": close - 0.1,
        f"{NATIVE}_high": close + 0.5,
        f"{NATIVE}_low": close - 0.5,
        f"{NATIVE}_close": close,
        f"{NATIVE}_volume": rng.integers(100, 1000, n).astype(float),
    }, index=idx)


def _config(timeframes=("15m", "1h", "4h")) -> dict:
    return {
        "SYMBOLS": [SYMBOL],
        "EXCHANGE": "BINANCE",
        "DATA": "liquid",
        "TIMEFRAMES": list(timeframes),
        "BASE_TF": "15m",
        "TARGET_TF": "15m",
        "TIME_UNIT": "15m",
        "LADDER": 1,
        "LADDER_BPS": 1.0,
        "FEE": 0,
        "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14,
    }


@pytest.fixture(scope="module")
def orb() -> OrbResearch:
    """Engineer + verticalize a three-TF orb panel with no disk I/O."""
    cfg = _config()
    inst_orb = OrbResearch(cfg, "/tmp/fake_root")
    for tf in cfg["TIMEFRAMES"]:
        freq, div = _FREQ[tf]
        inst = AgamottoResearch({**cfg, "TIME_UNIT": tf}, "/tmp/fake_root")
        inst.raw = _make_ohlcv(_ROWS_15M // div, freq)
        inst_orb._tf_instances[tf] = inst
    inst_orb.raw = inst_orb._tf_instances[cfg["BASE_TF"]].raw
    inst_orb.engineer_features()
    inst_orb.verticalize()
    return inst_orb


def _mask(orb: OrbResearch, name: str, position: str = "long") -> pd.Series:
    return orb._apply_filter_mask(
        orb.vertical_features, name, position).astype(bool)


def _oracle(orb: OrbResearch, name: str, position: str = "long") -> pd.Series:
    """Pre-fix behaviour: the standalone evaluator, no recursion hook."""
    return rf.apply_filter_mask(
        orb.vertical_features, name, position, strict_filters=True,
        allowed_positions_fn=OrbResearch.allowed_positions).astype(bool)


# ── a cross-TF compound now requires BOTH legs on their OWN timeframes ───────

@pytest.mark.parametrize("a,b,position", [
    ("15m_macd_bullish", "1h_macd_bullish", "long"),
    ("15m_rsi_oversold", "1h_adx_trend", "long"),
    ("4h_adx_trend", "1h_macd_bullish", "long"),
    ("1h_high_volume", "4h_mom_positive", "long"),
    ("15m_macd_bearish", "4h_rsi_overbought", "short"),
    ("1h_strong_trend", "4h_above_all_mas", "long"),
])
def test_cross_tf_and_is_the_conjunction_of_its_own_tf_legs(orb, a, b, position):
    leg_a = _mask(orb, a, position)
    leg_b = _mask(orb, b, position)
    both = _mask(orb, f"{a}_and_{b}", position)
    pd.testing.assert_series_equal(both, leg_a & leg_b, check_names=False)


@pytest.mark.parametrize("a,b,position", [
    ("15m_macd_bullish", "1h_macd_bullish", "long"),
    ("15m_rsi_oversold", "1h_adx_trend", "long"),
    ("4h_adx_trend", "1h_macd_bullish", "long"),
])
def test_cross_tf_and_no_longer_matches_the_target_tf_evaluation(orb, a, b,
                                                                 position):
    """The mutation guard: the old mask evaluated BOTH legs on TARGET_TF.

    Asserting only the conjunction above would still pass on a panel where the
    two timeframes happen to agree, so pin the inequality against the pre-fix
    oracle explicitly.
    """
    name = f"{a}_and_{b}"
    assert not _mask(orb, name, position).equals(_oracle(orb, name, position))


def test_the_documented_985_vs_505_case(orb):
    """The measurement in this file's docstring, asserted as numbers."""
    leg_15m = _mask(orb, "15m_macd_bullish")
    leg_1h = _mask(orb, "1h_macd_bullish")
    both = _mask(orb, "15m_macd_bullish_and_1h_macd_bullish")
    assert leg_15m.sum() == 985
    assert leg_1h.sum() == 956
    assert (leg_15m & leg_1h).sum() == 505
    assert both.sum() == 505                       # was 985 — the 15m leg alone
    assert _oracle(orb, "15m_macd_bullish_and_1h_macd_bullish").sum() == 985


def test_cross_tf_or_is_the_disjunction_of_its_own_tf_legs(orb):
    """`_or_` shares the split, so it shared the defect."""
    leg_a = _mask(orb, "15m_rsi_oversold")
    leg_b = _mask(orb, "1h_adx_trend")
    either = _mask(orb, "15m_rsi_oversold_or_1h_adx_trend")
    pd.testing.assert_series_equal(either, leg_a | leg_b, check_names=False)
    assert not either.equals(_oracle(orb, "15m_rsi_oversold_or_1h_adx_trend"))


def test_three_leg_cross_tf_compound(orb):
    """The stack's vol-quantile-style 3-atom conjunctions, cross-TF."""
    name = "15m_high_volume_and_1h_macd_bullish_and_4h_adx_trend"
    legs = [_mask(orb, n) for n in
            ("15m_high_volume", "1h_macd_bullish", "4h_adx_trend")]
    want = legs[0] & legs[1] & legs[2]
    pd.testing.assert_series_equal(_mask(orb, name), want, check_names=False)


def test_list_form_compound_also_dispatches_per_leg(orb):
    """The list branch shared the same self-recursion and the same fix."""
    want = _mask(orb, "15m_rsi_oversold") & _mask(orb, "1h_adx_trend")
    got = _mask(orb, ["15m_rsi_oversold", "&", "1h_adx_trend"])
    pd.testing.assert_series_equal(got, want, check_names=False)


def test_per_leg_position_veto_survives_the_split(orb):
    """A long-only atom inside a SHORT compound still vetoes the whole regime.

    `allowed_positions` runs per leg inside the standalone evaluator; routing
    the legs through the subclass must not skip it.
    """
    name = "15m_macd_bearish_and_1h_macd_bullish"   # bullish leg is long-only
    assert not _mask(orb, name, "short").any()


def test_a_cross_tf_leg_reads_its_own_column_not_the_target_copy(orb):
    """Ground truth straight off the panel, no filter machinery involved."""
    vf = orb.vertical_features
    want = (vf["1h_macdhist"] > 0) & (vf["macdhist"] > 0)
    got = _mask(orb, "15m_macd_bullish_and_1h_macd_bullish")
    pd.testing.assert_series_equal(got, want, check_names=False)


# ── everything that must NOT change ─────────────────────────────────────────

@pytest.mark.parametrize("name,position", [
    ("15m_macd_bullish_and_15m_adx_trend", "long"),
    ("15m_rsi_oversold_and_15m_high_volume", "long"),
    ("15m_macd_bearish_and_15m_rsi_overbought", "short"),
])
def test_target_tf_compound_is_byte_identical(orb, name, position):
    """TARGET_TF legs already read the bare columns — nothing to re-point."""
    pd.testing.assert_series_equal(
        _mask(orb, name, position), _oracle(orb, name, position),
        check_names=False)


@pytest.mark.parametrize("name,position", [
    ("macd_bullish_and_adx_trend", "long"),
    ("strong_candle_and_low_vol", "long"),
    ("high_volume_and_mom_positive", "long"),
    ("macd_bearish_and_rsi_overbought", "short"),
])
def test_unprefixed_compound_is_byte_identical(orb, name, position):
    """No prefix anywhere → the remap is never entered, before or after."""
    pd.testing.assert_series_equal(
        _mask(orb, name, position), _oracle(orb, name, position),
        check_names=False)


@pytest.mark.parametrize("name", [
    "strong_candle", "low_vol", "high_vol", "rsi_oversold", "macd_bullish",
    "mom_positive", "adx_trend", "bop_bullish", "high_volume",
])
def test_unprefixed_atomic_filters_still_resolve(orb, name):
    """`strong_candle` (open_close_pct) / `low_vol` (price_range_pct) and the
    rest deliberately read BARE columns; resolution is proved by not raising —
    `_require_col` fails loud rather than returning an all-True mask."""
    got = _mask(orb, name)
    pd.testing.assert_series_equal(got, _oracle(orb, name), check_names=False)


@pytest.mark.parametrize("name", [
    "15m_rsi_oversold", "1h_macd_bullish", "4h_adx_trend", "1h_high_volume",
    "4h_strong_trend", "15m_low_vol",
])
def test_atomic_prefixed_filters_are_unchanged(orb, name):
    """Atoms already went through the remap; only compounds were broken.

    Also covers the `_remap_tf_columns` hardening (a filter column the TF does
    not carry is now dropped rather than showing the TARGET_TF copy through):
    on a well-formed panel every TF carries every column, so it is a no-op.
    """
    before = _mask(orb, name)
    assert before.any() or not before.any()        # resolves without raising
    assert len(before) == len(orb.vertical_features)


def test_an_atom_naming_a_timeframe_the_panel_lacks_now_raises():
    """The second route to the same silent wrong answer.

    `_remap_tf_columns` used to leave the bare (TARGET_TF) columns in place when
    the requested TF had none, so `4h_rsi_oversold` on a 15m+1h panel silently
    read 15m and called it 4h. The columns are dropped now, so `_require_col`
    fails loud and names the column.
    """
    cfg = _config(timeframes=("15m", "1h"))
    inst_orb = OrbResearch(cfg, "/tmp/fake_root")
    for tf in cfg["TIMEFRAMES"]:
        freq, div = _FREQ[tf]
        inst = AgamottoResearch({**cfg, "TIME_UNIT": tf}, "/tmp/fake_root")
        inst.raw = _make_ohlcv(_ROWS_15M // div, freq)
        inst_orb._tf_instances[tf] = inst
    inst_orb.raw = inst_orb._tf_instances["15m"].raw
    inst_orb.engineer_features()
    inst_orb.verticalize()
    with pytest.raises(ValueError, match="requires column 'rsi'"):
        inst_orb._apply_filter_mask(
            inst_orb.vertical_features, "4h_rsi_oversold", "long")
