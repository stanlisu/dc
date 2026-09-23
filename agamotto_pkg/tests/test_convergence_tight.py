"""Unit tests for the `convergence_tight` own-state regime (2026-09-22 design,
thresholds LOOSENED 2026-09-23 in two calibration passes — see
agamotto/research.py::engineer_features for the full before/after table and
the real-data fire-rate evidence).

`convergence_tight` = range-compression (ATR(10)/ATR(60) in the bottom 40th
percentile of its own trailing 60-bar distribution) AND proximity to a
rolling-20-bar high (lagged 1 bar, within 4.5%) AND a >=3-bar streak of both
AND MA9/MA21 convergence (stable spread, < 0.80% of close) (see
agamotto/research.py::engineer_features and
agamotto/research_filters.py::apply_filter_mask). All four sub-conditions are
computed per-symbol, on rolling windows that only look backward — this file
checks the mask fires on a genuine "coiling" pattern, stays off on a series
that never compresses, and never reads a future bar.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "agamotto_pkg/src")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib")

from agamotto.research import AgamottoResearch  # noqa: E402
from agamotto.research_filters import apply_filter_mask  # noqa: E402

SYM = "AAA"


def _cfg() -> dict:
    return {
        "EXCHANGE": "BINANCE", "DATA": "liquid", "TIME_UNIT": "1h",
        "LADDER": 2, "LADDER_BPS": 5.0, "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14, "FEE": 2.25,
        "SYMBOLS": [f"BINANCE_PERP_{SYM}_USDT"],
    }


def _tight_case(seed: int = 0):
    """Uptrend -> settle -> a >=5-bar tight coil near a flat level.

    Needs >252 valid ATR(10)/ATR(60) ratio points before the coil (the
    compression flag is a percentile of its OWN trailing 252-bar window), plus
    the coil itself long enough to clear the 5-bar streak and the MA9/MA21
    convergence windows.
    """
    rng = np.random.default_rng(seed)
    n_trend, n_settle, n_tight = 400, 15, 20
    n = n_trend + n_settle + n_tight
    close = np.empty(n)
    close[:n_trend] = np.linspace(80, 99.5, n_trend) + rng.standard_normal(n_trend) * 0.3
    close[n_trend:n_trend + n_settle] = 99.7 + rng.standard_normal(n_settle) * 0.15
    close[n_trend + n_settle:] = 100.0 + rng.standard_normal(n_tight) * 0.02

    high = np.empty(n)
    low = np.empty(n)
    # Phase 1: wide daily range (ATR(10) and ATR(60) both ~elevated).
    high[:n_trend] = close[:n_trend] + (0.4 + rng.random(n_trend) * 0.3)
    low[:n_trend] = close[:n_trend] - (0.4 + rng.random(n_trend) * 0.3)
    # Phase 2: narrowing range.
    high[n_trend:n_trend + n_settle] = close[n_trend:n_trend + n_settle] + (0.1 + rng.random(n_settle) * 0.1)
    low[n_trend:n_trend + n_settle] = close[n_trend:n_trend + n_settle] - (0.1 + rng.random(n_settle) * 0.1)
    # Phase 3: the coil — tiny range, flat level. ATR(10) collapses fast;
    # ATR(60) is still dragging most of phase 1, so the ratio drops hard.
    high[n_trend + n_settle:] = close[n_trend + n_settle:] + (0.02 + rng.random(n_tight) * 0.02)
    low[n_trend + n_settle:] = close[n_trend + n_settle:] - (0.02 + rng.random(n_tight) * 0.02)

    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    return close, high, low, idx


def _normal_case(seed: int = 1, n: int = 435):
    """A steady downtrend that never revisits its own trailing high.

    Under the strict 2026-09-22 thresholds (252-bar lookback, 1.5% proximity,
    5-bar streak) a flat-range random walk was rare enough to never coincide
    with all four sub-conditions by chance. Under the pass-1 loosened
    thresholds (60-bar lookback, bottom 35th pct, 3% proximity, 3-bar streak)
    a flat-range random walk DOES spuriously satisfy all four over a 435-bar
    sample (measured: 35 false-positive bars) — the compression leg is a
    percentile of the series' OWN trailing window, so a flat-range series
    puts ~35% of its bars below that threshold by construction, and a plain
    random walk wanders back near its own recent high often enough, by
    chance, to complete the other three legs too. A monotonically WIDENING
    range does not fix this either (measured: still 10/435 false positives —
    the ratio's rise saturates well before the sample ends, so the back half
    of the series compresses relative to its own trailing window just the
    same). The pass-2 thresholds (40th pct, 4.5% proximity, 0.80% MA-converge)
    are looser still, so the same argument applies a fortiori.

    The fix instead blocks the PROXIMITY leg structurally: a steady
    downtrend (drift 0.25/bar, small noise) declines roughly 5 points every
    20 bars — far more than even the pass-2 4.5% proximity band — so `close`
    is never within band of its own trailing-20-bar high (lagged 1 bar) at
    any point in the sample (measured: 0/435 bars across 20 seeds at the 3%
    band; the 4.5% band is strictly easier to clear by staying away from).
    Proximity is one of four AND'd legs, so blocking it alone makes the whole
    mask provably False regardless of what compression/streak/MA-converge do.
    """
    rng = np.random.default_rng(seed)
    drift = 0.25
    close = 150 - np.cumsum(np.full(n, drift)) + rng.standard_normal(n) * 0.3
    half_range = 0.4 + rng.random(n) * 0.2
    high = close + half_range
    low = close - half_range
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    return close, high, low, idx


def _convergence_tight_column(close, high, low, idx) -> pd.Series:
    raw = pd.DataFrame({
        f"{SYM}_open": close, f"{SYM}_high": high, f"{SYM}_low": low,
        f"{SYM}_close": close, f"{SYM}_volume": 1000.0,
    }, index=idx)
    r = AgamottoResearch(_cfg(), "/tmp/unused")
    r.raw = raw
    r.engineer_features()
    return r.features[f"{SYM}_convergence_tight"]


def test_fires_on_a_genuine_coil():
    col = _convergence_tight_column(*_tight_case())
    assert bool(col.tail(10).all()), "expected the last 10 coiled bars to all fire"


def test_does_not_fire_on_a_steady_downtrend():
    col = _convergence_tight_column(*_normal_case())
    assert int(col.sum()) == 0, "a steady downtrend must never sit near its own trailing high"


def test_causal_perturbing_a_future_bar_does_not_change_the_past():
    close, high, low, idx = _tight_case()
    before = _convergence_tight_column(close, high, low, idx)

    close2, high2, low2 = close.copy(), high.copy(), low.copy()
    t_future = len(close2) - 1
    close2[t_future] *= 3.0
    high2[t_future] *= 3.0
    low2[t_future] *= 3.0
    after = _convergence_tight_column(close2, high2, low2, idx)

    assert before.iloc[:-1].tolist() == after.iloc[:-1].tolist()


def test_apply_filter_mask_reads_the_precomputed_column():
    """apply_filter_mask must not recompute anything — it just reads/casts the
    column research.engineer_features already built (position-invariant)."""
    df = pd.DataFrame({"convergence_tight": [True, False, True]})
    for position in ("long", "short"):
        got = apply_filter_mask(df, "convergence_tight", position)
        assert got.tolist() == [True, False, True]


def test_apply_filter_mask_treats_nan_as_false_not_true():
    """Real incident, 2026-09-22: engineer_features computes this column on a
    SHORT per-symbol index then `.reindex()`s it onto the wide multi-symbol
    panel, which fills a later-listed symbol's pre-listing rows with NaN and
    upcasts bool->object to hold it. `.astype(bool)` alone turns those NaNs
    into Python `True` (NaN is truthy) — every symbol's pre-listing gap then
    "fires". Measured: ARM/TEM/NNE/ARKB (all later-listed) supplied ~2,260 of
    2,272 pooled fires across the adamantium sweep, every one dated before
    that symbol's own first real bar. fillna(False) before astype(bool) is the
    fix this test locks in."""
    df = pd.DataFrame({"convergence_tight": pd.array([True, np.nan, False, np.nan], dtype=object)})
    for position in ("long", "short"):
        got = apply_filter_mask(df, "convergence_tight", position)
        assert got.tolist() == [True, False, False, False]


def test_multi_symbol_reindex_gap_does_not_fire_for_the_shorter_history_symbol():
    """End-to-end reproduction through the real multi-symbol panel path (not
    just the isolated apply_filter_mask unit above): a second symbol whose
    data starts well after the first must show convergence_tight=False for
    every row before its own first real bar, not True."""
    close, high, low, idx = _tight_case()
    n_gap = 200
    raw = pd.DataFrame({
        f"{SYM}_open": close, f"{SYM}_high": high, f"{SYM}_low": low,
        f"{SYM}_close": close, f"{SYM}_volume": 1000.0,
    }, index=idx)

    late_close, late_high, late_low, _ = _tight_case(seed=7)
    # Starts n_gap bars after SYM's own index and extends beyond it — a later
    # listing with its own full history, not a slice of SYM's date range.
    late_idx = pd.date_range(idx[n_gap], periods=len(late_close), freq="1h")
    late = pd.DataFrame({
        "LATE_open": late_close, "LATE_high": late_high, "LATE_low": late_low,
        "LATE_close": late_close, "LATE_volume": 1000.0,
    }, index=late_idx)

    cfg = _cfg()
    cfg["SYMBOLS"] = [f"BINANCE_PERP_{SYM}_USDT", "BINANCE_PERP_LATE_USDT"]
    r = AgamottoResearch(cfg, "/tmp/unused")
    r.raw = raw.combine_first(late).reindex(columns=list(raw.columns) + list(late.columns))
    r.engineer_features()

    late_col = r.features["LATE_convergence_tight"]
    gap_rows = late_col.loc[late_col.index < late_idx[0]]
    assert len(gap_rows) > 0, "test setup: expected a real pre-listing gap to check"
    assert not bool(gap_rows.any()), (
        f"{int(gap_rows.sum())} rows fired convergence_tight=True before LATE's "
        "own first real bar — the reindex/NaN-truthy bug is back."
    )


def test_apply_filter_mask_raises_when_column_missing():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="convergence_tight"):
        apply_filter_mask(df, "convergence_tight", "long")


def test_allowed_positions_is_both():
    from agamotto.research_filters import allowed_positions
    assert allowed_positions("convergence_tight") == ["long", "short"]
