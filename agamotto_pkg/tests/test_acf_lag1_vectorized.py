"""Regression guard for the vectorized `{base}_acf_lag1` feature.

`engineer_features` used to compute the rolling lag-1 autocorrelation with

    hist_return.rolling(window=w).apply(
        lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) >= 4 else 0.0,
        raw=False,
    ).fillna(0.0)

which materializes two throwaway `pd.Series` per window (~19k per live cycle at
28 symbols x 700 bars) and cost ~0.78 s/symbol. It is now

    hist_return.rolling(window=w - 1).corr(hist_return.shift(1)).fillna(0.0)

`Series.autocorr(lag=1)` over a w-wide window IS `corr(x[:-1], x[1:])`, i.e. a
(w-1)-wide rolling correlation against the lag-1 shift, so the two are the same
function. This feature feeds the TRAINING pipeline for every kline algo
(orb/aether/scepter/vomir all reach it through `AgamottoResearch`), so a real
(non-float-noise) drift here silently invalidates every deployed weight.
These tests pin the equivalence so it cannot drift later.
"""
import numpy as np
import pandas as pd
import pytest

W = 14
# Worst observed |old - new| over 6 real 15m symbols x 1500 bars was 1.04e-13,
# and 4.9e-11 at the smallest supported window (w=4, only 3 pairs per window).
# The feature's own range is [-1, 1], so this is ~9 orders of magnitude tighter
# than anything a model could resolve.
TOL = 1e-9
TOL_W14 = 1e-11


def _old_acf_lag1(hist_return: pd.Series, stats_window: int) -> pd.Series:
    """The retired implementation, verbatim. Do not 'improve' it."""
    return hist_return.rolling(window=stats_window).apply(
        lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) >= 4 else 0.0,
        raw=False,
    ).fillna(0.0)


def _new_acf_lag1(hist_return: pd.Series, stats_window: int) -> pd.Series:
    """The shipped expression, kept byte-for-byte in sync with research.py."""
    return hist_return.rolling(window=stats_window - 1).corr(
        hist_return.shift(1)).fillna(0.0)


def _brute_acf_lag1(hist_return: pd.Series, stats_window: int) -> pd.Series:
    """Independent ground truth, written from the DEFINITION rather than from
    either implementation: pairwise-complete Pearson on (x[1:], x[:-1]) of the
    raw w-wide window, gated by rolling's min_periods=w non-NaN count."""
    v = hist_return.to_numpy(dtype=float)
    v = np.where(np.isinf(v), np.nan, v)   # what pandas' _prep_values does
    out = np.full(len(v), np.nan)
    for i in range(stats_window - 1, len(v)):
        x = v[i - stats_window + 1: i + 1]
        if np.count_nonzero(~np.isnan(x)) < stats_window:
            continue
        a, b = x[1:], x[:-1]
        m = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[m], b[m]
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            continue
        with np.errstate(all="ignore"):
            out[i] = np.corrcoef(a, b)[0, 1]
    return pd.Series(out, index=hist_return.index).fillna(0.0)


def _assert_equivalent(s: pd.Series, w: int, tol: float = TOL, label: str = ""):
    old = _old_acf_lag1(s, w)
    new = _new_acf_lag1(s, w)
    assert list(old.index) == list(new.index), f"{label}: index drift"
    o, n = old.to_numpy(), new.to_numpy()
    nan_mismatch = int((np.isnan(o) != np.isnan(n)).sum())
    assert nan_mismatch == 0, f"{label}: {nan_mismatch} rows differ in NaN-ness"
    d = np.abs(o - n)
    worst = float(np.nanmax(d)) if d.size else 0.0
    assert worst <= tol, f"{label}: max|old-new|={worst:.3e} > {tol:.0e}"
    return worst


# --------------------------------------------------------------------------
# 1. Random-walk returns across several "symbols" (seeds) and window sizes
# --------------------------------------------------------------------------

def _returns(seed: int, n: int = 1500, vol: float = 3e-4) -> pd.Series:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    return pd.Series(
        close, index=pd.date_range("2026-01-01", periods=n, freq="15min")
    ).pct_change(fill_method=None)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_matches_the_retired_lambda_on_realistic_returns(seed):
    _assert_equivalent(_returns(seed), W, TOL_W14, f"seed={seed}")


@pytest.mark.parametrize("w", [4, 5, 7, 10, 14, 20, 30, 50, 100])
def test_matches_across_window_sizes(w):
    _assert_equivalent(_returns(11), w, TOL, f"w={w}")


def test_both_implementations_match_an_independent_ground_truth():
    """Guards against the two implementations sharing a wrong assumption."""
    s = _returns(21)
    brute = _brute_acf_lag1(s, W).to_numpy()
    for name, impl in (("old", _old_acf_lag1), ("new", _new_acf_lag1)):
        got = impl(s, W).to_numpy()
        worst = float(np.nanmax(np.abs(got - brute)))
        assert worst <= TOL, f"{name} vs brute-force: {worst:.3e}"


def test_the_feature_is_actually_populated():
    """A tolerance test passes trivially if both sides are all-zero."""
    new = _new_acf_lag1(_returns(3), W)
    assert int((new != 0).sum()) > 1000
    assert new.abs().max() <= 1.0 + 1e-12


def test_warmup_rows_are_zero_and_the_first_real_row_is_row_w():
    """`hist_return[0]` is NaN (pct_change), so min_periods only clears at row w.
    Both forms must agree on WHERE the warmup ends, not just on the values."""
    old, new = _old_acf_lag1(_returns(4), W), _new_acf_lag1(_returns(4), W)
    assert (old.iloc[:W] == 0.0).all()
    assert (new.iloc[:W] == 0.0).all()
    assert old.iloc[W] != 0.0 and new.iloc[W] != 0.0


# --------------------------------------------------------------------------
# 2. Edge cases — where a "mathematically equivalent" rewrite usually breaks
# --------------------------------------------------------------------------

def _edge_cases() -> dict[str, pd.Series]:
    rng = np.random.default_rng(7)
    cases: dict[str, pd.Series] = {}

    for L in (1, 3, 4, 13, 14, 15, 16, 30):
        cases[f"len={L}"] = pd.Series(rng.normal(0, 1e-3, L))

    s = pd.Series(rng.normal(0, 1e-3, 200)); s.iloc[0] = np.nan
    cases["leading NaN"] = s
    s = pd.Series(rng.normal(0, 1e-3, 200)); s.iloc[0] = np.nan; s.iloc[100] = np.nan
    cases["single mid NaN"] = s
    s = pd.Series(rng.normal(0, 1e-3, 300)); s.iloc[0] = np.nan; s.iloc[50:70] = np.nan
    cases["NaN block"] = s
    s = pd.Series(rng.normal(0, 1e-3, 200)); s.iloc[-5:] = np.nan
    cases["trailing NaN block"] = s
    cases["all NaN"] = pd.Series([np.nan] * 60)

    # zero-variance windows: correlation is 0/0 and must land on the SAME side
    # (NaN -> fillna(0.0)) in both forms.
    s = pd.Series(rng.normal(0, 1e-3, 300)); s.iloc[100:160] = 0.0
    cases["constant run"] = s
    cases["all constant 0.0"] = pd.Series(np.zeros(60))
    cases["all constant 5.0"] = pd.Series(np.full(60, 5.0))
    cases["near-constant 1e6"] = pd.Series(1e6 + rng.normal(0, 1e-12, 300))

    # a zero close makes pct_change return inf; pandas coerces it to NaN
    s = pd.Series(rng.normal(0, 1e-3, 200)); s.iloc[80] = np.inf; s.iloc[120] = -np.inf
    cases["inf values"] = s
    c = pd.Series(rng.uniform(90, 110, 200)); c.iloc[100] = 0.0
    cases["close hits zero"] = c.pct_change(fill_method=None)
    c = pd.Series(rng.uniform(90, 110, 300)); c.iloc[150:200] = 100.0
    cases["flat price run"] = c.pct_change(fill_method=None)

    cases["huge magnitude"] = pd.Series(rng.normal(0, 1e12, 300))
    cases["tiny magnitude"] = pd.Series(rng.normal(0, 1e-12, 300))

    e = rng.normal(0, 1e-3, 400); ar = np.zeros(400)
    for i in range(1, 400):
        ar[i] = 0.9 * ar[i - 1] + e[i]
    cases["AR(1) rho=0.9"] = pd.Series(ar)
    cases["alternating +-1"] = pd.Series(np.tile([1.0, -1.0], 150))
    cases["monotone ramp"] = pd.Series(np.arange(300, dtype=float))
    return cases


@pytest.mark.parametrize("name", sorted(_edge_cases()))
def test_edge_cases(name):
    _assert_equivalent(_edge_cases()[name], W, TOL, name)


def test_empty_series():
    s = pd.Series([], dtype=float)
    assert len(_old_acf_lag1(s, W)) == 0
    assert len(_new_acf_lag1(s, W)) == 0


# --------------------------------------------------------------------------
# 3. End-to-end through the real engine (orb/aether/scepter/vomir inherit it)
# --------------------------------------------------------------------------

def _ohlcv(prefix="BTCUSDT", n=600):
    rng = np.random.default_rng(0)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    return pd.DataFrame({
        f"{prefix}_open": c,
        f"{prefix}_high": c * (1 + np.abs(rng.normal(0, 4e-4, n))),
        f"{prefix}_low": c * (1 - np.abs(rng.normal(0, 4e-4, n))),
        f"{prefix}_close": c,
        f"{prefix}_volume": np.abs(rng.normal(1000, 100, n)),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min"))


def _cfg(stats_window=14):
    return {"LADDER": 2, "LADDER_BPS": 1.0, "FEE": 0.0,
            "MA_PERIODS": [7, 25, 99], "STATS_WINDOW": stats_window}


def test_engineer_features_column_matches_the_retired_implementation():
    from agamotto.research import AgamottoResearch

    raw = _ohlcv()
    r = AgamottoResearch.__new__(AgamottoResearch)
    r.config = _cfg()
    r.raw = raw
    r.engineer_features()

    got = r.features["BTCUSDT_acf_lag1"]
    expected = _old_acf_lag1(
        raw["BTCUSDT_close"].pct_change(fill_method=None), W)

    assert not got.isna().any(), "acf_lag1 must be fully filled"
    worst = float(np.nanmax(np.abs(got.to_numpy() - expected.to_numpy())))
    assert worst <= TOL_W14, f"engineer_features drifted: {worst:.3e}"
    assert int((got != 0).sum()) > 400, "column is degenerate, test is vacuous"


def test_stats_window_below_four_fails_loud():
    """The retired lambda silently emitted an all-zero column for w < 4 (its
    `len(x) >= 4` guard). The vectorized form would emit real numbers instead,
    so the degenerate config must raise rather than quietly change meaning."""
    from agamotto.research import AgamottoResearch

    for bad in (0, 1, 2, 3):
        r = AgamottoResearch.__new__(AgamottoResearch)
        r.config = _cfg(stats_window=bad)
        r.raw = _ohlcv(n=100)
        with pytest.raises(ValueError, match="STATS_WINDOW must be >= 4"):
            r.engineer_features()
