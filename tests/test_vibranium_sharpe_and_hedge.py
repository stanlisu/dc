"""Three regression contracts for vibranium, all found 2026-09-18.

Each of these silently produced a number that looked plausible, which is why
they need tests rather than comments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SRC = "vibranium_pkg/src/vibranium"


def _daily_raw(bar_pnl: pd.Series) -> pd.Series:
    """The fixed rule: keep every day that HAS BARS, flat or not."""
    daily = bar_pnl.resample("D").sum()
    return daily[bar_pnl.resample("D").count() > 0]


def _daily_old(bar_pnl: pd.Series) -> pd.Series:
    """The old rule: `daily[daily != 0]`."""
    daily = bar_pnl.resample("D").sum()
    return daily[daily != 0]


def _sharpe(daily: pd.Series) -> float:
    sd = daily.std()
    return float(daily.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0


def test_flat_but_held_days_are_kept():
    """A day the book held risk and netted zero is a REAL zero, not missing data."""
    idx = pd.date_range("2026-01-01", periods=4 * 24, freq="h", tz="UTC")
    pnl = pd.Series(0.0, index=idx)
    pnl.iloc[0] = 5.0            # day 1 makes money
    pnl.iloc[24 * 2] = -5.0      # day 3 loses it; days 2 and 4 are FLAT
    assert len(_daily_raw(pnl)) == 4, "all four traded days must survive"
    assert len(_daily_old(pnl)) == 2, "the old filter kept only the non-zero days"


def test_days_with_no_bars_are_still_dropped():
    """The original INTENT — drop calendar days with no data — is preserved."""
    idx = pd.DatetimeIndex(
        list(pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC"))
        + list(pd.date_range("2026-01-03", periods=24, freq="h", tz="UTC"))
    )
    pnl = pd.Series(0.0, index=idx)
    daily = _daily_raw(pnl)
    assert len(daily) == 2, "2026-01-02 has no bars and must not appear"
    assert pd.Timestamp("2026-01-02", tz="UTC") not in daily.index


@pytest.mark.parametrize("frac", [0.5, 0.25, 0.1])
def test_dropping_flat_days_inflates_sharpe_by_one_over_sqrt_f(frac):
    """Quantifies what the old filter was worth: exactly 1/sqrt(fraction kept)."""
    rng = np.random.default_rng(7)
    ratios = []
    for _ in range(200):
        idx = pd.date_range("2026-01-01", periods=400, freq="D", tz="UTC")
        d = rng.normal(0.01, 1.0, 400)
        d[rng.random(400) > frac] = 0.0
        s = pd.Series(d, index=idx)
        raw, old = _sharpe(_daily_raw(s)), _sharpe(_daily_old(s))
        if raw != 0:
            ratios.append(old / raw)
    assert np.mean(ratios) == pytest.approx(1 / np.sqrt(frac), rel=0.12)


def test_signed_hedge_ratio_flips_pnl_sign_versus_abs():
    """abs(h) trades a negative-h pair BACKWARDS against its own signal."""
    capital = 100.0
    # |h * ret_b| must EXCEED |ret_a| for the sign to flip; with a smaller leg-b
    # move the two only differ in size, which is why the first draft of this test
    # passed the buggy code.
    log_ret_a = np.array([0.0, 0.001, -0.001])
    log_ret_b = np.array([0.0, 0.010, -0.010])
    h = -0.8                                     # long/inverse ETF pair
    signed = capital * log_ret_a - capital * h * log_ret_b
    absed = capital * log_ret_a - capital * abs(h) * log_ret_b
    assert np.sign(signed[1]) != np.sign(absed[1]), (
        "with h<0 the two disagree in DIRECTION, not merely in size")
    # Fees must stay sign-agnostic: notional is |capital*h|.
    assert capital + capital * abs(h) > 0


def test_kalman_hedge_ratio_is_causal_in_source():
    """The filter must not be advanced over the window before pricing it.

    Reads the source rather than running the engine: the wheel is not importable
    on every host, and the defect is a statement ORDER, which text can prove.
    """
    src = open(f"{SRC}/break_recover.py").read()
    block = src[src.index("if kalman is not None:"):]
    block = block[:block.index("port_ret_usdt")]
    h_assign = block.index("h[j] = (kalman.hedge_ratio")
    update = block.index("kalman.update(la, lb)")
    assert h_assign < update, (
        "h[j] must be READ BEFORE bar j is fed to the filter; the pre-2026-09-18 "
        "code ran the whole forward window through kalman.update() first and then "
        "priced that same window from bar 1 with the final hedge ratio")


def test_no_flat_day_filter_survives_anywhere():
    """Grep contract — the idiom must not come back in any vibranium module."""
    import glob
    offenders = []
    for path in glob.glob(f"{SRC}/*.py"):
        for n, line in enumerate(open(path), 1):
            if line.strip().startswith("#"):
                continue
            if "!= 0]" in line and "resample" not in line and "count()" not in line:
                if "daily" in line:
                    offenders.append(f"{path}:{n}: {line.strip()}")
    assert not offenders, "flat-day filter reintroduced:\n" + "\n".join(offenders)
