"""`vol_ret_lag*` is NaN, never +inf, after a zero-volume bar.

`vol_ret` was `volume.pct_change()`, which is +inf on the bar after a
zero-volume bar. Measured 2026-09-26 on shield2 over every 1m/15m filter
parquet: +inf in ~0.7% of agamotto base+stock 15m rows (US stocks leave
zero-volume bars) and 712 cells across base 1m. Step 2 turned the inf into NaN
before imputing, but live knull handed it to RobustScaler.transform, which
rejects inf; the bare `except` in `AgamottoTrading.predict` then dropped the
whole regime for that bar, for every symbol.

The rule is now `vol / vol.shift(1).where(vol.shift(1) != 0) - 1`. These tests
pin both halves: the zero-volume cells are NaN (reverting to pct_change fails
`test_bar_after_zero_volume_is_nan`), and every other cell is BIT-IDENTICAL to
pct_change, so no trained model sees a different number anywhere else.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "agamotto_pkg/src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("talib")

from agamotto.research import AgamottoResearch  # noqa: E402

N = 300
ZERO_AT = 100                      # a single zero-volume bar
ZERO_RUN = range(200, 205)         # a run: 0 -> 0 is NaN under either rule
CFG = {
    "EXCHANGE": "BINANCE", "DATA": "liquid", "TIME_UNIT": "1h",
    "LADDER": 2, "LADDER_BPS": 5.0, "MA_PERIODS": [7, 25, 99],
    "STATS_WINDOW": 14, "FEE": 2.25, "DUAL_HORIZON": True,
    "SYMBOLS": ["BINANCE_PERP_AAA_USDT"],
}


@pytest.fixture(scope="module")
def run():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2025-01-01", periods=N, freq="1h")
    close = 100.0 + np.cumsum(rng.standard_normal(N) * 0.5)
    vol = rng.integers(100, 1000, N).astype(float)
    vol[ZERO_AT] = 0.0
    vol[list(ZERO_RUN)] = 0.0
    raw = pd.DataFrame({
        "AAA_open": close - 0.05,
        "AAA_high": close + 0.5 + rng.random(N) * 0.2,
        "AAA_low": close - 0.5 - rng.random(N) * 0.2,
        "AAA_close": close,
        "AAA_volume": vol,
        "AAA_quote_volume": rng.integers(10_000, 100_000, N).astype(float),
        "AAA_number_of_trades": rng.integers(10, 100, N).astype(float),
        "AAA_taker_buy_quote_volume": rng.integers(1_000, 50_000, N).astype(float),
    }, index=idx)
    r = AgamottoResearch(CFG, "/tmp/unused")
    r.raw = raw
    r.engineer_features()
    return raw["AAA_volume"], r.features


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_bar_after_zero_volume_is_nan(run, lag):
    _, f = run
    col = f[f"AAA_vol_ret_lag{lag}"].to_numpy()
    # vol_ret[ZERO_AT + 1] divides by zero; lag k shifts it to ZERO_AT + 1 + k.
    assert np.isnan(col[ZERO_AT + 1 + lag])
    # The bar after the run: previous volume 0, current non-zero.
    assert np.isnan(col[ZERO_RUN[-1] + 1 + lag])


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_no_inf_anywhere(run, lag):
    _, f = run
    assert not np.isinf(f[f"AAA_vol_ret_lag{lag}"].to_numpy()).any()


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_every_other_cell_is_bit_identical_to_pct_change(run, lag):
    vol, f = run
    old = vol.pct_change(fill_method=None).shift(lag).to_numpy()
    new = f[f"AAA_vol_ret_lag{lag}"].to_numpy()
    keep = np.isfinite(old)
    assert keep.sum() > N - 20
    assert np.array_equal(new[keep], old[keep])
    # Where the old rule was NaN (warm-up, 0/0 inside the run) the new one is too.
    assert np.isnan(new[np.isnan(old)]).all()
    # And the ONLY cells that changed are the old +inf ones.
    changed = ~(keep | np.isnan(old))
    assert np.isposinf(old[changed]).all() and changed.sum() == 2
