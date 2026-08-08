"""Tick ladder sizing: base rung, LADDER-1 cap, float guard.

THE DEFECT THIS REPLACES (fixed 2026-08-08). `mjolnir/core/ladder.py` sized
positions as `size_long = floor(adverse / 1bp).clip(0, LADDER)` — "2026-06-20
refinement #1: use n layers, NOT 1 + n — no free base rung, so an excursion of
< 1bps fills nothing". A bar whose adverse excursion was under one step got
size 0, i.e. was labelled as a position that never opened.

The "no free base rung" premise holds only where the entry order needs price to
come to it. Nothing in the executor works that way:

  * `knull/ladder.py:170` — `_level[symbol] = 1` as soon as
    `filled_qty >= rung_qty`. Rung 1 fills at ENTRY, with no adverse move.
    Rungs 2..ladder_max each need a further LADDER_BPS against entry (`:177`
    advance loop, `:190` bps_frac, `:201,208` triggers), and
    `ladder_max = self._LADDER` (`:142`) with levels 1..ladder_max inclusive —
    so LADDER is the TOTAL rung count and only LADDER-1 rungs are reachable by
    moving.
  * `knull/execution_style.py:181,193-194` — LadderMakerStyle rests the entry
    post-only but `should_reprice_entry()` returns True, "Maker chases:
    _handle_entering runs its idempotent reprice". The entry follows the book.
  * `knull/execution_style.py:385,401` — TakerStyle sets `ioc_open` True and
    crosses the touch outright. `_regen_a26/setting.json` runs
    `TRADING_MODE: "Taker"` today and moves to LadderMaker next, so the base
    rung is real under both.

(marvel/knull and xmen/knull are line-identical for both files, checked
2026-08-08; the tick bot runs the xmen copy.)

So the discarded bars are exactly the ones where the trade was immediately
right — a no-dip bar is a long that never went underwater.

MEASURED on `_regen_a26` (mjolnir.base.30s_1, LADDER=1, limit_then_taker,
FEE=1.75, 28 symbols x 2026-03-01..06-30): see the commit message for the
zero-size fraction and the discarded net alpha in bps.

SOURCE OF TRUTH. The maths is `agamotto/ladder.py::compute_ladder_multiplier`
(dc `b696288`). mjolnir keeps a byte-equivalent COPY rather than importing it,
because xmen vendors only `mjolnir_pkg` — see that function's docstring.
`test_parity_with_agamotto_*` below is what keeps the two copies from drifting;
if it fails, the copies have diverged and one of them is wrong.
"""
import types

import numpy as np
import pandas as pd
import pytest

from agamotto.ladder import compute_ladder_multiplier as agamotto_multiplier
from mjolnir.core.ladder import (
    LADDER_STEP_BPS,
    compute_ladder_multiplier,
    compute_ladder_returns,
)

STEP = 0.0001  # 1 bp — LADDER_STEP_BPS * 1e-4
LADDER = 3


def _cfg(ladder=LADDER, fee=0.0, mode="ladder"):
    return {"LADDER": ladder, "FEE": fee, "LADDER_FILL_MODE": mode}


def _frame(close0, close1, low1, high1):
    """3-bar frame; row 0's label reads bar 1's low/high and close[1]/close[0]."""
    return pd.DataFrame({
        "close": [close0, close1, close1],
        "low": [close0, low1, close1],
        "high": [close0, high1, close1],
    })


def _size_long(df, ladder=LADDER):
    """Recover the long size: raw = price_return * size, and fee is 0 here."""
    out = compute_ladder_returns(_cfg(ladder=ladder), df, "close", "low", "high")
    pr = df["close"].iloc[1] / df["close"].iloc[0] - 1.0
    return out["return_long_raw"].iloc[0] / pr


def _size_short(df, ladder=LADDER):
    out = compute_ladder_returns(_cfg(ladder=ladder), df, "close", "low", "high")
    pr = df["close"].iloc[1] / df["close"].iloc[0] - 1.0
    return out["return_short_raw"].iloc[0] / pr


# --------------------------------------------------------------------------- #
# 1. The base rung — THE defect
# --------------------------------------------------------------------------- #
def test_no_dip_bar_opens_the_base_rung():
    """Price never trades below the signal close, then closes +5bp up.

    Old: adverse excursion 0 -> floor(0) = 0 rungs -> the bar was labelled 0.0,
    i.e. a position that never opened. It DID open: the executor fills rung 1 at
    entry. This is the immediately-right trade whose alpha was being thrown away.
    """
    df = _frame(100.0, 100.05, 100.00, 100.05)
    assert _size_long(df) == 1
    out = compute_ladder_returns(_cfg(), df, "close", "low", "high")
    assert out["return_long_raw"].iloc[0] == pytest.approx(0.0005, abs=1e-12)


def test_favourable_only_excursion_still_opens_one_rung():
    """A purely favourable move fills the entry rung and NO extra rung — the
    multiplier must never drop below 1 nor rise above it here."""
    df = _frame(100.0, 100.05, 100.00, 100.06)
    assert _size_long(df) == 1


def test_short_leg_base_rung_on_no_rally():
    """Mirror image: no upward excursion -> the short still opens rung 1."""
    df = _frame(100.0, 99.95, 99.95, 100.00)
    assert _size_short(df) == 1


# --------------------------------------------------------------------------- #
# 2. Step counting + the float guard
# --------------------------------------------------------------------------- #
def test_exact_one_step_dip_is_two_rungs():
    """1.0bp adverse = entry rung + 1 laddered rung."""
    df = _frame(100.0, 100.02, 99.99, 100.02)
    assert _size_long(df) == 2


def test_exact_two_step_dip_is_three_rungs_float_guard():
    """THE float guard. 0.02/100 / 0.0001 evaluates to 1.9999999999999998 in
    binary floating point, so a bare floor() returns 1 extra rung instead of 2
    and the bar is under-sized by a whole rung. `+1e-9` fixes it."""
    df = _frame(100.0, 100.03, 99.98, 100.03)
    raw = (100.0 - 99.98) / 100.0 / STEP
    assert raw < 2.0, "fixture no longer exercises the float-representation edge"
    assert _size_long(df) == 3


def test_over_cap_dip_is_clamped_to_ladder_total():
    """9bp adverse at LADDER=3: 1 entry rung + min(9, LADDER-1=2) = 3 TOTAL.

    LADDER is the total rung count (`knull/ladder.py:157`), so the cap on the
    laddered rungs is LADDER-1. Capping at LADDER instead would make the tick
    target mean one more rung than the kline target for the same config.
    """
    df = _frame(100.0, 99.95, 99.91, 100.00)
    assert _size_long(df) == LADDER == 3


@pytest.mark.parametrize("ladder", [0, 1])
def test_ladder_zero_or_one_is_entry_rung_only(ladder):
    """REVERSED 2026-08-08 (was `test_ladder_zero_is_accepted_and_not_collapsed`,
    which asserted the target was 0.0). With no laddered rungs the entry rung is
    still the entry rung, so the target is the plain return at size 1. Asserting
    0 modelled a position that never opened. 8 of the 10 live tick arms set
    LADDER=1, and `EXECUTORS.ltp.MAX_RUNGS_PER_LADDER` is 1 to match.
    """
    df = _frame(100.0, 100.05, 99.90, 100.10)  # big excursion both ways
    assert _size_long(df, ladder=ladder) == 1
    assert _size_short(df, ladder=ladder) == 1


def test_ladder_one_arm_no_longer_zeroes_the_no_dip_bar():
    """The live-arm case: LADDER=1 used to give size 0 or 1; now always 1."""
    df = _frame(100.0, 100.05, 100.00, 100.05)
    assert _size_long(df, ladder=1) == 1


# --------------------------------------------------------------------------- #
# 3. Degenerate prices
# --------------------------------------------------------------------------- #
def test_zero_close_is_nan_not_inf():
    """A zero close makes the forward return undefined. It must be NaN (dropped
    downstream), never +/-inf.

    Off raw `close`, pct_change gives inf. That used to be masked by accident —
    close_safe was NaN, so the old sizing floored to 0 rungs and `inf * 0` was
    NaN. With size >= 1 the inf would survive into the label, so `price_return`
    is now computed off close_safe.
    """
    df = pd.DataFrame({"close": [0.0, 100.0, 100.0],
                       "low": [0.0, 99.9, 100.0],
                       "high": [0.0, 100.1, 100.0]})
    out = compute_ladder_returns(_cfg(), df, "close", "low", "high")
    v = out["return_long_raw"].iloc[0]
    assert not np.isinf(v)
    assert np.isnan(v)


def test_nan_close_stays_nan():
    df = pd.DataFrame({"close": [np.nan, 100.0, 100.0],
                       "low": [np.nan, 99.9, 100.0],
                       "high": [np.nan, 100.1, 100.0]})
    out = compute_ladder_returns(_cfg(), df, "close", "low", "high")
    assert np.isnan(out["return_long_raw"].iloc[0])


def test_multiplier_never_leaves_one_to_ladder():
    """Invariant sweep over pathological adverse extremes."""
    close = pd.Series([100.0] * 7)
    adverse = pd.Series([100.0, 99.999, 99.0, 0.0, np.nan, 1e9, -50.0])
    m = compute_ladder_multiplier(close, adverse, LADDER, LADDER_STEP_BPS)
    assert m.min() >= 1
    assert m.max() <= LADDER


# --------------------------------------------------------------------------- #
# 4. Parity with agamotto — the anti-drift guard for the duplicated maths
# --------------------------------------------------------------------------- #
def _random_walk(n=400, seed=11):
    rng = np.random.default_rng(seed)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 3e-4, n))))
    return pd.DataFrame({
        "close": close,
        "high": close * (1 + np.abs(rng.normal(0, 4e-4, n))),
        "low": close * (1 - np.abs(rng.normal(0, 4e-4, n))),
    })


@pytest.mark.parametrize("ladder", [1, 3, 10])
def test_parity_with_agamotto_multiplier_long(ladder):
    """mjolnir's copy must equal agamotto's original bar-for-bar (>= 200 bars).

    A private copy of shared maths is how the tick and kline targets drifted
    apart in the first place; this is the guard that makes the copy safe.
    """
    df = _random_walk()
    close_safe = df["close"].replace(0, np.nan)
    low_next = df["low"].shift(-1)
    mine = compute_ladder_multiplier(close_safe, low_next, ladder, LADDER_STEP_BPS)
    theirs = agamotto_multiplier(close_safe, low_next, ladder, LADDER_STEP_BPS)
    assert len(mine) >= 200
    pd.testing.assert_series_equal(mine, theirs)


@pytest.mark.parametrize("ladder", [1, 3, 10])
def test_parity_with_agamotto_multiplier_short(ladder):
    """Same for the mirrored short leg."""
    df = _random_walk(seed=23)
    close_safe = df["close"].replace(0, np.nan)
    adverse = 2.0 * close_safe - df["high"].shift(-1)
    mine = compute_ladder_multiplier(close_safe, adverse, ladder, LADDER_STEP_BPS)
    theirs = agamotto_multiplier(close_safe, adverse, ladder, LADDER_STEP_BPS)
    pd.testing.assert_series_equal(mine, theirs)


def test_parity_with_agamotto_full_target_at_horizon_one():
    """End-to-end: at horizon_bars=1 and fill_mode='ladder', mjolnir's whole
    target must equal agamotto's `_compute_ladder_returns` bar-for-bar.

    Both are close-to-close at size = per-leg adverse rungs, so any divergence
    means one of the two engines is mislabelling the same market.
    """
    from agamotto.research import AgamottoResearch

    df = _random_walk(seed=5)
    cfg = {"LADDER": 4, "LADDER_BPS": LADDER_STEP_BPS, "FEE": 1.75,
           "LADDER_FILL_MODE": "ladder"}
    ag = AgamottoResearch.__new__(AgamottoResearch)
    ag.config = cfg
    expected = ag._compute_ladder_returns(df, "close", "low", "high")
    got = compute_ladder_returns(cfg, df, "close", "low", "high", horizon_bars=1)
    for col in ("return_long", "return_short",
                "return_long_raw", "return_short_raw"):
        pd.testing.assert_series_equal(
            got[col], expected[col], check_names=False)


# --------------------------------------------------------------------------- #
# 5. fill_mode overrides must be untouched by the sizing change
# --------------------------------------------------------------------------- #
def test_flat_mode_still_forces_size_one():
    """'flat' overrides size to a fixed 1 regardless of the excursion."""
    df = _frame(100.0, 100.05, 99.90, 100.10)  # would be many rungs
    fake = types.SimpleNamespace(config=_cfg(mode="flat"))
    out = compute_ladder_returns(fake.config, df, "close", "low", "high")
    assert out["return_long_raw"].iloc[0] == pytest.approx(0.0005, abs=1e-12)
    assert out["return_short_raw"].iloc[0] == pytest.approx(0.0005, abs=1e-12)


def test_limit_then_taker_still_overrides_the_exit_price():
    """'limit_then_taker' changes the EXIT PRICE, not the sizing rule: the
    leftover path must still book close[t+2h], now at the base-rung floor."""
    close = [100.00, 100.02, 99.95, 99.90]
    low = [100.00, 99.99, 99.94, 99.90]
    high = [100.00, 100.03, 99.96, 99.90]
    df = pd.DataFrame({"close": close, "low": low, "high": high})
    out = compute_ladder_returns(
        _cfg(mode="limit_then_taker"), df, "close", "low", "high")
    # close_h = 100.02; the (t+h, t+2h] window high is 99.96 < close_h, so the
    # maker close does NOT fill and the leftover takers out at close_2h = 99.95.
    size = 2  # 1.0bp dip -> entry rung + 1
    assert out["return_long_raw"].iloc[0] == pytest.approx(
        (99.95 / 100.00 - 1.0) * size, rel=1e-9)
