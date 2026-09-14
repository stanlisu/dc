"""Tick ladder: base rung + per-rung entry pricing (2026-09-14).

Two defects, fixed together because they live in the same formula and a
fix to one changes how you'd test the other:

1. BASE RUNG (was dc #79/`0744ea0`, opened 2026-08-08, never merged). Sizing
   used to be `floor(adverse / step).clip(0, LADDER)` — "no free base rung" —
   so a bar whose adverse excursion was under one step got size 0: labelled
   as a position that never opened. The executor's rung 1 fills AT ENTRY
   with no adverse move needed (`knull/ladder.py:170`), so the discarded
   bars were exactly the ones where the trade was immediately right.

2. PER-RUNG ENTRY PRICING (the "separate ticket" `6b17fa3`/PR #74 flagged
   for kline on 2026-09-10 and explicitly left unfixed for tick:
   "mjolnir_pkg core/ladder.py:287-290 still books `ret * size`"). The
   target used to book EVERY rung at the anchor (`price_return * k`). A real
   k-rung stack fills rung j at `(j-1)*LADDER_BPS` against the anchor, so
   each rung earns from its own price — and the correction's SIGN can flip
   inside the `(k-1)*step` band.

Both together mean this target column CHANGES from what every existing
mjolnir/stormbreaker tick filter parquet was built on. That is not a defect
in this file — the parquets and every model/threshold/IC/Sharpe derived from
them must be regenerated; see the module docstring.

NOT under test here: `size_short = high_layers` being correlated with the
realized return (TODO.md P4, "the short drought... diagnosed, formula
unfixed"). That is a separate, still-open research question with no
committed fix — `high_layers`/`low_layers` are unchanged by this file.
"""
import numpy as np
import pandas as pd
import pytest

from agamotto.ladder import (
    compute_ladder_multiplier as agamotto_multiplier,
    compute_ladder_return as agamotto_return,
)
from agamotto.research import AgamottoResearch
from mjolnir.core.ladder import (
    compute_ladder_multiplier,
    compute_ladder_return,
    compute_ladder_returns,
    resolve_ladder_bps,
)

BP = 1e-4
STEP_BPS = 1.0
LADDER = 3


def _cfg(ladder=LADDER, fee=0.0, mode="ladder", step_bps=STEP_BPS):
    return {"LADDER": ladder, "LADDER_BPS": step_bps, "FEE": fee,
            "LADDER_FILL_MODE": mode}


def _frame(close0, close1, low1, high1):
    """3-bar frame; row 0's label reads bar 1's low/high and close[1]/close[0]."""
    return pd.DataFrame({
        "close": [close0, close1, close1],
        "low": [close0, low1, close1],
        "high": [close0, high1, close1],
    })


def _random_walk(n=400, seed=11):
    rng = np.random.default_rng(seed)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 3e-4, n))))
    return pd.DataFrame({
        "close": close,
        "high": close * (1 + np.abs(rng.normal(0, 4e-4, n))),
        "low": close * (1 - np.abs(rng.normal(0, 4e-4, n))),
    })


# --------------------------------------------------------------------------- #
# 1. resolve_ladder_bps — required, valid, and must agree with the executor
# --------------------------------------------------------------------------- #
def test_missing_ladder_bps_raises():
    cfg = {"LADDER": LADDER, "FEE": 0.0, "LADDER_FILL_MODE": "ladder"}
    with pytest.raises(KeyError, match="LADDER_BPS is required"):
        resolve_ladder_bps(cfg)


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_ladder_bps_raises(bad):
    with pytest.raises(ValueError, match="LADDER_BPS"):
        resolve_ladder_bps(_cfg(step_bps=bad))


@pytest.mark.parametrize("bad", [None, "", "1.0", True, [1.0]])
def test_non_numeric_ladder_bps_raises(bad):
    with pytest.raises(ValueError, match="LADDER_BPS must be a number"):
        resolve_ladder_bps(_cfg(step_bps=bad))


def test_executors_block_disagreeing_with_top_level_raises():
    """The venue overlay wins at boot, so a disagreeing per-venue LADDER_BPS
    IS the live rung spacing — the target would train at a spacing the
    executor never fills at."""
    cfg = _cfg()
    cfg["EXECUTORS"] = {"ltp": {"LADDER_BPS": 1.0}, "sumo": {"LADDER_BPS": 2.0}}
    with pytest.raises(ValueError, match=r"EXECUTORS\.sumo\.LADDER_BPS=2\.0"):
        resolve_ladder_bps(cfg)


def test_legacy_executor_block_is_checked_too():
    """pred_mjolnir.base.{5s,15s}_1 carry LADDER_BPS under the legacy
    single-venue `executor` block, not `EXECUTORS`."""
    cfg = _cfg()
    cfg["executor"] = {"EXEC_VENUE": "ltp", "LADDER_BPS": 2.0}
    with pytest.raises(ValueError, match=r"executor\.LADDER_BPS=2\.0"):
        resolve_ladder_bps(cfg)


def test_agreeing_or_absent_executor_blocks_pass():
    cfg = _cfg()
    cfg["EXECUTORS"] = {"ltp": {"LADDER_BPS": 1.0},
                        "sumo": {"LADDER_BPS": 1.0, "OKX_ACCOUNT": "x"}}
    assert resolve_ladder_bps(cfg) == 1.0
    cfg["EXECUTORS"]["binance"] = {"CAPITAL": 100}  # omits the key -> no conflict
    assert resolve_ladder_bps(cfg) == 1.0


# --------------------------------------------------------------------------- #
# 2. compute_ladder_multiplier — the base rung fix
# --------------------------------------------------------------------------- #
def test_no_dip_never_zeroes_the_multiplier():
    """Price never trades below the signal close -> still 1 rung, not 0."""
    close = pd.Series([100.0])
    m = compute_ladder_multiplier(close, pd.Series([100.0]), LADDER, STEP_BPS)
    assert m.iloc[0] == 1


def test_favourable_only_excursion_still_one_rung():
    close = pd.Series([100.0])
    m = compute_ladder_multiplier(close, pd.Series([100.06]), LADDER, STEP_BPS)
    assert m.iloc[0] == 1


def test_exact_one_step_dip_is_two_rungs():
    close = pd.Series([100.0])
    m = compute_ladder_multiplier(close, pd.Series([99.99]), LADDER, STEP_BPS)
    assert m.iloc[0] == 2


def test_float_guard_on_exact_two_step_dip():
    """0.02/100 / 0.0001 evaluates to 1.9999999999999998 in binary floating
    point; a bare floor() would under-size by a whole rung."""
    close = pd.Series([100.0])
    raw = (100.0 - 99.98) / 100.0 / (STEP_BPS * 1e-4)
    assert raw < 2.0, "fixture no longer exercises the float-representation edge"
    m = compute_ladder_multiplier(close, pd.Series([99.98]), LADDER, STEP_BPS)
    assert m.iloc[0] == 3


def test_over_cap_dip_clamps_to_ladder_total():
    close = pd.Series([100.0])
    m = compute_ladder_multiplier(close, pd.Series([99.91]), LADDER, STEP_BPS)
    assert m.iloc[0] == LADDER == 3


@pytest.mark.parametrize("ladder", [0, 1])
def test_ladder_zero_or_one_is_entry_rung_only(ladder):
    close = pd.Series([100.0])
    m = compute_ladder_multiplier(close, pd.Series([90.0]), ladder, STEP_BPS)
    assert m.iloc[0] == 1


def test_multiplier_never_leaves_one_to_ladder():
    close = pd.Series([100.0] * 7)
    adverse = pd.Series([100.0, 99.999, 99.0, 0.0, np.nan, 1e9, -50.0])
    m = compute_ladder_multiplier(close, adverse, LADDER, STEP_BPS)
    assert m.min() >= 1
    assert m.max() <= LADDER


@pytest.mark.parametrize("ladder", [1, 3, 10])
def test_parity_with_agamotto_multiplier_long(ladder):
    df = _random_walk()
    close_safe = df["close"].replace(0, np.nan)
    low_next = df["low"].shift(-1)
    mine = compute_ladder_multiplier(close_safe, low_next, ladder, STEP_BPS)
    theirs = agamotto_multiplier(close_safe, low_next, ladder, STEP_BPS)
    assert len(mine) >= 200
    pd.testing.assert_series_equal(mine, theirs)


@pytest.mark.parametrize("ladder", [1, 3, 10])
def test_parity_with_agamotto_multiplier_short(ladder):
    df = _random_walk(seed=23)
    close_safe = df["close"].replace(0, np.nan)
    adverse = 2.0 * close_safe - df["high"].shift(-1)
    mine = compute_ladder_multiplier(close_safe, adverse, ladder, STEP_BPS)
    theirs = agamotto_multiplier(close_safe, adverse, ladder, STEP_BPS)
    pd.testing.assert_series_equal(mine, theirs)


# --------------------------------------------------------------------------- #
# 3. compute_ladder_return — the per-rung entry pricing fix
# --------------------------------------------------------------------------- #
def test_one_rung_is_exactly_the_price_return_both_legs():
    r = pd.Series([0.0010, -0.0025, 0.0])
    k = pd.Series([1, 1, 1])
    for side in ("long", "short"):
        out = compute_ladder_return(r, k, step_bps=1.0, side=side)
        np.testing.assert_allclose(out.to_numpy(), r.to_numpy(), rtol=0, atol=1e-15)


def test_long_k2_sign_flips_versus_return_times_k():
    """Anchor 100, exit 99.996 (k=2 at 1bp step):
      rung 1 @ 100.00 : 99.996/100.00 - 1 = -0.40 bp
      rung 2 @  99.99 : 99.996/ 99.99 - 1 = +0.60 bp
      sum              = +0.20 bp     old formula: -0.40 x 2 = -0.80 bp
    """
    a, exit_px, step = 100.0, 99.996, 1.0
    r = pd.Series([exit_px / a - 1.0])
    k = pd.Series([2])
    got = compute_ladder_return(r, k, step_bps=step, side="long").iloc[0]
    rung1 = exit_px / a - 1.0
    rung2 = exit_px / (a * (1 - step * BP)) - 1.0
    assert got == pytest.approx(rung1 + rung2, abs=1e-15)
    assert got > 0, "average-entry books a small GAIN here"
    assert r.iloc[0] * 2 < 0, "the old formula books a LOSS on the same bar"


def test_short_k2_mirror_sign_flips_and_stays_unsigned():
    a, exit_px, step = 100.0, 100.004, 1.0
    r = pd.Series([exit_px / a - 1.0])
    k = pd.Series([2])
    got = compute_ladder_return(r, k, step_bps=step, side="short").iloc[0]
    rung1 = exit_px / a - 1.0
    rung2 = exit_px / (a * (1 + step * BP)) - 1.0
    assert got == pytest.approx(rung1 + rung2, abs=1e-15)
    assert got < 0 < r.iloc[0] * 2, "unsigned column flips sign vs return*k"


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_gap_from_return_times_k_is_the_entry_improvement_sum(k):
    """Pinned so nobody later 'discovers' a huge effect here: at LADDER=2,
    1bp step, the fix is worth at most ~1bp."""
    r, s = 0.0007, 1.0
    out = compute_ladder_return(pd.Series([r]), pd.Series([k]),
                                step_bps=s, side="long").iloc[0]
    expected_gap = sum((1 + r) * (j - 1) * s * BP / (1 - (j - 1) * s * BP)
                       for j in range(1, k + 1))
    assert out - r * k == pytest.approx(expected_gap, abs=1e-15)
    if k == 2:
        assert 0 < out - r * k < 1.1 * BP


def test_nan_price_return_stays_nan_never_zero():
    r = pd.Series([0.001, np.nan])
    k = pd.Series([2, 2])
    out = compute_ladder_return(r, k, step_bps=1.0, side="long")
    assert np.isnan(out.iloc[1]) and not np.isnan(out.iloc[0])


def test_bad_side_raises():
    with pytest.raises(ValueError, match="side"):
        compute_ladder_return(pd.Series([0.0]), pd.Series([1]), step_bps=1.0, side="LONG")


def test_k_below_one_or_non_integer_raises():
    with pytest.raises(ValueError):
        compute_ladder_return(pd.Series([0.0]), pd.Series([0]), step_bps=1.0, side="long")
    with pytest.raises(ValueError):
        compute_ladder_return(pd.Series([0.0]), pd.Series([1.5]), step_bps=1.0, side="long")


def test_non_positive_step_raises():
    with pytest.raises(ValueError, match="step"):
        compute_ladder_return(pd.Series([0.0]), pd.Series([2]), step_bps=0.0, side="long")


def test_misaligned_index_raises_rather_than_silently_realigning():
    r = pd.Series([0.001, 0.002], index=[0, 1])
    k = pd.Series([1, 1], index=[1, 2])
    with pytest.raises(ValueError, match="index"):
        compute_ladder_return(r, k, step_bps=1.0, side="long")


def test_parity_with_agamotto_return_function():
    """The aggregation itself is algorithm-only (no tick/kline-specific
    logic) — mjolnir's copy must equal agamotto's original bar-for-bar."""
    r = pd.Series(np.random.default_rng(7).normal(0, 5e-4, 300))
    for side in ("long", "short"):
        for k_max in (1, 2, 5):
            k = pd.Series(np.random.default_rng(3).integers(1, k_max + 1, 300))
            mine = compute_ladder_return(r, k, step_bps=1.3, side=side)
            theirs = agamotto_return(r, k, step_bps=1.3, side=side)
            pd.testing.assert_series_equal(mine, theirs)


# --------------------------------------------------------------------------- #
# 4. compute_ladder_returns — the full pipeline, base rung + per-rung pricing
# --------------------------------------------------------------------------- #
def test_full_pipeline_no_dip_bar_opens_base_rung_at_exact_return():
    """The defect this whole file exists to fix: a no-dip, +5bp bar used to
    be labelled 0.0 (never opened); it must now book exactly the price
    return at rung 1."""
    df = _frame(100.0, 100.05, 100.00, 100.05)
    out = compute_ladder_returns(_cfg(), df, "close", "low", "high")
    assert out["return_long_raw"].iloc[0] == pytest.approx(0.0005, abs=1e-12)


def test_full_pipeline_prices_each_rung_from_its_own_entry():
    """3-rung stack (9bp dip, LADDER=3 caps it): raw must equal the
    compute_ladder_return aggregate, NOT price_return * 3."""
    df = _frame(100.0, 99.95, 99.91, 100.00)
    out = compute_ladder_returns(_cfg(), df, "close", "low", "high")
    pr = 99.95 / 100.0 - 1.0
    naive = pr * 3
    expected = compute_ladder_return(
        pd.Series([pr]), pd.Series([3]), step_bps=STEP_BPS, side="long").iloc[0]
    assert out["return_long_raw"].iloc[0] == pytest.approx(expected, abs=1e-12)
    assert out["return_long_raw"].iloc[0] != pytest.approx(naive, abs=1e-9)


def test_zero_close_is_nan_not_inf():
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


def test_flat_mode_still_forces_size_one_and_is_unaffected_by_the_pricing_fix():
    df = _frame(100.0, 100.05, 99.90, 100.10)  # would be many rungs under "ladder"
    out = compute_ladder_returns(_cfg(mode="flat"), df, "close", "low", "high")
    assert out["return_long_raw"].iloc[0] == pytest.approx(0.0005, abs=1e-12)
    assert out["return_short_raw"].iloc[0] == pytest.approx(0.0005, abs=1e-12)


def test_limit_then_taker_prices_the_leftover_exit_per_rung():
    close = [100.00, 100.02, 99.95, 99.90]
    low = [100.00, 99.99, 99.94, 99.90]
    high = [100.00, 100.03, 99.96, 99.90]
    df = pd.DataFrame({"close": close, "low": low, "high": high})
    out = compute_ladder_returns(_cfg(mode="limit_then_taker"), df, "close", "low", "high")
    # close_h = 100.02; window high 99.96 < close_h -> leftover taker-exits at
    # close_2h = 99.95. size = 2 (1.0bp dip -> entry rung + 1).
    exit_ret = 99.95 / 100.00 - 1.0
    expected = compute_ladder_return(
        pd.Series([exit_ret]), pd.Series([2]), step_bps=STEP_BPS, side="long").iloc[0]
    assert out["return_long_raw"].iloc[0] == pytest.approx(expected, rel=1e-9)


def test_fee_is_charged_per_rung_after_the_pricing_fix():
    """dip to 99.99 -> 1bp adverse -> size_long=2; rally to 100.02 -> 2bp
    adverse -> size_short=3. The two legs are sized independently and must
    each pay fee * ITS OWN size, not a shared one."""
    df = _frame(100.0, 100.02, 99.99, 100.02)
    fee = 1.75
    fee_cost = fee / 1e4 * 2.0
    out = compute_ladder_returns(_cfg(fee=fee), df, "close", "low", "high")
    size_long = compute_ladder_multiplier(
        pd.Series([100.0]), pd.Series([99.99]), LADDER, STEP_BPS).iloc[0]
    size_short = compute_ladder_multiplier(
        pd.Series([100.0]), pd.Series([2 * 100.0 - 100.02]), LADDER, STEP_BPS).iloc[0]
    assert (size_long, size_short) == (2, 3)
    raw = out["return_long_raw"].iloc[0]
    assert out["return_long"].iloc[0] == pytest.approx(raw - fee_cost * size_long, abs=1e-12)
    raw_s = out["return_short_raw"].iloc[0]
    assert out["return_short"].iloc[0] == pytest.approx(raw_s + fee_cost * size_short, abs=1e-12)


def test_ladder_zero_or_one_no_longer_zeroes_the_target():
    """REVERSED 2026-08-08 (dc #79) / re-pinned here: `LADDER: 0`/`1` mean
    'entry rung only' (size 1), not 'no position'. 8 of 10 live tick arms
    set LADDER=1, matching EXECUTORS.ltp.MAX_RUNGS_PER_LADDER=1."""
    df = _frame(100.0, 100.05, 99.90, 100.10)
    for ladder in (0, 1):
        out = compute_ladder_returns(_cfg(ladder=ladder), df, "close", "low", "high")
        pr = 100.05 / 100.0 - 1.0
        assert out["return_long_raw"].iloc[0] == pytest.approx(pr, abs=1e-12)
        assert out["return_short_raw"].iloc[0] == pytest.approx(pr, abs=1e-12)


def test_parity_with_agamotto_full_target_at_horizon_one():
    """End-to-end: at horizon_bars=1 and fill_mode='ladder', mjolnir's whole
    target must equal agamotto's `_compute_ladder_returns` bar-for-bar — same
    sizing, same per-rung pricing, same fee handling."""
    rw = _random_walk(seed=5)
    cfg = {"LADDER": 4, "LADDER_BPS": STEP_BPS, "FEE": 1.75,
           "LADDER_FILL_MODE": "ladder"}
    ag = AgamottoResearch.__new__(AgamottoResearch)
    ag.config = cfg
    expected = ag._compute_ladder_returns(rw, "close", "low", "high")
    got = compute_ladder_returns(cfg, rw, "close", "low", "high", horizon_bars=1)
    for col in ("return_long", "return_short",
                "return_long_raw", "return_short_raw"):
        pd.testing.assert_series_equal(got[col], expected[col], check_names=False)


# --------------------------------------------------------------------------- #
# 5. LADDER / LADDER_FILL_MODE still fail fast (unchanged behaviour, re-pinned
#    because compute_ladder_returns is rewritten in this change)
# --------------------------------------------------------------------------- #
def test_ladder_missing_raises():
    cfg = {"LADDER_BPS": STEP_BPS, "FEE": 0.0, "LADDER_FILL_MODE": "ladder"}
    with pytest.raises(KeyError, match="LADDER is required"):
        compute_ladder_returns(cfg, _random_walk(), "close", "low", "high")


@pytest.mark.parametrize("bad", [None, "", False, 1.5, -1])
def test_ladder_non_whole_or_negative_raises(bad):
    cfg = _cfg(ladder=bad)
    with pytest.raises(ValueError, match="LADDER"):
        compute_ladder_returns(cfg, _random_walk(), "close", "low", "high")


def test_unknown_fill_mode_raises():
    with pytest.raises(ValueError, match="LADDER_FILL_MODE"):
        compute_ladder_returns(_cfg(mode="bogus"), _random_walk(), "close", "low", "high")
