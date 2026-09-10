"""RED phase: the kline ladder target must price the laddered AVERAGE ENTRY, not
multiply one anchor-referenced return by the rung count.

What the current target does (research.py `_compute_ladder_returns` and the inline
copy in `engineer_features`): `return_long_raw = price_return * size_long`, where
`price_return = close[t+1]/close[t] - 1` and `size_long = k` = the rung count that
would have filled (ladder.py `compute_ladder_multiplier`, in [1, LADDER]). Every
rung is booked AT THE ANCHOR. On the live 15m arm (LADDER=2) that reads as
`y_true_raw == 2.000000 x close-to-close return` on 4/4 sampled rows (2026-09-10).

What a real ladder does (tasks/lessons.md 2026-08-04, TODO P0.6's open half): rung
1 fills at the anchor (post-only but chased, ladder.py docstring); rung j>=2 fills
`(j-1)*LADDER_BPS` against it. Each rung earns from ITS OWN entry price:

    long   sum_{j=1..k} (exit/ent_j - 1),   ent_j = a*(1 - (j-1)*step)
    short  sum_{j=1..k} (exit/ent_j - 1),   ent_j = a*(1 + (j-1)*step)   [UNSIGNED]

The short column stays UNSIGNED (a market return measured from each rung's entry)
because the consumer applies the trade direction — marvel books
`signal * y_true_raw` — and negating here would double-sign the short book
(mm_target.py:37-48). Fees stay per rung, as they are today.

The correction is small on LADDER=2 (at most ~1 bp per 2-rung stack) but its SIGN
can flip inside the (k-1)*step band, and it is the half of P0.6 that TODO.md says
"is larger than the part that was fixed. Do not treat P0.6 as closed."

These tests are written BEFORE `compute_ladder_return` exists and must fail on
ImportError. The rung-COUNT helper `compute_ladder_multiplier` is NOT under test —
it is correct and unchanged; only the aggregation over those k rungs changes.
"""
import numpy as np
import pandas as pd
import pytest

from agamotto.ladder import compute_ladder_return  # noqa: F401 — does not exist yet (RED)

BP = 1e-4


def _s(*vals):
    return pd.Series(list(vals), dtype=float)


# --------------------------------------------------------------------------- #
# k = 1 is the identity: one rung at the anchor earns exactly the price return.
# --------------------------------------------------------------------------- #
def test_one_rung_is_exactly_the_price_return_both_legs():
    r = _s(0.0010, -0.0025, 0.0)
    k = pd.Series([1, 1, 1])
    for side in ("long", "short"):
        out = compute_ladder_return(r, k, step_bps=1.0, side=side)
        np.testing.assert_allclose(out.to_numpy(), r.to_numpy(), rtol=0, atol=1e-15)


# --------------------------------------------------------------------------- #
# THE sign-flip case from the spec (Researcher, 2026-09-10). Long, anchor 100,
# next-bar low 99.98 (k=2 at step 1 bp), exit 99.996.
#   rung 1 @ 100.00 : 99.996/100.00 - 1 = -0.40 bp
#   rung 2 @  99.99 : 99.996/ 99.99 - 1 = +0.60 bp   (= 0.006/99.99)
#   sum              = +0.20 bp            current formula: -0.40 x 2 = -0.80 bp
# --------------------------------------------------------------------------- #
def test_long_k2_sign_flips_versus_return_times_k():
    a, exit_px, step = 100.0, 99.996, 1.0
    r = _s(exit_px / a - 1.0)
    k = pd.Series([2])
    got = compute_ladder_return(r, k, step_bps=step, side="long").iloc[0]
    rung1 = exit_px / a - 1.0
    rung2 = exit_px / (a * (1 - step * BP)) - 1.0
    assert got == pytest.approx(rung1 + rung2, abs=1e-15)
    assert got > 0, "average-entry books a small GAIN here"
    assert r.iloc[0] * 2 < 0, "the old formula books a LOSS on the same bar"


def test_short_k2_mirror_sign_flips_and_stays_unsigned():
    """Short mirror: anchor 100, next-bar HIGH 100.02 (k=2), exit 100.004.
    Stored UNSIGNED — a market return from each rung's entry — so the value
    here is what the consumer NEGATES for the short book:
      rung 1 @ 100.00 : 100.004/100.00 - 1 = +0.40 bp
      rung 2 @ 100.01 : 100.004/100.01 - 1 = -0.60 bp
      sum              = -0.20 bp  -> short PnL +0.20 bp after the consumer's -1
    Old: +0.40 x 2 = +0.80 bp -> short PnL -0.80 bp. Sign flips."""
    a, exit_px, step = 100.0, 100.004, 1.0
    r = _s(exit_px / a - 1.0)
    k = pd.Series([2])
    got = compute_ladder_return(r, k, step_bps=step, side="short").iloc[0]
    rung1 = exit_px / a - 1.0
    rung2 = exit_px / (a * (1 + step * BP)) - 1.0
    assert got == pytest.approx(rung1 + rung2, abs=1e-15)
    assert got < 0 < r.iloc[0] * 2, "unsigned column flips sign vs return*k"


# --------------------------------------------------------------------------- #
# The exact size of the correction: the difference from `return * k` is the sum
# of the per-rung entry improvements. For rung j the extra is
#   (1+r)/(1-(j-1)s) - (1+r)  =  (1+r) * (j-1)s / (1-(j-1)s)
# so to first order the gap is (1+r) * s * k(k-1)/2 -- at most ~1 bp on a
# 2-rung, 1 bp ladder. Pinned so nobody later "discovers" a huge effect here.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_gap_from_return_times_k_is_the_entry_improvement_sum(k):
    r, s = 0.0007, 1.0
    out = compute_ladder_return(_s(r), pd.Series([k]), step_bps=s, side="long").iloc[0]
    expected_gap = sum((1 + r) * (j - 1) * s * BP / (1 - (j - 1) * s * BP)
                       for j in range(1, k + 1))
    assert out - r * k == pytest.approx(expected_gap, abs=1e-15)
    if k == 2:
        assert 0 < out - r * k < 1.1 * BP, "LADDER=2, 1 bp step: the fix is worth ~1 bp"


# --------------------------------------------------------------------------- #
# Vectorised over a Series with mixed k, including k=1 rows, and index preserved.
# --------------------------------------------------------------------------- #
def test_vectorised_mixed_k_matches_rowwise_and_keeps_index():
    idx = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    r = pd.Series([0.001, -0.002, 0.0003, -0.0001], index=idx)
    k = pd.Series([1, 2, 3, 2], index=idx)
    out = compute_ladder_return(r, k, step_bps=1.0, side="long")
    assert out.index.equals(idx)
    for i in range(4):
        row = compute_ladder_return(pd.Series([r.iloc[i]]), pd.Series([k.iloc[i]]),
                                    step_bps=1.0, side="long").iloc[0]
        assert out.iloc[i] == pytest.approx(row, abs=1e-15)


def test_nan_price_return_stays_nan_never_zero():
    """The last bar has no next close, so price_return is NaN there. It must stay
    NaN — a 0.0 would be a silent 'flat' label on a bar that has no label."""
    r = _s(0.001, np.nan)
    k = pd.Series([2, 2])
    out = compute_ladder_return(r, k, step_bps=1.0, side="long")
    assert np.isnan(out.iloc[1]) and not np.isnan(out.iloc[0])


# --------------------------------------------------------------------------- #
# Degenerate inputs RAISE, never default (CLAUDE.md).
# --------------------------------------------------------------------------- #
def test_bad_side_raises():
    with pytest.raises(ValueError, match="side"):
        compute_ladder_return(_s(0.0), pd.Series([1]), step_bps=1.0, side="LONG")


def test_k_below_one_or_non_integer_raises():
    with pytest.raises(ValueError):
        compute_ladder_return(_s(0.0), pd.Series([0]), step_bps=1.0, side="long")
    with pytest.raises(ValueError):
        compute_ladder_return(_s(0.0), pd.Series([1.5]), step_bps=1.0, side="long")


def test_non_positive_step_raises():
    with pytest.raises(ValueError, match="step"):
        compute_ladder_return(_s(0.0), pd.Series([2]), step_bps=0.0, side="long")


def test_misaligned_index_raises_rather_than_silently_realigning():
    r = pd.Series([0.001, 0.002], index=[0, 1])
    k = pd.Series([1, 1], index=[1, 2])
    with pytest.raises(ValueError, match="index"):
        compute_ladder_return(r, k, step_bps=1.0, side="long")
