"""Unit tests for MjolnirResearch._compute_ladder_returns fill modes.

Covers the three LADDER_FILL_MODE values:
  - "ladder"            : whole opened stack marked-to-market at close[t+h]
  - "flat"              : fixed size 1, taker at the horizon close
  - "limit_then_taker"  : maker close at close[t+h] if the (t+h, t+2h] window
                          reaches it, else taker-close the leftover at close[t+2h]

The method only reads self.config (LADDER, FEE, LADDER_FILL_MODE), so we drive
it through a lightweight SimpleNamespace rather than constructing the full
MjolnirResearch (which has heavy __init__ deps).

Run (dc deobfuscated source must win over the installed obfuscated copy):
  PYTHONPATH=/home/stan/sandbox/dc/mjolnir_pkg/src \
    /opt/miniconda3/envs/py313/bin/python -m pytest \
    mjolnir_pkg/tests/test_ladder_returns.py -q
"""
import types

import numpy as np
import pandas as pd
import pytest

from mjolnir.core.research import MjolnirResearch

STEP = 0.0001  # 1 bp
FEE = 1.75     # taker bps; fee_cost = FEE/1e4 * 2 = 3.5 bps round trip
FEE_COST = FEE / 10000.0 * 2.0
LADDER = 3


def _call(df, mode, horizon_bars=1, ladder=LADDER, fee=FEE):
    fake = types.SimpleNamespace(
        config={"LADDER": ladder, "FEE": fee, "LADDER_FILL_MODE": mode})
    return MjolnirResearch._compute_ladder_returns(
        fake, df, "close", "low", "high", horizon_bars=horizon_bars)


# Hand-built single-symbol OHLC frame (8 bars, horizon_bars=1 -> t+1, t+2 are
# the next two bars). Key rows:
#   idx 0: LONG penetrated  — dip 3.5bp (n=3), close_h=100.02, high[t+2]=100.05
#          >= close_h -> exit at close_h=100.02 -> ret_long = +0.0002.
#          SHORT penetrated — rise 3.5bp (n=3), low[t+2]=99.93 <= close_h ->
#          exit at close_h -> ret_short = +0.0002 (UNSIGNED market move; the
#          short LOST here, which the consumer books via signal = -1).
#   idx 3: LONG leftover    — dip 3.5bp (n=3), close_h=99.98, high[t+2]=99.97
#          < close_h -> NOT penetrated -> taker exit at close_2h=close[5]=99.95
#          -> ret_long = -0.0005 (strictly worse than the penetrated -0.0002).
CLOSE = [100.00, 100.02, 100.00, 100.00, 99.98, 99.95, 99.96, 99.97]
LOW = [99.95, 99.965, 99.93, 99.92, 99.965, 99.90, 99.90, 99.90]
HIGH = [100.05, 100.035, 100.05, 100.06, 100.02, 99.97, 100.00, 100.00]


@pytest.fixture
def df():
    return pd.DataFrame({"close": CLOSE, "low": LOW, "high": HIGH})


def test_entry_sizing_unchanged(df):
    """n_long/n_short still equal the entry-side penetration layers (raw/return
    scale linearly with size, so size = return_long_raw / per-unit gross)."""
    out = _call(df, "limit_then_taker")
    # idx0: dip & rise both 3.5bp -> 3 layers each, capped at LADDER=3.
    # per-unit gross long at idx0 = +0.0002 -> raw = 0.0002 * 3.
    assert out["return_long_raw"].iloc[0] == pytest.approx(0.0002 * 3, rel=1e-6)
    # UNSIGNED (2026-07-29): the short's own exit path moved +0.0002. It is NOT
    # negated here — marvel's engine applies signal = -1. Matches agamotto.
    assert out["return_short_raw"].iloc[0] == pytest.approx(0.0002 * 3, rel=1e-6)


def test_limit_then_taker_penetrated_long(df):
    out = _call(df, "limit_then_taker")
    # exit at close_h=100.02 -> gross +0.0002, fee 0.00035, size 3.
    assert out["return_long"].iloc[0] == pytest.approx((0.0002 - FEE_COST) * 3, rel=1e-6)
    # Fee is ADDED on the short (unsigned target): it must fall below -fee to pay.
    assert out["return_short"].iloc[0] == pytest.approx((0.0002 + FEE_COST) * 3, rel=1e-6)


def test_limit_then_taker_leftover_long(df):
    out = _call(df, "limit_then_taker")
    # NOT penetrated -> taker exit at close_2h=99.95 -> gross -0.0005.
    assert out["return_long"].iloc[3] == pytest.approx((-0.0005 - FEE_COST) * 3, rel=1e-6)
    # The leftover (taker-at-t+2) exit is strictly worse than the would-be maker
    # exit at close_h (gross -0.0002) — leftover honestly eats the later move.
    maker_hypothetical = (-0.0002 - FEE_COST) * 3
    assert out["return_long"].iloc[3] < maker_hypothetical


def test_limit_then_taker_full_window_required(df):
    """The final 2h bars have no close[t+2h] -> NaN target (dropped downstream)."""
    out = _call(df, "limit_then_taker", horizon_bars=1)
    # 2h = 2 bars: last two rows NaN, the one before is finite.
    assert out["return_long"].iloc[-1] != out["return_long"].iloc[-1]  # NaN
    assert out["return_long"].iloc[-2] != out["return_long"].iloc[-2]  # NaN
    assert np.isfinite(out["return_long"].iloc[-3])


def test_horizon_2_window(df):
    """horizon_bars=2 -> t+1 = +2 bars, t+2 = +4 bars; last 4 rows NaN."""
    out = _call(df, "limit_then_taker", horizon_bars=2)
    assert out["return_long"].iloc[-4:].isna().all()
    assert np.isfinite(out["return_long"].iloc[-5])


def test_ladder_mode_is_close_to_close(df):
    """Regression guard: 'ladder' == (price_return - fee) * entry_layers."""
    out = _call(df, "ladder")
    price_return = df["close"].pct_change(1, fill_method=None).shift(-1)
    # entry layers from the next-bar excursion (horizon_bars=1).
    low_next = df["low"].shift(-1)
    high_next = df["high"].shift(-1)
    n_long = np.floor(((df["close"] - low_next) / df["close"]) / STEP).clip(0, LADDER).fillna(0)
    n_short = np.floor(((high_next - df["close"]) / df["close"]) / STEP).clip(0, LADDER).fillna(0)
    exp_long = (price_return - FEE_COST) * n_long
    exp_short = (price_return + FEE_COST) * n_short
    pd.testing.assert_series_equal(
        out["return_long"], exp_long.rename("return_long"), check_names=True)
    pd.testing.assert_series_equal(
        out["return_short"], exp_short.rename("return_short"), check_names=True)


def test_penetrated_equals_ladder(df):
    """When penetrated, limit_then_taker exits at close[t+h] == the ladder
    close-to-close exit, so the two modes agree on that row (idx0)."""
    a = _call(df, "limit_then_taker")["return_long"].iloc[0]
    b = _call(df, "ladder")["return_long"].iloc[0]
    assert a == pytest.approx(b, rel=1e-9)


def test_flat_mode_size_one(df):
    """'flat' = fixed size 1 at the close-to-close return."""
    out = _call(df, "flat")
    price_return = df["close"].pct_change(1, fill_method=None).shift(-1)
    assert out["return_long"].iloc[0] == pytest.approx(price_return.iloc[0] - FEE_COST, rel=1e-6)
    assert out["return_long_raw"].iloc[0] == pytest.approx(price_return.iloc[0], rel=1e-6)


def test_unknown_mode_raises(df):
    with pytest.raises(ValueError, match="LADDER_FILL_MODE"):
        _call(df, "bogus")


# ── LADDER is REQUIRED (no magic default, no `X or Y`) ───────────────────────
# Was `int(config.get("LADDER", 1) or 0)`: an absent key silently became 1 (all
# six pred_stormbreaker.base.*_1 arms), and a null/""/False value silently
# became 0. LADDER caps the rungs of the TARGET, so either path changes what
# the model is trained on with nothing in the logs. CLAUDE.md bans both the
# magic default and the `get(K, X) or Y` idiom.

def test_ladder_missing_raises(df):
    fake = types.SimpleNamespace(config={"FEE": FEE, "LADDER_FILL_MODE": "ladder"})
    with pytest.raises(KeyError, match="LADDER is required"):
        MjolnirResearch._compute_ladder_returns(
            fake, df, "close", "low", "high", horizon_bars=1)


@pytest.mark.parametrize("bad", [None, "", False, 1.5, -1])
def test_ladder_non_whole_or_negative_raises(df, bad):
    """The old `or 0` swallowed None/""/False into a silent 0 rung cap."""
    with pytest.raises(ValueError, match="LADDER"):
        _call(df, "ladder", ladder=bad)


def test_ladder_zero_is_accepted_and_not_collapsed(df):
    """`LADDER: 0` is a legitimate value (entry rung only -> size 0) and must
    survive as 0, not be re-read as the old default of 1."""
    out = _call(df, "ladder", ladder=0)
    assert (out["return_long_raw"].fillna(0.0) == 0.0).all()
    assert (out["return_short_raw"].fillna(0.0) == 0.0).all()
