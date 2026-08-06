"""Kline ladder sizing: per-leg, base rung, LADDER_BPS step. No `min()` gate.

THE DEFECT THIS REPLACES. `agamotto/research.py:245` and `orb/research.py:599`
both sized every bar as `size = min(long_layers, short_layers)` — requiring BOTH
a downward AND an upward excursion within the next bar. The upward requirement
has no counterpart in the executor:

  * `knull/ladder.py:126-296` — rungs fill on ADVERSE moves only. Rung 1 fills at
    entry (`:165-172`, `_level = 1` as soon as `filled_qty >= rung_qty`); rungs
    2..LADDER each need `LADDER_BPS` against the position. Nothing anywhere
    requires a favourable excursion.
  * `knull/base_executor.py:3890-3891` — "Hold position until next signal (FLAT
    or direction flip)" -> `_place_exit_at_book`, chased into the book with no
    give-up (`:4166-4172`). The exit has no recovery precondition.
  * The only bounce-conditioned exit is MarketMaker mode
    (`knull/market_maker.py:122-219`), which NO setting.json enables.

So a long that ladders into a falling market realizes its whole loss live, and
was labelled exactly 0.0. Measured on `pred_agamotto.base.15m_1`
(`filter_r029_and_r001_long.parquet`), losing bars only: 20,901 bars with no
intrabar bounce averaged -36.7bp of real loss and were labelled 0.0bp, while the
186,199 bars that did bounce averaged a MILDER -33.0bp and were amplified to
-260.4bp. The label punished the smaller losses and ignored the bigger ones.

MJOLNIR DIVERGENCE (deliberate, not adopted here). `mjolnir/core/ladder.py:110`
made the same per-leg fix on 2026-06-20 but WITHOUT a base rung: "use n layers,
NOT 1 + n — no free base rung, so an excursion of < 1bps fills nothing." That
holds only if the entry order needs price to come to it. For kline it does not:
`knull/execution_style.py:185-196` (LadderMaker, the kline mode) rests the entry
post-only but `should_reprice_entry()` returns True — "Maker chases" — so the
entry follows the book and fills without an adverse move; Taker mode fills
outright. Kline therefore keeps the base rung. Tick is out of scope here.
"""
import numpy as np
import pandas as pd
import pytest

from agamotto.ladder import compute_ladder_multiplier

LADDER = 10
STEP_BPS = 1.0


def _cfg(**over):
    cfg = {"LADDER": LADDER, "LADDER_BPS": STEP_BPS, "FEE": 0.0}
    cfg.update(over)
    return cfg


# --------------------------------------------------------------------------- #
# The pure sizing function
# --------------------------------------------------------------------------- #
def test_dip_without_bounce_is_sized_not_zeroed():
    """THE defect. Next bar digs 10bp below close and never trades above it.

    Old: short_layers == 0 -> min(...) == 0 -> the bar vanishes from the label.
    New: the long ladder filled 10 rungs and the loss is multiplied by them.
    """
    close = pd.Series([100.0])
    low_next = pd.Series([99.90])   # 10bp dip -> rungs fill all the way down

    size = compute_ladder_multiplier(close, low_next, ladder=LADDER,
                                     step_bps=STEP_BPS)

    assert int(size.iloc[0]) == 10, "1 base rung + 9 reachable extras = LADDER"


def test_the_two_legs_are_independent():
    """A long's size must not depend on the short's excursion, or vice versa.

    Same bar, viewed from each side: it dips 10bp and never rises. The long
    ladders fully; the short only ever fills its entry rung.
    """
    close = pd.Series([100.0])
    low_next = pd.Series([99.90])    # 10bp below
    high_next = pd.Series([100.0])   # never above close

    long_size = compute_ladder_multiplier(close, low_next, ladder=LADDER,
                                          step_bps=STEP_BPS)
    # short leg: caller passes the mirrored distance (high above close)
    short_size = compute_ladder_multiplier(close, 2 * close - high_next,
                                           ladder=LADDER, step_bps=STEP_BPS)

    assert int(long_size.iloc[0]) == 10
    assert int(short_size.iloc[0]) == 1, "no upward excursion -> entry rung only"


# --------------------------------------------------------------------------- #
# Config plumbing — CLAUDE.md forbids silent fallbacks on numeric config
# --------------------------------------------------------------------------- #
def test_missing_LADDER_raises_rather_than_defaulting():
    """`int(config.get("LADDER", 1) or 0)` is the banned `get(K,X) or Y` idiom
    (CLAUDE.md; the FEE bug 2026-04-27). A defaulted ladder silently resizes
    every trade in the book."""
    from agamotto.ladder import ladder_params

    with pytest.raises(KeyError, match="LADDER"):
        ladder_params({"LADDER_BPS": 1.0, "FEE": 0.0})


def test_missing_LADDER_BPS_raises_rather_than_defaulting():
    """The step was hardcoded to 0.0001, which silently matched only configs
    with LADDER_BPS == 1.0. On 1h/4h/1d LADDER_BPS was null, so target and
    executor could not be reconciled at all."""
    from agamotto.ladder import ladder_params

    with pytest.raises(KeyError, match="LADDER_BPS"):
        ladder_params({"LADDER": 10, "FEE": 0.0})


def test_step_bps_scales_rung_width():
    """2bp dip is 1 extra rung at a 2bp step, 2 extra rungs at a 1bp step."""
    close = pd.Series([100.0])
    low_next = pd.Series([99.98])

    assert int(compute_ladder_multiplier(
        close, low_next, ladder=LADDER, step_bps=2.0).iloc[0]) == 2
    assert int(compute_ladder_multiplier(
        close, low_next, ladder=LADDER, step_bps=1.0).iloc[0]) == 3


# --------------------------------------------------------------------------- #
# Integration: the research engines that build the actual target columns
#
# One bar, engineered so the two legs disagree maximally:
#   bar 0 close = 100.00
#   bar 1  low  =  99.90  (10bp BELOW  -> long ladder fills to the cap)
#          high = 100.00  (never above -> short ladder gets its entry rung only)
#          close=  99.90  -> price_return = -0.001 (-10bp)
#
# Old label: min(long=10, short=0) = 0  -> BOTH legs recorded 0.0. The bar
# vanished, despite a real 10bp loss on a fully-laddered long.
# New label: long -10bp x 10 rungs = -0.01 ; short -10bp x 1 rung = -0.001.
# --------------------------------------------------------------------------- #
DIP_NO_BOUNCE = pd.DataFrame({
    "open":  [100.0, 100.0],
    "high":  [100.0, 100.00],
    "low":   [100.0,  99.90],
    "close": [100.0,  99.90],
})


def test_orb_labels_the_dip_no_bounce_loss_instead_of_zeroing_it():
    pytest.importorskip("orb.research", reason="orb package not installed")
    from orb.research import OrbResearch

    orb = OrbResearch.__new__(OrbResearch)
    orb.config = _cfg()

    out = orb._compute_ladder_returns(
        DIP_NO_BOUNCE, close_col="close", low_col="low", high_col="high")

    assert out["return_long_raw"].iloc[0] == pytest.approx(-0.01, abs=1e-9), (
        "fully-laddered long into a falling bar must book 10 rungs of loss")
    assert out["return_short_raw"].iloc[0] == pytest.approx(-0.001, abs=1e-9), (
        "short never got an adverse (upward) move: entry rung only")


def test_agamotto_labels_the_dip_no_bounce_loss_instead_of_zeroing_it():
    from agamotto.research import AgamottoResearch

    ag = AgamottoResearch.__new__(AgamottoResearch)
    ag.config = _cfg()

    out = ag._compute_ladder_returns(
        DIP_NO_BOUNCE, close_col="close", low_col="low", high_col="high")

    assert out["return_long_raw"].iloc[0] == pytest.approx(-0.01, abs=1e-9)
    assert out["return_short_raw"].iloc[0] == pytest.approx(-0.001, abs=1e-9)


def test_both_engines_agree_bar_for_bar():
    """orb inherits from agamotto; a divergent private copy is how the two
    silently drifted apart in the first place."""
    pytest.importorskip("orb.research", reason="orb package not installed")
    from orb.research import OrbResearch
    from agamotto.research import AgamottoResearch

    rng = np.random.default_rng(0)
    n = 200
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 3e-4, n))))
    df = pd.DataFrame({
        "open": close,
        "close": close,
        "high": close * (1 + np.abs(rng.normal(0, 4e-4, n))),
        "low": close * (1 - np.abs(rng.normal(0, 4e-4, n))),
    })

    orb = OrbResearch.__new__(OrbResearch); orb.config = _cfg()
    ag = AgamottoResearch.__new__(AgamottoResearch); ag.config = _cfg()

    a = ag._compute_ladder_returns(df, "close", "low", "high")
    o = orb._compute_ladder_returns(df, "close", "low", "high")

    pd.testing.assert_frame_equal(a, o)


# --------------------------------------------------------------------------- #
# Per-leg LADDER (2026-08-06)
#
# MEASURED: the two legs want opposite ladder depths. Direction-only IC
# (features vs the PLAIN return, so uncontaminated by the size term), comparing
# all bars against bars where size hit the cap:
#
#                  all bars   size>=cap    the multiplier upweights...
#     15m long      0.0293      0.0261     LESS predictable bars
#      1h long      0.0188      0.0174     LESS predictable bars
#     15m short     0.0490      0.0580     MORE predictable bars
#      1h short     0.0223      0.0270     MORE predictable bars
#
# and a LADDER sweep on 1.7-6.4M rows agrees: top-16 |IC| falls monotonically
# with LADDER on longs (15m 0.0293 -> 0.0220 at L=20) and rises monotonically on
# shorts (15m 0.0490 -> 0.0638). So one shared LADDER is wrong for one leg
# whichever value is chosen.
#
# CAVEAT recorded here because the config cannot: `LADDER` is also what the
# EXECUTOR fills (knull/ladder.py:157 `ladder_max = self._LADDER`). A per-leg
# TARGET ladder that differs from it models rungs that will not fill — the same
# defect as the min() gate. These keys are for research arms; reconcile with the
# executor before any of it is deployed.
# --------------------------------------------------------------------------- #
def test_per_leg_ladders_are_read_independently():
    from agamotto.ladder import ladder_params

    long_l, short_l, step = ladder_params(
        {"LADDER_LONG": 2, "LADDER_SHORT": 10, "LADDER_BPS": 1.0})

    assert (long_l, short_l, step) == (2, 10, 1.0)


def test_per_leg_ladders_actually_size_the_two_legs_differently():
    """The whole point: same bar, same excursions, different rung counts."""
    from agamotto.research import AgamottoResearch

    r = AgamottoResearch.__new__(AgamottoResearch)
    r.config = {"LADDER_LONG": 2, "LADDER_SHORT": 10, "LADDER_BPS": 1.0, "FEE": 0.0}
    # bar 1 digs 10bp below AND rallies 10bp above -> both ladders saturate,
    # so each leg reports exactly its own cap.
    df = pd.DataFrame({
        "open":  [100.0, 100.00],
        "high":  [100.0, 100.10],
        "low":   [100.0,  99.90],
        "close": [100.0,  99.90],
    })
    out = r._compute_ladder_returns(df, "close", "low", "high")
    pr = -0.001

    assert out["return_long_raw"].iloc[0] == pytest.approx(pr * 2, abs=1e-12)
    assert out["return_short_raw"].iloc[0] == pytest.approx(pr * 10, abs=1e-12)


def test_plain_LADDER_still_works_for_both_legs():
    """DEPRECATED back-compat: 159 kline settings carry only LADDER. The chain
    must fall back to it rather than breaking every experiment at once."""
    from agamotto.ladder import ladder_params

    assert ladder_params({"LADDER": 7, "LADDER_BPS": 1.0}) == (7, 7, 1.0)


def test_per_leg_key_overrides_the_shared_one_for_that_leg_only():
    from agamotto.ladder import ladder_params

    assert ladder_params(
        {"LADDER": 7, "LADDER_LONG": 2, "LADDER_BPS": 1.0}) == (2, 7, 1.0)


def test_no_ladder_key_at_all_still_raises():
    """The fallback chain must END in a fail-fast (CLAUDE.md)."""
    from agamotto.ladder import ladder_params

    with pytest.raises(KeyError, match="LADDER"):
        ladder_params({"LADDER_BPS": 1.0})


# --------------------------------------------------------------------------- #
# The CROSS-TF target (2026-08-07)
#
# OrbResearch has TWO target paths and 2026-08-06 fixed only one. `verticalize()`
# builds the cross-TF target (TARGET_TF != BASE_TF) from the exit bar's own
# low/high, and kept a private copy of the old maths: `min(long_layers,
# short_layers)`, a step hardcoded to 0.0001 (silently correct only when
# LADDER_BPS == 1.0), and one shared LADDER read through the banned
# `get(K, X) or Y`. All five production orb settings are BASE_TF == TARGET_TF and
# take the same-TF path, which is exactly why this branch drifted unnoticed.
# --------------------------------------------------------------------------- #
NATIVE = "BTCUSDT"
SYMBOL = "BINANCE_PERP_BTC_USDT"


def _orb_cross_tf(cfg, entry_close, exit_close, exit_low, exit_high):
    """Run OrbResearch.verticalize() over a cross-TF fixture.

    load()/engineer_features() are bypassed: verticalize() consumes
    `self.features`, and every column lookup other than the four cross-TF ones
    is guarded by `in self.features.columns`.

    Args are array-likes of equal length; returns the vertical_features frame.
    """
    pytest.importorskip("orb.research", reason="orb package not installed")
    from orb.research import OrbResearch

    n = len(entry_close)
    orb = OrbResearch.__new__(OrbResearch)
    orb.config = {**cfg, "SYMBOLS": [SYMBOL]}
    orb.timeframes = ["15m", "1h"]
    orb.base_tf = "15m"
    orb.target_tf = "1h"          # != base_tf -> the cross-TF branch
    orb.features = pd.DataFrame({
        f"15m_{NATIVE}_close": entry_close,
        f"1h_{NATIVE}_exit_close": exit_close,
        f"1h_{NATIVE}_exit_low": exit_low,
        f"1h_{NATIVE}_exit_high": exit_high,
        "year": [2026] * n,
        "month": [1] * n,
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min"))
    orb.verticalize()
    return orb.vertical_features


def test_cross_tf_dip_no_bounce_is_sized_not_zeroed():
    """The same defect as the same-TF path, on the branch that kept the copy.

    Exit bar digs 10bp under entry and never trades above it. Old:
    short_layers == 0 -> min(...) == 0 -> BOTH legs labelled 0.0 despite a real
    10bp loss on a fully-laddered long.
    """
    out = _orb_cross_tf(_cfg(), [100.0], [99.90], [99.90], [100.00])

    assert out["return_long_raw"].iloc[0] == pytest.approx(-0.01, abs=1e-9), (
        "fully-laddered long into a falling exit bar must book 10 rungs of loss")
    assert out["return_short_raw"].iloc[0] == pytest.approx(-0.001, abs=1e-9), (
        "short never got an adverse (upward) move: entry rung only")


def test_cross_tf_per_leg_ladders_size_the_legs_differently():
    """Same bar, same excursions, different rung counts — the point of the split."""
    cfg = _cfg(LADDER_LONG=2, LADDER_SHORT=10)
    # Exit bar digs 10bp below AND rallies 10bp above, so both ladders saturate
    # and each leg reports exactly its own cap.
    out = _orb_cross_tf(cfg, [100.0], [99.90], [99.90], [100.10])
    pr = -0.001

    assert out["return_long_raw"].iloc[0] == pytest.approx(pr * 2, abs=1e-12)
    assert out["return_short_raw"].iloc[0] == pytest.approx(pr * 10, abs=1e-12)


def test_cross_tf_short_target_is_not_negated():
    """`return_short` was `-(raw + fee) * size` while `return_short_raw` was
    `raw * size` — the two short columns of the SAME block disagreed in sign, so
    one was necessarily wrong. Agamotto and orb's own same-TF path both keep the
    short target in forward-return space; a NEGATIVE threshold under
    `y_pred < thresh` (CLAUDE.md) depends on it. Negated, a good short scored
    HIGH and the leg selected backwards.
    """
    out = _orb_cross_tf(_cfg(FEE=2.25), [100.0], [99.90], [99.90], [100.10])
    short, short_raw = out["return_short"].iloc[0], out["return_short_raw"].iloc[0]

    assert short < 0 and short_raw < 0, (
        f"a falling exit bar must give a NEGATIVE short target on both columns, "
        f"got return_short={short}, return_short_raw={short_raw}")
    # Fee makes the short target strictly less adverse, never sign-flipped.
    assert short > short_raw


def test_cross_tf_honours_LADDER_BPS():
    """The step was hardcoded to 0.0001, so LADDER_BPS was silently ignored
    unless it happened to be 1.0."""
    out = _orb_cross_tf(_cfg(LADDER_BPS=2.0), [100.0], [99.90], [99.90], [100.00])

    # 10bp dip at 2bp per rung = 5 extra rungs + the entry rung = 6.
    # The hardcoded 0.0001 gave 10 extra, capped at LADDER-1 = 9, i.e. 10.
    assert out["return_long_raw"].iloc[0] == pytest.approx(-0.001 * 6, abs=1e-12)


def test_cross_tf_matches_the_same_tf_engine_bar_for_bar():
    """Degenerate the cross-TF fixture onto the same-TF one: when the exit bar
    IS the next base bar, both paths must label identically. A private copy of
    the maths is how they drifted apart in the first place."""
    from agamotto.research import AgamottoResearch

    rng = np.random.default_rng(7)
    n = 200
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 3e-4, n))))
    df = pd.DataFrame({
        "open": close,
        "close": close,
        "high": close * (1 + np.abs(rng.normal(0, 4e-4, n))),
        "low": close * (1 - np.abs(rng.normal(0, 4e-4, n))),
    })

    cfg = _cfg(LADDER_LONG=2, LADDER_SHORT=10, FEE=2.25)
    ag = AgamottoResearch.__new__(AgamottoResearch)
    ag.config = cfg
    expected = ag._compute_ladder_returns(df, "close", "low", "high")

    # exit bar == next base bar
    got = _orb_cross_tf(
        cfg,
        entry_close=df["close"].tolist(),
        exit_close=df["close"].shift(-1).tolist(),
        exit_low=df["low"].shift(-1).tolist(),
        exit_high=df["high"].shift(-1).tolist(),
    )

    cols = ["return_long", "return_short", "return_long_raw", "return_short_raw"]
    for col in cols:
        pd.testing.assert_series_equal(
            got[col].iloc[:-1].reset_index(drop=True),      # last row has no exit bar
            expected[col].iloc[:-1].reset_index(drop=True),
            check_names=False, rtol=1e-12, atol=1e-15)


def test_cross_tf_missing_ladder_key_raises():
    """The fail-fast must reach this branch too, not just the same-TF one."""
    with pytest.raises(KeyError, match="LADDER"):
        _orb_cross_tf({"LADDER_BPS": 1.0, "FEE": 0.0},
                      [100.0], [99.90], [99.90], [100.10])
