"""MM-realistic target (Phase B of the 2026-08-10 MM design).

The guarantees, and why each exists:

  1. DEGENERATE EQUIVALENCE. LADDER=1, patience=0, taker fee 0 must reproduce the
     close-to-close ladder target EXACTLY. This is the bridge to the behaviour
     every existing kline result was produced under, and the strongest single
     test that the new arithmetic did not drift.
  2. NO LOOKAHEAD. Maker credit never uses an extreme from the entry bar T+1.
     A bar whose T+1 high clears exit_px AND whose T+1 low fills the whole ladder
     must still resolve TAKER — otherwise the target assumes low-before-high and
     manufactures the free lunch the design's "Why 15m only" section warns about.
  3. SIGN CONVENTION. The target is in PRICE-RETURN space, not PnL space: the PnL
     engine multiplies a short leg by -1 itself. Getting this backwards inverts
     the whole short book silently, which is why it is pinned rather than
     commented.
  4. CENSUS PARITY. dc ships standalone and cannot import marvel's
     `gauntlet/mm_fill_census.py`, so the two implementations of the same fill
     economics are held together by golden values computed by hand here.
  5. FEE ABSORPTION. The target is already net; a non-zero FEE would double
     charge at seven downstream sites, so it must raise.
  6. Branch exclusivity, size bounds, and the short-leg mirror.
"""
import numpy as np
import pandas as pd
import pytest

from agamotto.mm_target import (
    TARGET_MODE_LADDER,
    TARGET_MODE_MM,
    bar_seconds,
    compute_mm_target,
    expiry_offset,
    mm_params,
    resolve_leg,
    target_mode,
)

BAR_SEC = 900  # 15m


def _cfg(**over):
    cfg = {
        "TIME_UNIT": "15m",
        "FEE": 0.0,
        "LADDER_LONG": 2,
        "LADDER_SHORT": 2,
        "LADDER_BPS": 1.0,
        "MM_PROFIT_AIM": 1.0,
        "MM_PATIENCE_SEC": 900.0,
        "MM_TAKER_FEE_BPS": 1.75,
    }
    cfg.update(over)
    return cfg


def _bars(close, low, high):
    idx = pd.date_range("2025-01-01", periods=len(close), freq="15min")
    return pd.DataFrame({"close": close, "low": low, "high": high}, index=idx)


# --------------------------------------------------------------------------
# 4. Census parity — golden values computed by hand from the design's formulas
# --------------------------------------------------------------------------

def test_maker_branch_golden_value():
    """Row 0: a 1bp dip fills rung 2, and T+2's high clears the 1bp aim.

    dip   = (100 - 99.99)/100 = 1.0 bp -> n = 1 -> size = 2
    vwap  = 100 - 100*(1/2)*1bp        = 99.995
    exit  = 99.995 * (1 + 1bp)         = 100.0049995
    T+2 high = 100.05 >= exit          -> MAKER
    pnl   = size * aim                 = 2 * 1e-4
    """
    bars = _bars(close=[100.0, 100.0, 100.02, 100.0, 100.0],
                 low=[100.0, 99.99, 100.0, 100.0, 100.0],
                 high=[100.0, 100.0, 100.05, 100.0, 100.0])
    out = compute_mm_target(bars, "close", "low", "high", _cfg())
    assert out["return_long_raw"].iloc[0] == pytest.approx(2e-4, rel=1e-12)


def test_taker_branch_golden_value():
    """Row 1: no dip -> size 1; T+2's high misses the aim -> taker at close.

    size = 1, vwap = 100, exit = 100.01, T+2 high = 100 -> TAKER
    close_exit = close[3] = 100 -> delta = 0
    pnl = 1 * (0 - 1.75bp) = -1.75e-4
    """
    bars = _bars(close=[100.0, 100.0, 100.02, 100.0, 100.0],
                 low=[100.0, 99.99, 100.0, 100.0, 100.0],
                 high=[100.0, 100.0, 100.05, 100.0, 100.0])
    out = compute_mm_target(bars, "close", "low", "high", _cfg())
    assert out["return_long_raw"].iloc[1] == pytest.approx(-1.75e-4, rel=1e-12)


def test_expiry_offset_matches_census():
    """900s patience on a 900s bar -> expiry at T+2, one maker-window bar.

    300s -> expiry at T+1 -> the window is EMPTY and every row is taker, which
    is what put 40 of the census's 120 cells at p=0.
    """
    assert expiry_offset(900, BAR_SEC) == 2
    assert expiry_offset(300, BAR_SEC) == 1
    assert expiry_offset(1800, BAR_SEC) == 3


def test_patience_shorter_than_a_bar_is_all_taker():
    bars = _bars(close=[100.0] * 6, low=[100.0] * 6, high=[101.0] * 6)
    res = resolve_leg(bars["close"], bars["low"].shift(-1),
                      bars["high"].shift(-1), bars.index, leg="long", ladder=2,
                      ladder_bps=1.0, aim_bps=1.0, patience_sec=300.0,
                      taker_fee_bps=1.75)
    assert not res["is_maker"].dropna().any()


# --------------------------------------------------------------------------
# 1. Degenerate equivalence with the close-to-close ladder target
# --------------------------------------------------------------------------

def test_degenerate_reproduces_close_to_close_target():
    """LADDER=1, patience=0, fee=0 -> exactly `price_return`, both legs.

    With one rung the VWAP is the close and size is 1; with no patience the
    maker window is empty, so every row prices at the next close. That is
    precisely `close.pct_change().shift(-1)` — the current target with size 1.
    """
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 300)))
    bars = _bars(close=close, low=close * 0.999, high=close * 1.001)
    cfg = _cfg(LADDER_LONG=1, LADDER_SHORT=1, MM_PATIENCE_SEC=0.0,
               MM_TAKER_FEE_BPS=0.0)

    out = compute_mm_target(bars, "close", "low", "high", cfg)
    expected = bars["close"].pct_change(fill_method=None).shift(-1)

    # The MM row prices at T+1's close when patience is 0, so it is defined one
    # bar further back than the plain shift; compare where both exist.
    both = out["return_long_raw"].notna() & expected.notna()
    assert both.sum() > 250
    # atol at one ULP of the intermediate: `pct_change` computes
    # `diff()/shift()` while this module computes `(exit - vwap)/vwap`. Those are
    # algebraically identical and round differently — measured max absolute
    # difference 1.1e-16 on values of ~1e-5, i.e. one ULP of a ~100-magnitude
    # price divided by 100. Anything larger is real drift, so the bound is tight,
    # not permissive.
    np.testing.assert_allclose(out["return_long_raw"][both],
                               expected[both], rtol=1e-9, atol=1e-15)
    # The short column is in price-return space too, so it matches unnegated —
    # exactly like the existing `return_short_raw = price_return * size_short`.
    np.testing.assert_allclose(out["return_short_raw"][both],
                               expected[both], rtol=1e-9, atol=1e-15)


# --------------------------------------------------------------------------
# 2. No lookahead
# --------------------------------------------------------------------------

def test_entry_bar_extreme_never_grants_maker_credit():
    """T+1 clears the aim and fills the ladder; T+2 does not. Must be TAKER.

    This is the free lunch the design refuses to buy: crediting T+1's high would
    assume low-before-high within the entry bar.
    """
    bars = _bars(
        close=[100.0, 100.0, 100.0, 100.0],
        low=[100.0, 99.90, 100.0, 100.0],     # T+1 low fills the whole ladder
        high=[100.0, 105.0, 100.0, 100.0],    # T+1 high clears any aim
    )
    res = resolve_leg(bars["close"], bars["low"].shift(-1),
                      bars["high"].shift(-1), bars.index, leg="long", ladder=2,
                      ladder_bps=1.0, aim_bps=1.0, patience_sec=900.0,
                      taker_fee_bps=1.75)
    assert res["is_maker"].iloc[0] == False  # noqa: E712 — NaN-safe compare


# --------------------------------------------------------------------------
# 3. Sign convention
# --------------------------------------------------------------------------

def test_short_column_is_negated_true_pnl():
    """`return_short_raw` must be -(true short PnL).

    The canonical engine does `rets = ret_col * (-1)` for a short leg
    (evaluate_regimes.py:196-197). Emitting true short PnL here would flip it a
    second time and invert the short book.
    """
    bars = _bars(close=[100.0, 100.0, 99.98, 100.0, 100.0],
                 low=[100.0, 99.95, 100.0, 100.0, 100.0],
                 high=[100.0, 100.01, 100.0, 100.0, 100.0])
    cfg = _cfg()
    out = compute_mm_target(bars, "close", "low", "high", cfg)
    short_true = resolve_leg(bars["close"], bars["low"].shift(-1),
                             bars["high"].shift(-1), bars.index, leg="short",
                             ladder=2, ladder_bps=1.0, aim_bps=1.0,
                             patience_sec=900.0, taker_fee_bps=1.75)["pnl"]
    both = out["return_short_raw"].notna() & short_true.notna()
    assert both.sum() > 0
    np.testing.assert_allclose(out["return_short_raw"][both],
                               -short_true[both], rtol=1e-12)


# --------------------------------------------------------------------------
# 5. Fee absorption
# --------------------------------------------------------------------------

def test_nonzero_fee_raises_rather_than_double_charging():
    with pytest.raises(ValueError, match="requires FEE == 0"):
        mm_params(_cfg(FEE=2.25))


def test_missing_fee_raises():
    cfg = _cfg()
    del cfg["FEE"]
    with pytest.raises(KeyError, match="FEE missing"):
        mm_params(cfg)


@pytest.mark.parametrize("key", ["MM_PROFIT_AIM", "MM_PATIENCE_SEC",
                                 "MM_TAKER_FEE_BPS"])
def test_every_mm_knob_is_required(key):
    cfg = _cfg()
    del cfg[key]
    with pytest.raises(KeyError, match=key):
        mm_params(cfg)


# --------------------------------------------------------------------------
# 6. Branch exclusivity, size bounds, short mirror
# --------------------------------------------------------------------------

def test_every_resolvable_row_is_priced_exactly_once():
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 400)))
    bars = _bars(close=close,
                 low=close * (1 - abs(rng.normal(0, 0.001, 400))),
                 high=close * (1 + abs(rng.normal(0, 0.001, 400))))
    res = resolve_leg(bars["close"], bars["low"].shift(-1),
                      bars["high"].shift(-1), bars.index, leg="long", ladder=2,
                      ladder_bps=1.0, aim_bps=1.0, patience_sec=900.0,
                      taker_fee_bps=1.75)
    resolvable = res["pnl"].notna()
    # A priced row always has a branch, and a branch always has a price.
    assert (res["is_maker"].notna() == resolvable).all()
    assert resolvable.sum() > 380


@pytest.mark.parametrize("ladder", [1, 2, 5])
def test_size_stays_in_bounds_on_degenerate_input(ladder):
    bars = _bars(close=[100.0, 0.0, np.nan, 100.0, 100.0, 100.0],
                 low=[100.0, np.nan, 100.0, 200.0, 100.0, 100.0],   # inverted
                 high=[100.0, 100.0, 100.0, 50.0, 100.0, 100.0])
    res = resolve_leg(bars["close"], bars["low"].shift(-1),
                      bars["high"].shift(-1), bars.index, leg="long",
                      ladder=ladder, ladder_bps=1.0, aim_bps=1.0,
                      patience_sec=900.0, taker_fee_bps=1.75)
    assert res["size"].min() >= 1
    assert res["size"].max() <= ladder


def test_short_leg_mirrors_the_long_leg():
    """A mirrored book must give the short leg the long leg's branch and size."""
    close = [100.0] * 8
    dip, rip = 0.03, 0.04
    long_bars = _bars(close=close,
                      low=[100 - dip] * 8, high=[100 + rip] * 8)
    short_bars = _bars(close=close,
                       low=[100 - rip] * 8, high=[100 + dip] * 8)
    kw = dict(ladder=2, ladder_bps=1.0, aim_bps=1.0, patience_sec=900.0,
              taker_fee_bps=1.75)
    lo = resolve_leg(long_bars["close"], long_bars["low"].shift(-1),
                     long_bars["high"].shift(-1), long_bars.index,
                     leg="long", **kw)
    sh = resolve_leg(short_bars["close"], short_bars["low"].shift(-1),
                     short_bars["high"].shift(-1), short_bars.index,
                     leg="short", **kw)
    np.testing.assert_array_equal(lo["size"].to_numpy(), sh["size"].to_numpy())
    # Compare branch labels only where BOTH resolve. `is_maker` is NaN on the
    # unresolvable tail, and NaN != NaN under object-dtype array comparison, so
    # an unmasked assert fails on rows that are in fact identical.
    resolvable = lo["is_maker"].notna() & sh["is_maker"].notna()
    assert resolvable.sum() > 0
    np.testing.assert_array_equal(
        lo["is_maker"][resolvable].to_numpy().astype(bool),
        sh["is_maker"][resolvable].to_numpy().astype(bool))
    # The tail must be unresolvable on BOTH legs, not silently dropped on one.
    np.testing.assert_array_equal(lo["is_maker"].isna().to_numpy(),
                                  sh["is_maker"].isna().to_numpy())
    np.testing.assert_allclose(lo["pnl"].dropna(), sh["pnl"].dropna(), rtol=1e-6)


# --------------------------------------------------------------------------
# Mode + timeframe guards
# --------------------------------------------------------------------------

def test_target_mode_defaults_to_ladder_but_rejects_unknown():
    assert target_mode({}) == TARGET_MODE_LADDER
    assert target_mode({"TARGET_MODE": "mm"}) == TARGET_MODE_MM
    with pytest.raises(ValueError, match="TARGET_MODE"):
        target_mode({"TARGET_MODE": "taker"})


@pytest.mark.parametrize("tf", ["1h", "4h", "1d", None])
def test_coarser_timeframes_are_refused(tf):
    bars = _bars(close=[100.0] * 4, low=[100.0] * 4, high=[100.0] * 4)
    with pytest.raises(ValueError, match="15m"):
        compute_mm_target(bars, "close", "low", "high", _cfg(TIME_UNIT=tf))


def test_both_call_sites_agree_in_mm_mode():
    """`engineer_features` and `_compute_ladder_returns` must not drift apart.

    They are two copies of the same target arithmetic — the design's stated
    reason for putting it in a shared module — and a private copy is exactly how
    the cross-TF and same-TF ladder engines drifted apart before. If one switches
    to MM and the other does not, training and scoring silently price different
    strategies.
    """
    from agamotto.research import AgamottoResearch

    rng = np.random.default_rng(11)
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="15min")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 3e-4, n))),
                      index=idx)
    sym = "BINANCE_PERP_BTC_USDT"
    raw = pd.DataFrame({
        f"{sym}_open": close,
        f"{sym}_close": close,
        f"{sym}_high": close * (1 + np.abs(rng.normal(0, 4e-4, n))),
        f"{sym}_low": close * (1 - np.abs(rng.normal(0, 4e-4, n))),
        f"{sym}_volume": pd.Series(np.abs(rng.normal(1e3, 10, n)), index=idx),
    }, index=idx)

    cfg = _cfg(TARGET_MODE="mm")
    ag = AgamottoResearch.__new__(AgamottoResearch)
    ag.config = cfg
    expected = ag._compute_ladder_returns(
        raw.rename(columns={f"{sym}_close": "close", f"{sym}_low": "low",
                            f"{sym}_high": "high"}),
        "close", "low", "high")

    ag.raw = raw
    ag.engineer_features()

    for col in ["return_long", "return_short", "return_long_raw",
                "return_short_raw"]:
        pd.testing.assert_series_equal(
            ag.features[f"{sym}_{col}"].reset_index(drop=True),
            expected[col].reset_index(drop=True),
            check_names=False, rtol=1e-12, atol=1e-15)


def test_bar_seconds_uses_total_seconds_not_asi8():
    """A [us]-unit index must still read as 900s (CLAUDE.md duration rule)."""
    idx = pd.date_range("2025-01-01", periods=5, freq="15min").as_unit("us")
    assert bar_seconds(idx) == 900.0
