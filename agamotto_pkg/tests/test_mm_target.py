"""Minute-resolved MM target (2026-08-10 revision of the Phase B design).

WHY THIS MODEL EXISTS. The first cut resolved the MarketMaker lifecycle from 15m bars and
implied **-28.8% of capital per day** against a live book that made money. Two lines caused
it: the maker window started at offset 2 (discarding bar T+1, where the ladder actually fills
and where a bounce to +1bp is most likely), and the taker exit was marked at T+2's close, up
to a full bar after the crossing really fires. `PASSIVE_SEC 600 + CROSSING_SEC 300 = 900 s` is
exactly one 15m bar, so the whole lifecycle lives inside T+1 and needs sub-bar data.

The guarantees, and why each exists:

  1. DEGENERATE EQUIVALENCE. LADDER=1, an unreachable aim and a zero taker fee must reproduce
     the close-to-close target EXACTLY. That is the bridge to every existing kline result.
  2. NO INTRA-MINUTE LOOKAHEAD. The exit resting during minute j is priced off fills up to
     j-1. A minute that both fills rungs and prints through the exit must not bank the
     post-dip size — that would assume low-before-high.
  3. SIZE AT A MAKER EXIT IS n(j-1), NOT THE FINAL COUNT. Once the reduce-only close fills the
     position is flat and the remaining entry rungs are cancelled. Using the final size banks
     the aim on rungs never held, and erases the asymmetry that defines the mode.
  4. SIGN CONVENTION. Price-return space, not PnL space — the engine applies the position
     sign itself. Getting it backwards inverts the short book silently.
  5. FEE ABSORPTION and required knobs.
  6. INCOMPLETE MINUTE COVERAGE resolves to NaN, never a guess.
"""
import numpy as np
import pandas as pd
import pytest

from agamotto.mm_target import (
    TARGET_MODE_LADDER,
    TARGET_MODE_MM,
    compute_mm_target,
    minute_matrices,
    mm_params,
    patience_minutes,
    resolve_leg,
    target_mode,
)

BARS_PER_15M = 15
START = "2025-01-01 00:00"


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
        "TARGET_MODE": "mm",
    }
    cfg.update(over)
    return cfg


def _one_bar(minutes, close_t=100.0):
    """One 15m signal bar at `close_t`, with `minutes` = list of (high, low, close) for T+1.

    The 15m bar opens at START and closes at START+15m; its T+1 minutes are START+15m
    onwards, which is exactly what `minute_matrices` reaches for.
    """
    sig_idx = pd.DatetimeIndex([pd.Timestamp(START)])
    df15 = pd.DataFrame({"close": [close_t], "high": [close_t], "low": [close_t]},
                        index=sig_idx)
    # Two signal bars' worth of minutes so bar spacing is inferrable; only T+1 is read.
    m_idx = pd.date_range(pd.Timestamp(START) + pd.Timedelta(minutes=15),
                          periods=len(minutes), freq="1min")
    minute = pd.DataFrame(minutes, columns=["high", "low", "close"], index=m_idx)
    return df15, minute


def _resolve_one(minutes, *, leg="long", ladder=2, ladder_bps=1.0, aim_bps=1.0,
                 taker_fee_bps=1.75, close_t=100.0):
    """Resolve a single constructed bar and return its one-row result."""
    df15, minute = _one_bar(minutes, close_t=close_t)
    high_m, low_m, close_m = minute_matrices(df15.index, minute, len(minutes), 900.0)
    return resolve_leg(df15["close"], high_m, low_m, close_m, leg=leg, ladder=ladder,
                       ladder_bps=ladder_bps, aim_bps=aim_bps,
                       taker_fee_bps=taker_fee_bps).iloc[0]


def _flat(n, px=100.0):
    return [(px, px, px)] * n


# --------------------------------------------------------------------------
# Golden values — hand-computed from the design's formulas
# --------------------------------------------------------------------------

def test_maker_on_base_rung_only():
    """No dip: size 1, exit at 100*(1+1bp)=100.01, cleared by minute 1's high.

    pnl = size * aim = 1 * 1e-4
    """
    minutes = _flat(15)
    minutes[1] = (100.02, 100.0, 100.0)          # high clears 100.01
    r = _resolve_one(minutes)
    assert bool(r["is_maker"]) is True
    assert r["size"] == 1
    assert r["pnl"] == pytest.approx(1e-4, rel=1e-12)


def test_maker_after_a_rung_fills():
    """m0 dips to 99.985 -> rung 1 fills (99.99), VWAP 99.995, exit 100.0049995.

    m1's high of 100.01 clears it. size = n(0) = 2 -> pnl = 2 * 1e-4.
    """
    minutes = _flat(15)
    minutes[0] = (100.0, 99.985, 99.99)
    minutes[1] = (100.01, 99.99, 100.0)
    r = _resolve_one(minutes)
    assert bool(r["is_maker"]) is True
    assert r["size"] == 2
    assert r["avg_cost"] == pytest.approx(99.995, rel=1e-12)
    assert r["pnl"] == pytest.approx(2e-4, rel=1e-12)


def test_taker_when_the_aim_is_never_reached():
    """Flat book: exit 100.01 never printed -> cross at the last minute's close.

    pnl = 1 * ((100 - 100)/100 - 1.75bp) = -1.75e-4
    """
    r = _resolve_one(_flat(15))
    assert bool(r["is_maker"]) is False
    assert r["pnl"] == pytest.approx(-1.75e-4, rel=1e-12)


def test_taker_marks_at_the_last_minute_close_not_a_later_bar():
    """The crossing fires at the END of T+1. Marking at T+2 was the old defect."""
    minutes = _flat(15)
    minutes[14] = (100.0, 100.0, 99.50)          # -50 bp at the expiry minute
    r = _resolve_one(minutes)
    assert bool(r["is_maker"]) is False
    expected = 1 * ((99.50 - 100.0) / 100.0 - 1.75e-4)
    assert r["pnl"] == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# The two rules that make this model different from the 15m one
# --------------------------------------------------------------------------

def test_size_at_maker_exit_is_the_count_before_the_fill():
    """Rung 2 fills at m3, exit clears at m5, a deeper dip arrives at m9.

    The deeper dip must NOT count: the position was already flat. Final count would be 3.
    """
    minutes = _flat(15)
    minutes[3] = (100.0, 99.985, 99.99)          # fills rung 1 (99.99), not rung 2 (99.98)
    minutes[5] = (100.01, 99.99, 100.0)          # clears exit 100.0049995
    minutes[9] = (100.0, 99.975, 99.98)          # would fill rung 2 -- too late
    r = _resolve_one(minutes, ladder=3)
    assert bool(r["is_maker"]) is True
    assert r["size"] == 2, "used the final rung count instead of the count at the exit"
    assert r["pnl"] == pytest.approx(2e-4, rel=1e-12)


def test_no_intra_minute_lookahead_on_the_fill_minute():
    """A minute that both fills the whole ladder and prints far through the exit.

    The exit resting during that minute was priced off the PREVIOUS minute's fills, so the
    size banked is 1 -- not the 3 the dip would have produced. Crediting 3 would assume
    low-before-high inside the minute.
    """
    minutes = _flat(15)
    minutes[1] = (105.0, 99.97, 100.0)           # fills rungs 2 and 3 AND clears any exit
    r = _resolve_one(minutes, ladder=3)
    assert bool(r["is_maker"]) is True
    assert r["size"] == 1
    assert r["pnl"] == pytest.approx(1e-4, rel=1e-12)


def test_ladder_fills_in_minute_order():
    """Rungs are unfilled until the minute their price is reached."""
    minutes = _flat(15)
    minutes[7] = (100.0, 99.975, 99.98)          # reaches rung 1 and rung 2
    minutes[8] = (100.01, 99.98, 100.0)          # clears the 3-rung exit
    r = _resolve_one(minutes, ladder=3)
    assert bool(r["is_maker"]) is True
    assert r["size"] == 3
    assert r["avg_cost"] == pytest.approx(100.0 * (1 - 1e-4), rel=1e-12)


# --------------------------------------------------------------------------
# Degenerate equivalence with the close-to-close target
# --------------------------------------------------------------------------

def _synthetic(n_bars, seed):
    """A minute series plus the 15m bars aggregated FROM it, so they cannot disagree."""
    n_min = (n_bars + 1) * BARS_PER_15M
    rng = np.random.default_rng(seed)
    m_idx = pd.date_range(START, periods=n_min, freq="1min")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 2e-4, n_min)))
    minute = pd.DataFrame({
        "high": close * (1 + np.abs(rng.normal(0, 1e-4, n_min))),
        "low": close * (1 - np.abs(rng.normal(0, 1e-4, n_min))),
        "close": close,
    }, index=m_idx)
    sig_idx = m_idx[::BARS_PER_15M][:n_bars]
    # A 15m bar closes at the close of its LAST minute.
    c15 = close[BARS_PER_15M - 1::BARS_PER_15M][:n_bars]
    df15 = pd.DataFrame({"close": c15, "high": c15, "low": c15}, index=sig_idx)
    return df15, minute


def test_degenerate_reproduces_close_to_close_target():
    """LADDER=1, unreachable aim, zero taker fee -> exactly the next-bar return.

    With one rung the VWAP is the close and size is 1; with an unreachable aim every bar
    resolves taker at the close of T+1, which IS `close.pct_change().shift(-1)`.
    """
    df15, minute = _synthetic(200, seed=0)
    cfg = _cfg(LADDER_LONG=1, LADDER_SHORT=1, MM_PROFIT_AIM=1e6, MM_TAKER_FEE_BPS=0.0)
    out = compute_mm_target(df15, "close", "low", "high", cfg, minute_bars=minute)

    expected = df15["close"].pct_change(fill_method=None).shift(-1)
    both = out["return_long_raw"].notna() & expected.notna()
    assert both.sum() > 190
    # atol at one ULP of the intermediate: pct_change is diff()/shift() while this module
    # computes (exit - vwap)/vwap -- algebraically identical, different rounding.
    np.testing.assert_allclose(out["return_long_raw"][both], expected[both],
                               rtol=1e-9, atol=1e-15)
    # Short is in price-return space too, so it matches UNNEGATED, exactly like the existing
    # return_short_raw = price_return * size_short.
    np.testing.assert_allclose(out["return_short_raw"][both], expected[both],
                               rtol=1e-9, atol=1e-15)


# --------------------------------------------------------------------------
# Sign convention, mirroring, coverage
# --------------------------------------------------------------------------

def test_short_column_is_negated_true_pnl():
    df15, minute = _synthetic(80, seed=3)
    cfg = _cfg()
    out = compute_mm_target(df15, "close", "low", "high", cfg, minute_bars=minute)
    high_m, low_m, close_m = minute_matrices(df15.index, minute, BARS_PER_15M, 900.0)
    short_true = resolve_leg(df15["close"], high_m, low_m, close_m, leg="short", ladder=2,
                             ladder_bps=1.0, aim_bps=1.0, taker_fee_bps=1.75)["pnl"]
    both = out["return_short_raw"].notna() & short_true.notna()
    assert both.sum() > 0
    np.testing.assert_allclose(out["return_short_raw"][both], -short_true[both], rtol=1e-12)


def test_short_leg_mirrors_the_long_leg():
    """A book mirrored about the close gives the short leg the long leg's branch and size."""
    up = _flat(15)
    up[0] = (100.0, 99.985, 99.99)
    up[1] = (100.01, 99.99, 100.0)
    lo = _resolve_one(up, leg="long")

    dn = _flat(15)
    dn[0] = (100.015, 100.0, 100.01)
    dn[1] = (100.01, 99.99, 100.0)
    sh = _resolve_one(dn, leg="short")

    assert lo["size"] == sh["size"]
    assert bool(lo["is_maker"]) == bool(sh["is_maker"])
    assert lo["pnl"] == pytest.approx(sh["pnl"], rel=1e-6)


def test_incomplete_minute_coverage_is_nan_not_guessed():
    minutes = _flat(15)
    minutes[6] = (np.nan, np.nan, np.nan)
    r = _resolve_one(minutes)
    assert np.isnan(r["pnl"])
    assert np.isnan(r["size"])


def test_missing_minute_bars_raises_rather_than_falling_back():
    df15, _ = _synthetic(20, seed=5)
    with pytest.raises(ValueError, match="requires 1m bars"):
        compute_mm_target(df15, "close", "low", "high", _cfg(), minute_bars=None)


@pytest.mark.parametrize("ladder", [1, 2, 5])
def test_size_stays_in_bounds(ladder):
    minutes = _flat(15)
    minutes[2] = (100.0, 90.0, 95.0)             # a crash deeper than every rung
    r = _resolve_one(minutes, ladder=ladder)
    assert 1 <= r["size"] <= ladder


# --------------------------------------------------------------------------
# Config guards
# --------------------------------------------------------------------------

def test_nonzero_fee_raises_rather_than_double_charging():
    with pytest.raises(ValueError, match="requires FEE == 0"):
        mm_params(_cfg(FEE=2.25))


@pytest.mark.parametrize("key", ["MM_PROFIT_AIM", "MM_PATIENCE_SEC", "MM_TAKER_FEE_BPS"])
def test_every_mm_knob_is_required(key):
    cfg = _cfg()
    del cfg[key]
    with pytest.raises(KeyError, match=key):
        mm_params(cfg)


def test_patience_must_be_whole_minutes():
    with pytest.raises(ValueError, match="whole number of minutes"):
        mm_params(_cfg(MM_PATIENCE_SEC=890.0))


def test_patience_minutes_matches_the_live_lifecycle():
    assert patience_minutes(900) == 15          # PASSIVE_SEC 600 + CROSSING_SEC 300
    assert patience_minutes(1800) == 30
    with pytest.raises(ValueError):
        patience_minutes(0)


def test_target_mode_defaults_to_ladder_but_rejects_unknown():
    assert target_mode({}) == TARGET_MODE_LADDER
    assert target_mode({"TARGET_MODE": "mm"}) == TARGET_MODE_MM
    with pytest.raises(ValueError, match="TARGET_MODE"):
        target_mode({"TARGET_MODE": "taker"})


@pytest.mark.parametrize("tf", ["1h", "4h", "1d", None])
def test_coarser_signal_grids_are_refused(tf):
    df15, minute = _synthetic(20, seed=6)
    with pytest.raises(ValueError, match="15m"):
        compute_mm_target(df15, "close", "low", "high", _cfg(TIME_UNIT=tf),
                          minute_bars=minute)


def test_minute_matrices_reads_T_plus_1_not_T():
    """The window must start at the close of T, i.e. the first minute of T+1."""
    df15, minute = _synthetic(4, seed=7)
    high_m, low_m, close_m = minute_matrices(df15.index, minute, BARS_PER_15M, 900.0)
    # Bar 0 opens at START and closes at START+15m; its window is minutes 15..29.
    np.testing.assert_allclose(close_m[0], minute["close"].to_numpy()[15:30], rtol=1e-12)


def test_both_call_sites_agree_in_mm_mode(tmp_path):
    """`engineer_features` and `_compute_ladder_returns` must not drift apart.

    They are two copies of the same target arithmetic, and a private copy is exactly how the
    cross-TF and same-TF ladder engines drifted apart before. This also exercises
    `_load_minute_bars`, since engineer_features reads the 1m tree off disk while the direct
    call is handed the frame.
    """
    from agamotto.research import AgamottoResearch

    sym = "BTCUSDT"
    df15, minute = _synthetic(120, seed=11)

    # Lay the 1m tree out exactly as sync_klines.sh writes it.
    sym_dir = tmp_path / "data" / "BINANCEFUTURES" / "1m" / "liquid" / sym
    sym_dir.mkdir(parents=True)
    on_disk = minute.copy()
    # NORMALISE THE UNIT EXPLICITLY (CLAUDE.md): a raw int64 view returns the index's OWN
    # unit, which is [us] for pd.date_range on pandas 3 — `// 1_000_000` on that yields
    # SECONDS, not milliseconds, and silently puts every minute off-grid.
    on_disk["open_time_ms"] = (
        on_disk.index.to_numpy(dtype="datetime64[ns]").astype("int64") // 1_000_000)
    on_disk.to_csv(sym_dir / f"{sym}_2025-01_1m.csv", index=False)

    cfg = _cfg()
    ag = AgamottoResearch.__new__(AgamottoResearch)
    ag.config = cfg
    ag.home_root = str(tmp_path)

    expected = ag._compute_ladder_returns(df15, "close", "low", "high", minute_bars=minute)

    raw = pd.DataFrame({
        f"{sym}_open": df15["close"], f"{sym}_close": df15["close"],
        f"{sym}_high": df15["high"], f"{sym}_low": df15["low"],
        f"{sym}_volume": pd.Series(1000.0, index=df15.index),
    }, index=df15.index)
    ag.raw = raw
    ag.engineer_features()

    for col in ["return_long", "return_short", "return_long_raw", "return_short_raw"]:
        pd.testing.assert_series_equal(
            ag.features[f"{sym}_{col}"].reset_index(drop=True),
            expected[col].reset_index(drop=True),
            check_names=False, rtol=1e-12, atol=1e-15)


def test_minute_matrices_is_gap_safe():
    """A missing minute becomes NaN rather than shifting every later bar."""
    df15, minute = _synthetic(4, seed=8)
    minute = minute.drop(minute.index[20])
    high_m, low_m, close_m = minute_matrices(df15.index, minute, BARS_PER_15M, 900.0)
    assert np.isnan(close_m[0, 5])               # minute 20 == window slot 5 of bar 0
    assert np.isfinite(close_m[0, 6])            # its neighbour is untouched
