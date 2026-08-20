#!/usr/bin/env python3
"""Differential parity: C++ `agamotto::engineerFeatures` == the REAL reference.

Stage 2.2 + 2.3 + 2.4 + 2.5 gate — the COMPLETE 65-column feature panel.
Generates a raw OHLCV panel, drives it through BOTH
`AgamottoResearch.engineer_features()` (the actual research.py, imported — never
reimplemented here) and `tests/feature_parity_driver`, and compares every
implemented column cell by cell.

Four things this harness does that sentinel_core's does NOT, each because
agamotto differs from mjolnir in a way that a copied harness would hide:

1. **NaN is compared AS NaN.** sentinel_core's harness runs
   `np.where(np.isfinite(x), x, 0.0)` on both sides before diffing. That is
   correct for mjolnir, whose engine sanitises non-finite cells to 0.0
   panel-wide, and WRONG here: agamotto sanitises nothing (its only fill is
   `X.fillna(0.0)` on the selected model columns of the single scored row,
   trading.py:700, and inf is never touched). Under the sentinel rule a column
   that returned 0.0 where pandas returns NaN would PASS, and a NaN that fails
   to propagate is precisely what turns a regime that cannot fire into one that
   fires on every bar. So every cell is first classified
   finite / NaN / +inf / -inf and the two classification arrays must be
   EQUAL before any value is looked at. Classifying (rather than just
   `isnan`) also stops +inf on one side and NaN on the other from cancelling
   out as "both non-finite".

2. **The panel is exactly PANEL_BARS = 699 rows.** trading.py:443
   `load_data(limit=700)` -> :480 `tail(limit)` -> :485 `iloc[:-1]`.
   `price_range_pct_q50` is `rolling(700, min_periods=1)`, i.e. an EXPANDING
   median at this width, so its values depend on the ROW COUNT: at 700 or 1000
   rows this harness would compare numbers live never computes.

3. **q80/q90/q95 must be ENTIRELY NaN, and that is asserted.** They are
   `rolling(700, min_periods=700)` (research.py:371-376), so on 699 rows they
   are NaN everywhere by construction. This is live behaviour under an open
   production finding (marvel PR #532,
   docs/findings/2026-08-19-vol-quantile-regimes-inert-live.md: 53 of 62
   deployed regimes cannot fire because `x > NaN` is False). It is reproduced,
   not fixed, and pinned here so it cannot become incidental.

4. **Two price scales.** BTC-like (~64000) and 1000PEPE-like (~0.0045). The
   `+1e-8` epsilons are ABSOLUTE and inline per expression, so on BTC they are
   a 1.6e-13 perturbation (invisible) and on 1000PEPE a 2.2e-6 one — a
   sixth-significant-figure move in `price_range_pct`, which is a top-5
   IC-selected feature. A BTC-only harness cannot tell `+1e-8` from `+0`.
   A third scenario injects NaN holes and a zero-volume bar so the NaN masks
   and the +/-inf cells are exercised rather than merely permitted.

Stage 2.3 adds two checks that only the TA-Lib block needs:

5. **A FIRST-VALID-INDEX assertion, per column.** Every TA-Lib output is
   COMPACTED — the C function returns `outNBElement` values whose first one
   belongs at input index `outBegIdx`, and the wrapper additionally skips
   leading NaNs. Place the payload at the wrong offset and an entire column is
   shifted by its lookback, which a value diff of the OVERLAPPING region can
   still pass (a 14-row shift on a slow-moving indicator like `adx` is a small
   relative difference on most rows). The classification arrays would catch it
   only if the head length changed; a shift that keeps the same NaN count would
   not move them at all. So the first non-NaN row index of every column is
   compared against the reference's, per column, and printed as a table.

6. **A TA-Lib VERSION GATE, and proof the reference actually ran TA-Lib.**
   research.py:501-554 wraps the whole block in `except Exception` and only
   logs a warning, so a reference environment WITHOUT TA-Lib silently produces
   a panel MISSING all 30 of these columns rather than failing. Graded
   naively that reads as a pass over the remaining 24. This harness therefore
   (a) refuses to run unless `talib.__ta_version__` equals the version the C++
   is pinned to, and (b) asserts the 29 TA-Lib columns are PRESENT in the
   reference panel before comparing anything.

Stage 2.5 adds two more, both for `_safe` (features_scalefree.py:61-63):

7. **A FIFTH SCENARIO THAT ACTUALLY EXERCISES `_safe`.** On ordinary
   market-shaped data not one of the seven scale-free divisions ever meets a
   zero or an overflowing denominator — the four older scenarios report 0/0/0
   branch hits on every column — so `_safe` would be graded on its pass-through
   arm alone, exactly the trap stage 2.3 found when all three original scenarios
   turned out to have `begidx == 0`. Scenario 5 constructs a `close == 0.0` bar,
   a `close == 1e-305` bar, a 26-bar flat run and a 25-bar zero-volume run, and
   `safe_branch_coverage` ASSERTS the resulting branch-hit counts rather than
   printing them. It distinguishes DISCRIMINATING zero-denominator cells (where
   omitting step 1 would change the answer) from `0/0` cells that would be NaN
   either way, so the coverage claim is what was measured.

8. **A NARROW, PROVEN WAIVER FOR A TA-LIB DEFECT.** ta_NATR.c:334-338 leaves its
   output UNWRITTEN when `|close| < TA_EPSILON`, so the reference is not a
   function of its inputs on those cells. See `natr_unstable_rows` — the waiver
   is derived from the defect, proven per-row by re-running the reference, and
   fails the gate on any non-determinism it does not explain.

Usage:
    # macOS / clang
    tests/run_feature_parity.sh
    # rocky8 / gcc, inside the build image
    tests/run_feature_parity.sh --linux

    # or directly, --driver is a COMMAND (shlex-split), so it can be a docker run
    python3 tests/feature_parity.py --driver ./build/feature_parity_driver
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DC_ROOT = HERE.parents[1]
sys.path.insert(0, str(DC_ROOT / "agamotto_pkg" / "src"))

# ---------------------------------------------------------------------------
# PANEL WIDTH IS A CORRECTNESS PARAMETER — see the module docstring, point 2,
# and src/feature_engine.hpp `PANEL_BARS`. Read from the header rather than
# retyped, so the two cannot drift.
# ---------------------------------------------------------------------------


def _panel_bars_from_header() -> int:
    src = (DC_ROOT / "agamotto_core" / "src" / "feature_engine.hpp").read_text()
    for line in src.splitlines():
        if line.startswith("constexpr size_t PANEL_BARS"):
            return int(line.split("=")[1].strip().rstrip(";"))
    raise SystemExit(
        "cannot read PANEL_BARS from src/feature_engine.hpp — refusing to guess "
        "the panel width, which decides what price_range_pct_q50 computes")


PANEL_BARS = _panel_bars_from_header()


# ---------------------------------------------------------------------------
# THE TA-LIB PIN. Same shape, and the same reasoning, as
# sentinel_core/tests/feature_parity.py `_pinned_talib_version` /
# `require_reference_talib`: the version is read from the C++ side's own source
# of truth rather than retyped here, because a version difference in TA-Lib is a
# SILENT parity break (indicator internals have changed across releases), not a
# rounding difference.
#
# agamotto has TWO pins that must agree, so both are read and cross-checked:
#   * agamotto_core/CMakeLists.txt `TALIB_PINNED_VERSION` — what the LIBRARY and
#     the host driver build against (CMake refuses a mismatched ta-lib outright);
#   * sentinel_core/Dockerfile.build `ARG TALIB_VERSION` — what is installed in
#     mjolnir-core-build:latest, i.e. what the LINUX driver links.
# If those two ever drift, the two toolchain legs of this gate are grading
# against different math while both print PASS.
# ---------------------------------------------------------------------------


def _pinned_talib_version() -> str:
    cmake = DC_ROOT / "agamotto_core" / "CMakeLists.txt"
    m = re.search(r'set\(TALIB_PINNED_VERSION\s+"([^"]+)"\)', cmake.read_text())
    if not m:
        raise SystemExit(
            f"cannot read TALIB_PINNED_VERSION from {cmake} — refusing to guess "
            "the version the C++ was built against")
    pinned = m.group(1)

    dockerfile = DC_ROOT / "sentinel_core" / "Dockerfile.build"
    d = re.search(r"^ARG\s+TALIB_VERSION=(\S+)", dockerfile.read_text(), re.M)
    if not d:
        raise SystemExit(f"cannot read ARG TALIB_VERSION from {dockerfile}")
    if d.group(1) != pinned:
        raise SystemExit(
            f"=== BLOCKED: the two TA-Lib pins disagree — CMakeLists says "
            f"{pinned}, {dockerfile.name} says {d.group(1)} ===\n"
            "  The host leg and the linux leg of this gate would be grading "
            "against different\n  indicator math while both print PASS.")
    return pinned


def require_reference_talib() -> str:
    """Fail loudly unless the reference will take the TA-Lib path, at the pin."""
    pinned = _pinned_talib_version()
    try:
        import talib  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "=== BLOCKED: the reference has no TA-Lib — this is NOT a parity "
            f"result ===\n  ({type(exc).__name__}: {exc})\n"
            "  research.py:501-554 wraps the whole indicator block in "
            "`except Exception` and only\n"
            "  logs a warning, so the reference panel would silently come back "
            "MISSING all 30\n"
            "  stage-2.3 columns instead of failing, and the remaining 24 "
            "would still print PASS.\n"
            f"  Production runs TA-Lib {pinned}; install the SAME version, e.g.:\n"
            f"    python3 -m pip install 'TA-Lib=={pinned}'  # wheel ships the "
            "C library\n"
            "  and confirm with "
            "`python3 -c \"import talib; print(talib.__ta_version__)\"`."
        ) from exc

    # The WRAPPER version is not what computes indicators — the C library is.
    # talib.__ta_version__ reports the library, so ask that.
    raw = getattr(talib, "__ta_version__", None)
    if raw is None:
        raise SystemExit(
            "talib exposes no __ta_version__ — cannot confirm the C library "
            f"matches the pinned {pinned}, and an unverified version is a "
            "silent parity break, not a rounding difference.")
    if isinstance(raw, bytes):
        raw = raw.decode()
    got = raw.split()[0]
    if got != pinned:
        raise SystemExit(
            f"=== BLOCKED: TA-Lib version mismatch — reference {got}, C++ "
            f"pinned {pinned} ===\n"
            "  TA-Lib has changed indicator internals across releases, so this "
            "is a silent\n  parity break rather than a rounding difference. "
            "Align the reference to the pin.")
    return got


def assert_reference_used_talib(panel, enc, where: str) -> None:
    """Defence in depth: prove the panel really came from the TA-Lib path.

    `require_reference_talib` checks the PRECONDITION (the module imports at the
    pin); this checks the OUTCOME. research.py's `except Exception` swallows any
    failure inside the block — a TA-Lib that imports but raises on some input
    would leave exactly the same hole, and the run would grade the C++ against
    a panel that never had these columns.
    """
    missing = [c for c in _TALIB_NAMES if enc(c) not in panel.columns]
    if missing:
        raise SystemExit(
            f"=== BLOCKED: {where} is MISSING {len(missing)} of the "
            f"{len(_TALIB_NAMES)} TA-Lib columns ===\n"
            f"  {missing}\n"
            "  research.py:554 swallows any exception in the indicator block "
            "into a\n"
            "  `logger.warning`, so this is what a broken or absent TA-Lib "
            "looks like from\n"
            "  here: a panel that is simply short 29 columns. Do NOT read the "
            "comparison\n  below as parity — re-run with the warning visible "
            "(logging at WARNING) to see\n  the underlying error.")


SYMBOL = "BINANCE_PERP_BTC_USDT"
NATIVE = "BTCUSDT"

RAW_COLS = ["open", "high", "low", "close", "volume", "quote_volume",
            "taker_buy_quote_volume", "number_of_trades"]

# ---------------------------------------------------------------------------
# The column contract. `IMPLEMENTED` is the EXACT set stage 2.2 emits; the C++
# side must emit this set and no other, and every one of them must exist in the
# reference panel. That is what stops the coverage from silently shrinking:
# sentinel_core's harness records that a refactor once dropped its comparison
# from 155 columns to 55 while still printing PASS, and a >= floor alone would
# not have caught an engine that emitted the right COUNT of the wrong columns.
# ---------------------------------------------------------------------------
_REAL_NAMES_CODED = [
    "price_range", "price_range_pct", "price_range_pct_q50",
    "price_range_pct_q80", "price_range_pct_q90", "price_range_pct_q95",
    "open_close_diff", "open_close_pct", "high_open_pct", "low_open_pct",
    "ret_lag1", "ret_lag2", "ret_lag3",
    "vol_ratio", "vol_ret_lag1", "vol_ret_lag2", "vol_ret_lag3",
    "quote_vol_ratio", "buy_pressure", "trade_intensity",
]

# ---------------------------------------------------------------------------
# Stage 2.3. The 29 TA-Lib columns (25 calls: MACD, STOCH, STOCHRSI and BBANDS
# each yield two kept outputs, and MACD's signal line and BBANDS' middle band
# are DISCARDED by the reference) — research.py:508-553, in reference order.
#
# `parkinson_vol` is listed SEPARATELY because it is NOT a TA-Lib call
# (research.py:545-548 is numpy + a pandas rolling mean). It is inside the
# reference's `try:` and therefore shares the block's fate, but it is not part
# of the "did the reference actually run TA-Lib?" evidence — it would compute
# fine with TA-Lib uninstalled, so including it in that check would weaken it.
# ---------------------------------------------------------------------------
_TALIB_NAMES = [
    "rsi", "rsi_7", "rsi_28",
    "macd", "macdhist",
    "stoch_k", "stoch_d",
    "cci", "adx", "dx", "plus_di", "minus_di", "mom", "roc", "willr", "cmo",
    "trix", "ultosc",
    "stochrsi_k", "stochrsi_d",
    "obv", "ad", "mfi", "bop",
    "atr", "natr",
    "bb_upper", "bb_lower",
    "sar",
]
_REAL_NAMES_CODED += _TALIB_NAMES + ["parkinson_vol"]

# ---------------------------------------------------------------------------
# Stage 2.4 — the rolling return-moment stats (research.py:557-593). All four
# are computed on `hist_return`, NOT on close.
#
# `std` is the reason stage 2.3 stopped one column short. Its obfuscation code
# is `f085`, and sentinel_core/src/talib_block.cpp:133 emits the SAME code from
# `TA_STDDEV(close, 14)` — the standard deviation of PRICE (~1e4 on BTC), not of
# RETURNS (~4e-3). Had the TA-Lib block emitted f085, this harness would have
# compared a price-scale column against a return-scale reference and reported a
# value diff, which reads like a numeric bug rather than the wrong quantity.
_STATS_NAMES = ["std", "skew", "kurt", "acf_lag1"]

# ---------------------------------------------------------------------------
# Stage 2.5 — the scale-free level transforms (features_scalefree.py:113-129,
# called at research.py:639-646 with window=20, obv_is_cumulative=False), in the
# reference's dict order.
_SCALE_FREE_NAMES = ["sar_dist", "bb_pctb", "bb_width", "macd_norm",
                     "macdhist_norm", "obv_slope", "ad_slope"]

_REAL_NAMES_CODED += _STATS_NAMES + _SCALE_FREE_NAMES
# Uncoded on purpose: dc/obfuscation/map.json has no entry for the three MAs or
# for the `close` passthrough, so both sides carry the real name.
_REAL_NAMES_UNCODED = ["close", "mvg1", "mvg2", "mvg3"]

# The three q* columns whose all-NaN-ness is the pinned production property.
VOL_Q_COLS = ["price_range_pct_q80", "price_range_pct_q90", "price_range_pct_q95"]

# ---------------------------------------------------------------------------
# LOOKAHEAD. Every one of these reads shift(-1) or shift(-2) on close/high/low:
# they are TARGET construction for the trainer, not features, and a live engine
# computing one would be reading a bar that has not closed. Declared, so they
# are reported as deliberately absent rather than quietly missing — and
# cross-checked against the engine's output so "not implemented" cannot quietly
# become "implemented after all".
# ---------------------------------------------------------------------------
EXPECTED_ABSENT_TARGETS = [
    "return",              # research.py:456  price_return = hist_return.shift(-1)
    "return_long",         # research.py:429  (price_return - fee) * size_long
    "return_short",        # research.py:432
    "return_long_raw",     # research.py:434
    "return_short_raw",    # research.py:435
    "return_dip",          # research.py:452  low.shift(-1)  / close - 1
    "return_rip",          # research.py:453  high.shift(-1) / close - 1
    # DUAL_HORIZON only (not set by this harness, listed so the intent is on
    # record if a future arm turns it on): ret_2bar, return_long_2bar,
    # return_short_2bar, return_long_2bar_raw, return_short_2bar_raw.
]


def make_panel(n: int, price: float, seed: int, holes: bool,
               leads: dict | None = None, safe_branch: bool = False) -> pd.DataFrame:
    """A synthetic but realistically-shaped 15m OHLCV panel for ONE symbol.

    `price` sets the SCALE, which is the whole point of running this twice: the
    inline +1e-8 epsilons are absolute, so they are invisible at 64000 and a
    6th-significant-figure effect at 0.0045.
    """
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0, 0.004, n)
    close = price * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[price], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.003, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread * rng.uniform(0.2, 1.0, n)

    volume = rng.lognormal(6.0, 1.0, n)
    quote_volume = volume * close
    taker_buy_quote_volume = quote_volume * rng.uniform(0.2, 0.8, n)
    number_of_trades = np.floor(rng.lognormal(5.0, 0.8, n))

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "quote_volume": quote_volume,
        "taker_buy_quote_volume": taker_buy_quote_volume,
        "number_of_trades": number_of_trades,
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min"))

    if holes:
        # A multi-symbol wide frame is built with pd.concat(axis=1)
        # (research.py:225), so symbols with different histories DO leave NaN
        # holes in a real panel. Shapes chosen to exercise the min_periods
        # accounting the 2.1 primitives implement: a singleton, a run shorter
        # than the MA window, a run LONGER than it, and a trailing hole.
        for col in ("open", "high", "low", "close"):
            df.iloc[13, df.columns.get_loc(col)] = np.nan
            df.iloc[100:104, df.columns.get_loc(col)] = np.nan
            df.iloc[300:312, df.columns.get_loc(col)] = np.nan
            df.iloc[n - 2, df.columns.get_loc(col)] = np.nan
        df.iloc[50:53, df.columns.get_loc("volume")] = np.nan
        df.iloc[200, df.columns.get_loc("quote_volume")] = np.nan
        # Zero denominators, so +/-inf really appears on both sides:
        #   volume == 0    -> vol_ret = pct_change over 0  -> +inf on the next bar
        #   quote_volume 0 -> buy_pressure divides by (0 + 1e-8), a huge finite
        # A harness that only ever sees finite cells cannot tell an engine that
        # propagates inf from one that swallows it.
        df.iloc[400, df.columns.get_loc("volume")] = 0.0
        df.iloc[420, df.columns.get_loc("quote_volume")] = 0.0
        # A flat bar: open == high == low == close makes price_range 0 and
        # price_range_pct 0, and feeds the rolling constant-window guards.
        for col in ("open", "high", "low", "close"):
            df.iloc[500, df.columns.get_loc(col)] = float(df.iloc[500]["close"])

    if leads:
        # STAGGERED LEADING NaNs — the only thing that exercises the TA-Lib
        # wrapper's `check_begidx1..4`.
        #
        # Every generated `talib.*` wrapper skips the leading NaNs of its
        # inputs: it calls the C function on `data + begidx`, where `begidx` is
        # the MAX over the inputs' individual first-valid indices, and writes
        # the result at `begidx + lookback`. Interior NaNs are NOT skipped and
        # DO poison. With every column starting at row 0 (the other three
        # scenarios) `begidx` is 0 everywhere and a C++ that ignored the skip
        # entirely would pass — so the counts here are deliberately DIFFERENT
        # per column, which makes each call site's max() observable:
        #
        #   RSI/MOM/ROC/CMO/TRIX/MACD/STOCHRSI/BBANDS  begidx = close        = 4
        #   SAR                                        = max(high, low)      = 7
        #   STOCH/CCI/ADX/DX/*_DI/WILLR/ULTOSC/ATR/NATR= max(h, l, c)        = 7
        #   BOP                                        = max(o, h, l, c)     = 7
        #   OBV                                        = max(c, volume)      = 9
        #   AD/MFI                                     = max(h, l, c, volume)= 9
        #
        # This is ALSO the scenario in which the `holes` panel is weak: its
        # first hole at row 13 lands inside the warmup of every recursive
        # indicator (RSI's Wilder smoothing, ADX, TRIX, MACD), so those columns
        # come back entirely NaN there and their VALUES go unverified. Here the
        # panel is clean after the head, so they carry real numbers, and the
        # single late hole at row 600 checks that mid-panel poisoning starts at
        # the right row rather than not at all.
        for col, k in leads.items():
            df.iloc[:k, df.columns.get_loc(col)] = np.nan
        df.iloc[600, df.columns.get_loc("high")] = np.nan

    if safe_branch:
        # STAGE 2.5: force BOTH branches of `_safe`
        # (features_scalefree.py:61-63) to actually execute. Stage 2.3 learned
        # this the hard way — all three of the original scenarios happened to
        # have begidx == 0, so an entire class of TA-Lib placement bug could not
        # have been caught by any of them. The same trap applies here: on
        # ordinary market-shaped data NOT ONE of the seven scale-free divisions
        # ever meets a zero or an overflowing denominator, so `_safe` would be
        # graded purely on its pass-through path.
        #
        # Four constructions, each aimed at a specific cell (the counts are
        # ASSERTED, not hoped for — see `safe_branch_coverage`):
        #
        # (a) close == 0.0 EXACTLY, one bar. This is the only construction that
        #     makes step 1 OBSERVABLE: the denominator is zero while the
        #     numerator is NOT (`close - sar` = -sar, `macd`, `bb_upper -
        #     bb_lower`), so a naive `num / den` yields +/-inf where the
        #     reference yields NaN. Without it, an implementation missing step 1
        #     entirely would still pass — see (b) and (c).
        df.iloc[350, df.columns.get_loc("close")] = 0.0

        # (b) close == 1e-305 for one bar, i.e. a price that collapses far
        #     enough that a BTC-scale numerator OVERFLOWS. This is the only
        #     construction that reaches STEP 2. The magnitude is not arbitrary:
        #     step 2 needs |num / den| > DBL_MAX, and with `sar`/`macd`/the band
        #     span at ~1e4 that requires close < ~3.5e-304. 1e-305 is the
        #     mildest value that gets there; anything larger leaves a finite
        #     ratio and step 2 goes unexercised, which is precisely the hole
        #     `--negative-safeinf` proves this closes.
        df.iloc[360, df.columns.get_loc("close")] = 1e-305

        # (c) A 26-bar FLAT run (rows 399..424): 20 identical closes make
        #     TA-Lib's BBANDS stddev exactly 0, so `bb_upper == bb_lower` and
        #     bb_pctb's denominator is zero on rows 418..424.
        #     It DISCRIMINATES, which is not obvious and was measured rather
        #     than assumed: the naive argument says sd == 0 forces close == the
        #     mean, so the numerator `close - bb_lower` would be zero too and
        #     0/0 is NaN either way. It is not zero. TA-Lib's STDDEV clamps its
        #     cancelled variance to exactly 0 while its SMA is a RUNNING SUM
        #     divided by 20, and on this run that rounds to 8.00355e-11 away
        #     from close — so the numerator is -8.00355338e-11 and, without step
        #     1, all seven cells would be -inf rather than NaN.
        #     The run also drives the stage-2.4 constant-window guards on the
        #     zero-return stretch: std -> 0, skew -> 0, kurt -> -3.
        flat = float(df["close"].iloc[399])
        for col in ("open", "high", "low", "close"):
            df.iloc[400:425, df.columns.get_loc(col)] = flat

        # (d) A 25-bar ZERO-VOLUME run (rows 200..224) makes
        #     `volume.rolling(20, min_periods=20).sum()` exactly 0 on rows
        #     219..224, the obv_slope / ad_slope denominator.
        #     THESE SIX CELLS DO NOT DISCRIMINATE, and cannot: volume is
        #     non-negative, so a 20-bar sum of zero forces all 20 bars to zero,
        #     which leaves OBV and AD constant across the run, which makes their
        #     `.diff(14)` — the numerator, and a 14-bar window nested INSIDE the
        #     20-bar one — exactly zero as well. 0/0 is NaN with or without step
        #     1 (measured: 6/0). Kept because it is the only thing that drives
        #     the obv_slope/ad_slope denominator to zero at all, and reported
        #     rather than counted as coverage it does not provide. The step-1
        #     evidence comes from (a) and (c).
        df.iloc[200:225, df.columns.get_loc("volume")] = 0.0
    return df


# ---------------------------------------------------------------------------
# `_safe` BRANCH COVERAGE (stage 2.5).
#
# The value/mask diff already grades every cell, but it cannot say WHICH code
# path produced the match: a scenario in which no denominator is ever zero
# grades `_safe` on its pass-through arm alone and still prints PASS. This
# counts the cells that reach each branch, from the REFERENCE panel and the raw
# input, and REQUIRES the counts to be non-zero on the scenario built for it.
#
# The (num, den) pairs below are re-derived here rather than read out of the
# engine. That is deliberate and it is not a second implementation of the
# feature: nothing here is COMPARED against anything, it only counts branch
# hits. The values themselves are graded by the cell-by-cell diff, against
# research.py, as everything else is.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# A TA-LIB 0.6.4 DEFECT, REPRODUCED AND WAIVED — NOT WORKED AROUND.
#
# ta_NATR.c:334-338 (the pinned 0.6.4 source, read):
#
#       tempValue = inClose[today];
#       if( !TA_IS_ZERO(tempValue) )
#          outReal[outIdx] = (prevATR/tempValue)*100.0;
#       else
#          outReal[0] = 0.0;          /* <-- outReal[0], NOT outReal[outIdx] */
#       outIdx++;
#
# The `else` writes index 0 — a copy-paste of the initialisation twelve lines
# above (`outReal[0] = 0.0;` at :323, where index 0 IS the right target). Two
# separate consequences, and only one of them is deterministic:
#
#   * `outReal[outIdx]` is NEVER WRITTEN. The caller reads back whatever its
#     output buffer already held. The Python wrapper allocates a fresh
#     uninitialised array, so the reference returns HEAP GARBAGE that changes
#     between identical calls (measured: 1.63e+69, 61686.73, 2.24e-314 from
#     three calls on the same inputs). The C++ reuses one scratch buffer across
#     indicator calls, so it reads back the previous indicator's value. NEITHER
#     is right; the reference is simply not a function of its inputs there.
#   * `outReal[0]` IS clobbered to 0.0 — the FIRST emitted NATR value, an
#     unrelated and previously-correct cell, is destroyed. That part is
#     deterministic and BOTH sides reproduce it (natr[14] == 0.0 on the
#     reference and on the C++), so it is compared, not waived.
#
# TA_IS_ZERO(v) is |v| < TA_EPSILON, and TA_EPSILON is 1e-14 (ta_utility.h:257,
# :259) — an ABSOLUTE threshold. No Binance perp trades below 1e-14, so this is
# not reachable on today's feed; it is reachable for any instrument quoted below
# it, and the failure mode is a `natr` column of uninitialised memory with
# nothing logged. REPORTED, NOT FIXED (upstream library).
#
# The waiver is deliberately unable to grow on its own:
#   * the affected rows are DERIVED from the defect (|close| < TA_EPSILON), and
#   * each one must independently be PROVEN non-deterministic by re-running the
#     reference's own `talib.NATR` and observing it disagree with itself, and
#   * any non-determinism OUTSIDE the derived set fails the gate outright, and
#   * a row that turns out to be stable is NOT waived — it is compared.
# If TA-Lib ever fixes this, the proof stops succeeding and the cells go back
# under the diff with no edit here.
# ---------------------------------------------------------------------------
TA_EPSILON = 1e-14
NATR_TRIALS = 16


def natr_unstable_rows(raw: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Rows where the reference's `natr` is not a function of its inputs.

    Returns (waived rows, count of unexplained non-deterministic rows).
    """
    import talib  # noqa: PLC0415

    h = raw["high"].to_numpy(float)
    lo = raw["low"].to_numpy(float)
    c = raw["close"].to_numpy(float)
    predicted = np.abs(c) < TA_EPSILON

    outs = [talib.NATR(h, lo, c, timeperiod=14) for _ in range(NATR_TRIALS)]
    base = outs[0]
    unstable = np.zeros(base.shape, dtype=bool)
    for o in outs[1:]:
        unstable |= ~((o == base) | (np.isnan(o) & np.isnan(base)))

    unexplained = int((unstable & ~predicted).sum())
    return np.flatnonzero(unstable & predicted), unexplained


def _safe_pairs(ref: pd.DataFrame, raw: pd.DataFrame):
    span = ref["bb_upper"] - ref["bb_lower"]
    vol_sum = raw["volume"].astype(float).rolling(20, min_periods=20).sum()
    return {
        "sar_dist":      (ref["close"] - ref["sar"], ref["close"]),
        "bb_pctb":       (ref["close"] - ref["bb_lower"], span),
        "bb_width":      (span, ref["close"]),
        "macd_norm":     (ref["macd"], ref["close"]),
        "macdhist_norm": (ref["macdhist"], ref["close"]),
        "obv_slope":     (ref["obv"], vol_sum),
        "ad_slope":      (ref["ad"], vol_sum),
    }


def safe_branch_coverage(ref: pd.DataFrame, raw: pd.DataFrame, cpp: pd.DataFrame,
                         enc, require: bool) -> int:
    """Print, and optionally require, `_safe` branch-hit counts. Returns failures."""
    failures = 0
    tot_z, tot_zd, tot_inf = 0, 0, 0
    rows = []
    for name, (num, den) in _safe_pairs(ref, raw).items():
        num = num.to_numpy(float)
        den = den.to_numpy(float)
        zero = den == 0.0                     # step 1 fires (== catches -0.0)
        # DISCRIMINATING step-1 cells: the ones where omitting step 1 would
        # change the answer. den == 0 with num == 0 gives NaN either way.
        zero_disc = zero & np.isfinite(num) & (num != 0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            after1 = num / np.where(zero, np.nan, den)
        infs = np.isinf(after1)               # step 2 fires
        tot_z += int(zero.sum())
        tot_zd += int(zero_disc.sum())
        tot_inf += int(infs.sum())
        rows.append((name, int(zero.sum()), int(zero_disc.sum()), int(infs.sum())))

        # Whatever the branch, the OUTPUT must be NaN on both sides. The
        # classification diff already enforces ref == cpp; this pins that the
        # agreed-on value is NaN rather than something they agree on wrongly.
        code = enc(name)
        if code in cpp.columns:
            hit = zero | infs
            v = cpp[code].to_numpy(float)
            bad = int((hit & ~np.isnan(v)).sum())
            if bad:
                print(f"=== FAIL: {name} is NOT NaN on {bad} cells where _safe "
                      "must map the division to NaN ===")
                failures += 1

    print("[feat_parity] _safe branch coverage (den==0 / of those DISCRIMINATING "
          "/ overflow->inf):")
    for i in range(0, len(rows), 4):
        print("    " + "  ".join(f"{n}={z}/{zd}/{f}" for n, z, zd, f in rows[i:i + 4]))
    print(f"    totals: den==0 {tot_z} cells ({tot_zd} discriminating), "
          f"step-2 overflow {tot_inf} cells")

    if require:
        if tot_zd == 0:
            print("=== FAIL: no DISCRIMINATING zero-denominator cell — step 1 of "
                  "_safe (den.replace(0.0, nan)) is unexercised, so an "
                  "implementation without it would pass ===")
            failures += 1
        if tot_inf == 0:
            print("=== FAIL: no overflow cell — step 2 of _safe "
                  "(replace([inf,-inf], nan)) is unexercised ===")
            failures += 1
    return failures


def reference_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Run the REAL research.py over `raw` and return its engineered frame.

    Nothing is reimplemented: `AgamottoResearch` is constructed with its `.raw`
    set directly (bypassing `load()`, which only reads CSVs off disk) and
    `engineer_features()` is called. If research.py changes, this harness sees
    the change.
    """
    import agamotto.research as research  # noqa: PLC0415

    wide = raw.rename(columns={c: f"{NATIVE}_{c}" for c in raw.columns})
    config = {
        "SYMBOLS": [SYMBOL],
        "TIME_UNIT": "15m",
        # Required by research.py with NO defaults (CLAUDE.md): FEE and
        # LADDER_BPS/LADDER feed the TARGET columns, which stage 2.2 does not
        # implement — but engineer_features computes them unconditionally, so
        # they must be present or it raises. Values taken from the deployed arm
        # (marvel/gauntlet/pred_agamotto.base.15m_1/setting.json).
        "FEE": 0.0,
        "LADDER_BPS": 1.0,
        "LADDER": 1,
        "STATS_WINDOW": 14,
        # MA_PERIODS is deliberately ABSENT: the deployed setting.json carries
        # no such key, so the research.py:461 default [7, 25, 99] is what
        # production actually uses and what the C++ hardcodes.
    }
    ar = research.AgamottoResearch(config, home_root=str(DC_ROOT))
    ar.raw = wide
    ar.engineer_features()
    if ar.features is None:
        raise SystemExit("engineer_features() produced no panel")

    # Assert the default really bound, rather than trusting the comment above.
    for mvg in ("mvg1", "mvg2", "mvg3"):
        if f"{NATIVE}_{mvg}" not in ar.features.columns:
            raise SystemExit(f"reference emitted no {mvg} — MA block did not run")

    out = ar.features.rename(
        columns={c: c[len(NATIVE) + 1:] for c in ar.features.columns
                 if c.startswith(f"{NATIVE}_")})
    return out


def encoder():
    """name -> obfuscation code, with pass-through for uncoded names.

    The C++ emits CODED column keys (no real feature name compiles into the
    .so). The reference emits real names. Encode the reference before comparing
    or only `close`/`mvg*` line up and the run reports a meaningless PASS over
    four columns.
    """
    mp = json.loads((DC_ROOT / "obfuscation" / "map.json").read_text())["features"]
    return lambda c: mp.get(c, c)


def classify(a: np.ndarray) -> np.ndarray:
    """0 finite, 1 NaN, 2 +inf, 3 -inf.

    Compared for EQUALITY before any value is diffed. NaN-vs-NaN and
    +inf-vs-+inf are matches; NaN-vs-0.0, NaN-vs-+inf and +inf-vs--inf are
    failures. See the module docstring, point 1.
    """
    out = np.zeros(a.shape, dtype=np.int8)
    out[np.isnan(a)] = 1
    out[np.isposinf(a)] = 2
    out[np.isneginf(a)] = 3
    return out


def run_driver(driver_cmd: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Feed the raw panel to the C++ driver as CSV, read its panel back."""
    buf = io.StringIO()
    # %.17g on the C++ side, repr-precision here: both round-trip a double
    # exactly, so the seam adds no error of its own.
    raw.to_csv(buf, index=False, float_format="%.17g", na_rep="nan")
    proc = subprocess.run(shlex.split(driver_cmd), input=buf.getvalue(),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"driver failed (rc={proc.returncode}):\n{proc.stderr}")
    lines = proc.stdout.strip().splitlines()
    if len(lines) < 2:
        raise SystemExit(f"driver produced no rows:\n{proc.stderr}")
    return pd.DataFrame([[float(x) for x in ln.split(",")] for ln in lines[1:]],
                        columns=lines[0].split(","))


def compare(name: str, ref: pd.DataFrame, cpp: pd.DataFrame, enc, tol: float,
            raw: pd.DataFrame, require_safe_cov: bool) -> int:
    """Return the number of FAILURES for one scenario (0 == pass)."""
    print(f"\n--- scenario: {name} ---")
    # Stage 2.5 branch coverage, on the UNRENAMED reference (it reads real
    # names) and before the rename below rebinds `ref`.
    failures_pre = safe_branch_coverage(ref, raw, cpp, enc, require_safe_cov)
    ref = ref.rename(columns={c: enc(c) for c in ref.columns})

    implemented = {enc(c) for c in _REAL_NAMES_CODED} | set(_REAL_NAMES_UNCODED)
    cpp_cols = set(cpp.columns)
    failures = failures_pre

    # (d) COVERAGE, in the strict form: the engine must emit EXACTLY the
    # declared set. A `>= MIN_SHARED` floor alone would bless an engine that
    # emitted the right number of the wrong columns.
    MIN_SHARED = len(implemented)
    if cpp_cols != implemented:
        print(f"=== FAIL: C++ emitted {len(cpp_cols)} columns, contract declares "
              f"{len(implemented)} ===")
        print(f"    unexpected: {sorted(cpp_cols - implemented)}")
        print(f"    missing   : {sorted(implemented - cpp_cols)}")
        failures += 1
    absent_from_ref = implemented - set(ref.columns)
    if absent_from_ref:
        print(f"=== FAIL: {len(absent_from_ref)} implemented columns do not exist "
              f"in the reference panel: {sorted(absent_from_ref)} ===")
        failures += 1

    shared = sorted(cpp_cols & set(ref.columns))
    if len(shared) < MIN_SHARED:
        print(f"=== FAIL: only {len(shared)} columns compared (contract requires "
              f"{MIN_SHARED}) — the rest are UNVERIFIED ===")
        failures += 1

    # LOOKAHEAD: declared absent, and PROVEN absent from the engine.
    target_codes = {enc(c) for c in EXPECTED_ABSENT_TARGETS}
    leaked = sorted(target_codes & cpp_cols)
    if leaked:
        print(f"=== FAIL: the engine emitted LOOKAHEAD target columns {leaked} ===")
        failures += 1
    not_in_ref = sorted(c for c in EXPECTED_ABSENT_TARGETS if enc(c) not in ref.columns)
    if not_in_ref:
        print(f"=== FAIL: EXPECTED_ABSENT_TARGETS {not_in_ref} are not in the "
              "reference either — the declaration is stale ===")
        failures += 1
    print(f"[feat_parity] EXPECTED_ABSENT (lookahead targets, deliberately not "
          f"implemented): {', '.join(EXPECTED_ABSENT_TARGETS)}")

    later_stages = sorted(set(ref.columns) - implemented - target_codes)
    print(f"[feat_parity] later stages / passthrough, not compared "
          f"({len(later_stages)}): {', '.join(later_stages[:12])}"
          f"{' ...' if len(later_stages) > 12 else ''}")

    if len(ref) != len(cpp):
        print(f"=== FAIL: rows ref={len(ref)} cpp={len(cpp)} ===")
        return failures + 1

    # (3) the pinned production property: all-NaN vol-quantile cutoffs.
    for col in VOL_Q_COLS:
        code = enc(col)
        for side, frame in (("reference", ref), ("C++", cpp)):
            if code not in frame.columns:
                continue
            v = frame[code].to_numpy(float)
            if not np.isnan(v).all():
                print(f"=== FAIL: {col} is NOT all-NaN on the {side} side "
                      f"({int((~np.isnan(v)).sum())} finite cells) — at "
                      f"{PANEL_BARS} rows < min_periods={700} it must be. See "
                      "marvel PR #532. ===")
                failures += 1

    # The TA-Lib NATR defect (ta_NATR.c:334-338). Derived from the defect AND
    # independently proven non-deterministic; anything unexplained fails here
    # rather than being absorbed. See `natr_unstable_rows`.
    natr_waived, natr_unexplained = natr_unstable_rows(raw)
    if natr_unexplained:
        print(f"=== FAIL: the reference disagrees with itself on "
              f"{natr_unexplained} natr cells that the TA-Lib NATR defect does "
              "NOT explain (|close| >= TA_EPSILON) — an unknown source of "
              "non-determinism, not a known one ===")
        failures += 1
    waived_cells = {}
    if natr_waived.size:
        waived_cells[enc("natr")] = natr_waived
        print(f"[feat_parity] WAIVED {natr_waived.size} natr cells at rows "
              f"{natr_waived.tolist()} — |close| < TA_EPSILON=1e-14 there, and "
              f"ta_NATR.c:334-338 leaves outReal[outIdx] UNWRITTEN (its `else` "
              "writes outReal[0]). Proven non-deterministic over "
              f"{NATR_TRIALS} identical reference calls. Library defect, "
              "reported not fixed. Every other cell of natr IS compared, "
              "including natr[14], which the same bug deterministically "
              "clobbers to 0.0 on both sides.")

    def _keep(code: str, a: np.ndarray) -> np.ndarray:
        """Mask of cells that count, i.e. everything except a proven waiver."""
        m = np.ones(a.shape, dtype=bool)
        if code in waived_cells:
            m[waived_cells[code]] = False
        return m

    # (5) FIRST-VALID-INDEX, per column. See the module docstring, point 5.
    #
    # This is the only check that catches a TA-Lib `outBegIdx` placement error.
    # A column placed at the wrong offset is SHIFTED by its lookback: the value
    # diff over the overlapping region can still pass on a slow-moving
    # indicator, and the NaN classification only moves if the head LENGTH
    # changed. The first non-NaN row index does move, always, and it is exactly
    # what `place(n, begidx, outBeg, outCnt, raw)` decides.
    #
    # It also pins the wrapper's leading-NaN skip: `talib.RSI` calls the C
    # function on `data + begidx`, so a C++ that ignored `begidx` would place
    # the head `begidx` rows early and nothing else in this harness would care.
    fvi_bad, fvi_table = [], []
    for c in shared:
        r = pd.to_numeric(ref[c], errors="coerce").to_numpy(float)
        v = cpp[c].to_numpy(float)

        def _first(a):
            ok = np.flatnonzero(~np.isnan(a))
            return int(ok[0]) if ok.size else None

        fr, fv = _first(r), _first(v)
        fvi_table.append((c, fr, fv))
        if fr != fv:
            fvi_bad.append((c, fr, fv))
    print(f"[feat_parity] first-valid-index (ref == cpp) over {len(fvi_table)} "
          "columns:")
    for i in range(0, len(fvi_table), 6):
        print("    " + "  ".join(
            f"{c}={'-' if fr is None else fr}"
            for c, fr, _ in fvi_table[i:i + 6]))
    if fvi_bad:
        print(f"=== FAIL: {len(fvi_bad)} columns start at a DIFFERENT row than "
              "the reference — a compacted TA-Lib output is misplaced, which "
              "SHIFTS the whole column ===")
        for c, fr, fv in fvi_bad[:20]:
            print(f"   {c}: ref first valid row {fr}, cpp {fv}")
        failures += 1

    mask_bad, val_bad = [], []
    for c in shared:
        r = pd.to_numeric(ref[c], errors="coerce").to_numpy(float)
        v = cpp[c].to_numpy(float)
        keep = _keep(c, r)
        kr, kv = classify(r), classify(v)
        diff_mask = (kr != kv) & keep
        nmask = int(diff_mask.sum())
        if nmask:
            i = int(np.argmax(diff_mask))
            mask_bad.append((c, nmask, i, r[i], v[i]))
            continue  # a mask break is the failure; the values are moot
        fin = (kr == 0) & keep
        if not fin.any():
            continue
        d = np.abs(r[fin] - v[fin]) / np.maximum(np.abs(r[fin]), 1e-300)
        k = int((d > tol).sum())
        if k:
            val_bad.append((c, k, float(d.max())))

    print(f"[feat_parity] compared {len(shared)} columns x {len(ref)} rows")
    if mask_bad:
        print(f"=== FAIL: {len(mask_bad)} columns differ in NaN/inf CLASSIFICATION ===")
        for c, k, i, rv, vv in mask_bad[:20]:
            print(f"   {c}: {k} cells, first row {i}: ref={rv!r} cpp={vv!r}")
        failures += 1
    if val_bad:
        print(f"=== FAIL: {len(val_bad)} columns differ in VALUE ===")
        for c, k, mx in val_bad[:20]:
            print(f"   {c}: {k} cells, max rel diff {mx:.3e}")
        failures += 1
    if not failures:
        print(f"=== PASS: {len(shared)} columns identical, masks and values "
              f"(rel tol={tol}) ===")
    return failures


# ---------------------------------------------------------------------------
# THE SCENARIOS. Module-level, not local to main(), because
# tests/regime_parity.py (Phase 3) imports and reuses them: the gate must be
# graded on the SAME five panels the feature engine was, including the PEPE
# price scale and the NaN/leading-NaN/_safe-branch constructions. Two
# generators would drift, and the drift would be invisible — each harness would
# go on passing against its own idea of a panel.
SCENARIOS = [
    # (name, price, seed, NaN holes, leading NaNs, _safe-branch construction)
    ("BTC-like  ~64000", 64000.0, 20260819, False, None, False),
    ("PEPE-like ~0.0045", 0.0045, 20260820, False, None, False),
    ("BTC-like + NaN holes, zero volume, flat bar", 64000.0, 20260821, True,
     None, False),
    # Stage 2.3: the TA-Lib wrapper's leading-NaN skip. Counts differ per
    # column ON PURPOSE so each call site's check_begidxN max() is
    # observable — see make_panel.
    ("BTC-like + staggered LEADING NaNs (o5 h3 l7 c4 v9) + late hole",
     64000.0, 20260822, False,
     {"open": 5, "high": 3, "low": 7, "close": 4, "volume": 9}, False),
    # Stage 2.5: the ONLY scenario in which `_safe` does anything. See
    # make_panel's `safe_branch` block and `safe_branch_coverage`; the
    # branch-hit counts are ASSERTED here, not merely printed, because the
    # other four scenarios never produce a single zero or overflowing
    # denominator and would grade `_safe` on its pass-through arm alone.
    ("BTC-like + _safe branches (close==0, close==1e-305, 26-bar flat run, "
     "25-bar zero-volume run)", 64000.0, 20260823, False, None, True),
]



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True,
                    help="command that reads the raw CSV on stdin and prints "
                         "the engineered panel (shlex-split, so a `docker run "
                         "-i ...` line works)")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    # BEFORE anything else: without TA-Lib at the pin, research.py silently
    # returns a panel missing 30 columns and the remaining 24 would still pass.
    ta_ver = require_reference_talib()

    print(f"[feat_parity] pandas {pd.__version__}  numpy {np.__version__}")
    print(f"[feat_parity] reference TA-Lib C library {ta_ver} (pinned)")
    print(f"[feat_parity] PANEL_BARS={PANEL_BARS} (trading.py:443/:480/:485)")
    print(f"[feat_parity] driver: {args.driver}")

    enc = encoder()
    scenarios = SCENARIOS
    failures = 0
    for name, price, seed, holes, leads, safe_branch in scenarios:
        raw = make_panel(PANEL_BARS, price, seed, holes, leads, safe_branch)
        ref = reference_panel(raw)
        assert_reference_used_talib(
            ref.rename(columns={c: enc(c) for c in ref.columns}), enc,
            f"the reference panel for scenario {name!r}")
        cpp = run_driver(args.driver, raw)
        failures += compare(name, ref, cpp, enc, args.tol, raw, safe_branch)

    print()
    if failures:
        print(f"=== FEATURE PARITY FAILED ({failures} failure groups) ===")
        return 1
    print(f"=== FEATURE PARITY PASS: {len(scenarios)} scenarios ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
