#!/usr/bin/env python3
"""Golden generator for the pdops numeric primitives (Phase 2, stage 2.1).

Emits two CSVs:

  pdops_input.csv   the INPUT series — realistic 15m-shaped OHLCV derivatives
                    with deliberately injected NaN holes and constant runs.
  pdops_golden.csv  the EXPECTED output of every primitive at every
                    min_periods actually used by the agamotto reference,
                    computed by pandas itself.

The golden header is the SPEC. Each column name encodes the call the C++ must
make, e.g. ``rollskew|ret|14|14`` = ``ret.rolling(14, min_periods=14).skew()``.
``tests/pdops_parity_driver.cpp`` parses that header and dispatches, so the two
sides cannot drift apart: adding a column here automatically adds a C++ check,
and a spec the driver cannot parse is a hard failure, never a skip.

Columns prefixed ``NEG_`` are the NEGATIVE tests. They hold a DELIBERATELY
WRONG computation (population moments instead of the sample/bias-corrected
ones, nearest-rank instead of linear interpolation). The driver asserts the C++
does NOT match them. Without these, an implementation that silently used the
population form would pass a positive-only harness on any window where the two
happen to be close.

Usage:
    python tests/pdops_golden.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# pandas semantics are version-specific (the rolling kernels have changed
# repeatedly). Refuse to emit a golden from a version the C++ was not written
# against — a silent version drift would produce a golden that grades correct
# C++ as broken, or blesses broken C++.
REQUIRED_PANDAS = "2.3.3"

N_ROWS = 2000
SEED = 20260819

# The two windows the agamotto reference actually uses, and the min_periods
# each is called with. STATS_WINDOW default 14 (research.py:557); its
# acf_lag1 companion uses STATS_WINDOW-1 = 13; VOL_Q_WINDOW = 700
# (research.py:61) is called at BOTH min_periods=1 (price_range_pct_q50,
# research.py:363) and min_periods=700 (the high_vol_q* cutoffs,
# research.py:372-374).
STATS_WINDOW = 14
ACF_WINDOW = STATS_WINDOW - 1
VOL_Q_WINDOW = 700
VOL_Q_LEVELS = (0.80, 0.90, 0.95)


# ---------------------------------------------------------------------------
# input series
# ---------------------------------------------------------------------------
def build_inputs() -> pd.DataFrame:
    """Realistic 15m-shaped series with the NaN patterns that break parity.

    The holes are not decoration. Each one targets a specific way a
    hand-rolled rolling primitive diverges from pandas:

      * singleton NaN            -> min_periods counts OBSERVATIONS, not rows
      * a run shorter than w     -> partial windows drop below min_periods
      * a run LONGER than w      -> an entirely-NaN window (nobs == 0)
      * a trailing NaN           -> the last window is short
      * a CONSTANT run           -> pandas' num_consecutive_same_value guards
                                    (std/skew/kurt/sum) which force exact
                                    0 / -3 / prev_value*nobs instead of the
                                    floating-point residue
      * DIFFERENT masks on x and y -> rolling corr PAIRWISE-MASKS BOTH SERIES
                                    FIRST (rolling.py prep_binary:
                                    `X = x + 0*y; Y = y + 0*x`), so every mean
                                    AND every variance is over the pairwise
                                    observations. An implementation that takes
                                    each series' own non-NaN values computes a
                                    plausible number that is a DIFFERENT
                                    statistic — measured 100%+ off on the rows
                                    after a hole (research.py:589 correlates
                                    hist_return with hist_return.shift(1),
                                    whose masks differ by one row at every
                                    hole, so this is the live case)
    """
    rng = np.random.default_rng(SEED)
    n = N_ROWS

    step = rng.normal(0.0, 0.0015, n)          # ~15m BTC vol
    close_clean = 64000.0 * np.exp(np.cumsum(step))
    open_ = np.concatenate([[64000.0], close_clean[:-1]])
    high = np.maximum(open_, close_clean) * (1.0 + np.abs(rng.normal(0, 8e-4, n)))
    low = np.minimum(open_, close_clean) * (1.0 - np.abs(rng.normal(0, 8e-4, n)))
    volume = np.exp(rng.normal(6.0, 0.8, n))

    # research.py:362 — the EPS is 1e-8 written inline, NOT mjolnir's 1e-10.
    prp = (high - low) / (open_ + 1e-8)
    # research.py:382 — computed from the CLEAN close, before the holes below.
    ret = pd.Series(close_clean).pct_change(fill_method=None).to_numpy()

    close = close_clean.copy()

    # constant runs (longer than STATS_WINDOW so a whole window is constant)
    prp[300:316] = prp[299]
    volume[500:520] = 1234.5

    # NaN holes — deliberately DIFFERENT masks per series
    close[1000] = np.nan
    close[1200:1203] = np.nan

    ret[200:203] = np.nan
    ret[1500:1520] = np.nan

    prp[100] = np.nan
    prp[137] = np.nan
    prp[400:404] = np.nan
    prp[900:930] = np.nan          # > STATS_WINDOW: an all-NaN 14-window

    volume[50] = np.nan
    volume[700:706] = np.nan
    volume[n - 1] = np.nan         # trailing hole

    # +/-inf is NOT a NaN and NOT a huge number to pandas: Rolling._prep_values
    # (rolling.py:375-378) rewrites every inf to NaN before any rolling kernel
    # sees it. A hand-rolled primitive that lets an inf into a Kahan
    # accumulator poisons every LATER window of that series, not just the one
    # containing it. `pct_change` produces infs from a zero denominator
    # (research.py:475 divides by a rolling volume mean), so this is a real
    # path, not a synthetic one.
    infy = ret.copy()
    infy[600] = np.inf
    infy[601] = -np.inf
    infy[1100:1103] = np.inf

    # --- the ZERO-VARIANCE / FP-CONTRACTION regression pair ----------------
    #
    # `flat` is piecewise constant in blocks LONGER than STATS_WINDOW, so many
    # 14-windows are exactly constant and pandas' same-value guard forces
    # var == 0. Paired against the varying `wobble`, rolling corr then divides
    # by an exactly-zero denominator on those rows, and the ONLY thing deciding
    # NaN vs +/-inf is whether `mean(X*Y) - mean(X)*mean(Y)` cancelled to
    # exactly 0.0. That is a last-bit predicate, and it is precisely what a
    # fused multiply-add changes (see the banner in feature_engine.cpp).
    #
    # In the original panel this path was reached only by luck — `vol` happened
    # to carry a constant run at 1234.5 — so a reseed or a data reshuffle would
    # have retired the coverage silently. These two series reach it on purpose.
    #
    # The block values are deliberately NON-DYADIC (0.037-spaced, offsets
    # ending in .37), because a dyadic constant makes mean(X*c) == mean(X)*c
    # exact and the +/-inf branch would never fire. One dyadic block (1024.0)
    # is kept so the exactly-zero-numerator branch fires too, and
    # `report_regression_coverage` ASSERTS both branches are actually reached —
    # otherwise this spec could pass vacuously.
    #
    # EVERY block is kept inside ONE NARROW scale band (~1.0e3 .. 1.3e3). That
    # is a hard constraint, not an aesthetic. An earlier draft spread the
    # blocks over 0.1 .. 6.1e4; a 14-window straddling such a boundary cancels
    # over many decades, and pandas cannot reproduce THAT even against ITSELF —
    # measured, moving only the frame start: 6.0e-4 (std), 1.9e1 (skew), 3.3e4
    # (kurt) relative self-disagreement, and 18-29 rows of NaN-MASK disagreement
    # in skew/kurt. Gating against that measures the reference's own noise. A
    # narrow band keeps every mixed window well-conditioned, so the whole series
    # is gateable at the normal 1e-12 and the ONLY interesting thing left is the
    # constant-window predicate this spec exists for.
    flat = np.empty(n)
    blk = 23                                   # > STATS_WINDOW, not a divisor of it
    for k, s0 in enumerate(range(0, n, blk)):
        seg = flat[s0:s0 + blk]
        lvl = k % 4
        if lvl == 0:                           # varying: partial/mixed windows
            seg[:] = 1100.0 + (close_clean[s0:s0 + seg.size] - 64000.0) * 1e-2
        elif lvl == 1:
            seg[:] = 1000.137 + 0.037 * k                        # non-dyadic
        elif lvl == 2:
            seg[:] = 1234.37 + 0.01 * k                          # non-dyadic
        else:
            seg[:] = 1024.0                                      # DYADIC
    flat[820:829] = np.nan                     # a hole inside a constant block
    wobble = volume * (1.0 + 0.5 * np.sin(np.arange(n) * 0.37))
    wobble[1400:1405] = np.nan                 # a mask that differs from flat's

    # --- `nearflat`: the NEGATIVE-VARIANCE / zsqrt regression ---------------
    #
    # Blocks that are near-constant but NOT constant, so pandas' same-value
    # guard does NOT fire and the streaming Welford residue is computed for
    # real. At a base of 1024 with a 1e-7 perturbation the true variance is
    # ~2.5e-15 while one update's roundoff is ~1e-13, so the residue SWAMPS the
    # statistic and pandas' rolling var comes out NEGATIVE — measured, 170 of
    # 2000 rows, between -1.4e-10 and -3.5e-9.
    #
    # That is not a curiosity, it is a NaN-mask predicate: calc_var has no
    # `if result < 0` clamp, so a port that does the obvious `sqrt(var)` returns
    # NaN on every one of those rows, while pandas' Rolling.std is `zsqrt`
    # (common.py:149-161), which maps negative -> 0.0. This series is what
    # caught that bug in the port.
    #
    # It IS gateable, unlike the near-constant skew/kurt case, because only the
    # SIGN of the residue is consulted and the sign is robust: -1e-10 sits five
    # decades above the ~1e-15 that a fused multiply-add moves it by. Verified
    # directly — recomputing every row with and without an FMA at the ssqdm
    # update gives 0 sign changes in 1965 rows — so std is EXACTLY 0.0 there on
    # any toolchain, not a value that happens to round the same way.
    nearflat = np.empty(n)
    for k, s0 in enumerate(range(0, n, blk)):
        seg = nearflat[s0:s0 + blk]
        if k % 3 == 0:                         # varying: well-conditioned rows
            seg[:] = 1024.0 + (close_clean[s0:s0 + seg.size] - 64000.0) * 1e-2
        else:
            eps = 1e-7 * float(2 ** ((k // 3) % 5))
            seg[:] = 1024.0 + eps * (np.arange(seg.size) % 2)
    nearflat[1700] = np.nan

    df = pd.DataFrame({
        "close": close,
        "ret": ret,
        "prp": prp,
        "vol": volume,
        "infy": infy,
        "flat": flat,
        "wobble": wobble,
        "nearflat": nearflat,
    })
    # An explicit second series for corr, so the corr check does not silently
    # depend on shift() also being right.
    df["retlag1"] = df["ret"].shift(1)
    return df


# ---------------------------------------------------------------------------
# the spec: every primitive at every min_periods the reference uses
# ---------------------------------------------------------------------------
def build_specs() -> list[str]:
    w = STATS_WINDOW
    q = VOL_Q_WINDOW
    lv = ";".join(f"{x:g}" for x in VOL_Q_LEVELS)
    specs: list[str] = []

    # --- elementwise -------------------------------------------------------
    for s in ("close", "ret", "prp", "vol"):
        specs.append(f"diff|{s}")
        specs.append(f"shift|{s}|1")
        specs.append(f"shift|{s}|3")
    for s in ("close", "vol"):
        specs.append(f"pctchange|{s}|1")
        specs.append(f"pctchange|{s}|3")
        specs.append(f"diffn|{s}|14")          # research.py:538 obv/ad .diff(14)

    # --- rollSum (features_scalefree.py:115 volume.rolling(w, min_periods=w)) -
    specs += [f"rollsum|vol|{w}|{w}", f"rollsum|vol|{w}|1",
              f"rollsum|vol|{q}|{q}", f"rollsum|vol|{q}|1",
              f"rollsum|prp|{w}|{w}"]

    # --- rollMean (research.py:466-498 min_periods=1) -----------------------
    specs += ["rollmean|close|7|1", "rollmean|vol|7|1",
              f"rollmean|prp|{w}|{w}", f"rollmean|close|{w}|1",
              f"rollmean|prp|{q}|{q}", f"rollmean|prp|{q}|1"]

    # --- rollStd (research.py:569 rolling(w) => min_periods=w) --------------
    specs += [f"rollstd|ret|{w}|{w}", f"rollstd|close|{w}|1",
              f"rollstd|prp|{w}|{w}", f"rollstd|vol|{w}|{w}",
              f"rollstd|prp|{q}|{q}", f"rollstd|close|{q}|1"]

    # --- rollSkew / rollKurt (research.py:570-571) -------------------------
    # GATED on the shapes the reference actually feeds them. research.py:570-571
    # call .skew()/.kurt() on `hist_return` ONLY — a zero-mean ~1e-3 series —
    # and the 700-window case is included because the same code path serves it.
    specs += [f"rollskew|ret|{w}|{w}", f"rollskew|ret|{w}|1",
              f"rollskew|prp|{q}|{q}"]
    specs += [f"rollkurt|ret|{w}|{w}", f"rollkurt|ret|{w}|1",
              f"rollkurt|prp|{q}|{q}"]
    # PROBE_ = computed and REPORTED by the driver, NOT gated. These are
    # ill-conditioned inputs the reference never sends to skew/kurt: a
    # price-scale series (close ~6.4e4) and windows dominated by a constant
    # run (vol, prp at w=14). pandas' rolling skew/kurt accumulate RAW moments
    # (sum x, x^2, x^3, x^4) rather than centred ones, so on a price-scale
    # series the 4th raw moment is ~1e19 while the 4th CENTRED moment is ~1e7
    # — a cancellation of 1e12 that makes the last 12 digits of the answer an
    # artifact of accumulation order. Measured 2026-08-19: pandas cannot even
    # reproduce ITSELF there — the same 14-bar window computed from a frame
    # starting 1300 rows earlier differs by 5.6e-8 relative (see the
    # `--repro` diagnostic below). Gating these would be gating noise; they
    # are kept because a REGRESSION in them (orders of magnitude, not digits)
    # would still expose a real formula error.
    specs += [f"PROBE_rollskew|prp|{w}|{w}", f"PROBE_rollskew|close|{w}|1",
              f"PROBE_rollskew|vol|{w}|{w}", f"PROBE_rollskew|close|{q}|1"]
    specs += [f"PROBE_rollkurt|prp|{w}|{w}", f"PROBE_rollkurt|close|{w}|1",
              f"PROBE_rollkurt|vol|{w}|{w}", f"PROBE_rollkurt|close|{q}|1"]

    # --- rollCorr (research.py:589-590 rolling(w-1).corr(x.shift(1))) -------
    specs += [f"rollcorr|ret|retlag1|{ACF_WINDOW}|{ACF_WINDOW}",
              f"rollcorr|ret|retlag1|{ACF_WINDOW}|1",
              f"rollcorr|close|vol|{w}|{w}",       # different NaN masks
              f"rollcorr|prp|vol|{w}|1",           # different NaN masks, mp=1
              f"rollcorr|prp|vol|{q}|{q}"]

    # --- ZERO-VARIANCE / FP-CONTRACTION regression -------------------------
    # Constant windows => var == 0 => the result is decided purely by whether
    # the numerator cancelled to exactly 0.0 (NaN) or landed 1 ULP off
    # (+/-inf). BOTH orders, so the vX == 0 and vY == 0 branches each fire.
    # This is the spec that fails if FP contraction is ever re-enabled at a
    # mask-deciding site; see feature_engine.cpp's banner and pdRound().
    #
    # PROBE_, and deliberately so — PROBE does NOT weaken this test. The driver
    # gates NaN MASKS at zero tolerance on PROBE columns too, and the mask is
    # the entire point here: `report_regression_coverage` below shows these
    # carry 600+ rows of zero-denominator division split across both outcomes.
    # What PROBE drops is the VALUE comparison, which cannot be gated at 1e-12
    # on these series because the denominator is a rolling variance and
    # roll_var's last bit is not portable: the pandas 2.3.3 arm64 wheel
    # contracts add_var's ssqdm update into an FMA, a baseline-x86-64 wheel
    # cannot, and the port must pick one (it picks the source form). On a
    # constant/near-constant window that 1 ULP is amplified to ~1e-11.
    #
    # The ORIGINAL defect stays covered by GATED columns: rollcorr|close|vol|
    # 14|14 and rollcorr|prp|vol|14|1 are the two that actually failed on
    # clang, and they remain gated on value as well as mask.
    specs += [f"PROBE_rollcorr|wobble|flat|{w}|{w}",
              f"PROBE_rollcorr|flat|wobble|{w}|{w}",
              f"PROBE_rollcorr|flat|wobble|{w}|1",
              f"PROBE_rollcorr|flat|close|{w}|1"]
    # The same constant windows through the other kernels. mean/sum/quantile
    # are EXACT on constant windows (no cancellation at all), so those stay
    # fully GATED; std/skew/kurt inherit roll_var's non-portable last bit.
    specs += [f"rollmean|flat|{w}|1", f"rollsum|flat|{w}|{w}",
              f"rollquantile|flat|{w}|{w}|0.5"]
    specs += [f"PROBE_rollstd|flat|{w}|{w}", f"PROBE_rollstd|flat|{w}|1",
              f"PROBE_rollskew|flat|{w}|{w}", f"PROBE_rollkurt|flat|{w}|{w}"]
    # `nearflat`: the NEGATIVE-variance rows. pandas' Rolling.std is zsqrt, so
    # negative variance must come out as EXACTLY 0.0, not sqrt(negative) = NaN.
    # That is a MASK statement, which PROBE gates at zero tolerance — and it is
    # portable, because only the SIGN of the residue is consulted and the sign
    # survives an FMA (verified: 0 sign changes in 1965 rows).
    specs += [f"PROBE_rollstd|nearflat|{w}|{w}", f"PROBE_rollstd|nearflat|{w}|1",
              f"PROBE_rollcorr|nearflat|wobble|{w}|{w}"]
    # skew/kurt on `nearflat` are ABSENT, not PROBE. Their NaN mask is decided
    # by `B <= 1e-14` where B is pure cancellation residue at this scale, and
    # pandas cannot reproduce that mask against ITSELF — moving only the frame
    # start flips 101-113 rows. Since the driver gates masks even on PROBE
    # columns (correctly), there is no honest way to include them: the golden
    # would be pinning a coin flip. Do not add them back.

    # --- inf handling: _prep_values rewrites +/-inf to NaN before EVERY
    # rolling kernel, so these must come out identical to the NaN-hole case.
    specs += [f"rollsum|infy|{w}|{w}", f"rollmean|infy|{w}|1",
              f"rollstd|infy|{w}|{w}", f"rollskew|infy|{w}|{w}",
              f"rollkurt|infy|{w}|{w}", f"rollcorr|infy|ret|{w}|{w}",
              f"rollcorr|ret|infy|{w}|1"]

    # --- rollQuantile ------------------------------------------------------
    # research.py:363 (700, min_periods=1, q=0.5) and :372-374 (700/700, q*)
    specs.append(f"rollquantile|prp|{q}|1|0.5")
    for lvl in VOL_Q_LEVELS:
        specs.append(f"rollquantile|prp|{q}|{q}|{lvl:g}")
    # the exact-integer interpolation path (idx_with_fraction == idx)
    specs += [f"rollquantile|infy|{w}|{w}|0.5",
              f"rollquantile|vol|{w}|1|0", f"rollquantile|vol|{w}|1|1",
              f"rollquantile|ret|{w}|{w}|0.25", f"rollquantile|ret|{w}|{w}|0.5",
              f"rollquantile|ret|{w}|{w}|0.75", f"rollquantile|close|{w}|1|0.5"]

    # --- rollQuantiles: k levels from ONE sort, must equal the singles ------
    for i in range(len(VOL_Q_LEVELS)):
        specs.append(f"rollquantiles|prp|{q}|{q}|{lv}|{i}")

    # --- NEGATIVE tests ----------------------------------------------------
    # If the C++ MATCHES any of these, the bias correction / interpolation is
    # missing and the positive tests were passing by coincidence.
    specs += [f"NEG_popskew|ret|{w}|{w}", f"NEG_popskew|prp|{w}|{w}",
              f"NEG_popkurt|ret|{w}|{w}", f"NEG_popkurt|prp|{w}|{w}",
              f"NEG_nearestquantile|prp|{q}|{q}|0.8",
              f"NEG_nearestquantile|ret|{w}|{w}|0.25"]
    return specs


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def _pop_skew(x: pd.Series, w: int, mp: int) -> np.ndarray:
    """POPULATION g1 — the WRONG one. m3 / m2**1.5, no n/((n-1)(n-2)) factor."""
    def f(v: np.ndarray) -> float:
        v = v[~np.isnan(v)]
        n = v.size
        if n < 3:
            return np.nan
        d = v - v.mean()
        m2 = (d ** 2).mean()
        m3 = (d ** 3).mean()
        if m2 <= 0:
            return np.nan
        return m3 / m2 ** 1.5
    return x.rolling(w, min_periods=mp).apply(f, raw=True).to_numpy()


def _pop_kurt(x: pd.Series, w: int, mp: int) -> np.ndarray:
    """POPULATION excess g2 — the WRONG one. m4/m2**2 - 3."""
    def f(v: np.ndarray) -> float:
        v = v[~np.isnan(v)]
        n = v.size
        if n < 4:
            return np.nan
        d = v - v.mean()
        m2 = (d ** 2).mean()
        m4 = (d ** 4).mean()
        if m2 <= 0:
            return np.nan
        return m4 / m2 ** 2 - 3.0
    return x.rolling(w, min_periods=mp).apply(f, raw=True).to_numpy()


def _nearest_quantile(x: pd.Series, w: int, mp: int, q: float) -> np.ndarray:
    """NEAREST-RANK quantile — the WRONG one. No interpolation between ranks."""
    def f(v: np.ndarray) -> float:
        v = np.sort(v[~np.isnan(v)])
        n = v.size
        if n == 0:
            return np.nan
        return float(v[min(n - 1, int(np.ceil(q * n)) - 1 if q > 0 else 0)])
    return x.rolling(w, min_periods=mp).apply(f, raw=True).to_numpy()


def evaluate(df: pd.DataFrame, spec: str) -> np.ndarray:
    f = spec.split("|")
    # PROBE_ marks a REPORTED-but-not-gated column; the maths is identical.
    op = f[0][6:] if f[0].startswith("PROBE_") else f[0]
    if op == "diff":
        return df[f[1]].diff().to_numpy()
    if op == "shift":
        return df[f[1]].shift(int(f[2])).to_numpy()
    if op == "pctchange":
        return df[f[1]].pct_change(periods=int(f[2]), fill_method=None).to_numpy()
    if op == "diffn":
        return df[f[1]].diff(int(f[2])).to_numpy()

    if op in ("rollsum", "rollmean", "rollstd", "rollskew", "rollkurt"):
        r = df[f[1]].rolling(int(f[2]), min_periods=int(f[3]))
        return getattr(r, op[4:])().to_numpy()

    if op == "rollcorr":
        r = df[f[1]].rolling(int(f[3]), min_periods=int(f[4]))
        return r.corr(df[f[2]]).to_numpy()

    if op == "rollquantile":
        r = df[f[1]].rolling(int(f[2]), min_periods=int(f[3]))
        return r.quantile(float(f[4])).to_numpy()

    if op == "rollquantiles":
        levels = [float(v) for v in f[4].split(";")]
        r = df[f[1]].rolling(int(f[2]), min_periods=int(f[3]))
        # The k-level form must be IDENTICAL to k separate calls; that is the
        # whole claim being tested, so the golden is the separate call.
        return r.quantile(levels[int(f[5])]).to_numpy()

    if op == "NEG_popskew":
        return _pop_skew(df[f[1]], int(f[2]), int(f[3]))
    if op == "NEG_popkurt":
        return _pop_kurt(df[f[1]], int(f[2]), int(f[3]))
    if op == "NEG_nearestquantile":
        return _nearest_quantile(df[f[1]], int(f[2]), int(f[3]), float(f[4]))

    raise SystemExit(f"unknown spec op {op!r} in {spec!r}")


def _fmt(v: float) -> str:
    return "nan" if np.isnan(v) else repr(float(v)) if False else f"{v:.17g}"


def write_csv(path: Path, names: list[str], cols: list[np.ndarray]) -> None:
    n = len(cols[0])
    with path.open("w") as fh:
        fh.write(",".join(names) + "\n")
        for i in range(n):
            fh.write(",".join(_fmt(c[i]) for c in cols) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True,
                    help="directory to write pdops_input.csv / pdops_golden.csv")
    args = ap.parse_args()

    if pd.__version__ != REQUIRED_PANDAS:
        raise SystemExit(
            f"=== BLOCKED: pandas {pd.__version__}, this golden is defined "
            f"against {REQUIRED_PANDAS} ===\n"
            "  The rolling kernels (roll_var/roll_skew/roll_kurt/roll_quantile)\n"
            "  are version-specific. Emitting a golden from another version\n"
            "  would grade the C++ against different math.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = build_inputs()
    in_names = ["close", "ret", "prp", "vol", "infy", "retlag1",
                "flat", "wobble", "nearflat"]
    write_csv(out / "pdops_input.csv", in_names,
              [df[c].to_numpy() for c in in_names])

    specs = build_specs()
    cols = [evaluate(df, s) for s in specs]
    write_csv(out / "pdops_golden.csv", specs, cols)

    n_nan = sum(int(np.isnan(c).sum()) for c in cols)
    n_cell = sum(c.size for c in cols)
    n_probe = sum(1 for s in specs if s.startswith("PROBE_"))
    n_neg = sum(1 for s in specs if s.startswith("NEG_"))
    print(f"pandas {pd.__version__} / numpy {np.__version__}")
    print(f"rows={len(df)} inputs={len(in_names)} specs={len(specs)} "
          f"(gated={len(specs) - n_probe - n_neg} probe={n_probe} neg={n_neg}) "
          f"cells={n_cell} nan_cells={n_nan} ({100.0 * n_nan / n_cell:.1f}%)")
    print(f"wrote {out / 'pdops_input.csv'}")
    print(f"wrote {out / 'pdops_golden.csv'}")
    print()
    report_regression_coverage(df, specs, cols)
    print()
    repro_diagnostic(df)
    return 0


def report_regression_coverage(df: pd.DataFrame, specs: list[str],
                               cols: list[np.ndarray]) -> None:
    """ASSERT the zero-variance regression spec is not vacuous.

    `rollcorr|*|flat|*` only pins the FP-contraction defect if the golden
    actually contains BOTH outcomes of the 0-denominator division:

        numerator cancelled to exactly 0.0  ->  0/0  ->  NaN
        numerator landed 1 ULP off          ->  x/0  ->  +/-inf

    A golden with only the NaN outcome would still pass against a broken build
    that hard-coded "zero variance -> NaN", and a golden with only the inf
    outcome would pass one that never checked. Both must be present, so this
    counts them and RAISES if either is missing — a reseed or a tweak to the
    block values that quietly retires the coverage fails the generator instead
    of silently weakening the gate.

    Reference for the branch itself: rolling.py corr_func computes
    `numerator/denominator` with no guard, and this is measured, not assumed —
    on `rollcorr|close|vol|14|14` pandas 2.3.3 emits NaN on rows 514/517/518
    and +/-inf on rows 513/515/516/519, all with denominator == 0.0.
    """
    by_spec = dict(zip(specs, cols))
    print("zero-variance regression coverage (constant windows, den == 0):")
    total_nan = total_inf = 0
    for spec in specs:
        if "rollcorr|" not in spec or "flat" not in spec:
            continue
        _, a, b, w, mp = spec.replace("PROBE_", "").split("|")
        x, y = df[a].to_numpy(float), df[b].to_numpy(float)
        X, Y = x + 0 * y, y + 0 * x
        with np.errstate(all="ignore"):
            X = np.where(np.isinf(X), np.nan, X)
            Y = np.where(np.isinf(Y), np.nan, Y)
            sx = pd.Series(X).rolling(int(w), min_periods=int(mp))
            sy = pd.Series(Y).rolling(int(w), min_periods=int(mp))
            den = sx.var(ddof=1).to_numpy() * sy.var(ddof=1).to_numpy()
        out = by_spec[spec]
        zero_den = (den == 0.0) & np.isfinite(den)
        n_nan = int((zero_den & np.isnan(out)).sum())
        n_inf = int((zero_den & np.isinf(out)).sum())
        total_nan += n_nan
        total_inf += n_inf
        print(f"  {spec:34s} zero-den rows={int(zero_den.sum()):4d} "
              f"-> NaN={n_nan:4d}  +/-inf={n_inf:4d}")
    if total_nan == 0 or total_inf == 0:
        raise SystemExit(
            "=== BLOCKED: the zero-variance regression spec is VACUOUS ===\n"
            f"  NaN outcomes={total_nan} inf outcomes={total_inf}; both must be\n"
            "  non-zero or the spec cannot detect FP contraction at the corr\n"
            "  numerator. Fix build_inputs()'s `flat` block values (a dyadic\n"
            "  constant gives only NaN, a non-dyadic one gives mostly inf).")

    # --- the NEAR-constant windows (`nearflat`) ----------------------------
    #
    # More than one distinct value but a relative spread below 1e-6, so pandas'
    # same-value guard does NOT fire and the streaming Welford residue is
    # genuinely computed. At this scale the residue swamps the true variance and
    # pandas' rolling var goes NEGATIVE, which is the input to the OTHER mask
    # predicate this suite pins: Rolling.std is zsqrt (common.py:149-161), so
    # negative -> 0.0, whereas a naive sqrt(var) gives NaN.
    #
    # Assert the negative rows exist. Unlike skew/kurt's `B <= 1e-14` — which
    # is pure cancellation noise here and which pandas cannot even reproduce
    # against itself (hence PROBE_) — only the SIGN of this residue is
    # consulted, and the sign is robust across an FMA at the ssqdm update.
    nf = df["nearflat"]
    w = STATS_WINDOW
    a = nf.to_numpy(float)
    near = np.zeros(len(a), dtype=bool)
    for i in range(w - 1, len(a)):
        win = a[i - w + 1:i + 1]
        if np.isnan(win).any():
            continue
        u = np.unique(win)
        if u.size > 1 and (u.max() - u.min()) / max(abs(u.max()), 1e-300) < 1e-6:
            near[i] = True
    var1 = nf.rolling(w, min_periods=w).var(ddof=1).to_numpy()
    std1 = by_spec[f"PROBE_rollstd|nearflat|{w}|{w}"]
    n_neg = int((near & (var1 < 0)).sum())
    n_zero = int((near & (var1 < 0) & (std1 == 0.0)).sum())
    print(f"  near-constant windows={int(near.sum()):4d} -> pandas var < 0 on "
          f"{n_neg:4d}; zsqrt gives std == 0.0 on {n_zero:4d}")
    if n_neg == 0 or n_zero != n_neg:
        raise SystemExit(
            "=== BLOCKED: the negative-variance / zsqrt regression is VACUOUS ===\n"
            f"  negative-var rows={n_neg}, of which std == 0.0 on {n_zero}.\n"
            "  Every negative-variance row must yield EXACTLY 0.0 (zsqrt), or\n"
            "  the spec cannot distinguish pandas' zsqrt from a naive sqrt.\n"
            "  Retune the eps sweep in build_inputs()'s `nearflat` blocks.")


def repro_diagnostic(df: pd.DataFrame) -> None:
    """pandas' rolling var/skew/kurt are NOT window-local — they stream, and
    skew/kurt additionally pre-centre by a WHOLE-ARRAY statistic. So the SAME
    window computed from a frame that starts earlier is a DIFFERENT number.

    This measures that self-disagreement. It is the achievable floor for any
    port: no implementation can agree with pandas better than pandas agrees
    with itself, and the live core will compute over a ~700-bar panel while
    research computes over years of bars. Printed, never asserted — the
    numbers are what justify which specs are gated and which are PROBE.
    """
    cut = 1300
    print("pandas-vs-pandas panel-start reproducibility (SAME window, frame "
          f"truncated to start at row {cut}):")
    print(f"  {'series':8s} {'w':>4s} {'stat':6s} {'max rel self-disagreement':>26s}")
    for col, w in (("ret", STATS_WINDOW), ("close", STATS_WINDOW),
                   ("prp", VOL_Q_WINDOW), ("close", VOL_Q_WINDOW)):
        s = df[col]
        for stat in ("std", "skew", "kurt"):
            a = getattr(s.rolling(w, min_periods=w), stat)().to_numpy()[cut:]
            tail = s.iloc[cut - w + 1:].reset_index(drop=True)
            b = getattr(tail.rolling(w, min_periods=w), stat)().to_numpy()[w - 1:]
            k = min(len(a), len(b))
            a, b = a[:k], b[:k]
            m = ~np.isnan(a) & ~np.isnan(b)
            with np.errstate(all="ignore"):
                rel = np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-300)
            worst = float(np.nanmax(rel)) if rel.size else 0.0
            print(f"  {col:8s} {w:4d} {stat:6s} {worst:26.3e}")


if __name__ == "__main__":
    sys.exit(main())
