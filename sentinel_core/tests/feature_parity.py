#!/usr/bin/env python3
"""Differential parity: C++ FeatureEngine == reference MjolnirFeatures.compute().

Drives the SAME synthetic event stream through both stacks (C++: bars+features
in one driver; Python: the already-parity-verified reference bar builder, then
the reference feature engine) and compares every shared column, cell by cell.

Columns the C++ side has not implemented yet (the TA-Lib block) are reported
explicitly as MISSING rather than quietly skipped — a parity harness that hides
unimplemented columns reports a green that means nothing.

Usage:
    python tests/feature_parity.py --ref-bar <live_bar.py> --ref-feat <features.py> \
        --driver ./feature_parity_driver
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bar_parity import (  # noqa: E402
    BAR_SEC, TARGET_SEC, gen_events, load_reference, load_reference_pkg)

# ----------------------------------------------------------------------------
# Reference-environment gate — the TA-Lib asymmetry
# ----------------------------------------------------------------------------
# The C++ side links libta-lib UNCONDITIONALLY: CMakeLists.txt raises
# FATAL_ERROR when find_library misses it, and Dockerfile.build asserts the
# pinned version by interrogating the built library. The REFERENCE has no such
# gate — features.py wraps its whole TA-Lib block in `except ImportError` and
# drops to `_numpy_indicators`, which stubs adx/dx/cci/willr/stoch/sar/obv/ad/
# mfi/plus_di/minus_di/cmo to NaN outright, hand-rolls a non-Wilder RSI with
# min_periods=1, and emits a MACD with no warmup NaN.
#
# So a reference without TA-Lib compares the C++ against DIFFERENT MATH and
# reports failures that are pure environment. That is exactly what happened on
# dev105 (2026-08-04): five regimes "failed" — rsi_oversold, rsi_overbought,
# macd_bullish, macd_bearish, adx_trend — and adx_trend fired 0 times on the
# reference for the dull reason that `NaN > 25` is False on every bar. The C++
# was correct throughout; installing TA-Lib 0.6.4 made all 30 masks identical
# with no source change.
#
# Refuse to run instead of grading against a degraded reference. Such a harness
# is worse than none: it manufactures failures that send the next person to
# debug correct C++, and it would just as happily bless a WRONG C++ that
# matched the numpy stubs.

# Set to NaN by features.py `_numpy_indicators`; real numbers on the TA-Lib
# path. If every one of these is entirely NaN, the reference took the fallback.
_TALIB_ONLY_COLS = ("stoch_k", "stoch_d", "cci", "adx", "dx", "plus_di",
                    "minus_di", "willr", "cmo", "sar", "obv", "ad", "mfi")


def _pinned_talib_version() -> str:
    """The version the C++ links, read from its single source of truth."""
    dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile.build"
    m = re.search(r"^ARG\s+TALIB_VERSION=(\S+)", dockerfile.read_text(), re.M)
    if not m:
        raise SystemExit(
            f"cannot read ARG TALIB_VERSION from {dockerfile} — refusing to "
            "guess the pin the C++ was built against")
    return m.group(1)


def require_reference_talib() -> str:
    """Fail loudly unless the reference will take the TA-Lib path, at the pin.

    Returns the underlying TA-Lib C library version actually in use.
    """
    pinned = _pinned_talib_version()
    try:
        import talib  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "=== BLOCKED: the reference has no TA-Lib — this is NOT a parity "
            f"result ===\n  ({type(exc).__name__}: {exc})\n"
            "  features.py would silently fall back to `_numpy_indicators`, "
            "which stubs\n"
            "  adx/dx/cci/willr/stoch/sar/obv/ad/mfi to NaN and hand-rolls "
            "RSI/MACD, so every\n"
            "  TA-Lib-derived comparison would grade the C++ against different "
            "math.\n"
            f"  Production runs TA-Lib {pinned}; install the SAME version, e.g.:\n"
            "    TA_INCLUDE_PATH=$HOME/.local/include "
            "TA_LIBRARY_PATH=$HOME/.local/lib \\\n"
            f"      python3.11 -m pip install --user 'TA-Lib=={pinned}'\n"
            "  (the C library itself can be copied out of the "
            "mjolnir-core-build image,\n"
            "   which pins it; then export "
            "LD_LIBRARY_PATH=$HOME/.local/lib)") from exc

    # The WRAPPER version is not the thing that computes indicators — the C
    # library is. talib.__ta_version__ reports the library, so ask that.
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
            "is a silent\n"
            "  parity break rather than a rounding difference. Align the "
            "reference to the pin.")
    return got


def assert_reference_used_talib(panel, where: str) -> None:
    """Defence in depth: prove the panel actually came from the TA-Lib path.

    ``require_reference_talib`` checks the precondition; this checks the
    OUTCOME, so a future refactor that reroutes features.py past TA-Lib for
    some other reason cannot quietly reintroduce the stub panel.
    """
    present = [c for c in _TALIB_ONLY_COLS if c in panel.columns]
    if not present:
        raise SystemExit(
            f"=== BLOCKED: {where} has none of the TA-Lib columns "
            f"{list(_TALIB_ONLY_COLS)} ===\n"
            "  Nothing TA-Lib-derived could be verified.")
    if all(panel[c].isna().all() for c in present):
        raise SystemExit(
            f"=== BLOCKED: every TA-Lib column in {where} is entirely NaN "
            "— the reference took `_numpy_indicators` ===\n"
            "  TA-Lib imports, yet the panel carries its stub signature, so "
            "the reference is\n"
            "  NOT the math the C++ implements. Do not read the comparison "
            "below as parity.")


def ref_bars_df(mod, events):
    bldr = mod.LiveBarBuilder(bar_sec=BAR_SEC, target_sec=TARGET_SEC)
    sym, rows, idx = "S", [], []
    for line in events:
        f = line.split(",")
        kind, ts = f[0], int(f[1])
        if kind == "T":
            b = bldr.on_trade(sym, float(f[2]), float(f[3]), bool(int(f[4])), ts, int(f[5]))
            if b is not None:
                bucket = (ts // (BAR_SEC * 1000)) * (BAR_SEC * 1000) - BAR_SEC * 1000
                rows.append(b)
                idx.append(pd.Timestamp(bucket, unit="ms", tz="UTC"))
        elif kind == "B":
            bldr.on_book_ticker(sym, float(f[2]), float(f[3]), float(f[4]), float(f[5]), ts)
        elif kind == "D":
            bids = [[float(f[2 + i]), float(f[7 + i])] for i in range(5)]
            asks = [[float(f[12 + i]), float(f[17 + i])] for i in range(5)]
            bldr.on_depth(sym, bids, asks, ts)
        elif kind == "M":
            bldr.on_mark_price(sym, float(f[2]), float(f[3]), float(f[4]), float(f[5]), ts)
        elif kind == "L":
            bldr.on_liquidation(sym, "BUY" if int(f[2]) else "SELL", float(f[3]), ts)
        elif kind == "O":
            bldr.set_open_interest(sym, float(f[2]), ts)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-bar", required=True)
    ap.add_argument("--ref-feat", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ta_ver = require_reference_talib()
    print(f"[feature_parity] reference TA-Lib {ta_ver}")

    events = gen_events(seed=args.seed)

    bar_mod = load_reference(Path(args.ref_bar))
    bars = ref_bars_df(bar_mod, events)

    # Package member, not standalone file: features.py imports
    # `.features_scalefree` relatively (dc 3fe8e57), which only resolves with a
    # parent package. The src root is discovered from the file itself.
    feat_mod = load_reference_pkg(Path(args.ref_feat))
    fe = feat_mod.MjolnirFeatures(feature_windows=[30, 60, 300, 900],
                                  bar_tf="5s", target_tf="5s",
                                  ta_price_source="close")
    ref = fe.compute(bars)
    assert_reference_used_talib(ref, "reference panel")

    proc = subprocess.run([args.driver, str(BAR_SEC), str(TARGET_SEC)],
                          input="\n".join(events), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"driver failed: {proc.stderr}")
    lines = proc.stdout.strip().splitlines()
    cpp = pd.DataFrame([[float(x) for x in ln.split(",")] for ln in lines[1:]],
                       columns=lines[0].split(","))

    # The C++ engine emits CODED column names (no real name compiles into the
    # .so). Encode the reference's columns the same way before comparing —
    # otherwise only the passthrough columns line up and the run reports a
    # meaningless PASS over a fraction of the panel.
    mp = json.loads((Path(__file__).resolve().parents[2] / "obfuscation" / "map.json")
                    .read_text())["features"]

    def enc(col: str) -> str:
        m2 = re.match(r"^(.*)_roll(\d+)_(mean|std)$", col)
        if m2 and m2.group(1) in mp:
            return f"{mp[m2.group(1)]}_roll{m2.group(2)}_{m2.group(3)}"
        return mp.get(col, col)

    ref = ref.rename(columns={c: enc(c) for c in ref.columns})

    print(f"[feat_parity] rows ref={len(ref)} cpp={len(cpp)}  "
          f"cols ref={len(ref.columns)} cpp={len(cpp.columns)}")
    if len(ref) != len(cpp):
        print("FAIL: row count differs")
        return 1

    shared = sorted(set(ref.columns) & set(cpp.columns))
    missing = sorted(set(ref.columns) - set(cpp.columns))
    extra = sorted(set(cpp.columns) - set(ref.columns))

    bad, bad_cols = 0, []
    for c in shared:
        r = pd.to_numeric(ref[c], errors="coerce").to_numpy(float)
        v = cpp[c].to_numpy(float)
        r = np.where(np.isfinite(r), r, 0.0)
        v = np.where(np.isfinite(v), v, 0.0)
        d = np.abs(r - v) / np.maximum(1.0, np.abs(r))
        k = int((d > args.tol).sum())
        if k:
            bad += k
            bad_cols.append((c, k, float(d.max())))

    print(f"[feat_parity] compared {len(shared)} shared columns")
    # Coverage guard: the codes refactor once dropped this to 55 while still
    # printing PASS. A comparison that silently shrinks is not a pass.
    MIN_SHARED = 155
    if len(shared) < MIN_SHARED:
        print(f"=== FAIL: only {len(shared)} columns compared (expected >= {MIN_SHARED}) — "
              f"the rest are UNVERIFIED ===")
        return 1
    if missing:
        print(f"[feat_parity] NOT YET IMPLEMENTED in C++ ({len(missing)}): "
              f"{', '.join(missing[:30])}{' ...' if len(missing) > 30 else ''}")
    if extra:
        print(f"[feat_parity] extra in C++ ({len(extra)}): {', '.join(extra[:15])}")
    if bad_cols:
        print(f"=== FAIL: {len(bad_cols)} columns differ ({bad} cells) ===")
        for c, k, mx in bad_cols[:20]:
            print(f"   {c}: {k} cells, max rel diff {mx:.3e}")
        return 1

    print(f"=== PASS: all {len(shared)} shared columns identical (tol={args.tol}) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
