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
from bar_parity import BAR_SEC, TARGET_SEC, gen_events, load_reference  # noqa: E402


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

    events = gen_events(seed=args.seed)

    bar_mod = load_reference(Path(args.ref_bar))
    bars = ref_bars_df(bar_mod, events)

    feat_mod = load_reference(Path(args.ref_feat))
    fe = feat_mod.MjolnirFeatures(feature_windows=[30, 60, 300, 900],
                                  bar_tf="5s", target_tf="5s")
    ref = fe.compute(bars)

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
