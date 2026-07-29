#!/usr/bin/env python3
"""M1 EXIT GATE — variant-A parity against live y_pred.

Extracts every LIVE FIRING from decisions_<date>.csv, has the C++ chain re-score
the bot's own dumped bars at those exact points, and compares.

The reference established the target: scoring frozen weights over the bot's own
bars reproduces live y_pred EXACTLY (corr 1.000, median delta 0.000 bps, 100% of
live firings). Anything less means the C++ chain diverges somewhere the
per-module parity tests did not reach.

Usage:
    python m1_gate.py --bars bars_<d>.csv --decisions decisions_<d>.csv \
        --driver ./m1_gate_driver --weights <export_root> [--limit N]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N firings")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    dec = pd.read_csv(args.decisions)
    fired = dec[dec["y_pred"].notna()].copy()
    if fired.empty:
        print("=== FAIL: no live firings in the decisions file — nothing to verify ===")
        return 1

    fired["bar_ts_ns"] = (
        pd.to_datetime(fired["bar_ts"], utc=True, format="mixed").astype("int64"))
    # regime column carries e.g. "<name>_short"; the exported weight dir uses
    # exactly that name.
    fired["regime_dir"] = fired["regime"].astype(str)
    if args.limit:
        fired = fired.head(args.limit)

    print(f"[m1_gate] live firings: {len(fired)}  "
          f"regimes: {sorted(fired['regime_dir'].unique())}")

    # Written beside the bars file, not into a temp dir: the driver runs inside
    # a container and only the data directory is mounted.
    tasks = str(Path(args.bars).parent / "_m1_tasks.csv")
    with open(tasks, "w") as th:
        th.write("symbol,bar_ts_ns,regime_dir\n")
        for _, r in fired.iterrows():
            th.write(f"{r['symbol']},{r['bar_ts_ns']},{r['regime_dir']}\n")

    proc = subprocess.run([args.driver, args.bars, tasks, args.weights],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        raise SystemExit(f"driver failed rc={proc.returncode}")
    sys.stderr.write(proc.stderr[-800:])

    lines = [ln for ln in proc.stdout.strip().splitlines() if ln]
    got = pd.DataFrame([ln.split(",") for ln in lines[1:]],
                       columns=lines[0].split(","))
    if got.empty:
        print("=== FAIL: driver scored 0 rows ===")
        return 1
    got["bar_ts_ns"] = got["bar_ts_ns"].astype("int64")
    got["y_pred_cpp"] = got["y_pred_cpp"].astype(float)

    m = fired.merge(got, on=["symbol", "bar_ts_ns"], how="inner")
    print(f"[m1_gate] matched {len(m)} / {len(fired)} firings")
    if m.empty:
        print("=== FAIL: no firings matched the dumped bars ===")
        return 1

    ref = m["y_pred"].to_numpy(float)
    cpp = m["y_pred_cpp"].to_numpy(float)
    d = np.abs(ref - cpp)
    corr = float(np.corrcoef(ref, cpp)[0, 1]) if len(ref) > 1 else float("nan")
    within = int((d <= args.tol).sum())

    print(f"[m1_gate] corr={corr:.6f}  max_abs_diff={d.max():.3e}  "
          f"median_abs_diff={np.median(d):.3e}")
    print(f"[m1_gate] within tol({args.tol}): {within}/{len(m)} "
          f"({100.0*within/len(m):.1f}%)")
    # bps framing, matching how the reference reported it
    print(f"[m1_gate] median delta: {np.median(d)*1e4:.4f} bps")

    # A partial match is a FAIL: unmatched firings are unverified, not absent.
    if len(m) < len(fired):
        print(f"=== FAIL: {len(fired) - len(m)} firings did not match a dumped bar "
              f"(unverified, not passing) ===")
        return 1
    if within != len(m):
        print("=== FAIL: C++ y_pred diverges from live ===")
        worst = m.reindex(np.argsort(-d)).head(5)
        for _, r in worst.iterrows():
            print(f"   {r['symbol']} {r['bar_ts']}: live={r['y_pred']:.10g} "
                  f"cpp={r['y_pred_cpp']:.10g}")
        return 1

    print(f"=== PASS: all {len(m)} live firings reproduced within {args.tol} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
