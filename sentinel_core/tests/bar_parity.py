#!/usr/bin/env python3
"""Differential parity test: C++ BarBuilder == Python reference LiveBarBuilder.

Generates an adversarial event stream (bucket gaps, future-stamped events that
must be buffered, out-of-order arrivals, empty bars, multi-bucket jumps), drives
BOTH implementations over the identical stream, and compares every emitted bar
field by field.

This is the gate for the bar builder: bar CONTENT accounted for ~40% of the
measured live-vs-sim divergence, so a builder that merely compiles is worthless.

Usage:
    python tests/bar_parity.py --ref /path/to/knull/live_bar.py --driver ./build/bar_parity_driver
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import subprocess
import sys
from pathlib import Path

BAR_SEC = 5
TARGET_SEC = 30   # exercise the non-trivial cycle_progress / secs_to_boundary path


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("live_bar_ref", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load reference module: {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: @dataclass resolves field annotations via
    # sys.modules[cls.__module__], which fails if the module isn't there yet.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def gen_events(seed: int = 7, n: int = 4000) -> list[str]:
    """Adversarial stream. Deliberately includes the cases that break naive ports."""
    rng = random.Random(seed)
    ev: list[str] = []
    t = 1_700_000_000_000          # epoch ms, bucket-aligned
    px = 30000.0

    for i in range(n):
        # Irregular time steps; occasionally jump several buckets to force the
        # gap-walk path (empty bars carrying last-value fields).
        if rng.random() < 0.04:
            t += rng.randint(2, 5) * BAR_SEC * 1000     # multi-bucket gap
        else:
            t += rng.randint(50, 2500)

        px = max(1.0, px + rng.gauss(0, 5))
        qty = round(abs(rng.gauss(1.0, 0.5)) + 0.001, 8)
        ev.append(f"T,{t},{px:.8f},{qty:.8f},{int(rng.random() < 0.5)},1")

        # Book ticker, sometimes stamped in a FUTURE bucket -> must be buffered,
        # not applied to the still-open bar (this is the lookahead trap).
        if rng.random() < 0.6:
            bts = t + (BAR_SEC * 1000 if rng.random() < 0.25 else 0)
            ev.append(f"B,{bts},{px-1:.8f},{qty:.8f},{px+1:.8f},{qty:.8f}")

        if rng.random() < 0.4:
            dts = t + (BAR_SEC * 1000 * rng.randint(1, 2) if rng.random() < 0.2 else 0)
            bp = [f"{px - 1 - j:.8f}" for j in range(5)]
            bq = [f"{(j + 1) * 0.5:.8f}" for j in range(5)]
            ap = [f"{px + 1 + j:.8f}" for j in range(5)]
            aq = [f"{(j + 1) * 0.7:.8f}" for j in range(5)]
            ev.append("D,%d,%s,%s,%s,%s" % (dts, ",".join(bp), ",".join(bq),
                                            ",".join(ap), ",".join(aq)))

        if rng.random() < 0.2:
            mts = t + (BAR_SEC * 1000 if rng.random() < 0.2 else 0)
            ev.append(f"M,{mts},{px:.8f},{px-0.5:.8f},0.0001,0.0001")

        if rng.random() < 0.08:
            lts = t + (BAR_SEC * 1000 if rng.random() < 0.3 else 0)
            ev.append(f"L,{lts},{int(rng.random() < 0.5)},{rng.uniform(100, 9999):.8f}")

        # OI: the 2026-07-23 fix — a future-stamped snapshot must be buffered.
        if rng.random() < 0.1:
            ots = t + (BAR_SEC * 1000 if rng.random() < 0.5 else 0)
            ev.append(f"O,{ots},{rng.uniform(1e6, 2e6):.8f}")

    return ev


def run_reference(mod, events: list[str]) -> list[dict]:
    bldr = mod.LiveBarBuilder(bar_sec=BAR_SEC, target_sec=TARGET_SEC)
    sym = "S"
    bars: list[dict] = []
    for line in events:
        f = line.split(",")
        kind, ts = f[0], int(f[1])
        if kind == "T":
            b = bldr.on_trade(sym, float(f[2]), float(f[3]), bool(int(f[4])), ts, int(f[5]))
            if b is not None:
                # The reference emit() carries no bucket stamp, and by the time
                # it returns the builder has already reset() onto the INCOMING
                # bucket — so reading current_bucket here reports one bar too
                # late. The emitted bar is always the bucket immediately before
                # the incoming one (true both for the contiguous case and after
                # a gap walk, which stops at incoming - width).
                b = dict(b)
                b["bucket_ms"] = (ts // (BAR_SEC * 1000)) * (BAR_SEC * 1000) - BAR_SEC * 1000
                bars.append(b)
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
    return bars


def run_cpp(driver: Path, events: list[str]) -> list[dict]:
    proc = subprocess.run([str(driver), str(BAR_SEC), str(TARGET_SEC)],
                          input="\n".join(events), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"C++ driver failed rc={proc.returncode}\n{proc.stderr}")
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln]
    header = lines[0].split(",")
    out = []
    for ln in lines[1:]:
        vals = ln.split(",")
        out.append({k: float(v) for k, v in zip(header, vals)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="path to the reference live_bar.py")
    ap.add_argument("--driver", required=True, help="path to the built bar_parity_driver")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=7, help="event-stream seed")
    ap.add_argument("--seeds", type=int, default=1,
                    help="run N consecutive seeds starting at --seed (all must pass)")
    args = ap.parse_args()

    if args.seeds > 1:
        rc = 0
        for s in range(args.seed, args.seed + args.seeds):
            print(f"--- seed {s} ---")
            rc |= _run_one(args, s)
        return rc
    return _run_one(args, args.seed)


def _run_one(args, seed: int) -> int:
    events = gen_events(seed=seed)
    ref_bars = run_reference(load_reference(Path(args.ref)), events)
    cpp_bars = run_cpp(Path(args.driver), events)

    print(f"[bar_parity] events={len(events)} ref_bars={len(ref_bars)} cpp_bars={len(cpp_bars)}")
    if len(ref_bars) != len(cpp_bars):
        print(f"FAIL: bar COUNT differs ({len(ref_bars)} vs {len(cpp_bars)})")
        return 1

    # Compare only what the reference actually emits; bucket_ms is reconstructed
    # above and checked too.
    fields = sorted(set(ref_bars[0].keys()) & set(cpp_bars[0].keys()))
    missing = sorted(set(ref_bars[0].keys()) - set(cpp_bars[0].keys()))
    if missing:
        print(f"FAIL: C++ is missing reference fields: {missing}")
        return 1

    bad = 0
    for i, (r, c) in enumerate(zip(ref_bars, cpp_bars)):
        for k in fields:
            rv, cv = float(r[k]), float(c[k])
            denom = max(1.0, abs(rv))
            if abs(rv - cv) / denom > args.tol:
                if bad < 15:
                    print(f"  MISMATCH bar={i} field={k} ref={rv!r} cpp={cv!r}")
                bad += 1

    print(f"[bar_parity] fields compared: {len(fields)}")
    if bad:
        print(f"=== FAIL: {bad} field mismatches ===")
        return 1
    print(f"=== PASS: {len(ref_bars)} bars x {len(fields)} fields identical (tol={args.tol}) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
