#!/usr/bin/env python3
"""GATE: every tick-built bar must equal the venue's own kline, exactly.

RED-FIRST. This gate is written to FAIL against the current build. Measured
2026-08-21 on a QUIET bar (168 msg/s, 3.05 s of ring headroom): AAVE was
-12.21 pct, 1000PEPE -11.34 pct, AVAX -2.95 pct, with the absent trades 3-23x
average size. It is the executable form of "sentinel's bars must match what
knull reads".

WHAT IT IS NOT. Not a test of the feature math -- that is graded exactly by
feature_parity.py and is not in question. This grades the INPUT: whether the bar
handed to the panel is the bar the venue published.

WHY A GATE RATHER THAN A DIAGNOSTIC. The trade-id detector cannot answer this:
Binance burns ids without publishing trades, so gaps appear on nearly every bar
and mean nothing on their own -- 22 of 28 symbols showed hundreds of "missing"
ids with a trade COUNT matching exactly. Only `n_trades` and volume against the
venue's own bar separate real loss from phantom ids.

TOLERANCE. n_trades and OHLC must be EXACT. Summed float columns
(quote_volume, taker_buy_quote) are compared at 1e-9 relative, because both
sides accumulate doubles in a different order -- that is representation, not
loss. `volume` must be exact: it is summed from the same decimal quantities.

Usage:
  ./bar_fidelity_gate.py --logs '<glob of shard logs>' [--bars 1] [--symbols S,S]
  exit 0 = every compared bar matched; exit 1 = at least one lost data
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import urllib.request

FAPI = "https://fapi.binance.com/fapi/v1/klines"
EXACT = [("o", 1, "open"), ("h", 2, "high"), ("l", 3, "low"), ("c", 4, "close"),
         ("v", 5, "volume"), ("n", 8, "n_trades"), ("tbb", 9, "taker_buy_base")]
NEAR = [("qv", 7, "quote_volume"), ("tbq", 10, "taker_buy_quote")]
REL_TOL = 1e-9


def parse_bars(pattern: str) -> list[dict]:
    """Latest record per (file, bucket), preferring [AGBARR] over [AGBAR].

    [AGBAR] is written when a bar is BUILT, from the tick stream, and still
    carries whatever the feed failed to deliver. [AGBARR] is written when that
    bar is CORRECTED against the venue's own kline. Grading the build-time line
    would leave this gate red no matter how well the correction works -- it
    would be measuring the wrong record.
    """
    latest: dict[tuple[str, int], dict] = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f, errors="replace"):
            tag = None
            if "[AGBARR]" in line:
                tag = "[AGBARR]"
            elif "[AGBAR]" in line and "[AGBARQ]" not in line:
                tag = "[AGBAR]"
            if tag is None:
                continue
            d = dict(re.findall(r"(\w+)=([-\d.]+)", line.split(tag)[1]))
            if "open_ms" not in d:
                continue
            d["_corrected"] = (tag == "[AGBARR]")
            key = (f, int(d["open_ms"]))
            prev = latest.get(key)
            # a corrected record always supersedes a built one for the same
            # bucket in the same process
            if prev is None or (d["_corrected"] and not prev.get("_corrected")):
                latest[key] = d
    return list(latest.values())


def kline(symbol: str, open_ms: int) -> list | None:
    u = f"{FAPI}?symbol={symbol}&interval=15m&startTime={open_ms}&limit=1"
    with urllib.request.urlopen(u, timeout=30) as r:
        k = json.load(r)
    return k[0] if k else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--setting",
                    default="/home/stan/sandbox/marvel/gauntlet/"
                            "pred_agamotto.base.15m_1/setting.dryrun_baseline.json")
    ap.add_argument("--bars", type=int, default=1,
                    help="how many most-recent bar timestamps to grade")
    args = ap.parse_args()

    bars = parse_bars(args.logs)
    if not bars:
        print(f"NO BARS matched {args.logs} -- the gate cannot pass vacuously",
              file=sys.stderr)
        return 1
    stamps = sorted({int(b["open_ms"]) for b in bars})[-args.bars:]

    cfg = json.load(open(args.setting))
    syms = [re.match(r"^BINANCE_PERP_(.+)_USDT$", s).group(1) + "USDT"
            for s in cfg["SYMBOLS"]]

    graded = clean = 0
    failures: list[str] = []
    for ts in stamps:
        here = [b for b in bars if int(b["open_ms"]) == ts]
        for vs in syms:
            k = kline(vs, ts)
            if k is None:
                continue
            # match the sentinel bar to its symbol by open+close, which are
            # exact on both sides even when volume is short
            cand = [b for b in here
                    if float(b["o"]) == float(k[1]) and float(b["c"]) == float(k[4])]
            if not cand:
                continue          # this symbol produced no bar for this stamp
            b = cand[0]
            graded += 1
            bad = []
            for key, idx, name in EXACT:
                if float(b[key]) != float(k[idx]):
                    bad.append(f"{name} {float(b[key])!r} != {float(k[idx])!r}")
            for key, idx, name in NEAR:
                a_, e_ = float(b[key]), float(k[idx])
                if abs(a_ - e_) > REL_TOL * max(1.0, abs(e_)):
                    bad.append(f"{name} {a_!r} != {e_!r}")
            if bad:
                vol_pct = (float(b["v"]) - float(k[5])) / float(k[5]) * 100.0
                failures.append(f"  {vs:<13} bar {ts}  volume {vol_pct:+.2f}%  "
                                + "; ".join(bad[:2]))
            else:
                clean += 1

    corrected = sum(1 for b in bars if b.get("_corrected"))
    print(f"[bar_fidelity] graded {graded} bar(s) over {len(stamps)} stamp(s): "
          f"{clean} exact, {len(failures)} LOST DATA "
          f"({corrected} record(s) were corrected bars)")
    for f in failures:
        print(f)
    if graded == 0:
        print("  graded NOTHING -- refusing to report a pass", file=sys.stderr)
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
