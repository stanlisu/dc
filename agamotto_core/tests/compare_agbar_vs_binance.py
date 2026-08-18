#!/usr/bin/env python3
"""Diff [AGBAR] lines from a live AgamottoStrategy run against Binance's own klines.

This is the Phase 1 gate: the model trained on Binance klines, so tick-built
bars are only usable to the extent they reproduce them.

    python compare_agbar_vs_binance.py --log <strategy.log> --symbol BTCUSDT --interval 1m

Compares the nine kline columns. Tolerance is RELATIVE, not exact: both sides
accumulate the same trades into a double in a different order, so identical
values still differ in the last ULP. Exact equality is the wrong test for the
summed columns.

The first and last built bars are dropped: the first may be adjacent to the
partial bucket we joined on, and the last may still be open on Binance's side.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

AGBAR = re.compile(
    r"\[AGBAR\]\s+open_ms=(?P<open_ms>\d+)\s+o=(?P<o>[-\d.eE+]+)\s+h=(?P<h>[-\d.eE+]+)\s+"
    r"l=(?P<l>[-\d.eE+]+)\s+c=(?P<c>[-\d.eE+]+)\s+v=(?P<v>[-\d.eE+]+)\s+"
    r"qv=(?P<qv>[-\d.eE+]+)\s+n=(?P<n>\d+)\s+tbb=(?P<tbb>[-\d.eE+]+)\s+"
    r"tbq=(?P<tbq>[-\d.eE+]+)\s+aggr=(?P<aggr>\w+)\s+unclass=(?P<unclass>\d+)\s+"
    r"backfill=(?P<backfill>\d+)\s+recv_to_bar_us=(?P<lat>[-\d.eE+]+)"
)

COLS = ["open", "high", "low", "close", "volume", "quote_volume",
        "number_of_trades", "taker_buy_base", "taker_buy_quote"]


def parse_log(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = AGBAR.search(line)
            if not m:
                continue
            g = m.groupdict()
            out.append({
                "open_ms": int(g["open_ms"]),
                "vals": [float(g["o"]), float(g["h"]), float(g["l"]), float(g["c"]),
                         float(g["v"]), float(g["qv"]), float(g["n"]),
                         float(g["tbb"]), float(g["tbq"])],
                "aggr": g["aggr"],
                "unclass": int(g["unclass"]),
                "backfill": g["backfill"] == "1",
                "lat_us": float(g["lat"]),
            })
    return out


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> dict[int, list[float]]:
    url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}"
           f"&interval={interval}&startTime={start_ms}&endTime={end_ms}&limit=1500")
    with urllib.request.urlopen(url, timeout=20) as r:
        rows = json.loads(r.read().decode())
    # [open_time, open, high, low, close, volume, close_time, quote_volume,
    #  n_trades, taker_buy_base, taker_buy_quote, ignore]
    return {int(k[0]): [float(k[1]), float(k[2]), float(k[3]), float(k[4]),
                        float(k[5]), float(k[7]), float(k[8]),
                        float(k[9]), float(k[10])] for k in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", required=True, help="must match bar_sec (60 -> 1m, 900 -> 15m)")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    bars = [b for b in parse_log(args.log) if not b["backfill"]]
    if len(bars) < 3:
        print(f"only {len(bars)} live bar(s) in {args.log} — need >= 3 "
              f"(first and last are dropped as boundary-affected)")
        return 2

    bars = bars[1:-1]
    lo, hi = bars[0]["open_ms"], bars[-1]["open_ms"]
    ref = fetch_klines(args.symbol, args.interval, lo, hi)

    match = [0] * 9
    worst = [0.0] * 9
    worst_at = [0] * 9
    compared = 0
    missing = 0

    for b in bars:
        r = ref.get(b["open_ms"])
        if r is None:
            missing += 1
            continue
        compared += 1
        for i in range(9):
            got, exp = b["vals"][i], r[i]
            denom = max(1.0, abs(exp))
            rel = abs(got - exp) / denom
            if rel <= args.tol:
                match[i] += 1
            if rel > worst[i]:
                worst[i], worst_at[i] = rel, b["open_ms"]

    print(f"\ntick-built bars vs Binance {args.interval} klines for {args.symbol}")
    print(f"compared {compared} bars ({missing} had no Binance counterpart)\n")
    print(f"{'column':<18}{'match':>12}{'worst rel err':>16}")
    print("-" * 46)
    for i, c in enumerate(COLS):
        flag = "" if match[i] == compared else "   <-- MISMATCH"
        print(f"{c:<18}{match[i]:>6}/{compared:<5}{worst[i]:>16.3e}{flag}")

    aggr = {b["aggr"] for b in bars}
    unclass = sum(b["unclass"] for b in bars)
    lats = [b["lat_us"] for b in bars if b["lat_us"] >= 0]
    print(f"\naggressor_source: {sorted(aggr)}  (quote_rule => taker_buy_* are APPROXIMATIONS)")
    print(f"trades no rule could side: {unclass}")
    if lats:
        lats.sort()
        p = lambda q: lats[min(int(q * (len(lats) - 1) + 0.5), len(lats) - 1)]
        print(f"recv->bar us: n={len(lats)} min={lats[0]:.1f} p50={p(0.5):.1f} "
              f"p99={p(0.99):.1f} max={lats[-1]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
