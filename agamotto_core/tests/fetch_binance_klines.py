#!/usr/bin/env python3
"""Fetch closed Binance futures klines to the CSV the core ingests as backfill.

Agamotto needs 700 bars of history before its rolling windows are valid
(VOL_Q_WINDOW=700, min_periods=700 fails closed). The C++ core does no network
I/O by design — same as the mjolnir core, whose weights are likewise produced by
an external tool — so warmup arrives through this file.

    python fetch_binance_klines.py --symbol BTCUSDT --interval 15m --limit 700 \
        --out /home/stan/agamotto_test/config/backfill_BTCUSDT_15m.csv

Only CLOSED bars are written: the most recent kline is still open and would be
ingested as a complete bar that then disagrees with itself once it closes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request

INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
               "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

HEADER = ["open_ms", "open", "high", "low", "close", "volume",
          "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote"]


def fetch(symbol: str, interval: str, limit: int) -> list[list]:
    out: list[list] = []
    end = int(time.time() * 1000)
    step = INTERVAL_MS[interval]
    # Binance caps a page at 1500; page backwards until we have `limit` closed bars.
    while len(out) < limit:
        want = min(1500, limit - len(out))
        start = end - want * step
        url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}"
               f"&interval={interval}&startTime={start}&endTime={end}&limit={want}")
        with urllib.request.urlopen(url, timeout=20) as r:
            page = json.loads(r.read().decode())
        if not page:
            break
        out = page + out
        end = int(page[0][0]) - 1
    return out[-limit:] if len(out) > limit else out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", required=True, choices=sorted(INTERVAL_MS))
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = fetch(args.symbol, args.interval, args.limit)
    if not rows:
        print("no klines returned", file=sys.stderr)
        return 1

    step = INTERVAL_MS[args.interval]
    now_ms = int(time.time() * 1000)
    closed = [k for k in rows if int(k[0]) + step <= now_ms]
    dropped = len(rows) - len(closed)

    # Contiguity is the core's own precondition; catching a hole here gives a
    # readable error instead of an opaque ingestBackfill() == false.
    for i in range(1, len(closed)):
        gap = int(closed[i][0]) - int(closed[i - 1][0])
        if gap != step:
            print(f"non-contiguous klines at {closed[i][0]}: gap {gap}ms != {step}ms",
                  file=sys.stderr)
            return 1

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for k in closed:
            w.writerow([int(k[0]), k[1], k[2], k[3], k[4], k[5], k[7], int(k[8]), k[9], k[10]])

    print(f"wrote {len(closed)} closed {args.interval} bars to {args.out} "
          f"({dropped} open bar(s) dropped)")
    print(f"range {closed[0][0]} .. {closed[-1][0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
