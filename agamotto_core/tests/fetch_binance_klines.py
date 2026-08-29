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

THE BOOT SEAM, AND WHY --repeat-sec EXISTS
------------------------------------------
Because only closed bars are written, this file can NEVER contain the bucket the
strategy attaches to. That bucket is discarded as a partial (its early trades
happened before we connected), so the first bar the core builds is one bucket
later than this file's newest, and the core reports the difference as an
outstanding hole. It is repaired by RE-READING this same file once the bucket
has closed -- so the file has to be refreshed after startup, or the run holds
699 quarantined bars and never becomes warm.

    python fetch_binance_klines.py --symbol BTCUSDT --interval 15m --limit 700 \
        --out <bundle>/config/backfill_BTCUSDT_15m.csv --repeat-sec 60 &

--repeat-sec re-writes the file on an interval so the hole closes on its own
within one bar of opening. The write is ATOMIC (tmp file + os.replace): the
strategy reads this path live, and a half-written CSV would either fail to parse
or, worse, parse as a short run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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


def write_once(symbol: str, interval: str, limit: int, out: str) -> int:
    rows = fetch(symbol, interval, limit)
    if not rows:
        print("no klines returned", file=sys.stderr)
        return 1

    step = INTERVAL_MS[interval]
    now_ms = int(time.time() * 1000)
    closed = [k for k in rows if int(k[0]) + step <= now_ms]
    dropped = len(rows) - len(closed)
    if not closed:
        print("every kline returned was still open", file=sys.stderr)
        return 1

    # Contiguity is the core's own precondition; catching a hole here gives a
    # readable error instead of an opaque ingestBackfill() == false.
    for i in range(1, len(closed)):
        gap = int(closed[i][0]) - int(closed[i - 1][0])
        if gap != step:
            print(f"non-contiguous klines at {closed[i][0]}: gap {gap}ms != {step}ms",
                  file=sys.stderr)
            return 1

    # Atomic: the strategy re-reads this path live to close the boot seam, and a
    # torn read would parse as a SHORT run -- which looks exactly like a stale
    # file and would send the operator chasing the wrong thing.
    tmp = out + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for k in closed:
            w.writerow([int(k[0]), k[1], k[2], k[3], k[4], k[5], k[7], int(k[8]), k[9], k[10]])
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)

    print(f"wrote {len(closed)} closed {interval} bars to {out} "
          f"({dropped} open bar(s) dropped)")
    print(f"range {closed[0][0]} .. {closed[-1][0]}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", required=True, choices=sorted(INTERVAL_MS))
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat-sec", type=int, default=0,
                    help="re-write --out every N seconds instead of exiting. Required "
                         "alongside a live strategy: the bucket it attaches to is not "
                         "in the boot file and the seam only closes when the file is "
                         "refreshed after that bucket has closed.")
    args = ap.parse_args()

    rc = write_once(args.symbol, args.interval, args.limit, args.out)
    if args.repeat_sec <= 0:
        return rc
    if rc != 0:
        # The FIRST write is the one the strategy halts on if it is absent, so a
        # failure there is fatal rather than something to retry behind.
        print("first write failed -- not entering the refresh loop", file=sys.stderr)
        return rc

    while True:
        time.sleep(args.repeat_sec)
        rc = write_once(args.symbol, args.interval, args.limit, args.out)
        if rc != 0:
            # Keep going: the strategy already holds a usable file, the seam is
            # simply not closed yet, and it says so on every bar. Exiting here
            # would remove the only thing that can still close it.
            print("refresh failed; retrying next interval", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
