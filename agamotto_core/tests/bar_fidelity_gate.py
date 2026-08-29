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

IT GRADED 1 SYMBOL IN 14 UNTIL 2026-08-28, AND SAID SO IN A WAY NOBODY READ.
Two independent defects, both of which SHRANK the sample silently:

  1. `parse_bars` keyed its dedupe on (FILE, bucket). A shard log multiplexes
     14 symbols, so all 14 records for one bucket collided on one key and 13
     were thrown away. 2 shard logs x 4 buckets = 8 records survived out of
     112, and the gate reported "graded 7" as though that were the fleet.
  2. Symbols were matched to venue klines by (open, close) price, because
     [AGBAR] carried no product_id when this was written. That is not merely
     imprecise, it is BIASED TOWARD PASSING: a bar whose FIRST or LAST trade
     was among the ones the ring ate has a different open or close, matches
     nothing, and is skipped -- so the bars most likely to have lost data were
     the ones systematically excluded from the grade.

[AGBAR] carries product_id now (sentinel 170beb7), so both go away: records key
on (product_id, bucket) and the symbol comes from the contracts file. A symbol
that built NO bar for a bucket is reported as MISSING rather than skipped -- the
gate refuses to be quiet about a hole again.

BUILT vs CORRECTED. [AGBAR] is the bar as BUILT from ticks, and it is the bar
the model actually scored: reconcileFromBackfill runs AFTER scoring, on purpose
("the decision for a bar is taken on what was known when it closed"). [AGBARR]
is that bar after correction against the venue. Both are graded and reported
separately, because they answer different questions -- "did the correction
work" and "was the input to the decision right" -- and only the second one
prices a trade.

TOLERANCE. n_trades and OHLC are EXACT: n_trades is an integer and OHLC are
single observed prices, so neither accumulates. ALL FOUR summed columns --
volume, quote_volume, taker_buy_base, taker_buy_quote -- are compared at 1e-9
relative.

`volume` and `taker_buy_base` used to be in the exact list, on the reasoning
that they sum the same decimal quantities. They do, in a different ORDER, and
doubles are not associative: measured 2026-08-28, XRPUSDT read
volume 7185239.59999998 against the venue's 7185239.6 -- a -0.00 pct "failure"
on a bar that had every trade. That is representation. It cost nothing here
because the gate was only grading 7 bars, but at full fleet coverage it would
drown the real losses in noise. Real loss is SEVEN ORDERS OF MAGNITUDE larger
(1000PEPE -17.12 pct on the same run), so a 1e-9 tolerance cannot hide one.

Usage:
  ./bar_fidelity_gate.py --logs '<glob of shard logs>' [--bars 1]
        [--contracts /opt/data/Binance/contract_futures.csv] [--grade built|corrected]
  exit 0 = every compared bar matched; exit 1 = at least one lost data
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
import urllib.request

FAPI = "https://fapi.binance.com/fapi/v1/klines"
EXACT = [("o", 1, "open"), ("h", 2, "high"), ("l", 3, "low"), ("c", 4, "close"),
         ("n", 8, "n_trades")]
NEAR = [("v", 5, "volume"), ("qv", 7, "quote_volume"),
        ("tbb", 9, "taker_buy_base"), ("tbq", 10, "taker_buy_quote")]
REL_TOL = 1e-9


def parse_bars(pattern: str) -> dict:
    """-> {(product_id, bucket): {"built": rec|None, "corrected": rec|None}}

    Keyed on PRODUCT_ID, not on the file. A shard log multiplexes 14 symbols,
    so keying on (file, bucket) collapsed all 14 into one and silently threw
    away 13 -- see the banner. The two records are kept SEPARATELY rather than
    one superseding the other, because "did the correction work" and "was the
    input to the decision right" are different questions and the second one is
    the one that prices a trade.
    """
    out: dict = {}
    n_lines = 0
    for f in sorted(glob.glob(pattern)):
        for line in open(f, errors="replace"):
            if "[AGBARR]" in line:
                tag, slot = "[AGBARR]", "corrected"
            elif "[AGBAR]" in line and "[AGBARQ]" not in line:
                tag, slot = "[AGBAR]", "built"
            else:
                continue
            d = dict(re.findall(r"(\w+)=([-\d.]+)", line.split(tag)[1]))
            if "open_ms" not in d or "product_id" not in d:
                # A build without product_id on [AGBAR] cannot be attributed,
                # and guessing by price is what biased this gate toward passing.
                continue
            n_lines += 1
            key = (int(d["product_id"]), int(d["open_ms"]))
            out.setdefault(key, {"built": None, "corrected": None})[slot] = d
    if n_lines == 0 and glob.glob(pattern):
        print("bars found but NONE carried product_id -- this build predates "
              "sentinel 170beb7. Grade it with an older gate or redeploy.",
              file=sys.stderr)
    return out


def load_products(contracts: str, syms: list[str]) -> dict:
    """venue symbol -> product_id, from the contracts file the fleet uses.

    Raises on an unknown symbol rather than dropping it: a symbol silently
    missing from the map is a symbol silently missing from the grade, which is
    the whole failure being fixed here.
    """
    want, found = set(syms), {}
    with open(contracts, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("symbol") in want and row.get("product_id"):
                found[row["symbol"]] = int(row["product_id"])
    missing = sorted(want - set(found))
    if missing:
        raise SystemExit(f"{contracts} has no product_id for: {', '.join(missing)}")
    return found


def kline(symbol: str, open_ms: int) -> list | None:
    u = f"{FAPI}?symbol={symbol}&interval=15m&startTime={open_ms}&limit=1"
    with urllib.request.urlopen(u, timeout=30) as r:
        k = json.load(r)
    return k[0] if k else None


def compare(rec: dict, k: list) -> list:
    bad = []
    for key, idx, name in EXACT:
        if key not in rec:
            bad.append(f"{name} ABSENT from the log line")
            continue
        if float(rec[key]) != float(k[idx]):
            bad.append(f"{name} {float(rec[key])!r} != {float(k[idx])!r}")
    for key, idx, name in NEAR:
        if key not in rec:
            bad.append(f"{name} ABSENT from the log line")
            continue
        a_, e_ = float(rec[key]), float(k[idx])
        if abs(a_ - e_) > REL_TOL * max(1.0, abs(e_)):
            bad.append(f"{name} {a_!r} != {e_!r}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--setting",
                    default="/home/stan/sandbox/marvel/gauntlet/"
                            "pred_agamotto.base.15m_1/setting.dryrun_baseline.json")
    ap.add_argument("--contracts", default="/opt/data/Binance/contract_futures.csv")
    ap.add_argument("--bars", type=int, default=1,
                    help="how many most-recent bar timestamps to grade")
    ap.add_argument("--grade", choices=("built", "corrected", "both"),
                    default="both",
                    help="'built' is the bar the model SCORED; 'corrected' is "
                         "it after reconcileFromBackfill. Exit status follows "
                         "'built' when both are graded -- a decision taken on a "
                         "short bar is not rescued by fixing the panel after.")
    ap.add_argument("--symbols", help="comma-separated venue symbols to restrict to")
    args = ap.parse_args()

    recs = parse_bars(args.logs)
    if not recs:
        print(f"NO BARS matched {args.logs} -- the gate cannot pass vacuously",
              file=sys.stderr)
        return 1
    stamps = sorted({b for _, b in recs}) [-args.bars:]

    cfg = json.load(open(args.setting))
    syms = [re.match(r"^BINANCE_PERP_(.+)_USDT$", s).group(1) + "USDT"
            for s in cfg["SYMBOLS"]]
    if args.symbols:
        keep = {s.strip().upper() for s in args.symbols.split(",")}
        syms = [s for s in syms if s in keep]
    pid_of = load_products(args.contracts, syms)

    slots = ("built", "corrected") if args.grade == "both" else (args.grade,)
    graded = {s: 0 for s in slots}
    clean = {s: 0 for s in slots}
    failures = {s: [] for s in slots}
    missing: list[str] = []
    no_venue = 0

    for ts in stamps:
        for vs in syms:
            entry = recs.get((pid_of[vs], ts))
            if entry is None:
                missing.append(f"{vs}@{ts}")
                continue
            k = kline(vs, ts)
            if k is None:
                # No venue truth: UNKNOWN, never counted as clean. Reporting a
                # network blip as a pass is the failure this gate exists for.
                no_venue += 1
                continue
            for slot in slots:
                rec = entry.get(slot)
                if rec is None:
                    continue
                graded[slot] += 1
                bad = compare(rec, k)
                if bad:
                    vp = (float(rec["v"]) - float(k[5])) / float(k[5]) * 100.0
                    failures[slot].append(
                        f"  {vs:<13} bar {ts}  volume {vp:+.2f}%  "
                        + "; ".join(bad[:2]))
                else:
                    clean[slot] += 1

    expect = len(stamps) * len(syms)
    print(f"[bar_fidelity] {len(stamps)} stamp(s) x {len(syms)} symbol(s) "
          f"= {expect} symbol-bar(s) expected")
    for slot in slots:
        g, c, f = graded[slot], clean[slot], len(failures[slot])
        pct = f"{100.0 * c / g:.1f}%" if g else "n/a"
        print(f"  {slot:<9} graded {g:4d}  exact {c:4d} ({pct})  LOST DATA {f}")
        for line in failures[slot][:20]:
            print(line)
        if len(failures[slot]) > 20:
            print(f"    ... and {len(failures[slot]) - 20} more")
    if missing:
        print(f"  NO BAR BUILT for {len(missing)} symbol-bar(s): "
              f"{', '.join(missing[:10])}"
              + (f" ... (+{len(missing) - 10})" if len(missing) > 10 else ""))
    if no_venue:
        print(f"  UNKNOWN (venue lookup failed) for {no_venue} symbol-bar(s)")

    # The gate's verdict follows the BUILT bar when it was graded: that is the
    # one the decision was taken on. Falling back to 'corrected' would grade
    # the panel repair and call it the input.
    verdict_slot = "built" if "built" in slots else slots[0]
    if graded[verdict_slot] == 0:
        print("  graded NOTHING -- refusing to report a pass", file=sys.stderr)
        return 1
    return 1 if failures[verdict_slot] or missing else 0


if __name__ == "__main__":
    sys.exit(main())
