#!/usr/bin/env python3
"""ORB context-timeframe parity: knull orb (python) vs agamotto_core (C++), bar by bar.

THE REFERENCE IS THE REAL ORB CODE. `reference` builds `orb.trading.OrbTrading` on the
deployed arm (its stack, its pickled weights, its setting) and drives its OWN live path
-- `_fetch_and_prepare_data` (fetch every timeframe, drop the open bar, stamp
close_timestamp, engineer, align by backward as-of), `filter_signals`, `predict`,
`make_decision` -- for every base bar of a window. The only substitution is the
network: `orb.trading.fetch_futures_klines` is served from fixed CSVs, returning what
Binance would have returned AT THAT DECISION TIME (every bar opened at or before
D = T + base, the open one included, which orb then drops itself).

The C++ side (`orb_parity_driver`) reads the same CSVs and the exported weights and
runs the core's pieces: engineerFeatures (base, 799), engineerFeaturesContext (each
context timeframe, the newest 799 bars closed by D), contextAsofRow, atomMask,
the linear models and evaluateDecision -- the code that ships.

    fetch      --symbols ... --timeframes 15m,1h,4h,1d --start --end --out DIR
    reference  --setting S --stack CSV --home-root H --period W --klines DIR
               --start --end --out ref.jsonl [--workers N]
    compare    --ref ref.jsonl --cpp cpp.jsonl --rel-tol 1e-9 --abs-tol 1e-13

Rows (both sides, JSONL):
    {"T": open_ms, "symbol": "BTCUSDT", "regime": i, "fired": b, "y": float|null}
    {"T": open_ms, "symbol": "BTCUSDT", "decision": 1, "long": n, "short": n}

PASS = every (T, symbol, regime) agrees on `fired`, every fired `y` agrees to
--rel-tol (or
--abs-tol, for y near 0), and every (T, symbol) agrees on the long/short vote counts. Votes are the
reference's own comparison (prediction vs the stack row's optimal_threshold), and the
C++ side's is the centred per-side gate: the marvel generator refuses an orb arm whose
stack thresholds differ from that gate, so the two must agree here too.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
         "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000, "12h": 43_200_000,
         "1d": 86_400_000}
HEADER = ["open_ms", "open", "high", "low", "close", "volume", "quote_volume", "n_trades",
          "taker_buy_base", "taker_buy_quote"]
PANEL_CLOSED_BARS = 799


def _ms(s: str) -> int:
    import pandas as pd
    return int(pd.Timestamp(s, tz="UTC").value // 1_000_000)


# ---------------------------------------------------------------- fetch
def cmd_fetch(a) -> int:
    from agamotto.lib_binance import fetch_futures_klines
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start, end = _ms(a.start), _ms(a.end)
    for sym in a.symbols.split(","):
        for tf in a.timeframes.split(","):
            tf_ms = TF_MS[tf]
            lo = start - (PANEL_CLOSED_BARS + 8) * tf_ms
            lo = (lo // tf_ms) * tf_ms
            rows = fetch_futures_klines(sym, tf, lo, end + tf_ms, limit=1500)
            seen, keep = set(), []
            for r in rows:
                o = int(r[0])
                if o in seen or o >= end + tf_ms:
                    continue
                seen.add(o)
                keep.append(r)
            keep.sort(key=lambda r: int(r[0]))
            if not keep:
                raise SystemExit(f"{sym} {tf}: Binance returned no klines")
            p = out / f"backfill_{sym}_{tf}.csv"
            with p.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(HEADER)
                for r in keep:
                    # VERBATIM strings: both sides parse the same decimal text.
                    w.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[7], r[8], r[9], r[10]])
            print(f"{p.name}: {len(keep)} bars {keep[0][0]}..{keep[-1][0]}")
    return 0


# ---------------------------------------------------------------- reference
_STATE: dict = {}


def _load_klines(kdir: Path, symbols, tfs):
    data = {}
    for sym in symbols:
        for tf in tfs:
            p = kdir / f"backfill_{sym}_{tf}.csv"
            with p.open() as fh:
                r = csv.reader(fh)
                next(r)
                rows = [row for row in r]
            data[(sym, tf)] = rows
    return data


def _init_worker(a_dict):
    import logging
    logging.disable(logging.CRITICAL)
    import orb.trading as ot
    a = argparse.Namespace(**a_dict)
    setting = json.loads(Path(a.setting).read_text())
    symbols = [s for s in setting["SYMBOLS"]]
    natives = [ot._symbol_to_native(s) for s in symbols]
    tfs = setting["TIMEFRAMES"]
    data = _load_klines(Path(a.klines), natives, tfs)

    def fake_fetch(native, tf, start_ms, end_ms, limit=1500, **_):
        d = _STATE["D"]
        tf_ms = TF_MS[tf]
        rows = [r for r in data[(native, tf)] if int(r[0]) <= d]
        rows = rows[-limit:]
        return [[int(r[0]), r[1], r[2], r[3], r[4], r[5], int(r[0]) + tf_ms - 1, r[6],
                 int(r[7]), r[8], r[9], "0"] for r in rows]

    ot.fetch_futures_klines = fake_fetch
    ot.OrbTrading._dump_debug_features = lambda self: None
    ot.OrbTrading._calculate_sizes = lambda self: None
    cfg = dict(setting)
    cfg["REGIME_STACK_PATH"] = a.stack
    cfg["SIZES"] = [1.0] * len(symbols)
    cfg["LOT_SIZES"] = {}
    inst = ot.OrbTrading(cfg, home_root=a.home_root, period=a.period, skip_load=True)
    _STATE.update(inst=inst, symbols=symbols, natives=natives, ot=ot)


def _ref_one(T: int):
    inst, ot = _STATE["inst"], _STATE["ot"]
    base_ms = TF_MS[inst.config["TIME_UNIT"]]
    _STATE["D"] = T + base_ms
    inst._fetch_and_prepare_data(limit=ot.PANEL_CLOSED_BARS)
    inst._data_fresh = True
    import pandas as pd
    ts = pd.Timestamp(T, unit="ms")
    if inst.vertical_features["timestamp"].max() != ts:
        raise SystemExit(f"T={T}: the reference's newest base row is "
                         f"{inst.vertical_features['timestamp'].max()}, not {ts}")
    out = []
    preds = {}
    for i, regime in enumerate(inst.regime_stack):
        sig = inst.filter_signals(regime, save=False)
        fired_syms = set()
        if not sig.empty:
            fired_syms = set(sig.loc[sig["timestamp"] == ts, "symbol"])
        pred = inst.predict(sig, regime)
        ymap = {} if pred.empty else dict(zip(pred["symbol"], pred["prediction"]))
        for sym, nat in zip(_STATE["symbols"], _STATE["natives"]):
            fired = sym in fired_syms
            y = float(ymap[sym]) if sym in ymap else None
            if fired and y is None:
                raise SystemExit(f"T={T} {nat} regime {i}: fired but no prediction")
            preds[(sym, i)] = (fired, y, regime["position"], float(regime["threshold"]))
            out.append({"T": T, "symbol": nat, "regime": i, "fired": fired, "y": y})
    decisions = inst.make_decision()
    for sym, nat in zip(_STATE["symbols"], _STATE["natives"]):
        lc = sum(1 for (s, i), (f, y, p, thr) in preds.items()
                 if s == sym and f and p == "long" and y > thr)
        sc = sum(1 for (s, i), (f, y, p, thr) in preds.items()
                 if s == sym and f and p == "short" and y < thr)
        qty = decisions[sym][1]
        if (qty > 0) != (lc - sc > 0) or (qty < 0) != (lc - sc < 0):
            raise SystemExit(f"T={T} {nat}: make_decision qty {qty} disagrees with the "
                             f"counted net {lc - sc}")
        out.append({"T": T, "symbol": nat, "decision": 1, "long": lc, "short": sc})
    return out


def cmd_reference(a) -> int:
    setting = json.loads(Path(a.setting).read_text())
    base_ms = TF_MS[setting["TIME_UNIT"]]
    Ts = list(range(_ms(a.start), _ms(a.end), base_ms))
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker,
                             initargs=(vars(a),)) as ex, open(a.out, "w") as fh:
        for rows in ex.map(_ref_one, Ts, chunksize=4):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print(f"reference: {len(Ts)} base bars -> {a.out}")
    return 0


# ---------------------------------------------------------------- compare
def _read(p):
    regimes, decisions = {}, {}
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("decision"):
                decisions[(r["T"], r["symbol"])] = (r["long"], r["short"])
            else:
                regimes[(r["T"], r["symbol"], r["regime"])] = (r["fired"], r["y"])
    return regimes, decisions


def cmd_compare(a) -> int:
    rr, rd = _read(a.ref)
    cr, cd = _read(a.cpp)
    fail = 0
    if set(rr) != set(cr):
        print(f"FAIL: key sets differ: ref-only {len(set(rr) - set(cr))}, "
              f"cpp-only {len(set(cr) - set(rr))}")
        fail += 1
    fired_mis, y_mis, n_fired, worst = [], [], 0, 0.0
    for k in sorted(set(rr) & set(cr)):
        (rf, ry), (cf, cy) = rr[k], cr[k]
        if rf != cf:
            fired_mis.append(k)
            continue
        if rf:
            n_fired += 1
            rel = abs(ry - cy) / max(abs(ry), 1e-12)
            worst = max(worst, rel)
            # Relative OR absolute: near y = 0 a relative gap is pure rounding
            # (2026-10-10: |dy| ~ 2e-16 on y ~ 1.5e-7 read as 1.1e-8 relative).
            if not math.isfinite(cy) or (rel > a.rel_tol and abs(ry - cy) > a.abs_tol):
                y_mis.append((k, ry, cy))
    dec_mis = [k for k in rd if rd[k] != cd.get(k)]
    n_vote = sum(1 for v in rd.values() if v != (0, 0))
    print(f"regime rows: {len(rr)}  fired: {n_fired}  base bars x symbols: {len(rd)}  "
          f"with a vote: {n_vote}")
    print(f"fired mismatches: {len(fired_mis)}  y mismatches (> {a.rel_tol:g} rel and "
          f"> {a.abs_tol:g} abs): "
          f"{len(y_mis)}  worst rel |dy|: {worst:.3e}  decision mismatches: {len(dec_mis)}")
    for k in fired_mis[:10]:
        print(f"  fired  {k}: ref {rr[k]}  cpp {cr[k]}")
    for k, ry, cy in y_mis[:10]:
        print(f"  y      {k}: ref {ry!r}  cpp {cy!r}")
    for k in dec_mis[:10]:
        print(f"  vote   {k}: ref {rd[k]}  cpp {cd.get(k)}")
    if n_fired == 0:
        print("FAIL: nothing fired -- the comparison is vacuous")
        fail += 1
    fail += len(fired_mis) + len(y_mis) + len(dec_mis)
    print("PASS" if fail == 0 else f"FAIL ({fail})")
    return 0 if fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--symbols", required=True)
    f.add_argument("--timeframes", required=True)
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--out", required=True)
    r = sub.add_parser("reference")
    for k in ("--setting", "--stack", "--home-root", "--period", "--klines", "--start",
              "--end", "--out"):
        r.add_argument(k, required=True)
    r.add_argument("--workers", type=int, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--cpp", required=True)
    c.add_argument("--rel-tol", type=float, required=True)
    c.add_argument("--abs-tol", type=float, required=True,
                   help="a y gap under this is rounding whatever its relative size")
    a = ap.parse_args()
    return {"fetch": cmd_fetch, "reference": cmd_reference, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    sys.exit(main())
