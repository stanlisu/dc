#!/usr/bin/env python3
"""Run the C++ agamotto core over ALL deployed symbols and grade it against
the LIVE bot's own logged decisions.

WHY THIS EXISTS, AND WHAT IT IS NOT. Every gate before this gave the C++ core
one symbol. The deployed arm trades 28, and a rule that is right on BTCUSDT can
still be wrong on a symbol whose price is 0.0045 rather than 69000, whose bars
are thin, or whose regimes hold at different times. This grades all of them
against `agamotto_bridge`'s OWN log lines -- the decisions the bot actually
took -- rather than against a second copy of the reference.

IT DOES NOT MEASURE THE LIVE FEED. The SHM feed publisher on hydra subscribes to
`btcusdt@*` ONLY (/opt/infra_configs/btcusdt_all.json), so 27 of the 28 symbols
have no tick stream to build bars from. Bars here come from Binance REST klines
instead, which is sound for grading the FEATURE->REGIME->MODEL->DECISION chain
and says nothing about the bar builder. That is not a gap in coverage: the bar
builder was graded separately against Binance's own klines and matched 36/36
columns over 4 live bars, so REST klines are exactly what the tick path was
proven to reproduce.

ALIGNMENT. Only the NEWEST row of each panel is compared. Row k of a 699-row
panel was computed with k rows of history, not 699, so its rolling features
differ from what the bot -- which always holds a full window -- computed at that
bar. Comparing every row would manufacture disagreements that live never sees.
To grade several bars, the panel is re-sliced to end at each one, which is what
--bars does.

Usage:
  ./multi_symbol_parity.py --driver <decision_parity_driver> --weights <dir> \\
      --stack <filtered_optimal_regime_stack.csv> --setting <setting.json> \\
      --bridge-log <knull_*.log> [--bars 4] [--klines-dir /tmp/klines]
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

FAPI = "https://fapi.binance.com/fapi/v1/klines"
# The driver's column contract (decision_parity_driver.cpp:138-149). Anything
# else raises there, which is why the rename is explicit rather than a guess.
DRIVER_COLS = ["open", "high", "low", "close", "volume", "quote_volume",
               "taker_buy_quote_volume", "number_of_trades"]


def venue_symbol(marvel_symbol: str) -> str:
    """BINANCE_PERP_1000SHIB_USDT -> 1000SHIBUSDT."""
    m = re.match(r"^BINANCE_PERP_(.+)_USDT$", marvel_symbol)
    if not m:
        raise SystemExit(f"cannot parse venue symbol from {marvel_symbol!r}")
    return f"{m.group(1)}USDT"


def fetch_klines(symbol: str, interval: str, limit: int) -> list[list]:
    url = f"{FAPI}?symbol={symbol}&interval={interval}&limit={limit}"
    with urllib.request.urlopen(url, timeout=30) as r:
        rows = json.load(r)
    # Binance's last row is the OPEN bar. Drop it: a partial bar is exactly the
    # silently-low-volume row this whole project exists to avoid.
    now_ms = int(time.time() * 1000)
    closed = [k for k in rows if int(k[6]) <= now_ms]
    if len(closed) == len(rows):
        closed = rows[:-1]
    return closed


def to_driver_csv(klines: list[list]) -> str:
    out = [",".join(DRIVER_COLS)]
    for k in klines:
        # 0 open_ms 1 o 2 h 3 l 4 c 5 vol 6 close_ms 7 quote_vol 8 n_trades
        # 9 taker_buy_base 10 taker_buy_quote
        out.append(",".join([k[1], k[2], k[3], k[4], k[5], k[7], k[10], str(k[8])]))
    return "\n".join(out) + "\n"


def stack_specs(path: Path) -> str:
    """`r060_and_r075_long` + position -> `r060.r075:L`."""
    import csv
    specs = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["regime"]
            if name.startswith("__"):
                continue
            atoms = [a for a in name.replace("_long", "").replace("_short", "").split("_and_") if a]
            # The driver takes NUMERIC atom codes (decision_parity_driver.cpp:113
            # strtol), while the stack spells them rNNN. Convert explicitly and
            # fail on anything that is not rNNN -- a silently dropped atom would
            # widen the regime rather than error, which is the failure mode that
            # never announces itself.
            codes = []
            for a in atoms:
                m = re.fullmatch(r"r(\d{1,5})", a)
                if not m:
                    raise SystemExit(
                        f"{path}: regime {name!r} has atom {a!r} that is not rNNN")
                codes.append(str(int(m.group(1))))
            side = "L" if row["position"] == "long" else "S"
            specs.append(".".join(codes) + ":" + side)
    if not specs:
        raise SystemExit(f"{path}: no regimes")
    return ",".join(specs)


def run_driver(driver: str, weights: Path, specs: str, gate: dict, csv_text: str):
    cmd = shlex.split(driver) + [
        "--weights", str(weights), "--regimes", specs,
        "--threshold-long", repr(gate["tl"]), "--threshold-short", repr(gate["ts"]),
        "--center-long", repr(gate["cl"]), "--center-short", repr(gate["cs"]),
        "--reverse", str(gate["rev"]),
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, input=csv_text, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if p.returncode != 0:
        raise SystemExit(f"driver rc={p.returncode}: {p.stderr[:400]}")
    return p.stdout, wall_ms


def newest_decision(stdout: str, specs: str):
    """Last row of #decisions -> (fired, side, y_pred, winning spec)."""
    lines = stdout.splitlines()
    try:
        i = lines.index("#decisions")
    except ValueError:
        raise SystemExit("driver output has no #decisions section")
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("#")), len(lines))
    body = [l for l in lines[i + 2:j] if l.strip()]
    if not body:
        raise SystemExit("#decisions empty")
    f = body[-1].split(",")   # the driver emits CSV rows, not whitespace
    fired, side, y_pred, win = int(f[1]), int(f[2]), float(f[3]), int(f[-1])
    spec = specs.split(",")[win] if 0 <= win < len(specs.split(",")) else "-"
    return fired, side, y_pred, (spec if fired else "-")


def bridge_decisions(log: Path) -> dict:
    """{(symbol, 'HH:MM'): (side, regimes_text)} from the bot's own log."""
    out = {}
    pat = re.compile(
        r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}):\d{2},\d+ .*Decision for (\S+): (.+)$")
    for line in open(log, errors="replace"):
        m = pat.match(line.strip())
        if not m:
            continue
        _, hhmm, sym, rest = m.groups()
        side = "long" if rest.strip().endswith("long") else "short"
        out[(sym, hhmm)] = (side, rest.strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--stack", required=True, type=Path)
    ap.add_argument("--setting", required=True, type=Path)
    ap.add_argument("--bridge-log", type=Path)
    ap.add_argument("--bars", type=int, default=1,
                    help="how many recent bars to grade (panel re-sliced per bar)")
    ap.add_argument("--panel", type=int, default=699)
    ap.add_argument("--interval", default="15m")
    args = ap.parse_args()

    cfg = json.loads(args.setting.read_text())
    # No defaults: a defaulted width is the always-on gate (CLAUDE.md).
    gate = {"tl": cfg["THRESHOLD_LONG"], "ts": cfg["THRESHOLD_SHORT"],
            "cl": cfg["THRESHOLD_CENTER_LONG"], "cs": cfg["THRESHOLD_CENTER_SHORT"],
            "rev": int(cfg["REVERSE"])}
    symbols = cfg["SYMBOLS"]
    specs = stack_specs(args.stack)
    n_regimes = len(specs.split(","))
    print(f"[multi] {len(symbols)} symbols x {args.bars} bar(s), "
          f"{n_regimes} regimes, panel={args.panel}")
    print(f"[multi] gate long {gate['cl']}+{gate['tl']} | "
          f"short {gate['cs']}-{gate['ts']} | reverse={gate['rev']}")

    bridge = bridge_decisions(args.bridge_log) if args.bridge_log else {}
    need = args.panel + args.bars
    lat, rows = [], []
    for ms in symbols:
        vs = venue_symbol(ms)
        try:
            kl = fetch_klines(vs, args.interval, min(1500, need + 5))
        except Exception as exc:                      # noqa: BLE001
            print(f"  {vs:<14} FETCH FAILED: {exc}")
            continue
        if len(kl) < need:
            print(f"  {vs:<14} only {len(kl)} closed bars, need {need} -- skipped")
            continue
        for b in range(args.bars):
            end = len(kl) - b
            window = kl[end - args.panel:end]
            close_ms = int(window[-1][6])
            stdout, wall = run_driver(args.driver, args.weights, specs, gate,
                                      to_driver_csv(window))
            fired, side, y_pred, win = newest_decision(stdout, specs)
            lat.append(wall)
            hhmm = time.strftime("%H:%M", time.gmtime((close_ms + 1) / 1000))
            rows.append((ms, vs, hhmm, fired, side, y_pred, win))
    return report(rows, lat, bridge)


def report(rows, lat, bridge) -> int:
    print("\n=== C++ core decisions vs the live bot's own log ===")
    agree = disagree = unlogged = 0
    for ms, vs, hhmm, fired, side, y_pred, win in rows:
        cpp = "flat" if not fired else ("LONG" if side > 0 else "SHORT")
        b = bridge.get((ms, hhmm))
        if b is None:
            # The bot logs a Decision line ONLY when a regime fires, so absence
            # is a real datum: it means flat. Treated as such, not skipped.
            bot, mark = ("flat", "match" if not fired else "MISMATCH")
        else:
            bot = b[0].upper()
            mark = "match" if (fired and bot.lower() == ("long" if side > 0 else "short")) else "MISMATCH"
        if mark == "match":
            agree += 1
        else:
            disagree += 1
        if b is None and not fired:
            unlogged += 1
        if mark != "match" or fired:
            print(f"  {vs:<13} {hhmm}  cpp={cpp:<5} bot={bot:<5} "
                  f"y_pred={y_pred:+.8f} win={win:<28} {mark}")
    print(f"\n  compared={agree + disagree}  agree={agree}  disagree={disagree}"
          f"  (of which {unlogged} are agreed-flat)")
    if lat:
        lat = sorted(lat)
        n = len(lat)
        print(f"\n=== driver wall time, n={n} (panel + gate + models + vote, "
              f"INCLUDES process startup) ===")
        print(f"  mean={sum(lat)/n:.1f} ms  p50={lat[n//2]:.1f}  "
              f"p99={lat[min(n-1, int(n*0.99))]:.1f}  min={lat[0]:.1f}  max={lat[-1]:.1f}")
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())
