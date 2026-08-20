#!/usr/bin/env python3
"""STAGE 2.7 — reconcile the LIVE C++ core against the LIVE Python bot.

Every gate before this one compares the two sides on the SAME bars:
tests/feature_parity.py hands one CSV panel to `engineerFeatures` and to
`research.py` and diffs the results. That grades the MATH and is blind to
everything upstream of it. This script grades the CHAIN — feed -> kline builder
-> feature engine — against the thing production actually runs.

    SHM tick feed ---> KlineBuilder ---> engineerFeatures ---> panel CSV
    Binance WS kline -> knull bridge --> research.py -------> debug_features CSV

Those are two different data paths, and the whole point is that they can
disagree for two COMPLETELY different reasons:

    INPUT divergence   the two sides computed correct features over DIFFERENT
                       bars. Measured 2026-08-19: the bot's WS-fed klines drift
                       from Binance's official klines by one tick intermittently
                       (bot close 64261.1 vs Binance 64261.2 on the 08:15 bar),
                       while the tick-built C++ bars were 9/9 columns bit-exact
                       against Binance.
    ENGINE divergence  the two sides computed DIFFERENT features over the SAME
                       bars. This is the only kind that is a port bug.

A run that reports one number per column cannot tell those apart, and a feature
difference on a bar whose INPUTS already differ proves nothing about the engine.
So the report is strictly ordered: inputs first, features second, and every
feature disagreement is ATTRIBUTED to one cause or the other. A column flagged
`input-divergence` is not evidence against the port; a column flagged
`engine-divergence` is, and is the only thing worth chasing.

THE BINANCE REFERENCE IS THE ARBITER, not either side. "C++ != bot" says nothing
about which one is wrong. Both are compared against Binance's own klines
(--binance-csv, the file fetch_binance_klines.py already maintains for the
strategy's backfill) and the arbiter decides.

NAMES vs CODES. The bot's CSV carries REAL feature names; the engine emits
obfuscation CODES, because no real feature name compiles into the public
plugin. The mapping is dc/obfuscation/map.json, applied HERE, on the private
side. `close` and `mvg1/2/3` have no code and pass through — the same four
exceptions tests/feature_parity.py declares.

    python tests/live_reconcile.py --ssh-host hydra --bars 3
    python tests/live_reconcile.py --panel p.csv --agbar-log s.log \\
        --debug-dir ./debug --binance-csv bf.csv --symbol BINANCE_PERP_BTC_USDT

PHASE 5 adds STEP 6: the DECISION. Same methodology, one level up. The chain
now ends in a side, and a side is a boolean — so the question "did the two
agree" is answerable exactly, and the answer is worthless on a bar whose inputs
already differed. STEP 6 therefore reports THREE decisions per shared bar:

    (a) the C++ core's own, from its [AGDEC]/[AGPRED] log lines;
    (b) the REFERENCE decision path — utils.weights_io + agamotto.trading's
        dual_gate_filter + gauntlet.thresholds — run over the C++ PANEL row;
    (c) the same reference path run over the BOT's OWN debug_features row.

(a) vs (b) is the PORT: same features, same weights, same gate, so any
difference is a bug in this core and nothing else can explain it.
(b) vs (c) is the FEED: identical code over the two chains' features, so a
difference is attributable to STEP 1/1b/3 exactly as a feature difference is.
Splitting them is the whole point — "the bot went long and the core went flat"
is not a finding until you know which of those two it was.

Exit code is 0 when no ENGINE divergence is found. Input divergence is REPORTED,
never fatal: it is a property of the two feeds, not of this port, and failing on
it would train everyone to ignore the exit code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DC_ROOT = HERE.parents[1]

# The four columns dc/obfuscation/map.json has no code for, so both sides carry
# the real name. Identical to tests/feature_parity.py's _REAL_NAMES_UNCODED —
# if that list ever grows, this one must too or the new column silently drops
# out of the comparison instead of failing it.
UNCODED = ("close", "mvg1", "mvg2", "mvg3")

# Present in the bot's debug CSV and deliberately NOT engineered by the live
# core. The 7 lookahead TARGETS all read close/high/low at shift(-1) (a live
# engine computing one would be reading a bar that has not closed); year/month
# are calendar bookkeeping; symbol/timestamp are the index. Declared so they are
# reported as EXPECTED absences rather than counted as missing columns.
EXPECTED_BOT_ONLY = (
    "return", "return_long", "return_short", "return_long_raw", "return_short_raw",
    "return_dip", "return_rip",
    "year", "month", "symbol", "timestamp",
)

# The nine Binance kline columns, C++ [AGBAR] key -> --binance-csv header.
AGBAR_TO_BINANCE = {
    "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
    "qv": "quote_volume", "n": "n_trades", "tbb": "taker_buy_base",
    "tbq": "taker_buy_quote",
}

# Same tolerance the same-input parity gate uses. RELATIVE, never exact: the two
# sides accumulate the same trades into a double in a different order, so
# identical inputs still differ in the last ULP.
REL_TOL = 1e-9
# A second, far looser band. A column inside it is numerically the same feature
# computed over inputs that differ in their last digits; a column outside it is
# a different quantity. Reported separately so "1e-7 off because the close
# differed by one tick" is not filed next to "the sign is wrong".
REL_LOOSE = 1e-4

AGBAR_RE = re.compile(
    r"\[AGBAR\]\s+open_ms=(?P<open_ms>\d+)"
    r"\s+o=(?P<o>\S+)\s+h=(?P<h>\S+)\s+l=(?P<l>\S+)\s+c=(?P<c>\S+)"
    r"\s+v=(?P<v>\S+)\s+qv=(?P<qv>\S+)\s+n=(?P<n>\S+)"
    r"\s+tbb=(?P<tbb>\S+)\s+tbq=(?P<tbq>\S+)"
    r"\s+aggr=(?P<aggr>\S+)\s+unclass=(?P<unclass>\S+)"
    # Rule 7's dropped-trade detector, added AFTER the 2.6 bundle. OPTIONAL so
    # this script still parses a log from a pre-detector build — but a missing
    # group is reported as "unknown", never as zero: reading absent evidence as
    # "no trades were lost" is precisely the silence the detector exists to end.
    r"(?:\s+gaps=(?P<gaps>\S+)\s+missing=(?P<missing>\S+))?"
    r"\s+backfill=(?P<backfill>\S+)"
)
# PHASE 5. The decision, and the per-regime vote behind it.
AGDEC_RE = re.compile(
    r"\[AGDEC\]\s+bar_ts=(?P<bar_ts>\d+)"
    r"\s+fired=(?P<fired>\d+)\s+side=(?P<side>-?\d+)"
    r"\s+votes=(?P<n_long>\d+)L/(?P<n_short>\d+)S"
    r"\s+n_trig=(?P<n_trig>\d+)"
    r"\s+y_pred=(?P<y_pred>\S+)\s+thr=(?P<thr>\S+)\s+center=(?P<center>\S+)"
    r"\s+win=(?P<win>\S+)"
)
AGPRED_RE = re.compile(
    r"\[AGPRED\]\s+bar_ts=(?P<bar_ts>\d+)\s+(?P<regime>\S+)"
    r"\s+y_pred=(?P<y_pred>\S+)\s+vote=(?P<vote>\d+)"
)
# The BOT's OWN vote, from its own log. agamotto/trading.py:856 emits
#   "Decision for <symbol>: <regime> <side> + <regime> <side> ..."
# and only when at least one regime fired, so an absent line is "no votes" —
# which is why the CYCLE index below is built from the debug CSVs and not from
# these lines: a missing decision is data, and a source that only ever speaks up
# on a fire cannot distinguish it from a cycle that never ran.
BOT_DECISION_RE = re.compile(
    r"^(?P<t>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*Decision for (?P<sym>\S+): (?P<specs>.+)$"
)

AGGATE_BOOT_RE = re.compile(
    r"\[AGDEC\]\s+gate:\s+long y_pred > (?P<cl>\S+) \+ (?P<tl>\S+) = \S+"
    r"\s+\|\s+short y_pred < (?P<cs>\S+) - (?P<ts_>\S+) = \S+"
    r"\s+\|\s+reverse=(?P<rev>-?\d+)"
)

AGFEAT_RE = re.compile(
    r"\[AGFEAT\]\s+bar_ts=(?P<bar_ts>\d+)"
    r"\s+panel_rows=(?P<rows>-?\d+)\s+panel_cols=(?P<cols>-?\d+)"
    r"\s+panel_bar_ts=(?P<panel_bar_ts>-?\d+)"
    r"\s+compute_us=(?P<us>-?\d+)"
)


def f(x: str) -> float:
    """Parse a value that may be nan/inf. float() reads all three."""
    return float(x)


def reldiff(a: float, b: float) -> float:
    """Relative difference, with the non-finite cases decided FIRST.

    NaN-vs-NaN is a MATCH (0.0) and NaN-vs-anything is a total mismatch (inf),
    because the NaN mask is part of the contract: 53 deployed regimes cannot
    fire live precisely because three columns are NaN, and a comparison that
    treated NaN as "no data, skip" would report those columns as agreeing no
    matter what either side put there.
    """
    an, bn = math.isnan(a), math.isnan(b)
    if an or bn:
        return 0.0 if (an and bn) else float("inf")
    if math.isinf(a) or math.isinf(b):
        return 0.0 if (a == b) else float("inf")
    if a == b:
        return 0.0
    denom = max(abs(a), abs(b))
    return abs(a - b) / denom if denom > 0 else abs(a - b)


def fmt(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.10g}"


# ---------------------------------------------------------------------------
# Loading


def load_panel(path: Path) -> dict[int, dict[str, float]]:
    """The C++ panel dump: one row per bar, keys are CODES.

    The dump is APPEND-ONLY across restarts, so a bar can legitimately appear
    twice (a restart re-engineers the same bucket). The LAST occurrence wins:
    it is the one the currently-running build produced.
    """
    out: dict[int, dict[str, float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            # A RESTART re-writes the header mid-file (the strategy's
            # header-written flag is per process, and appending to the same path
            # is what makes the dump survive a restart at all). Skipping it here
            # rather than deduplicating on the strategy side keeps the writer a
            # plain append; parsing it as data would raise on the literal string.
            if row["bar_ts_ms"] == "bar_ts_ms":
                continue
            ts = int(row["bar_ts_ms"])
            out[ts] = {k: f(v) for k, v in row.items() if k != "bar_ts_ms"}
    return out


def load_agbar(paths: list[Path]) -> tuple[dict[int, dict], dict[int, dict]]:
    """[AGBAR] bar inputs and [AGFEAT] panel shape/cost, keyed by bucket open ms."""
    bars: dict[int, dict] = {}
    feats: dict[int, dict] = {}
    for p in paths:
        with p.open(errors="replace") as fh:
            for line in fh:
                m = AGBAR_RE.search(line)
                if m:
                    d = m.groupdict()
                    bars[int(d["open_ms"])] = {
                        **{k: f(d[k]) for k in ("o", "h", "l", "c", "v", "qv", "n", "tbb", "tbq")},
                        "aggr": d["aggr"],
                        "unclass": int(float(d["unclass"])),
                        # None means the BUILD did not report it; 0 means it
                        # reported none. Never collapsed into each other.
                        "gaps": None if d["gaps"] is None else int(float(d["gaps"])),
                        "missing": (None if d["missing"] is None
                                    else int(float(d["missing"]))),
                        "backfill": d["backfill"] == "1",
                    }
                    continue
                m = AGFEAT_RE.search(line)
                if m:
                    d = m.groupdict()
                    feats[int(d["bar_ts"])] = {
                        "rows": int(d["rows"]), "cols": int(d["cols"]),
                        "panel_bar_ts": int(d["panel_bar_ts"]), "us": int(d["us"]),
                    }
    return bars, feats


def load_agdec(paths: list[Path]) -> tuple[dict[int, dict], dict[int, dict], dict]:
    """[AGDEC] decisions, [AGPRED] per-regime votes, and the BOOT gate echo.

    The boot line is read rather than assumed: it is the gate the CORE reports
    it will compare against, and a reconciliation that took the gate from
    setting.json instead would silently pass while the deployed config carried
    a different one — which is exactly the drift make_sentinel_config --check
    exists to catch and which nothing here would otherwise see.
    """
    dec: dict[int, dict] = {}
    preds: dict[int, dict[str, dict]] = {}
    gate: dict = {}
    for p in paths:
        with p.open(errors="replace") as fh:
            for line in fh:
                m = AGGATE_BOOT_RE.search(line)
                if m:
                    d = m.groupdict()
                    g = {"threshold_long": f(d["tl"]),
                         "threshold_short": f(d["ts_"]),
                         "threshold_center_long": f(d["cl"]),
                         "threshold_center_short": f(d["cs"]),
                         "reverse": int(d["rev"])}
                    # TWO DIFFERENT GATES IN THE PULLED LOGS means the bundle
                    # was re-deployed mid-window, so the [AGDEC] lines are not
                    # all about the same strategy. Refused rather than
                    # last-one-wins: a table built across two gates reads as one
                    # run and would attribute one gate's decisions to the other.
                    if gate and gate != g:
                        raise SystemExit(
                            f"FATAL: two DIFFERENT decision gates appear in the "
                            f"pulled logs ({gate} vs {g}). The bundle changed "
                            f"mid-window; narrow the log selection to one run "
                            f"before reconciling.")
                    gate = g
                    continue
                m = AGDEC_RE.search(line)
                if m:
                    d = m.groupdict()
                    dec[int(d["bar_ts"])] = {
                        "fired": int(d["fired"]), "side": int(d["side"]),
                        "n_long": int(d["n_long"]), "n_short": int(d["n_short"]),
                        "n_trig": int(d["n_trig"]), "y_pred": f(d["y_pred"]),
                        "thr": f(d["thr"]), "center": f(d["center"]),
                        "win": d["win"],
                    }
                    continue
                m = AGPRED_RE.search(line)
                if m:
                    d = m.groupdict()
                    preds.setdefault(int(d["bar_ts"]), {})[d["regime"]] = {
                        "y_pred": f(d["y_pred"]), "vote": int(d["vote"])}
    return dec, preds, gate


def load_bot_decisions(path: Path, symbol: str, bar_ms: int) -> dict[int, dict]:
    """The BOT's own logged votes, keyed by the BAR they were taken on.

    THE CYCLE IS NOT THE BAR. The bridge wakes at the TIME_UNIT boundary plus
    DELAY and the bot drops the in-flight candle, so a decision logged at
    15:15:05 is about the bar that OPENED at 15:00. Keying on the log's own
    timestamp would compare adjacent bars — the error that produces a small,
    plausible and entirely wrong disagreement on every field at once, and the
    same one load_bot() exists to avoid on the feature CSVs.

    Absence of a line means NO regime fired: trading.py:850 logs only when
    long_count or short_count is nonzero. Callers must treat a missing key as
    "unknown", never as "zero" — this function cannot tell a quiet cycle from a
    cycle that never ran, and only the debug CSV index can.
    """
    import datetime as dt

    out: dict[int, dict] = {}
    if not path or not Path(path).exists():
        return out
    with Path(path).open(errors="replace") as fh:
        for line in fh:
            m = BOT_DECISION_RE.match(line.strip())
            if not m or m.group("sym") != symbol:
                continue
            t = dt.datetime.strptime(m.group("t"), "%Y-%m-%d %H:%M:%S")
            cycle_ms = int(t.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
            bar_ts = (cycle_ms // bar_ms) * bar_ms - bar_ms
            n_long = n_short = 0
            regimes = []
            for spec in m.group("specs").split(" + "):
                parts = spec.strip().split()
                if len(parts) < 2:
                    continue
                regimes.append((parts[0], parts[-1]))
                if parts[-1] == "long":
                    n_long += 1
                else:
                    n_short += 1
            out[bar_ts] = {"n_long": n_long, "n_short": n_short,
                           "regimes": sorted(regimes)}
    return out


def load_binance(path: Path) -> dict[int, dict[str, float]]:
    """Binance's own klines — the ARBITER. fetch_binance_klines.py's CSV."""
    out: dict[int, dict[str, float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            out[int(row["open_ms"])] = {k: f(v) for k, v in row.items() if k != "open_ms"}
    return out


def load_bot(debug_dir: Path, symbol: str) -> dict[int, dict[str, float]]:
    """The bot's debug_features_*.csv files, keyed by the BAR's open ms.

    The filename carries the CYCLE time and the `timestamp` column carries the
    BAR time; they differ by one bar period. Keying on the filename would
    silently compare adjacent bars, which is the one error that produces a
    small, plausible, entirely wrong disagreement in every column at once.
    """
    import datetime as dt

    out: dict[int, dict[str, float]] = {}
    for p in sorted(debug_dir.glob("debug_features_*.csv")):
        with p.open() as fh:
            for row in csv.DictReader(fh):
                if row.get("symbol") != symbol:
                    continue
                t = dt.datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                ts = int(t.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
                vals: dict[str, float] = {}
                for k, v in row.items():
                    if k in ("symbol", "timestamp"):
                        continue
                    # An EMPTY cell is pandas' NaN on the way out, not a zero.
                    # Reading it as 0.0 would make the three all-NaN
                    # vol-quantile columns look like they agreed with a
                    # zero-filled engine.
                    vals[k] = float("nan") if v.strip() == "" else f(v)
                out[ts] = vals
    return out


def pull(host: str, remote_glob: str, dest: Path, newest: int | None = None) -> list[Path]:
    """scp files off a host into dest; optionally only the `newest` of them.

    `newest` matters: the bot's debug dir holds ~96 files a day and had 886 of
    them (32 MB) the day this was written. Pulling the lot to read three bars
    is minutes of transfer for nothing, and a reconciliation nobody runs
    because it is slow is a reconciliation nobody runs.
    """
    listing = subprocess.run(
        ["ssh", host, f"ls -1t {remote_glob} 2>/dev/null"],
        capture_output=True, text=True, check=False).stdout.split()
    if not listing:
        raise SystemExit(f"FATAL: {host}:{remote_glob} matched no files")
    if newest is not None:
        listing = listing[:newest]
    # One scp for the whole batch: a call per file pays the SSH handshake N times.
    subprocess.run(["scp", "-q", *(f"{host}:{shlex.quote(f)}" for f in listing), str(dest)],
                   check=True)
    return sorted(dest.iterdir())


# ---------------------------------------------------------------------------
# Report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssh-host", default=None,
                    help="pull the C++ artifacts and the bot's debug CSVs off this host "
                         "(hydra) instead of reading local paths")
    ap.add_argument("--remote-bundle", default="~/agamotto_test",
                    help="the C++ strategy bundle on --ssh-host")
    ap.add_argument("--remote-debug", default="~/sandbox/marvel/debug",
                    help="the bot's debug_features output dir on --ssh-host")
    ap.add_argument("--panel", type=Path, default=None, help="C++ panel dump CSV")
    ap.add_argument("--agbar-log", type=Path, nargs="*", default=None,
                    help="C++ strategy log(s) carrying [AGBAR]/[AGFEAT]")
    ap.add_argument("--debug-dir", type=Path, default=None,
                    help="dir of the bot's debug_features_*.csv")
    ap.add_argument("--binance-csv", type=Path, default=None,
                    help="Binance's own klines (fetch_binance_klines.py output) — the ARBITER")
    ap.add_argument("--symbol", default="BINANCE_PERP_BTC_USDT")
    ap.add_argument("--map", type=Path, default=DC_ROOT / "obfuscation" / "map.json")
    ap.add_argument("--bars", type=int, default=3,
                    help="how many of the most recent SHARED bars to reconcile (>=3)")
    ap.add_argument("--counterfactual-driver", default=None,
                    help="path to feature_parity_driver. Runs the SAME engine over the "
                         "PURE BINANCE 699-bar window ending at each reconciled bar and "
                         "diffs THAT against the bot. See STEP 5 — this is what separates "
                         "'the engine is wrong' from 'the bars were wrong'.")
    # ---- PHASE 5: the decision reconciliation ---------------------------
    ap.add_argument("--stack", type=Path, default=HERE / "regime_stack_deployed.csv",
                    help="the DEPLOYED filtered_optimal_regime_stack.csv. STEP 6 "
                         "decides on exactly these regimes.")
    ap.add_argument("--raw-weights", type=Path, default=None,
                    help="the window_YYYY_MM_DD dir of ridge_*.pkl — what the "
                         "DEPLOYED loader reads. STEP 6 needs it to run the "
                         "reference decision path; without it STEP 6 is SKIPPED "
                         "and says so rather than reporting a decision it could "
                         "not check.")
    ap.add_argument("--marvel-root", type=Path, default=None,
                    help="marvel checkout, for utils.weights_io and "
                         "gauntlet.thresholds — THE deployed loader and THE gate. "
                         "Neither is reimplemented here.")
    ap.add_argument("--model", default="ridge", help="artifact prefix for STEP 6")
    ap.add_argument("--bridge-log", type=Path, default=None,
                    help="the knull agamotto_bridge log. Its 'Decision for <sym>:' "
                         "lines are the BOT'S OWN votes and are the only in-band "
                         "record of what the production bot actually decided; "
                         "STEP 6 grades the reference transcription against them.")
    ap.add_argument("--remote-bridge-log", default="~/agamotto_test/knull_baseline_*.log",
                    help="glob for --bridge-log when pulling over --ssh-host")
    args = ap.parse_args()

    tmp = None
    if args.ssh_host:
        tmp = tempfile.TemporaryDirectory(prefix="live_reconcile.")
        root = Path(tmp.name)
        (root / "panel").mkdir()
        (root / "log").mkdir()
        (root / "debug").mkdir()
        (root / "bf").mkdir()
        b = args.remote_bundle
        print(f"[pull] {args.ssh_host}:{b} + {args.remote_debug}")
        pull(args.ssh_host, f"{b}/log/*_panel_*.csv", root / "panel", newest=2)
        pull(args.ssh_host, f"{b}/log/tsAgamottoShadow-*.log", root / "log", newest=3)
        pull(args.ssh_host, f"{b}/config/backfill_*_15m.csv", root / "bf", newest=1)
        # 2 cycles per reconciled bar is generous slack for a restart or a
        # skipped cycle; the shared-bar intersection below decides what is
        # actually used.
        pull(args.ssh_host, f"{args.remote_debug}/debug_features_*.csv", root / "debug",
             newest=max(8, args.bars * 2))
        args.panel = sorted((root / "panel").glob("*.csv"))[-1]
        args.agbar_log = sorted((root / "log").glob("*.log"))
        args.debug_dir = root / "debug"
        args.binance_csv = sorted((root / "bf").glob("*.csv"))[-1]
        try:
            (root / "bridge").mkdir()
            pull(args.ssh_host, args.remote_bridge_log, root / "bridge", newest=1)
            args.bridge_log = sorted((root / "bridge").iterdir())[-1]
        except SystemExit as exc:
            # NAMED, not swallowed: without it STEP 6 loses its only in-band
            # record of what the bot actually decided, and must say so.
            print(f"[pull] WARNING: no bridge log ({exc}); STEP 6 will have no "
                  f"(d) source and the bot's OWN vote will be UNCHECKED")

    for name, v in (("--panel", args.panel), ("--agbar-log", args.agbar_log),
                    ("--debug-dir", args.debug_dir), ("--binance-csv", args.binance_csv)):
        if not v:
            print(f"FATAL: {name} is required (or use --ssh-host)", file=sys.stderr)
            return 2

    panel = load_panel(args.panel)
    agbar, agfeat = load_agbar([Path(p) for p in args.agbar_log])
    agdec, agpred, boot_gate = load_agdec([Path(p) for p in args.agbar_log])
    binance = load_binance(args.binance_csv)
    bot = load_bot(args.debug_dir, args.symbol)
    # The bar length is MEASURED off the Binance grid, never assumed: this
    # script is also run at bar_sec=60 for the accelerated parity loop, and a
    # hardcoded 15m would key every bot decision to the wrong bar there while
    # still producing a full, plausible table.
    _bt = sorted(binance)
    if len(_bt) < 2:
        raise SystemExit("FATAL: --binance-csv has fewer than 2 klines, so the bar "
                         "length cannot be measured.")
    bar_ms = min(b - a for a, b in zip(_bt, _bt[1:]))
    bot_dec = load_bot_decisions(args.bridge_log, args.symbol, bar_ms)
    enc_map = json.loads(args.map.read_text())["features"]

    print("=" * 78)
    print("STAGE 2.7 LIVE RECONCILIATION")
    print("=" * 78)
    print(f"  symbol        {args.symbol}")
    print(f"  C++ panel     {args.panel}  ({len(panel)} bars, "
          f"{len(next(iter(panel.values()))) if panel else 0} columns)")
    print(f"  C++ [AGBAR]   {len(agbar)} bars from {len(args.agbar_log)} log file(s)")
    print(f"  bot debug     {args.debug_dir}  ({len(bot)} bars for this symbol)")
    print(f"  Binance ref   {args.binance_csv}  ({len(binance)} klines)")
    print(f"  C++ [AGDEC]   {len(agdec)} decision(s), {len(agpred)} bar(s) with "
          f"per-regime votes")
    if boot_gate:
        print(f"  C++ gate      long {boot_gate['threshold_center_long']!r} + "
              f"{boot_gate['threshold_long']!r} | short "
              f"{boot_gate['threshold_center_short']!r} - "
              f"{boot_gate['threshold_short']!r} | reverse="
              f"{boot_gate['reverse']}   (read from the CORE's own boot line)")
    else:
        print("  C++ gate      NOT FOUND in the logs — the [AGDEC] boot line is "
              "absent, so this bundle predates Phase 5 or the log rolled.")

    shared = sorted(set(panel) & set(bot))
    if not shared:
        print("\nFATAL: no bar_ts is present in BOTH the C++ panel and the bot's CSV.")
        print(f"  C++ panel bars : {sorted(panel)[-4:]}")
        print(f"  bot CSV bars   : {sorted(bot)[-4:]}")
        print("  The two sides have not yet produced a common bar. Wait, or widen the logs.")
        return 2
    use = shared[-args.bars:]
    print(f"\n  shared bars   {len(shared)} total; reconciling the newest {len(use)}")
    if len(use) < args.bars:
        print(f"  WARNING: asked for {args.bars} bars, only {len(use)} are shared so far.")

    import datetime as dt

    def iso(ts: int) -> str:
        return dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")

    # ---------------------------------------------------------------- STEP 1
    # INPUTS FIRST. Nothing about the engine can be read off a feature diff
    # until this table is on the page.
    print()
    print("-" * 78)
    print("STEP 1 — BAR INPUT AGREEMENT (both sides vs Binance, the arbiter)")
    print("-" * 78)
    input_divergent: set[int] = set()
    print(f"{'bar (UTC)':<18}{'src':<8}{'col':<16}{'C++/bot':>20}{'Binance':>20}{'rel':>12}")
    for ts in use:
        b = binance.get(ts)
        a = agbar.get(ts)
        if b is None:
            print(f"{iso(ts):<18}{'--':<8}Binance kline ABSENT from --binance-csv "
                  f"— cannot arbitrate this bar")
            input_divergent.add(ts)
            continue
        worst_cpp = 0.0
        bad_cols = []
        if a is None:
            print(f"{iso(ts):<18}{'cpp':<8}[AGBAR] line ABSENT — the bar predates this log "
                  f"(backfilled, not built)")
        else:
            for k, bk in AGBAR_TO_BINANCE.items():
                r = reldiff(a[k], b[bk])
                worst_cpp = max(worst_cpp, r)
                if r > REL_TOL:
                    bad_cols.append((k, a[k], b[bk], r))
            for k, av, bv, r in bad_cols:
                print(f"{iso(ts):<18}{'cpp':<8}{AGBAR_TO_BINANCE[k]:<16}"
                      f"{fmt(av):>20}{fmt(bv):>20}{r:>12.2e}")
            if not bad_cols:
                print(f"{iso(ts):<18}{'cpp':<8}{'ALL 9 EXACT':<16}"
                      f"{'':>20}{'':>20}{worst_cpp:>12.2e}  aggr={a['aggr']} "
                      f"unclass={a['unclass']}")
            # THE TRADE-ID GAP DETECTOR — REPORTED, and deliberately NOT a
            # dirtiness verdict on its own.
            #
            # *** MEASURED 2026-08-20: IT FIRES ON BARS THAT ARE BIT-EXACT. ***
            # The 03:00/03:15/03:30/03:45 bars reported 198/47/24/22 "missing"
            # ids while ALL NINE columns matched Binance to 6e-15. They cannot
            # both be true: a bar that really lost 198 trades cannot reproduce
            # Binance's volume, quote_volume, trade count and both taker_buy_*
            # to fifteen digits. The publisher subscribes to BOTH btcusdt@trade
            # and btcusdt@aggTrade, and those carry DIFFERENT id sequences — an
            # aggTrade id is not a trade id — so a gap in the merged sequence is
            # not evidence of a dropped slot.
            #
            # THE ARBITER DECIDES, not the detector. Binance's own kline says
            # whether the inputs are right; the detector is a hint. Letting the
            # hint mark a bar dirty made every bar of this run UNATTRIBUTED and
            # printed "65 live differences" over a table of exact zeros — a
            # conservative default that destroys the attribution the whole
            # report exists to make.
            if a["missing"] is None:
                print(f"{iso(ts):<18}{'cpp':<8}{'trade-id gaps':<16}"
                      f"{'UNKNOWN':>20}{'(pre-detector build)':>20}")
            elif a["missing"] > 0:
                lost = 100.0 * a["missing"] / max(1.0, a["n"] + a["missing"])
                verdict_ = ("but ALL 9 COLUMNS ARE EXACT -> FALSE POSITIVE, "
                            "see the note in the source"
                            if not bad_cols else
                            "AND the columns differ -> a REAL loss")
                print(f"{iso(ts):<18}{'cpp':<8}{'trade-id gaps':<16}"
                      f"{a['gaps']:>20}{a['missing']:>20}{lost:>11.2f}%  {verdict_}")
        # The bot's debug CSV carries only `close` of the nine raw columns, so
        # `close` is the whole of the input comparison available on that side.
        # It is also the column that matters most: every return, every MA and
        # every TA-Lib indicator is a function of it.
        rb = reldiff(bot[ts]["close"], b["close"])
        tag = "EXACT" if rb <= REL_TOL else "DIFFERS"
        print(f"{iso(ts):<18}{'bot':<8}{'close':<16}"
              f"{fmt(bot[ts]['close']):>20}{fmt(b['close']):>20}{rb:>12.2e}  {tag}")
        if rb > REL_TOL or worst_cpp > REL_TOL or a is None:
            input_divergent.add(ts)

    print()
    if input_divergent:
        print(f"  {len(input_divergent)}/{len(use)} bar(s) have DIVERGENT INPUTS. Feature")
        print("  differences on those bars are NOT evidence about the engine.")
    else:
        print(f"  ALL {len(use)} bars: inputs agree with Binance on both sides.")

    # ------------------------------------------------------------ STEP 1b
    # A CLEAN BAR IS NOT ENOUGH. Almost every column here is a ROLLING
    # quantity: the TA-Lib block is 14 bars deep, the scale-free transforms are
    # 20, mvg3 is 99, and price_range_pct_q50 is a 700-bar median. One bad bar
    # therefore poisons the NEXT ~20 bars of every column that reads it, while
    # the bar it lands on may itself be perfectly clean.
    #
    # Judging attribution per bar alone would report exactly that as engine
    # divergence, which is how a correct port gets blamed for a dropped tick.
    print()
    print("-" * 78)
    print("STEP 1b — WINDOW INPUT AGREEMENT (a rolling column reads MANY bars)")
    print("-" * 78)
    live = sorted(agbar)
    dirty_window = []
    for ts in live:
        b = binance.get(ts)
        if b is None:
            continue
        worst = max(reldiff(agbar[ts][k], b[bk]) for k, bk in AGBAR_TO_BINANCE.items())
        # Same rule as STEP 1: the ARBITER decides. A trade-id gap on a bar
        # whose nine columns match Binance is carried into the listing as
        # information, not as contamination.
        lost = agbar[ts]["missing"]
        if worst > REL_TOL:
            dirty_window.append((ts, worst, lost))
    print(f"  live tick-built bars with a Binance kline to check: {len(live)}")
    if dirty_window:
        print(f"  DIRTY: {len(dirty_window)} of them differ from Binance —")
        for ts, worst, lost in dirty_window:
            extra = "" if not lost else f"  ({lost} trade id(s) skipped)"
            print(f"      {iso(ts)}  worst rel {worst:.2e}{extra}")
        print("  Every rolling column that reads high/low/volume/trades over a window")
        print("  containing one of those bars is CONTAMINATED, on bars whose own inputs")
        print("  are exact. STEP 5 is the only thing that can separate that from a port bug.")
    else:
        print("  none — every live-built bar matches Binance on all 9 columns.")
    gapped = [(t, agbar[t]["missing"]) for t in live
              if agbar[t]["missing"] and t in binance
              and max(reldiff(agbar[t][k], binance[t][bk])
                      for k, bk in AGBAR_TO_BINANCE.items()) <= REL_TOL]
    if gapped:
        tot = sum(m for _t, m in gapped)
        print(f"  FINDING: the trade-id gap detector reports {tot} skipped id(s) "
              f"across {len(gapped)} bar(s) whose NINE COLUMNS ARE EXACT against "
              f"Binance. Those two cannot both mean 'trades were lost'. The "
              f"publisher merges btcusdt@trade and btcusdt@aggTrade, which carry "
              f"DIFFERENT id sequences, so a gap in the merged stream is not "
              f"evidence of a dropped ring slot. Reported, not fixed — the fix "
              f"is to track the two sequences separately in the builder.")

    # ---------------------------------------------------------------- STEP 2
    print()
    print("-" * 78)
    print("STEP 2 — COLUMN COVERAGE (bot REAL names -> engine CODES via map.json)")
    print("-" * 78)
    bot_cols = set(next(iter(bot[ts] for ts in use)))
    cpp_cols = set(panel[use[-1]])

    def enc(name: str) -> str:
        return name if name in UNCODED else enc_map.get(name, name)

    bot_encoded = {enc(c): c for c in bot_cols if c not in EXPECTED_BOT_ONLY}
    shared_cols = sorted(set(bot_encoded) & cpp_cols)
    only_cpp = sorted(cpp_cols - set(bot_encoded))
    only_bot = sorted(set(bot_encoded) - cpp_cols)
    unmapped = sorted(c for c in bot_cols
                      if c not in EXPECTED_BOT_ONLY and c not in UNCODED and c not in enc_map)

    print(f"  engine panel        {len(cpp_cols)} columns")
    print(f"  bot CSV             {len(bot_cols)} columns "
          f"({len(EXPECTED_BOT_ONLY)} declared non-features: 7 lookahead targets, "
          f"year/month, symbol/timestamp)")
    print(f"  COMPARED            {len(shared_cols)} columns")
    print(f"  engine-only         {len(only_cpp)}  {only_cpp if only_cpp else ''}")
    print(f"  bot-only            {len(only_bot)}  "
          f"{[bot_encoded[c] for c in only_bot] if only_bot else ''}")
    if unmapped:
        print(f"  UNMAPPED in map.json {unmapped}")

    # ---------------------------------------------------------------- STEP 3
    print()
    print("-" * 78)
    print("STEP 3 — PER-COLUMN FEATURE AGREEMENT, WITH ATTRIBUTION")
    print("-" * 78)
    clean_bars = [t for t in use if t not in input_divergent]
    print(f"  bars with clean inputs: {len(clean_bars)}/{len(use)}"
          f"  {[iso(t) for t in clean_bars]}")
    print()

    rows = []
    for code in shared_cols:
        real = bot_encoded[code]
        worst_clean = 0.0
        worst_any = 0.0
        for ts in use:
            r = reldiff(panel[ts][code], bot[ts][real])
            worst_any = max(worst_any, r)
            if ts in clean_bars:
                worst_clean = max(worst_clean, r)
        rows.append((code, real, worst_clean, worst_any))

    # DELIBERATELY NOT CALLED "ENGINE-DIVERGENCE". This table compares the LIVE
    # panel, which is built on tick-built bars, against the bot, which is built
    # on kline bars — so a difference here is "the two chains differ", and the
    # cause is not decidable from this table. When STEP 1b found a dirty bar,
    # every rolling column reading it is contaminated for the next ~20 bars even
    # where the bar's own inputs are exact, and calling that an engine bug would
    # convict a correct port of a dropped tick. STEP 5 is the verdict.
    contaminated = " (window contaminated — see 1b/5)" if dirty_window else " (window clean)"

    def verdict(worst_clean: float, worst_any: float) -> str:
        if not clean_bars:
            return "UNATTRIBUTED (no bar had clean inputs)"
        if worst_clean <= REL_TOL:
            return "AGREE" if worst_any <= REL_TOL else "AGREE (differs only on dirty bars)"
        if worst_clean <= REL_LOOSE:
            return "DIVERGES-LIVE (small)" + contaminated
        return "DIVERGES-LIVE" + contaminated

    agree = [r for r in rows if verdict(r[2], r[3]).startswith("AGREE")]
    diverge = [r for r in rows if not verdict(r[2], r[3]).startswith("AGREE")]

    print(f"{'code':<8}{'real name':<24}{'worst rel (clean)':>19}"
          f"{'worst rel (all)':>18}   verdict")
    for code, real, wc, wa in sorted(diverge, key=lambda r: -r[2]):
        print(f"{code:<8}{real:<24}{wc:>19.3e}{wa:>18.3e}   {verdict(wc, wa)}")
    if not diverge:
        print("  (no column disagrees on a bar whose inputs agreed)")
    print()
    print(f"  AGREE               {len(agree)}/{len(rows)} columns "
          f"(rel <= {REL_TOL:g} on every clean-input bar)")
    print(f"  DIVERGES-LIVE       {len(diverge)}/{len(rows)} columns"
          f"{' — cause NOT decidable here; run STEP 5' if dirty_window else ''}")

    # The columns whose disagreement is EXPLAINED by the inputs, listed
    # separately so they are visibly accounted for rather than absorbed.
    input_only = [r for r in rows if r[2] <= REL_TOL < r[3]]
    if input_only:
        print(f"\n  {len(input_only)} column(s) disagree ONLY on input-divergent bars"
              f" — attributed to the FEED, not the engine:")
        for code, real, wc, wa in sorted(input_only, key=lambda r: -r[3])[:12]:
            print(f"      {code:<8}{real:<24} worst rel {wa:.3e}")

    # ---------------------------------------------------------------- STEP 4
    print()
    print("-" * 78)
    print("STEP 4 — NAMED CHECKS")
    print("-" * 78)
    # buy_pressure is taker_buy_quote / quote_volume. With an UNPOPULATED maker
    # flag the C++ side falls back to the quote rule and the column is an
    # approximation; the publisher build deployed 2026-08-19 populates it, so a
    # disagreement here is a FINDING, not the known limitation.
    bp = enc("buy_pressure")
    if bp in dict((r[0], r) for r in rows):
        r = dict((x[0], x) for x in rows)[bp]
        srcs = {agbar[t]["aggr"] for t in use if t in agbar}
        print(f"  buy_pressure ({bp}): worst rel (clean) {r[2]:.3e}   "
              f"aggressor_source seen: {sorted(srcs) or ['n/a — bar predates the log']}")
        if r[2] > REL_TOL and srcs and srcs != {"exact"}:
            print("      -> the maker flag is NOT exact on these bars; this is the known "
                  "quote-rule approximation, not an engine bug.")
        elif r[2] > REL_TOL:
            print("      -> FINDING: the maker flag IS exact and buy_pressure still "
                  "disagrees on clean inputs.")

    # The three vol-quantile cutoffs are NaN on a 699-row panel (min_periods=700).
    # Both sides must be NaN. If the bot is NOT NaN, the bot is engineering more
    # than 699 rows and the two are not comparable at all.
    for real in ("price_range_pct_q80", "price_range_pct_q90", "price_range_pct_q95"):
        code = enc(real)
        if code not in cpp_cols or real not in bot_cols:
            continue
        c_nan = all(math.isnan(panel[t][code]) for t in use)
        b_nan = all(math.isnan(bot[t][real]) for t in use)
        state = "both NaN (expected: min_periods=700 > 699 rows)" if (c_nan and b_nan) \
            else f"cpp_nan={c_nan} bot_nan={b_nan}  <-- MISMATCH"
        print(f"  {real:<22} {state}")

    # Panel shape and cost, from the strategy's own [AGFEAT] lines.
    got = [agfeat[t] for t in use if t in agfeat]
    if got:
        us = sorted(g["us"] for g in got)
        shapes = {(g["rows"], g["cols"]) for g in got}
        print(f"  panel shape            {shapes}")
        print(f"  feature_compute_us     min={us[0]} p50={us[len(us)//2]} max={us[-1]}"
              f"   over {len(us)} bar(s)")
        stamped = all(g["panel_bar_ts"] == t for t, g in
                      ((t, agfeat[t]) for t in use if t in agfeat))
        print(f"  panel stamped with its own bar: {stamped}")

    # ---------------------------------------------------------------- STEP 5
    # THE COUNTERFACTUAL, and the only step that can actually exonerate or
    # convict the engine when the live bars are not bit-exact.
    #
    # It runs the SAME engine binary over the PURE BINANCE 699-bar window ending
    # at the reconciled bar — the identical input the bot's own pipeline is
    # supposed to be working from — and diffs THAT against the bot's row. The
    # live panel is held out entirely.
    #
    #   agree here + disagree live  ->  the ENGINE is right and the BARS differ.
    #   disagree here               ->  the ENGINE differs, on inputs nothing
    #                                   can be blamed for.
    #
    # This is not the same test as tests/feature_parity.py: that one diffs the
    # engine against research.py run OFFLINE by the harness, on synthetic
    # panels. This diffs the engine against what the PRODUCTION BOT actually
    # emitted, in production, for that bar.
    cf_fail = None
    if args.counterfactual_driver:
        print()
        print("-" * 78)
        print("STEP 5 — COUNTERFACTUAL: the SAME engine over the PURE BINANCE window")
        print("-" * 78)
        order = sorted(binance)
        cf_fail = 0
        for ts in use:
            if ts not in binance:
                print(f"  {iso(ts)}: not in the Binance CSV — skipped")
                continue
            end = order.index(ts)
            start = end + 1 - 699
            if start < 0:
                print(f"  {iso(ts)}: only {end + 1} Binance bars precede it, 699 needed "
                      f"— skipped (widen --binance-csv)")
                continue
            win = order[start:end + 1]
            # Contiguity is a CORRECTNESS precondition, not a formality: a hole
            # would silently shift every rolling window and the run would still
            # print numbers.
            step = win[1] - win[0]
            if any(win[i + 1] - win[i] != step for i in range(len(win) - 1)):
                print(f"  {iso(ts)}: the Binance window has a HOLE — refusing to run it")
                cf_fail += 1
                continue
            hdr = ["open", "high", "low", "close", "volume", "quote_volume",
                   "taker_buy_quote", "n_trades"]
            # The driver's own column names; taker_buy_quote / n_trades are named
            # differently in the two CSVs and mapping them here is what keeps
            # buy_pressure and trade_intensity from being silently dropped.
            out = ["open,high,low,close,volume,quote_volume,"
                   "taker_buy_quote_volume,number_of_trades"]
            for w in win:
                r = binance[w]
                out.append(",".join(repr(r[c]) for c in hdr))
            proc = subprocess.run(shlex.split(args.counterfactual_driver),
                                  input="\n".join(out) + "\n",
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"  {iso(ts)}: driver failed rc={proc.returncode}: {proc.stderr.strip()}")
                cf_fail += 1
                continue
            lines = proc.stdout.strip().splitlines()
            names = lines[0].split(",")
            last = [float(x) for x in lines[-1].split(",")]
            cf = dict(zip(names, last))
            worst = []
            for code in shared_cols:
                r = reldiff(cf[code], bot[ts][bot_encoded[code]])
                if r > REL_TOL:
                    worst.append((r, code, bot_encoded[code], cf[code], bot[ts][bot_encoded[code]]))
            worst.sort(reverse=True)
            if not worst:
                print(f"  {iso(ts)}: ALL {len(shared_cols)} columns agree with the BOT "
                      f"to rel <= {REL_TOL:g}  -> ENGINE EXONERATED for this bar")
            else:
                cf_fail += len(worst)
                print(f"  {iso(ts)}: {len(worst)}/{len(shared_cols)} columns differ from the "
                      f"BOT on IDENTICAL Binance inputs -> ENGINE DIVERGENCE")
                for r, code, real, a, b in worst[:15]:
                    print(f"      {code:<8}{real:<24}{fmt(a):>18}{fmt(b):>18}{r:>12.2e}")

    # ---------------------------------------------------------------- STEP 6
    # THE DECISION. Same methodology as STEP 3/5, one level up.
    #
    # Three decisions per bar, and the two comparisons they make are about
    # completely different things:
    #
    #   (a) the C++ core's own          [AGDEC]/[AGPRED]
    #   (b) the REFERENCE path over the C++ PANEL row
    #   (c) the REFERENCE path over the BOT's debug_features row
    #
    #   (a) vs (b) is THE PORT. Same features, same weights, same gate — a
    #       difference here has no other explanation and is a bug in this core.
    #   (b) vs (c) is THE FEED. Identical code over two chains' features, so a
    #       difference is attributable exactly as STEP 3's are, and a decision
    #       difference on a bar STEP 1 already flagged proves nothing.
    #
    # The reference is IMPORTED, never rewritten: utils.weights_io.load_regime
    # (the loader the bot and tesseract call), agamotto.trading.dual_gate_filter
    # (the bot's own firing rule) and gauntlet.thresholds.signed_threshold (THE
    # definition of the per-leg centred gate).
    dec_engine_diff = None
    if args.raw_weights and args.marvel_root:
        print()
        print("-" * 78)
        print("STEP 6 — THE DECISION (port vs feed)")
        print("-" * 78)
        dec_engine_diff = 0
        sys.path.insert(0, str(Path(args.marvel_root).expanduser().resolve()))
        sys.path.insert(0, str(DC_ROOT / "agamotto_pkg" / "src"))
        try:
            import numpy as np
            import pandas as pd
            from utils.weights_io import load_regime
            from agamotto.research_filters import apply_filter_mask
            from agamotto.trading import dual_gate_filter
            from gauntlet.thresholds import signed_threshold
        except Exception as exc:   # noqa: BLE001 — reported, never swallowed
            print(f"  SKIPPED: cannot import the reference path ({exc}).")
            print("  STEP 6 refuses to reconcile a decision it cannot independently"
                  " compute — a 'both sides agree' printed from one side is not a"
                  " reconciliation.")
            dec_engine_diff = None
        if dec_engine_diff is not None:
            gate = boot_gate
            if not gate:
                print("  SKIPPED: the core never logged its gate, so there is no"
                      " authority for what it compared against. Reading the gate"
                      " from setting.json instead would pass while the DEPLOYED"
                      " config carried a different one.")
                dec_engine_diff = None
    if dec_engine_diff is not None:
        edges = {p: signed_threshold(gate[f"threshold_{p}"], p,
                                     center=gate[f"threshold_center_{p}"])
                 for p in ("long", "short")}
        print(f"  edges: long y_pred > {edges['long']!r}   "
              f"short y_pred < {edges['short']!r}   reverse={gate['reverse']}")

        stack_rows = []
        with args.stack.open() as fh:
            for row in csv.DictReader(fh):
                if str(row["regime"]).startswith("__"):
                    continue
                stack_rows.append((row["regime"], row["position"]))
        print(f"  stack: {len(stack_rows)} regime(s) from {args.stack}")

        # feature CODE -> real name, so apply_filter_mask (which reads REAL
        # column names) can be driven over the codes-only C++ panel. Same
        # decoder tests/regime_parity.py uses.
        _mp = json.loads(args.map.read_text())["features"]
        _rev = {v: k for k, v in _mp.items()}
        if len(_rev) != len(_mp):
            raise SystemExit("feature map is not bijective — a code decodes two ways")

        def dec_feature(c: str) -> str:
            return _rev.get(c, c)

        # The models, loaded ONCE. Only the regimes that can ever vote need a
        # model here, but all are loaded so a stack/weights mismatch surfaces as
        # the same boot failure the bot raises rather than as a quiet skip.
        arts = {}
        missing = []
        for name, _pos in stack_rows:
            try:
                arts[name] = load_regime(Path(args.raw_weights).expanduser(),
                                         regime_dir_name=name, model=args.model)
            except Exception as exc:   # noqa: BLE001 — named, never skipped silently
                missing.append((name, str(exc)[:80]))
        if missing:
            print(f"  {len(missing)} regime(s) have NO weights under "
                  f"{args.raw_weights} — they cannot vote on either side:")
            for name, why in missing[:5]:
                print(f"      {name}: {why}")

        def reference_decision(feats: dict) -> dict:
            """The reference path over ONE row of features. Nothing invented.

            THE REGIME GATE RUNS FIRST, and it is `research_filters
            .apply_filter_mask` — the same call `AgamottoResearch.filter_signals`
            makes in production, imported and never re-implemented. Skipping it
            is not a small simplification: `trading.py` predicts ONLY on rows its
            filter let through, so a version without the gate votes with all 62
            regimes including the 53 that CANNOT fire live (marvel PR #532), and
            reports 20+ voters on a bar where the bot reported none.

            The mask is evaluated on a ONE-ROW frame, which is correct here and
            only here: every predicate is a per-row comparison against columns
            the FEATURE ENGINE already computed over the whole 699-bar panel
            (`price_range_pct_q50` is a 700-bar rolling median and arrives as a
            COLUMN of that row). It is the ENGINE that must see the panel, not
            the gate. Running the engine on one row would be the real error.

            ``feats`` must be keyed by whatever ``art.feature_columns`` names.
            The callers below hand it a row carrying BOTH namespaces, because
            the exported metadata is CODED for the post-rollout weights and REAL
            for anything older — and a lookup that silently missed would drop
            the regime rather than fail, which reads as "it did not vote".
            """
            # ONE row, REAL column names — apply_filter_mask reads real names
            # (it decodes coded REGIME names itself, but its column reads are
            # research.py's own).
            named = {}
            for k, v in feats.items():
                named[dec_feature(k)] = v
            row_df = pd.DataFrame([named])

            # *** THE ONE-ROW EVALUATION IS ONLY EXACT WHILE THIS HOLDS. ***
            # research_filters carries a FALLBACK at four sites:
            #     df["price_range_pct_q50"] if present
            #     else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
            # On a one-row frame that fallback returns the value ITSELF, so
            # `price_range_pct > q50` becomes a tie and every high_vol / low_vol
            # regime silently stops firing — a gate that reports "no regime
            # held" for a reason that has nothing to do with the market.
            # The column IS engineered by both chains, so this is an assertion,
            # not a guard.
            for req in ("price_range_pct_q50", "price_range_pct"):
                if req not in named:
                    raise SystemExit(
                        f"FATAL: STEP 6 row has no {req!r}. research_filters "
                        "would fall back to a rolling quantile over ONE row, "
                        "which is the value itself, and every vol regime would "
                        "read as 'did not hold'. Refusing to report a "
                        "reconciliation computed that way.")

            recs = []
            unresolved = set()
            gated_out = 0
            for name, pos in stack_rows:
                art = arts.get(name)
                if art is None:
                    continue
                mask = apply_filter_mask(row_df, name, pos, strict_filters=True)
                if not bool(np.asarray(mask)[0]):
                    # The reference never predicts a filtered-out row at all.
                    gated_out += 1
                    continue
                cols = [str(c) for c in art.feature_columns]
                row = {}
                ok = True
                for c in cols:
                    if c not in feats:
                        ok = False
                        unresolved.add(c)
                        break
                    row[c] = feats[c]
                if not ok:
                    # trading.py:689-692 logs and returns an EMPTY frame when a
                    # selected column is missing — the regime makes no
                    # prediction. Reproduced, not filled with a guess.
                    continue
                X = pd.DataFrame([row], columns=cols)
                n_nan = int(X.isna().to_numpy().sum())
                X = X.fillna(0.0)                       # trading.py:697-700
                if np.isinf(X.to_numpy(dtype=float)).any():
                    # sklearn REFUSES inf; trading.py:744 catches and the regime
                    # returns an empty frame. Same outcome: no vote.
                    continue
                import warnings as _w
                with _w.catch_warnings():
                    _w.filterwarnings("ignore",
                                      message=".*does not have valid feature names.*")
                    y = float(np.asarray(art.predict(X))[0])
                recs.append({"regime": name, "position": pos, "prediction": y,
                             "opt_threshold": edges[pos], "nan_filled": n_nan})
            if not recs:
                return {"fired": 0, "side": 0, "n_long": 0, "n_short": 0,
                        "n_trig": 0, "preds": {}, "nan_filled": 0,
                        "gated_out": gated_out, "predicted": 0}
            if unresolved:
                # NAMED, never absorbed. A model column the row cannot supply
                # makes the regime abstain, and an abstention is
                # indistinguishable from a regime that simply did not clear its
                # gate unless it is said out loud.
                print(f"      NOTE: {len(unresolved)} model column(s) absent from "
                      f"this row, so the regimes selecting them ABSTAINED: "
                      f"{sorted(unresolved)[:6]}")
            df = pd.DataFrame(recs)
            longs, shorts = dual_gate_filter(df)        # <- THE reference call
            n_l, n_s = len(longs), len(shorts)
            net = n_l - n_s
            qty = net * gate["reverse"]                 # base_size > 0, so sign is this
            side = 0 if abs(qty) < 1e-9 else (1 if qty > 0 else -1)
            voted = set(longs["regime"]) | set(shorts["regime"])
            return {"fired": int(side != 0), "side": side, "n_long": n_l,
                    "n_short": n_s, "n_trig": n_l + n_s,
                    "preds": {r["regime"]: (r["prediction"],
                                            int(r["regime"] in voted)) for r in recs},
                    "nan_filled": int(df["nan_filled"].sum()),
                    "gated_out": gated_out, "predicted": len(recs)}

        def same(x, y):
            return (x["fired"], x["side"], x["n_long"], x["n_short"]) == \
                   (y["fired"], y["side"], y["n_long"], y["n_short"])

        print()
        print(f"{'bar (UTC)':<18}{'inputs':<10}{'source':<26}"
              f"{'fired':>6}{'side':>6}{'votes':>10}{'n_trig':>8}")
        feed_diff = 0
        bot_log_diff = 0
        checked = 0
        for ts in use:
            clean = "CLEAN" if ts not in input_divergent else "DIRTY"
            live = agdec.get(ts)
            # BOTH namespaces on each row. `art.feature_columns` is CODED for
            # the post-rollout weights and REAL for anything older, and the two
            # sides carry different ones natively: the panel is codes-only, the
            # bot CSV is real-names-only.
            cpp_row = dict(panel[ts])
            for code, real in bot_encoded.items():
                if code in cpp_row:
                    cpp_row.setdefault(real, cpp_row[code])
            bot_row = dict(bot[ts])
            for code, real in bot_encoded.items():
                if real in bot_row:
                    bot_row.setdefault(code, bot_row[real])
            ref_cpp = reference_decision(cpp_row)
            ref_bot = reference_decision(bot_row)
            if live is None:
                print(f"{iso(ts):<18}{clean:<10}{'C++ [AGDEC] ABSENT':<26}"
                      f"{'--':>6}{'--':>6}{'--':>10}{'--':>8}")
            else:
                print(f"{iso(ts):<18}{clean:<10}{'(a) C++ core, live':<26}"
                      f"{live['fired']:>6}{live['side']:>6}"
                      f"{str(live['n_long']) + 'L/' + str(live['n_short']) + 'S':>10}"
                      f"{live['n_trig']:>8}")
            for label, r in (("(b) reference / C++ panel", ref_cpp),
                             ("(c) reference / BOT feats", ref_bot)):
                print(f"{'':<18}{'':<10}{label:<26}"
                      f"{r['fired']:>6}{r['side']:>6}"
                      f"{str(r['n_long']) + 'L/' + str(r['n_short']) + 'S':>10}"
                      f"{r['n_trig']:>8}"
                      f"   gate let {r['predicted']}/{len(stack_rows)} through")
            # (d) THE BOT'S OWN VOTE, from its own log. This is the only source
            # in this table that is not something this script computed, and it
            # is what grades the four transcribed lines of make_decision against
            # what the production bot actually did.
            bd = bot_dec.get(ts)
            if args.bridge_log is None:
                pass
            elif bd is None:
                print(f"{'':<18}{'':<10}{'(d) BOT log, own vote':<26}"
                      f"{0:>6}{0:>6}{'0L/0S':>10}{0:>8}   (no line: the bot logs "
                      f"only when a regime fires)")
                bd = {"n_long": 0, "n_short": 0, "regimes": []}
            else:
                net_d = bd["n_long"] - bd["n_short"]
                side_d = (0 if net_d == 0
                          else (1 if net_d * gate["reverse"] > 0 else -1))
                print(f"{'':<18}{'':<10}{'(d) BOT log, own vote':<26}"
                      f"{int(side_d != 0):>6}{side_d:>6}"
                      f"{str(bd['n_long']) + 'L/' + str(bd['n_short']) + 'S':>10}"
                      f"{bd['n_long'] + bd['n_short']:>8}")
            if bd is not None and (bd["n_long"], bd["n_short"]) != \
                    (ref_bot["n_long"], ref_bot["n_short"]):
                bot_log_diff += 1
                print(f"{'':<18}  TRANSCRIPTION CHECK FAILED: the bot LOGGED "
                      f"{bd['n_long']}L/{bd['n_short']}S but the reference path "
                      f"over the bot's own features gives {ref_bot['n_long']}L/"
                      f"{ref_bot['n_short']}S. The four lines transcribed from "
                      f"make_decision do NOT reproduce the bot.")
                ref_voters = sorted(r for r, (_y, v) in ref_bot["preds"].items() if v)
                log_voters = sorted(r for r, _p in bd["regimes"])
                print(f"{'':<20}bot logged : {log_voters}")
                print(f"{'':<20}reference  : {ref_voters}")
                print(f"{'':<20}only in log: {sorted(set(log_voters) - set(ref_voters))}")
                print(f"{'':<20}only in ref: {sorted(set(ref_voters) - set(log_voters))}")
            if live is not None:
                checked += 1
                if not same(live, ref_cpp):
                    dec_engine_diff += 1
                    print(f"{'':<18}  *** PORT DIVERGENCE: the C++ decision differs "
                          f"from the reference over the SAME panel row. ***")
                    for name, (y, v) in sorted(ref_cpp["preds"].items()):
                        got = agpred.get(ts, {}).get(name)
                        if got is None and v == 0:
                            continue
                        gy = got["y_pred"] if got else float("nan")
                        gv = got["vote"] if got else 0
                        if gv != v or reldiff(gy, y) > 1e-9:
                            print(f"{'':<20}{name:<34} C++ y={fmt(gy)} vote={gv} | "
                                  f"ref y={fmt(y)} vote={v}")
            if not same(ref_cpp, ref_bot):
                feed_diff += 1
                tag = ("EXPECTED — this bar's inputs already diverged (STEP 1)"
                       if ts in input_divergent else
                       "on a bar whose INPUTS agreed; see STEP 1b/3 for the "
                       "rolling-window contamination")
                print(f"{'':<18}  FEED DIVERGENCE: same code, different features — {tag}")

        print()
        print(f"  (a) vs (b)  PORT : {dec_engine_diff} divergence(s) over {checked} "
              f"bar(s) with a live [AGDEC] line")
        print(f"  (b) vs (c)  FEED : {feed_diff} divergence(s) over {len(use)} bar(s)")
        if args.bridge_log is not None:
            print(f"  (c) vs (d)  TRANSCRIPTION : {bot_log_diff} divergence(s) — the "
                  f"reference path re-run over the bot's features vs what the BOT "
                  f"ITSELF logged")
        else:
            print("  (c) vs (d)  TRANSCRIPTION : NOT CHECKED (--bridge-log absent). "
                  "The four transcribed lines of make_decision are UNGRADED against "
                  "the bot's own record.")
        if checked == 0:
            print("  NOTE: no shared bar carried an [AGDEC] line, so the PORT "
                  "comparison is UNMEASURED — not passed.")
        # Only 9 of the 62 deployed regimes can fire at all (PR #532), so a run
        # in which nothing votes is the EXPECTED outcome and must not be read as
        # agreement. Said explicitly, because "0 divergences" over 0 votes is
        # the most convincing-looking empty result in this whole file.
        votes = sum(agdec[t]["n_trig"] for t in use if t in agdec)
        if votes == 0:
            print("  NOTE: NOT ONE regime voted on any reconciled bar. Only 9 of "
                  "the 62 deployed regimes can fire at all, so this is common — "
                  "but it means the vote comparison above is VACUOUS on these "
                  "bars: it agreed about nothing happening.")

    print()
    print("=" * 78)
    if cf_fail == 0:
        print(f"RESULT: ENGINE EXONERATED — over the PURE BINANCE window the engine matches "
              f"the LIVE BOT on all {len(shared_cols)} columns (STEP 5).")
        if diverge:
            print(f"        The {len(diverge)} live differences (STEP 3) are attributable to the "
                  f"tick-built bars listed in STEP 1b, not to the port.")
    elif diverge and cf_fail is None:
        print(f"RESULT: {len(diverge)} column(s) DIVERGE LIVE (STEP 3). Cause UNDECIDED: rerun "
              f"with --counterfactual-driver, which is the only step that can attribute them.")
    elif diverge:
        print(f"RESULT: {cf_fail} ENGINE DIVERGENCE(S) on identical Binance inputs — see STEP 5.")
    elif not clean_bars:
        print("RESULT: INCONCLUSIVE — no bar had clean inputs on both sides, so no "
              "feature difference can be attributed to the engine.")
    else:
        print(f"RESULT: NO ENGINE DIVERGENCE over {len(clean_bars)} clean-input bar(s), "
              f"{len(rows)} columns.")
    if dec_engine_diff:
        print(f"DECISION: {dec_engine_diff} PORT divergence(s) in STEP 6 — the C++ "
              f"decided differently from the reference over ITS OWN panel row.")
    elif dec_engine_diff == 0:
        print("DECISION: no PORT divergence in STEP 6 — over its own panel row the "
              "C++ decided exactly what the reference path decided.")
    else:
        print("DECISION: STEP 6 did not run (--raw-weights / --marvel-root absent, "
              "or no gate in the log) — the decision is UNRECONCILED, not agreed.")
    print("=" * 78)
    if tmp:
        tmp.cleanup()
    if dec_engine_diff:
        # A decision difference on identical features is unattributable to any
        # feed and is the only failure this whole file exists to find.
        return 1
    if cf_fail is not None:
        # STEP 5 is the authoritative verdict when it ran: it is the only
        # comparison in this file made on inputs both sides provably share.
        return 1 if cf_fail else 0
    # Input divergence is a property of the two FEEDS. Reported loudly, never
    # fatal: an exit code that goes red on something this port cannot fix is an
    # exit code everyone learns to ignore.
    return 1 if diverge or not clean_bars else 0


if __name__ == "__main__":
    sys.exit(main())
