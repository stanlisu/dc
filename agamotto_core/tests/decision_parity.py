#!/usr/bin/env python3
"""PHASE 5 gate: the REFERENCE decision path on one side, the C++ ``Decision`` on
the other, every regime x every row, on the SAME engineered panel.

WHAT THE REFERENCE IS, AND WHAT IT IS NOT
-----------------------------------------
Three imported functions, none of them re-implemented here:

  * ``gauntlet.thresholds.signed_threshold`` — where the SIGN lives. Long gets
    ``C+T``, short gets ``C-T``. This harness never spells that rule out; it
    calls the function marvel calls.
  * ``gauntlet.thresholds.read_threshold`` / ``read_threshold_center`` — the
    per-leg width and centre, read from the DEPLOYED ``setting.json`` with the
    same required-key discipline production uses (a missing key raises; a
    sub-floor width raises).
  * ``agamotto.trading.dual_gate_filter`` — where a regime row FIRES. It is the
    live bot's own function, handed a frame shaped exactly as ``predict()``
    builds one (``position`` / ``prediction`` / ``opt_threshold``), and it is
    what splits the votes into longs and shorts.

The vote arithmetic that follows (``net = long_count - short_count``,
``final_qty = base_size * net * reverse``, then ``side = sign(qty)``) is
transcribed here in four lines from ``trading.py::make_decision:846-864`` and
``knull/orb_bridge.py::_decisions_to_signals:155-169``, because those lines sit
inside a method that also fetches klines, prices symbols and rounds lot sizes —
there is no importable function to call. They are the only re-implemented lines
in this file and they are quoted in full at ``reference_decisions``.

THE PREDICTIONS ARE THE C++'s OWN, AND THAT IS DELIBERATE
---------------------------------------------------------
``tests/model_parity.py`` already grades the C++ ``y_pred`` against the DEPLOYED
sklearn pipeline at 1e-9 over 62 regimes x 699 rows x 5 scenarios. Re-grading it
here would fold that tolerance into a comparison that is supposed to be about
the GATE. So this gate takes the driver's per-regime predictions and asks one
question: given these numbers, does the reference decide what the C++ decided?

It is not circular. The C++ side's decision comes from ``decision_rule.cpp``;
the reference side's comes from ``dual_gate_filter`` plus four transcribed
lines. Neither can see the other, and a wrong comparison operator, a dropped
centre, a swapped leg or an inverted REVERSE changes one and not the other.

A driver that emitted MADE-UP y_pred would agree with itself here — so it is
``tests/model_parity.py``, not this gate, that proves those numbers are the
deployed model's, and the two must be run together. ``run_decision_parity.sh``
does exactly that, and refuses to report a green decision gate over predictions
that were never graded.

A DECISION IS A BOOLEAN, NOT A MEASUREMENT
------------------------------------------
``fired``, ``side``, ``n_triggered``, ``n_long``, ``n_short`` and ``net_count``
are compared EXACTLY. There is no tolerance on a side: a bar is long or it is
not, and "off by one vote" is a different trade. Only ``y_pred`` — a REPORTING
field — carries the 1e-9 tolerance, and it is graded against the reference's
prediction for the regime the C++ says it picked, so the tolerance never touches
anything that could change a trade.

WHAT ELSE IS ASSERTED, SO THE GATE CANNOT PASS TRIVIALLY
--------------------------------------------------------
  * the C++ floor constant EQUALS ``gauntlet.thresholds.ABS_THRESH_FLOOR``
    (the C++ cannot import it; this makes its copy a GATED one);
  * the C++ leg EDGES equal ``signed_threshold``'s, bit for bit;
  * both legs actually VOTE somewhere across the suite, and at least one bar
    FIRES and at least one does not — a rule that decided FLAT on every row
    would otherwise match a reference that did the same, forever;
  * every reported ``y_pred`` genuinely clears its reported ``centre +/- width``
    on its reported leg, and the reported regime is on the MAJORITY leg;
  * the reported ``|y_pred|`` is the largest among the REFERENCE's majority-leg
    voters, so the representative-regime rule is graded too — not just the
    number it produced.
"""
from __future__ import annotations

import argparse
import io
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DC_ROOT = HERE.parents[1]
sys.path.insert(0, str(DC_ROOT / "agamotto_pkg" / "src"))
sys.path.insert(0, str(HERE))

# The scenario generator is REUSED from the feature gate rather than
# re-implemented, for the same reason model_parity.py reuses it: two generators
# would drift, and the drift would be invisible.
import feature_parity as fp  # noqa: E402

PANEL_BARS = fp.PANEL_BARS
DEFAULT_STACK = HERE / "regime_stack_deployed.csv"

# The three trailing vol-quantile atoms. Every regime carrying one is inert live
# (marvel PR #532), so only the other nine can ever cast a vote.
VOL_Q_ATOM_CODES = {73, 74, 75}

# The two price scales the suite runs at. Every scenario is run at BOTH: the
# feature engine's inline epsilons are absolute, so they are invisible at 64000
# and a 6th-significant-figure effect at 0.0045 — and a decision is a comparison
# against a FIXED gate, so a scale that shifts y_pred shifts which side of the
# gate it lands on.
PRICE_SCALES = (64000.0, 0.0045)


def load_stack(path: Path) -> list[tuple[str, str]]:
    df = pd.read_csv(path)
    for col in ("regime", "position"):
        if col not in df.columns:
            raise SystemExit(f"{path}: no '{col}' column — is this a regime stack?")
    rows = [(str(r.regime), str(r.position)) for r in df.itertuples()
            if not str(r.regime).startswith("__")]
    if not rows:
        raise SystemExit(f"{path}: no regimes after dropping __summary__ rows")
    return rows


def stack_edges(path: Path, float_precision: str | None) -> dict[str, set]:
    """The DEPLOYED ``optimal_threshold`` column, per leg.

    Read so the gate this harness computes from setting.json can be checked
    against the number the LIVE BOT actually compares against — the two are
    independent expressions of `C +/- T` and nothing else in the system compares
    them.

    ``float_precision`` is a PARAMETER because the two readings differ:
    ``round_trip`` recovers the value the file literally spells, while
    pandas' DEFAULT parser is what ``utils.lib.load_regime_stack`` — and
    therefore the live bot — actually gets. See the ULP finding in main().
    """
    df = pd.read_csv(path, float_precision=float_precision)
    df = df[~df["regime"].astype(str).str.startswith("__")]
    out = {}
    for pos in ("long", "short"):
        rows = df[df["position"] == pos]
        out[pos] = set(float(v) for v in rows["optimal_threshold"])
    return out


def atoms_of(coded_name: str) -> list[int]:
    """`r029_and_r001_and_r073_long` -> [29, 1, 73]. Pure structure, no names."""
    body = coded_name.strip().lower()
    for suffix in ("_long", "_short"):
        if body.endswith(suffix):
            body = body[: -len(suffix)]
            break
    out = []
    for part in body.split("_and_"):
        part = part.strip()
        if not (part.startswith("r") and part[1:].isdigit()):
            raise SystemExit(
                f"regime {coded_name!r} has non-coded atom {part!r}. The core takes "
                "CODES; a stack carrying real names must be encoded first.")
        out.append(int(part[1:]))
    return out


def spec_arg(stack: list[tuple[str, str]]) -> str:
    parts = []
    for name, pos in stack:
        codes = ".".join(str(a) for a in atoms_of(name))
        parts.append(f"{codes}:{'L' if pos == 'long' else 'S'}")
    return ",".join(parts)


def run_driver(driver_cmd: str, weights: Path, raw: pd.DataFrame, regimes: str,
               gate: dict, extra: list[str] | None = None):
    """Feed the raw panel in; read the gate echo, the preds and the decisions."""
    buf = io.StringIO()
    raw.to_csv(buf, index=False, float_format="%.17g", na_rep="nan")
    cmd = (shlex.split(driver_cmd)
           + ["--weights", str(weights), "--regimes", regimes,
              "--threshold-long", repr(gate["threshold_long"]),
              "--threshold-short", repr(gate["threshold_short"]),
              "--center-long", repr(gate["threshold_center_long"]),
              "--center-short", repr(gate["threshold_center_short"]),
              "--reverse", str(gate["reverse"])]
           + (extra or []))
    proc = subprocess.run(cmd, input=buf.getvalue(), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"driver failed (rc={proc.returncode}):\n{proc.stderr}")
    lines = proc.stdout.splitlines()
    try:
        i_gate = lines.index("#gate")
        i_panel = lines.index("#panel")
        i_preds = lines.index("#preds")
        i_dec = lines.index("#decisions")
        i_meta = lines.index("#meta")
    except ValueError:
        raise SystemExit(
            "driver output has no #gate/#panel/#preds/#decisions/#meta sections — "
            f"refusing to guess its layout:\n{proc.stdout[:400]}")

    g = lines[i_gate + 1].split()
    echo = {"threshold_long": float(g[0]), "threshold_short": float(g[1]),
            "threshold_center_long": float(g[2]), "threshold_center_short": float(g[3]),
            "reverse": int(g[4]), "edge_long": float(g[5]), "edge_short": float(g[6])}

    panel_lines = lines[i_panel + 1:i_preds]
    panel = pd.DataFrame(
        [[float(x) for x in ln.split(",")] for ln in panel_lines[1:]],
        columns=panel_lines[0].split(","))

    pred_lines = [ln for ln in lines[i_preds + 1:i_dec] if ln.strip()]
    preds = pd.DataFrame(
        [[float(x) for x in ln.split(",")] for ln in pred_lines[1:]],
        columns=pred_lines[0].split(","))

    dec_lines = [ln for ln in lines[i_dec + 1:i_meta] if ln.strip()]
    dec = pd.DataFrame([ln.split(",") for ln in dec_lines[1:]],
                       columns=dec_lines[0].split(","))
    for c in ("row", "fired", "side", "n_triggered", "n_long", "n_short",
              "net_count", "winning_index"):
        dec[c] = dec[c].astype(int)
    for c in ("y_pred", "threshold", "threshold_center"):
        dec[c] = dec[c].astype(float)

    meta = {}
    for ln in lines[i_meta + 1:]:
        if not ln.strip():
            continue
        spec, dirname, n_feat = ln.split()
        meta[dirname] = {"spec": spec, "n_features": int(n_feat)}
    return echo, panel, preds, dec, meta


def reference_decisions(stack, preds: pd.DataFrame, gates: dict,
                        reverse: int) -> pd.DataFrame:
    """The REFERENCE decision for every row of one scenario.

    THE FIRING RULE IS IMPORTED, NOT WRITTEN. ``agamotto.trading.dual_gate_filter``
    is the live bot's own function; it is handed a frame in exactly the shape
    ``AgamottoTrading.predict`` builds — one row per (regime, bar) carrying
    ``position``, ``prediction`` and ``opt_threshold`` — and it returns the
    firing longs and the firing shorts. The deployed stack carries neither
    ``opt_threshold_2bar`` nor ``prediction_2bar``, so ``has_dual`` is False and
    the reference takes its 1-bar branch, which is what live does.

    ``opt_threshold`` is the SIGNED EDGE and is produced by the imported
    ``signed_threshold``, never spelled out here.

    THE FOUR TRANSCRIBED LINES. ``make_decision`` has no importable core, so its
    arithmetic is reproduced verbatim from trading.py:846-864:

        long_count = len(longs)
        short_count = len(shorts)
        net_count  = long_count - short_count
        final_qty  = base_size * net_count * reverse

    and the side comes from knull/orb_bridge.py:156/168:

        if abs(target_qty) < 1e-9 or price <= 0:  FLAT
        side = "LONG" if target_qty > 0 else "SHORT"

    ``base_size`` is ``CAPITAL / price`` — strictly positive — so it is carried
    as the positive constant it is rather than invented: it cannot change a
    sign, and pretending it can would be modelling the wrong thing.
    """
    from agamotto.trading import dual_gate_filter  # noqa: PLC0415  THE reference

    n_rows = len(preds)
    frames = []
    for i, (name, pos) in enumerate(stack):
        y = preds.iloc[:, i].to_numpy(dtype=float)
        # The reference never predicts a row its filter rejected — predict() is
        # handed filtered_signals — so a non-fired row is ABSENT from the frame,
        # not present with a NaN. Same for a row sklearn refuses (an inf in a
        # selected column makes trading.py:744 log and return an empty frame).
        keep = np.isfinite(y)
        if not keep.any():
            continue
        frames.append(pd.DataFrame({
            "row": np.flatnonzero(keep),
            "regime": name,
            "position": pos,
            "prediction": y[keep],
            "opt_threshold": gates[pos].edge,
        }))

    fired = np.zeros(n_rows, dtype=bool)
    side = np.zeros(n_rows, dtype=int)
    n_long = np.zeros(n_rows, dtype=int)
    n_short = np.zeros(n_rows, dtype=int)
    best_abs = np.full(n_rows, np.nan, dtype=float)

    if frames:
        sym_preds = pd.concat(frames, ignore_index=True)
        longs, shorts = dual_gate_filter(sym_preds)       # <- THE reference call
        lc = longs.groupby("row").size()
        sc = shorts.groupby("row").size()
        n_long[lc.index.to_numpy()] = lc.to_numpy()
        n_short[sc.index.to_numpy()] = sc.to_numpy()

        # trading.py:859-864 + orb_bridge.py:156/168. base_size = CAPITAL/price
        # and is strictly positive, so the qty's sign IS this product's.
        net_count = n_long - n_short
        base_size = 1.0                                    # any positive constant
        final_qty = base_size * net_count * reverse
        flat = np.abs(final_qty) < 1e-9
        side = np.where(flat, 0, np.where(final_qty > 0, 1, -1)).astype(int)
        fired = side != 0

        # The largest |prediction| among the MAJORITY leg's voters — the
        # representative-regime rule, computed from the REFERENCE's own firing
        # frames so the C++'s choice is graded rather than echoed.
        majority = np.where(net_count > 0, "long",
                            np.where(net_count < 0, "short", ""))
        voters = pd.concat([longs, shorts], ignore_index=True)
        if len(voters):
            voters = voters.assign(a=voters["prediction"].abs())
            maj_of_row = pd.Series(majority, index=np.arange(n_rows))
            keep = voters["position"].to_numpy() == maj_of_row.reindex(
                voters["row"].to_numpy()).to_numpy()
            tie = maj_of_row.reindex(voters["row"].to_numpy()).to_numpy() == ""
            sel = voters[keep | tie]
            if len(sel):
                m = sel.groupby("row")["a"].max()
                best_abs[m.index.to_numpy()] = m.to_numpy()
    else:
        net_count = np.zeros(n_rows, dtype=int)

    return pd.DataFrame({
        "fired": fired, "side": side, "n_long": n_long, "n_short": n_short,
        "n_triggered": n_long + n_short, "net_count": net_count,
        "best_abs": best_abs,
    })


def compare(scenario: str, stack, cpp: pd.DataFrame, ref: pd.DataFrame,
            preds: pd.DataFrame, gates: dict, tol: float, seen: dict) -> int:
    """FAILURE count for one scenario (0 == pass)."""
    print(f"\n--- scenario: {scenario} ---")
    failures = 0
    if len(cpp) != len(ref):
        print(f"=== FAIL: {len(cpp)} C++ decision rows vs {len(ref)} reference ===")
        return 1

    # ---- EXACT: everything that could change a trade ----------------------
    for col in ("fired", "side", "n_triggered", "n_long", "n_short", "net_count"):
        a = cpp[col].to_numpy()
        b = ref[col].to_numpy().astype(a.dtype)
        bad = np.flatnonzero(a != b)
        if bad.size:
            failures += 1
            first = int(bad[0])
            print(f"=== FAIL: {col} differs on {bad.size}/{len(a)} row(s); "
                  f"first at row {first}: C++={a[first]} reference={b[first]} ===")
            print(f"    (a decision is a boolean, not a measurement — there is no "
                  f"tolerance on {col})")

    # ---- the REPORTING fields --------------------------------------------
    # y_pred is graded against the prediction of the regime the C++ SAYS it
    # picked, so the 1e-9 never touches anything that could change a trade.
    win = cpp["winning_index"].to_numpy()
    have = win >= 0
    if have.any():
        rows = np.flatnonzero(have)
        got = cpp["y_pred"].to_numpy()[rows]
        want = preds.to_numpy(dtype=float)[rows, win[rows]]
        denom = np.maximum(np.maximum(np.abs(got), np.abs(want)), 1e-12)
        rel = np.abs(got - want) / denom
        bad = np.flatnonzero(rel > tol)
        if bad.size:
            failures += 1
            print(f"=== FAIL: y_pred does not match the reported regime's own "
                  f"prediction on {bad.size} row(s), max rel {rel.max():.3e} ===")

        # SELF-CONSISTENCY: the reported triple must explain itself. A gate that
        # reported a y_pred from one leg with the other leg's width would pass
        # every count above and be unreadable in the log.
        thr = cpp["threshold"].to_numpy()[rows]
        ctr = cpp["threshold_center"].to_numpy()[rows]
        side = cpp["side"].to_numpy()[rows]
        rev = 1 if gates["reverse"] > 0 else -1
        maj = side * rev            # the leg the representative regime is on
        clears = np.where(maj > 0, got > ctr + thr,
                          np.where(maj < 0, got < ctr - thr, True))
        bad = np.flatnonzero(~clears)
        if bad.size:
            failures += 1
            print(f"=== FAIL: on {bad.size} row(s) the reported y_pred does NOT "
                  f"clear the reported centre +/- width on the reported leg ===")

        # And the SELECTION rule itself, against the reference's voters.
        ref_best = ref["best_abs"].to_numpy()[rows]
        ok = np.isfinite(ref_best)
        d = np.abs(np.abs(got[ok]) - ref_best[ok])
        scale = np.maximum(np.abs(ref_best[ok]), 1e-12)
        bad = np.flatnonzero(d / scale > tol)
        if bad.size:
            failures += 1
            print(f"=== FAIL: the representative regime is NOT the largest "
                  f"|y_pred| among the reference's majority-leg voters on "
                  f"{bad.size} row(s) ===")
    else:
        print("  no row on this scenario produced a vote")

    # ---- non-triviality bookkeeping (asserted once, over the whole suite) --
    seen["fired"] += int(cpp["fired"].sum())
    seen["flat"] += int((~cpp["fired"].astype(bool)).sum())
    seen["long"] += int((cpp["side"] == 1).sum())
    seen["short"] += int((cpp["side"] == -1).sum())
    seen["votes_long"] += int(cpp["n_long"].sum())
    seen["votes_short"] += int(cpp["n_short"].sum())
    seen["rows"] += len(cpp)

    if not failures:
        print(f"  decisions: {len(cpp)} rows EXACT on fired/side/n_triggered/"
              f"n_long/n_short/net_count")
        print(f"             fired={int(cpp['fired'].sum())} "
              f"(long={int((cpp['side'] == 1).sum())} "
              f"short={int((cpp['side'] == -1).sum())}), "
              f"votes L={int(cpp['n_long'].sum())} S={int(cpp['n_short'].sum())}")
        last = cpp.iloc[-1]
        print(f"  newest row (the ONLY row live decides): fired={int(last['fired'])} "
              f"side={int(last['side'])} n_trig={int(last['n_triggered'])} "
              f"({int(last['n_long'])}L/{int(last['n_short'])}S) "
              f"y_pred={last['y_pred']:.9g}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--weights", required=True,
                    help="the export_agamotto_sentinel_weights.py OUTPUT dir")
    ap.add_argument("--marvel-root", required=True,
                    help="marvel checkout, for gauntlet.thresholds — THE definition "
                         "of the gate. Not reimplemented here")
    ap.add_argument("--setting", default=None,
                    help="the DEPLOYED setting.json the gate is read from "
                         "(default: <marvel>/gauntlet/pred_agamotto.base.15m_1/"
                         "setting.json)")
    ap.add_argument("--stack", default=str(DEFAULT_STACK))
    ap.add_argument("--tol", type=float, default=1e-9)
    # --- negative-control overrides. Each one is a DIFFERENT gate, and the run
    # must go RED. They are flags rather than source mutants because what they
    # must prove absent is a whole missing PROPERTY of the gate (its centre, its
    # per-leg-ness, its direction), not a wrong operator.
    ap.add_argument("--zero-center", action="store_true",
                    help="NEGATIVE CONTROL: drop the centre (the pre-2026-08-08 "
                         "zero-centred gate)")
    ap.add_argument("--swap-legs", action="store_true",
                    help="NEGATIVE CONTROL: give the long leg the short width and "
                         "vice versa")
    ap.add_argument("--invert-reverse", action="store_true",
                    help="NEGATIVE CONTROL: flip REVERSE")
    ap.add_argument("--force-threshold", type=float, default=None,
                    help="NEGATIVE CONTROL: force both widths to this value; used "
                         "with a sub-floor number, the DRIVER must REFUSE it")
    ap.add_argument("--expect-refusal", action="store_true",
                    help="the driver must exit non-zero naming the floor; a run "
                         "that produced decisions is the failure")
    ap.add_argument("--expect-red", action="store_true",
                    help="invert the exit code: for the negative controls, where "
                         "a PASS is the failure")
    args = ap.parse_args()

    marvel = Path(args.marvel_root).expanduser().resolve()
    if not (marvel / "gauntlet" / "thresholds.py").is_file():
        raise SystemExit(
            f"--marvel-root {marvel} has no gauntlet/thresholds.py. That module IS "
            "the definition of the gate; refusing to grade against a "
            "reimplementation of it.")
    sys.path.insert(0, str(marvel))

    # THE readers and THE sign rule. Imported, never re-declared — CLAUDE.md.
    # LEG_KEYS / LEG_CENTER_KEYS are the per-leg setting.json key names. They
    # are read here rather than spelled out so this harness cannot drift from
    # the reader: if marvel renames THRESHOLD_LONG, the printed provenance
    # follows and this line stops compiling rather than reporting a stale name.
    from gauntlet.thresholds import (  # noqa: E402,PLC0415
        ABS_THRESH_FLOOR, LEG_CENTER_KEYS, LEG_KEYS, read_leg_gates,
        read_threshold, read_threshold_center, signed_threshold,
    )

    setting_path = Path(args.setting) if args.setting else (
        marvel / "gauntlet" / "pred_agamotto.base.15m_1" / "setting.json")
    if not setting_path.is_file():
        raise SystemExit(f"setting.json not found: {setting_path}")
    with setting_path.open() as fh:
        setting = json.load(fh)

    src = f"setting.json({setting_path.name})"
    gate = {
        "threshold_long": read_threshold(setting, source=src, position="long"),
        "threshold_short": read_threshold(setting, source=src, position="short"),
        "threshold_center_long": read_threshold_center(setting, source=src,
                                                       position="long"),
        "threshold_center_short": read_threshold_center(setting, source=src,
                                                        position="short"),
        # trading.py:834 — `int(self.config.get("REVERSE", 1))`, 1 = identity.
        "reverse": int(setting["REVERSE"]) if "REVERSE" in setting else 1,
    }
    gates = read_leg_gates(setting, source=src)   # THE LegGate pair, imported

    # *** THE CONTROLS PERTURB ONE SIDE ONLY. ***
    # `gate` stays the DEPLOYED gate and is what the REFERENCE side uses;
    # `driver_gate` is what the C++ is handed. Perturbing both would hand the
    # two sides the same wrong number, they would agree, and the control would
    # report green over a gate that is not being watched at all — which is
    # exactly what the first draft of this harness did.
    driver_gate = dict(gate)
    label = "the DEPLOYED gate"
    if args.zero_center:
        driver_gate["threshold_center_long"] = 0.0
        driver_gate["threshold_center_short"] = 0.0
        label = "NEGATIVE CONTROL: the C++ is given a ZERO-CENTRED gate"
    if args.swap_legs:
        driver_gate["threshold_long"], driver_gate["threshold_short"] = (
            driver_gate["threshold_short"], driver_gate["threshold_long"])
        label = "NEGATIVE CONTROL: the C++ is given the legs SWAPPED"
    if args.invert_reverse:
        driver_gate["reverse"] = -driver_gate["reverse"]
        label = "NEGATIVE CONTROL: the C++ is given REVERSE INVERTED"
    if args.force_threshold is not None:
        driver_gate["threshold_long"] = args.force_threshold
        driver_gate["threshold_short"] = args.force_threshold
        label = (f"NEGATIVE CONTROL: the C++ is given both widths as "
                 f"{args.force_threshold!r}")

    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_dir():
        raise SystemExit(f"not a directory: {weights}")
    stack = load_stack(Path(args.stack))
    regimes = spec_arg(stack)
    inert = [n for n, _p in stack if set(atoms_of(n)) & VOL_Q_ATOM_CODES]
    firable = [n for n, _p in stack if not (set(atoms_of(n)) & VOL_Q_ATOM_CODES)]

    print(f"[decision_parity] pandas {pd.__version__}  numpy {np.__version__}")
    print(f"[decision_parity] PANEL_BARS={PANEL_BARS}")
    print(f"[decision_parity] setting:  {setting_path}")
    print(f"[decision_parity] keys:     "
          f"{LEG_KEYS['long']}/{LEG_KEYS['short']} + "
          f"{LEG_CENTER_KEYS['long']}/{LEG_CENTER_KEYS['short']} + REVERSE "
          f"(names imported from gauntlet.thresholds, never spelled here)")
    print(f"[decision_parity] stack:    {args.stack} ({len(stack)} regimes, "
          f"{len(inert)} inert / {len(firable)} firable)")
    print(f"[decision_parity] weights:  {weights}")
    print(f"[decision_parity] driver:   {args.driver}")
    print(f"[decision_parity] gate:     {label}")
    print(f"                  reference long  T={gate['threshold_long']!r} "
          f"C={gate['threshold_center_long']!r}")
    print(f"                  reference short T={gate['threshold_short']!r} "
          f"C={gate['threshold_center_short']!r}  reverse={gate['reverse']}")
    if driver_gate != gate:
        print(f"                  C++       long  T={driver_gate['threshold_long']!r} "
              f"C={driver_gate['threshold_center_long']!r}")
        print(f"                  C++       short T={driver_gate['threshold_short']!r} "
              f"C={driver_gate['threshold_center_short']!r}  "
              f"reverse={driver_gate['reverse']}")

    # ---- THE REFUSE-AT-LOAD CONTROL --------------------------------------
    if args.expect_refusal:
        raw = fp.make_panel(PANEL_BARS, 64000.0, 20260819, False, None, False)
        try:
            run_driver(args.driver, weights, raw, regimes, driver_gate)
        except SystemExit as e:
            msg = str(e)
            if "floor" in msg or "2 bps" in msg:
                print(f"\n=== REFUSED AT LOAD, as required ===\n{msg.strip()[:400]}")
                return 0
            print(f"\n=== FAIL: the driver failed, but NOT on the floor ===\n{msg}")
            return 1
        print("\n=== FAIL: a sub-floor gate was ACCEPTED and produced decisions. "
              "CLAUDE.md: |threshold| >= 2 bps, refused not clamped. ===")
        return 1

    failures = 0

    # ---- THE FLOOR AND THE SIGN RULE, GRADED AGAINST MARVEL ---------------
    floor_cmd = shlex.split(args.driver) + ["--print-floor"]
    got = subprocess.run(floor_cmd, capture_output=True, text=True)
    if got.returncode != 0:
        print(f"=== FAIL: --print-floor failed: {got.stderr} ===")
        failures += 1
    else:
        cpp_floor = float(got.stdout.strip())
        if cpp_floor != ABS_THRESH_FLOOR:
            failures += 1
            print(f"=== FAIL: the C++ floor is {cpp_floor!r}, marvel's "
                  f"ABS_THRESH_FLOOR is {ABS_THRESH_FLOOR!r} ===")
        else:
            print(f"\n  floor: C++ kAbsThreshFloor == "
                  f"gauntlet.thresholds.ABS_THRESH_FLOOR == {ABS_THRESH_FLOOR!r}")

    # The stack's own optimal_threshold column, as an independent witness of the
    # same two numbers. Reported, and asserted only on the UNMODIFIED gate — a
    # negative control is SUPPOSED to disagree with the deployed stack.
    if label == "the DEPLOYED gate":
        exact = stack_edges(Path(args.stack), "round_trip")
        as_bot_reads = stack_edges(Path(args.stack), None)
        for pos in ("long", "short"):
            want = signed_threshold(gate[f"threshold_{pos}"], pos,
                                    center=gate[f"threshold_center_{pos}"])
            if exact[pos] != {want}:
                failures += 1
                print(f"=== FAIL: signed_threshold gives {want!r} for {pos}, the "
                      f"DEPLOYED stack's optimal_threshold column spells "
                      f"{sorted(exact[pos])!r} ===")
            else:
                print(f"  {pos:5s} edge: signed_threshold -> {want!r} == every "
                      f"{pos} row of the deployed stack's optimal_threshold")
            # *** REFERENCE FINDING, REPORTED NOT FIXED. ***
            # utils.lib.load_regime_stack calls pd.read_csv with the DEFAULT
            # float parser, whose fast xstrtod path does not round-trip the last
            # digit. So the number the LIVE BOT compares against can differ from
            # the value the file spells — and therefore from `C +/- T` — by one
            # ULP. Measured here rather than assumed; it cannot move a decision
            # (a y_pred would have to land inside a 1e-19 window) but it means
            # `opt_threshold == signed_threshold(...)` is not an identity in
            # production, and anything that ever asserts it will be wrong.
            got = sorted(as_bot_reads[pos])
            if got != sorted(exact[pos]):
                print(f"  FINDING: pd.read_csv's DEFAULT float parser reads the "
                      f"{pos} optimal_threshold as {got!r}, not {sorted(exact[pos])!r} "
                      f"(delta {got[0] - sorted(exact[pos])[0]:.3g}, 1 ULP). "
                      f"utils.lib.load_regime_stack uses that default, so the LIVE "
                      f"BOT gates on the rounded value while this core computes "
                      f"C+/-T exactly. Reported, not fixed — the fix is "
                      f"float_precision='round_trip' in utils/lib.py.")

    seen = {"fired": 0, "flat": 0, "long": 0, "short": 0,
            "votes_long": 0, "votes_short": 0, "rows": 0}
    n_runs = 0
    for name, price, seed, holes, leads, safe_branch in fp.SCENARIOS:
        for scale in PRICE_SCALES:
            raw = fp.make_panel(PANEL_BARS, scale, seed, holes, leads, safe_branch)
            echo, panel, preds, dec, meta = run_driver(
                args.driver, weights, raw, regimes, driver_gate)
            n_runs += 1

            # The driver must be comparing against the gate it was HANDED, and
            # against the edges signed_threshold produces from it. Checked every
            # run: a driver that silently substituted its own would agree with
            # itself on every row.
            for k in ("threshold_long", "threshold_short", "threshold_center_long",
                      "threshold_center_short", "reverse"):
                if echo[k] != driver_gate[k]:
                    failures += 1
                    print(f"=== FAIL: the driver echoed {k}={echo[k]!r}, was handed "
                          f"{driver_gate[k]!r} ===")
            for pos, key in (("long", "edge_long"), ("short", "edge_short")):
                want = signed_threshold(driver_gate[f"threshold_{pos}"], pos,
                                        center=driver_gate[f"threshold_center_{pos}"])
                if echo[key] != want:
                    failures += 1
                    print(f"=== FAIL: the C++ {pos} edge is {echo[key]!r}, "
                          f"signed_threshold gives {want!r} — not bit-identical ===")

            # The REFERENCE's gate is built with the same imported
            # signed_threshold from the DEPLOYED values — never the driver's.
            ref_gates = {
                pos: gates[pos]._replace(
                    width=gate[f"threshold_{pos}"],
                    center=gate[f"threshold_center_{pos}"],
                    edge=signed_threshold(gate[f"threshold_{pos}"], pos,
                                          center=gate[f"threshold_center_{pos}"]))
                for pos in ("long", "short")}

            ref = reference_decisions(stack, preds, ref_gates, gate["reverse"])
            # `driver_gate` — NOT `gate` — for the self-consistency check: that
            # check asks whether the C++'s OWN reported triple explains itself,
            # which is a question about the gate the C++ was handed. The
            # fired/side comparison against the reference is what catches a
            # driver running on the wrong gate, and it is EXACT.
            failures += compare(f"{name}  @scale {scale:g}", stack, dec, ref,
                                preds, driver_gate, args.tol, seen)

    # ---- NON-TRIVIALITY ---------------------------------------------------
    # A rule that decided FLAT on every row would agree with a reference that did
    # the same, forever. Asserted, not printed.
    print(f"\n  suite: {n_runs} runs x {PANEL_BARS} rows = {seen['rows']} decisions; "
          f"fired={seen['fired']} flat={seen['flat']} "
          f"(long={seen['long']} short={seen['short']}); "
          f"votes L={seen['votes_long']} S={seen['votes_short']}")
    if not args.expect_red:
        if seen["fired"] == 0:
            failures += 1
            print("=== FAIL: NOTHING fired anywhere in the suite. A rule that "
                  "always decides flat matches a reference that does the same. ===")
        if seen["flat"] == 0:
            failures += 1
            print("=== FAIL: EVERY row fired — the gate is not gating. ===")
        if seen["votes_long"] == 0 or seen["votes_short"] == 0:
            failures += 1
            print(f"=== FAIL: one leg never voted (L={seen['votes_long']} "
                  f"S={seen['votes_short']}). A leg that cannot vote is a leg "
                  "whose comparison is ungraded. ===")

    print()
    if failures:
        print(f"=== DECISION PARITY FAILED ({failures} failure groups) ===")
        return 0 if args.expect_red else 1
    print(f"=== DECISION PARITY PASS: {len(stack)} regimes x {PANEL_BARS} rows x "
          f"{n_runs} runs ({len(fp.SCENARIOS)} scenarios x {len(PRICE_SCALES)} "
          f"price scales), fired/side EXACT, y_pred rel tol {args.tol:g} ===")
    return 1 if args.expect_red else 0


if __name__ == "__main__":
    sys.exit(main())
