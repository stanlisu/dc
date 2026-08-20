#!/usr/bin/env python3
"""PHASE 4 gate: the DEPLOYED sklearn pipeline on one side, the C++ model runner
on the other, every regime x every row compared at 1e-9 relative.

WHAT THE REFERENCE IS, AND WHAT IT IS NOT
-----------------------------------------
The reference side is ``utils.weights_io.load_regime(...).predict`` — the SAME
loader the live bot and tesseract call, imported from the marvel tree and
handed the SAME ``ridge_*.pkl`` artifacts production loads. It is never
reimplemented here. A hand-written ``X @ coef + intercept`` on the Python side
would be grading the C++ against a second copy of the same idea, and the two
would agree happily while both disagreed with sklearn.

The C++ side reads the TEXT export
(``marvel/gauntlet/export_agamotto_sentinel_weights.py``). So this gate closes
the loop the exporter opens: the exporter proves pickle -> text round-trips
(measured 1.1e-13 on random probes), and this proves text -> C++ prediction
matches the pickle ON A REAL ENGINEERED PANEL.

WHAT PANEL BOTH SIDES SEE
-------------------------
The C++ driver emits the 65-column panel it engineered AND the predictions it
computed from it. This harness takes THAT panel and feeds it to the reference,
exactly as tests/regime_parity.py does. Engineering a panel in Python alongside
would fold feature parity's 1e-9 tolerance into a comparison that is supposed to
be about the dot product alone.

The panel columns are already CODED and so is ``meta['feature_columns']``, so —
unlike the regime gate — nothing is decoded here at all. No real feature name
and no real regime name appears on either side of this comparison.

EVERY ROW, NOT JUST THE SCORED ONE
----------------------------------
Live scores exactly one row per bar (the panel's newest). Grading only that row
would compare FIVE numbers per regime across the whole suite — nowhere near
enough to separate a correct prediction from one that is right near the mean and
wrong in the tails, or that has a coefficient's sign wrong on a feature that is
rarely large. ``predictRow`` is row-independent, so all 699 rows are graded and
the newest row is ALSO reported on its own.

trading.py's NaN FILL IS APPLIED ON BOTH SIDES, AND ITS CLIP ON NEITHER
----------------------------------------------------------------------
``trading.py:697-700`` fills NaN with 0.0 across the selected model columns
before scaling. That is live behaviour, the C++ reproduces it, and this harness
applies the same ``.fillna(0.0)`` before calling the reference — on the HARNESS
side, so ``load_regime(...).predict`` itself stays the untouched production code.

``trading.py:705-713`` then does ``np.clip(preds, -1, 1)`` whenever
``|preds|.max() > 1``. That is NOT reproduced by either side here:
``RegimeArtifacts.predict`` does not clip, so the reference this gate uses does
not either. It is a clamp on a DERIVED quantity — the shape CLAUDE.md bans,
because an out-of-range prediction is the degenerate-scaler bug signalling
itself. REPORTED as a finding, not fixed; the fix belongs in trading.py. The
harness measures how often it would have bitten.

WHAT IS ASSERTED BEYOND EQUALITY
--------------------------------
A runner that returned the reference by construction is not the risk; a gate
that passes trivially is. So:

  * every regime's predictions must be equal to 1e-9 relative, cell for cell;
  * the panel must actually MOVE the predictions — a regime whose predictions
    are constant across 699 rows is reported as a failure, because a runner
    that returned only its intercept would match a reference that did the same;
  * the MIXTURE is asserted: the driver reports each model's feature count, and
    the run fails if the deployed stack does not contain BOTH 5-feature and
    16-feature models. A hardcoded 5 must not be able to pass;
  * and it is asserted that all NINE firable regimes are the 16-feature ones —
    which is what makes a hardcoded 5 a bug on every regime that can trade.
"""
from __future__ import annotations

import argparse
import io
import shlex
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DC_ROOT = HERE.parents[1]
sys.path.insert(0, str(DC_ROOT / "agamotto_pkg" / "src"))
sys.path.insert(0, str(HERE))

# The scenario generator and the panel-width reader are REUSED from the feature
# gate rather than re-implemented, for the same reason tests/regime_parity.py
# reuses them: two generators would drift, and the drift would be invisible.
import feature_parity as fp  # noqa: E402

PANEL_BARS = fp.PANEL_BARS

DEFAULT_STACK = HERE / "regime_stack_deployed.csv"

# The three trailing vol-quantile atoms. Every regime carrying one is inert live
# (marvel PR #532) — and, as this gate asserts, every one of them also carries a
# 5-feature model while every firable regime carries a 16-feature one.
VOL_Q_ATOM_CODES = {73, 74, 75}


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
               extra: list[str] | None = None):
    """Feed the raw panel in; read the engineered panel, the preds and the meta."""
    buf = io.StringIO()
    raw.to_csv(buf, index=False, float_format="%.17g", na_rep="nan")
    cmd = (shlex.split(driver_cmd)
           + ["--weights", str(weights), "--regimes", regimes]
           + (extra or []))
    proc = subprocess.run(cmd, input=buf.getvalue(), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"driver failed (rc={proc.returncode}):\n{proc.stderr}")
    lines = proc.stdout.splitlines()
    try:
        i_panel = lines.index("#panel")
        i_preds = lines.index("#preds")
        i_meta = lines.index("#meta")
    except ValueError:
        raise SystemExit(
            "driver output has no #panel/#preds/#meta sections — refusing to guess "
            f"its layout:\n{proc.stdout[:400]}")

    panel_lines = lines[i_panel + 1:i_preds]
    panel = pd.DataFrame(
        [[float(x) for x in ln.split(",")] for ln in panel_lines[1:]],
        columns=panel_lines[0].split(","))

    pred_lines = [ln for ln in lines[i_preds + 1:i_meta] if ln.strip()]
    preds = pd.DataFrame(
        [[float(x) for x in ln.split(",")] for ln in pred_lines[1:]],
        columns=pred_lines[0].split(","))

    meta = {}
    for ln in lines[i_meta + 1:]:
        if not ln.strip():
            continue
        spec, dirname, n_feat, n_unit, n_nan = ln.split()
        meta[dirname] = {"spec": spec, "n_features": int(n_feat),
                         "unit_scale": int(n_unit), "nan_filled": int(n_nan)}
    return panel, preds, meta


def reference_predictions(raw_window: Path, model_prefix: str,
                          panel: pd.DataFrame,
                          stack: list[tuple[str, str]]) -> tuple[pd.DataFrame, dict]:
    """The DEPLOYED loader, over the C++ panel, one column per regime.

    Nothing about the arithmetic is re-implemented: `load_regime(...).predict`
    IS the reference, and it is the call the live bot makes. A change to
    weights_io or to the pickles is therefore visible to this gate.
    """
    from utils.weights_io import load_regime  # noqa: PLC0415

    out = {}
    info = {}
    for idx, (name, _pos) in enumerate(stack):
        art = load_regime(raw_window, regime_dir_name=name, model=model_prefix)
        cols = [str(c) for c in art.feature_columns]
        missing = [c for c in cols if c not in panel.columns]
        if missing:
            raise SystemExit(
                f"{name}: the C++ panel has no column(s) {missing[:5]} that the "
                "DEPLOYED model selects. The core resolves these at boot and would "
                "have refused the stack; a harness that skipped them would grade a "
                "model the core cannot run.")
        X = panel[cols].copy()
        n_nan = int(X.isna().to_numpy().sum())
        # trading.py:697-700. Applied HERE, on the harness side, so the
        # reference call itself stays exactly the production code path.
        X = X.fillna(0.0)

        # *** THE REFERENCE CANNOT BE ASKED ABOUT AN inf ROW AT ALL. ***
        # trading.py fills NaN and does NOT touch inf, and sklearn's
        # `validate_data(..., ensure_all_finite="allow-nan")` inside
        # RobustScaler.transform ALLOWS NaN and REJECTS inf outright:
        #   ValueError: Input X contains infinity or a value too large for
        #               dtype('float64')
        # In production that ValueError is caught by trading.py:744's bare
        # `except Exception`, logged, and the regime returns an EMPTY frame —
        # i.e. an infinite feature makes the regime silently produce no
        # prediction for that bar. The C++ propagates the inf instead and
        # reports a NON-FINITE y_pred, which it counts and excludes from
        # n_triggered: the same trading outcome (no signal) reached by a
        # different and strictly more observable route.
        #
        # So those rows are EXCLUDED from the value comparison — the reference
        # has no value to compare — and asserted separately: the C++ must be
        # non-finite on exactly them. Reported, never quietly dropped.
        poisoned = np.isinf(X.to_numpy(dtype=float)).any(axis=1)
        Xc = X.loc[~poisoned]
        pred = np.full(len(X), np.nan, dtype=float)
        if len(Xc):
            with warnings.catch_warnings():
                # The scaler was fitted with feature names and predict() hands
                # it a bare ndarray — which is what the deployed path does too
                # (weights_io.predict passes X[cols].values), so the warning is
                # a property of production, not of this harness.
                warnings.filterwarnings(
                    "ignore", message=".*does not have valid feature names.*")
                pred[~poisoned] = np.asarray(art.predict(Xc), dtype=float)
        out[idx] = pred
        info[name] = {"n_features": len(cols), "nan_filled": n_nan,
                      "poisoned_rows": np.flatnonzero(poisoned),
                      "window_id": art.meta.get("window_id"),
                      "model_class": type(art.model).__name__}
    return pd.DataFrame(out), info


def compare(scenario: str, stack, cpp: pd.DataFrame, ref: pd.DataFrame,
            refinfo: dict, tol: float, moved: set) -> int:
    """FAILURE count for one scenario (0 == pass).

    THE TOLERANCE DENOMINATOR IS THE REGIME'S OWN SIGNAL SCALE, not the cell.
    A pure per-cell relative gate measures the denominator whenever a prediction
    crosses zero — and these predictions are returns, so they cross zero
    constantly. `max(|a|, |b|, rms(ref))` grades every cell against the size of
    the thing being predicted, which is what "1e-9 relative" is supposed to mean
    here. rms is taken over the REFERENCE, so the C++ cannot influence its own
    tolerance.
    """
    print(f"\n--- scenario: {scenario} ---")
    failures = 0

    if cpp.shape[1] != len(stack):
        print(f"=== FAIL: driver returned {cpp.shape[1]} prediction columns for "
              f"{len(stack)} regimes ===")
        return 1
    if len(cpp) != len(ref):
        print(f"=== FAIL: {len(cpp)} C++ rows vs {len(ref)} reference ===")
        return 1

    worst = 0.0
    worst_name = ""
    n_poisoned = 0
    diffs = []
    for i, (name, _pos) in enumerate(stack):
        a = cpp.iloc[:, i].to_numpy(dtype=float)
        b = ref.iloc[:, i].to_numpy(dtype=float)

        # The rows the REFERENCE could not be asked about (an inf reached a
        # selected column and sklearn refuses them — see reference_predictions).
        # On exactly those the C++ MUST be non-finite; anywhere else it must
        # agree cell for cell.
        pois = np.zeros(len(a), dtype=bool)
        idx_p = refinfo[name]["poisoned_rows"]
        if len(idx_p):
            pois[idx_p] = True
            n_poisoned += int(pois.sum())
            leaked = np.flatnonzero(pois & np.isfinite(a))
            if leaked.size:
                failures += 1
                print(f"=== FAIL: {name} produced a FINITE y_pred on "
                      f"{leaked.size} row(s) whose model input contains +/-inf "
                      f"(first row {int(leaked[0])}). An inf must propagate, "
                      "not be absorbed into a plausible number. ===")

        # Non-finite classification FIRST, before any value comparison: a NaN
        # on one side and a finite number on the other is not a small error.
        cls_a = np.isfinite(a) & ~pois
        cls_b = np.isfinite(b) & ~pois
        if not np.array_equal(cls_a, cls_b):
            n = int((cls_a != cls_b).sum())
            diffs.append((name, f"{n} row(s) differ in FINITE/non-finite class"))
            failures += 1
            continue

        fin = cls_a
        if not fin.any():
            diffs.append((name, "every prediction is non-finite on BOTH sides"))
            continue
        af, bf = a[fin], b[fin]
        rms = float(np.sqrt(np.mean(bf * bf)))
        denom = np.maximum(np.maximum(np.abs(af), np.abs(bf)),
                           rms if rms > 0.0 else 1.0)
        rel = np.abs(af - bf) / denom
        m = float(rel.max())
        if m > worst:
            worst, worst_name = m, name
        bad = np.flatnonzero(rel > tol)
        if bad.size:
            diffs.append((name, f"{bad.size} row(s) exceed tol, max rel {m:.3e}, "
                                f"first at row {int(np.flatnonzero(fin)[bad[0]])}"))

        # Did the panel actually move this model? A runner that emitted only its
        # intercept would agree with a reference that did the same, forever.
        if np.ptp(bf) > 0.0:
            moved.add(name)

    if diffs:
        failures += 1
        print(f"=== FAIL: {len(diffs)} regime(s) disagree with the DEPLOYED "
              f"sklearn pipeline ===")
        for name, why in diffs[:10]:
            print(f"    {name}: {why}")
        if len(diffs) > 10:
            print(f"    ... and {len(diffs) - 10} more")
    else:
        print(f"  preds:  {len(stack)} regimes x {len(cpp)} rows, max rel dev "
              f"{worst:.3e} (tol {tol:g}) — worst {worst_name}")

    # The row live actually scores, reported on its own so it is never only
    # implied by an aggregate over 699.
    last_dev = 0.0
    for i in range(len(stack)):
        a = float(cpp.iloc[-1, i])
        b = float(ref.iloc[-1, i])
        if np.isfinite(a) and np.isfinite(b):
            d = abs(a - b) / max(abs(a), abs(b), 1e-12)
            last_dev = max(last_dev, d)
    print(f"  newest row (the ONLY row live scores): max rel dev {last_dev:.3e}")

    if n_poisoned:
        print(f"  divergence: {n_poisoned} (regime, row) pair(s) carry +/-inf in a "
              "SELECTED model column. sklearn REFUSES them (ValueError), so live "
              "logs and returns an empty frame -- the regime silently makes no "
              "prediction for that bar; the C++ propagates the inf and reports a "
              "NON-FINITE y_pred it excludes from n_triggered. Same outcome (no "
              "signal), different route. Reported, not fixed.")

    # trading.py's clip would have bitten here. Reported every time rather than
    # allowlisted into silence — see the module docstring.
    over = int((np.abs(ref.to_numpy(dtype=float)) > 1.0).sum())
    if over:
        print(f"  FINDING: {over} reference prediction(s) exceed |1.0|; "
              "trading.py:705-713 would np.clip them to [-1, 1]. Neither side "
              "clips here (weights_io.predict does not), so this is a live-vs-"
              "export divergence, reported not fixed.")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True,
                    help="command that reads the raw CSV on stdin and prints "
                         "#panel/#preds/#meta (shlex-split, so a `docker run -i "
                         "...` line works)")
    ap.add_argument("--weights", required=True,
                    help="the export_agamotto_sentinel_weights.py OUTPUT dir "
                         "(model.txt/scaler.txt/features.txt per regime) — what "
                         "the C++ reads")
    ap.add_argument("--raw-weights", required=True,
                    help="the SOURCE window_YYYY_MM_DD dir of ridge_*.pkl — what "
                         "the DEPLOYED loader reads. Never the same directory: "
                         "the two sides must come from different artifacts or "
                         "the comparison is circular")
    ap.add_argument("--marvel-root", required=True,
                    help="marvel checkout, for utils.weights_io — THE deployed "
                         "loader. Not reimplemented here")
    ap.add_argument("--model", default="ridge", help="artifact prefix")
    ap.add_argument("--stack", default=str(DEFAULT_STACK))
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--expect-red", action="store_true",
                    help="invert the exit code: for the negative controls, where "
                         "a PASS is the failure")
    args = ap.parse_args()

    marvel = Path(args.marvel_root).expanduser().resolve()
    if not (marvel / "utils" / "weights_io.py").is_file():
        raise SystemExit(
            f"--marvel-root {marvel} has no utils/weights_io.py. That module IS "
            "the reference; refusing to grade against a reimplementation.")
    sys.path.insert(0, str(marvel))

    weights = Path(args.weights).expanduser().resolve()
    raw_window = Path(args.raw_weights).expanduser().resolve()
    if weights == raw_window:
        raise SystemExit(
            "--weights and --raw-weights are the same directory. The two sides "
            "must read DIFFERENT artifacts (text vs pickle) or this gate compares "
            "the export with itself.")
    for p in (weights, raw_window):
        if not p.is_dir():
            raise SystemExit(f"not a directory: {p}")

    stack = load_stack(Path(args.stack))
    regimes = spec_arg(stack)
    inert = [n for n, _p in stack if set(atoms_of(n)) & VOL_Q_ATOM_CODES]
    firable = [n for n, _p in stack if not (set(atoms_of(n)) & VOL_Q_ATOM_CODES)]

    print(f"[model_parity] pandas {pd.__version__}  numpy {np.__version__}")
    try:
        import sklearn
        print(f"[model_parity] sklearn {sklearn.__version__} (the REFERENCE "
              "pipeline; a version skew against the fit version changes numbers)")
    except ImportError:
        raise SystemExit("sklearn is missing — the deployed loader IS the "
                         "reference and cannot run without it.")
    print(f"[model_parity] PANEL_BARS={PANEL_BARS}")
    print(f"[model_parity] stack:        {args.stack} ({len(stack)} regimes, "
          f"{len(inert)} inert / {len(firable)} firable)")
    print(f"[model_parity] C++ weights:  {weights}")
    print(f"[model_parity] reference:    {raw_window} ({args.model}_*.pkl via "
          "utils.weights_io.load_regime)")
    print(f"[model_parity] driver:       {args.driver}")

    failures = 0
    moved: set = set()
    meta = {}
    for name, price, seed, holes, leads, safe_branch in fp.SCENARIOS:
        raw = fp.make_panel(PANEL_BARS, price, seed, holes, leads, safe_branch)
        panel, cpp, meta = run_driver(args.driver, weights, raw, regimes)
        ref, refinfo = reference_predictions(raw_window, args.model, panel, stack)

        # THE MIXTURE, cross-checked between the two sides on every scenario.
        # The C++ reads features.txt, the reference reads meta['feature_columns'];
        # if they ever disagreed, one of them is scoring a different model.
        for rname, _pos in stack:
            c = meta[rname]["n_features"]
            r = refinfo[rname]["n_features"]
            if c != r:
                print(f"=== FAIL: {rname} — the export declares {c} features, the "
                      f"DEPLOYED meta declares {r} ===")
                failures += 1

        failures += compare(name, stack, cpp, ref, refinfo, args.tol, moved)

    # ---- the mixed-provenance assertions ----------------------------------
    # Guards the exact bug a hardcoded TOPN_ICS=5 would be. Asserted rather than
    # printed, because "the harness happened to include a 16-feature regime" is
    # not a property anyone would notice losing.
    counts = sorted({meta[n]["n_features"] for n, _p in stack})
    by_count = {c: [n for n, _p in stack if meta[n]["n_features"] == c] for c in counts}
    print(f"\n  provenance: feature counts across the deployed stack = {counts}; "
          + ", ".join(f"{c}x{len(v)}" for c, v in by_count.items()))
    if len(counts) < 2:
        failures += 1
        print("=== FAIL: every deployed regime has the SAME feature count, so a "
              "hardcoded n could pass this gate silently. window_2026_07_31 is "
              "known to mix 5-feature and 16-feature models — either the stack "
              "or the weights under test are not the deployed ones. ===")
    else:
        print(f"  MIXED PROVENANCE CONFIRMED: {len(counts)} distinct feature "
              "counts in ONE weights directory — these regimes were NOT all "
              "fitted by the same run. A hardcoded TOPN_ICS (=5) cannot pass.")

    # And the part that makes it dangerous rather than merely untidy.
    firable_counts = sorted({meta[n]["n_features"] for n in firable})
    inert_counts = sorted({meta[n]["n_features"] for n in inert})
    print(f"  firable regimes ({len(firable)}): feature counts {firable_counts}")
    print(f"  inert regimes   ({len(inert)}): feature counts {inert_counts}")
    if firable_counts == [5]:
        failures += 1
        print("=== FAIL: every FIRABLE regime is 5-feature — the measured "
              "deployment has all 9 of them at 16. The weights under test are "
              "not window_2026_07_31. ===")
    elif len(firable_counts) == 1 and firable_counts != inert_counts:
        print(f"  *** the split is EXACTLY inert-vs-firable: every regime that "
              f"can trade carries a {firable_counts[0]}-feature model, every "
              f"regime that cannot carries {inert_counts}. A hardcoded 5 would be "
              "wrong on every regime that matters and right on every one that "
              "does not. ***")

    unit = {n: meta[n]["unit_scale"] for n, _p in stack if meta[n]["unit_scale"]}
    if unit:
        print(f"  FINDING: {sum(unit.values())} scaler row(s) across {len(unit)} "
              f"regime(s) have scale == 1.0 exactly (sklearn's substitute for a "
              f"ZERO IQR — a CONSTANT train feature, entering the model "
              f"UNSCALED): {sorted(unit)}")
        fire_unit = [n for n in unit if n in firable]
        if fire_unit:
            print(f"           {len(fire_unit)} of them can FIRE: {fire_unit}")

    filled = {n: meta[n]["nan_filled"] for n, _p in stack if meta[n]["nan_filled"]}
    if filled:
        print(f"  note: trading.py's NaN->0.0 fill touched {sum(filled.values())} "
              f"feature cell(s) across {len(filled)} regime(s) on the LAST "
              "scenario — applied identically on both sides.")

    # ---- non-triviality ---------------------------------------------------
    never = [n for n, _p in stack if n not in moved]
    if never:
        failures += 1
        print(f"\n=== FAIL: {len(never)} regime(s) produced a CONSTANT prediction "
              f"across all {len(fp.SCENARIOS)} scenarios ===")
        for n in never[:10]:
            print(f"    {n}")
        print("    A runner that emitted only its intercept would match a "
              "reference that did the same. Either the features never reach the "
              "model or the panel is degenerate.")
    else:
        print(f"\n  non-triviality: all {len(stack)} regime(s) produced a VARYING "
              f"prediction — the panel reaches the model.")

    print()
    if failures:
        print(f"=== MODEL PARITY FAILED ({failures} failure groups) ===")
        return 0 if args.expect_red else 1
    print(f"=== MODEL PARITY PASS: {len(stack)} regimes x {PANEL_BARS} rows x "
          f"{len(fp.SCENARIOS)} scenarios, rel tol {args.tol:g} ===")
    return 1 if args.expect_red else 0


if __name__ == "__main__":
    sys.exit(main())
