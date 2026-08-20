#!/usr/bin/env python3
"""PHASE 3 gate: the REAL research_filters on one side, the C++ regime gate on
the other, every regime x every row compared for EXACT boolean equality.

WHAT THIS COMPARES, AND WHY IT IS NOT tests/feature_parity.py's COMPARISON
--------------------------------------------------------------------------
The driver emits TWO things: the 65-column panel it engineered, and the mask it
computed FROM THAT PANEL. This harness takes the C++ panel, decodes its column
codes back to real names, hands the *identical numbers* to the reference's own
``research_filters.apply_filter_mask``, and diffs the two masks.

That is deliberate and it is the only way the comparison can be EXACT.
Engineering a panel in Python alongside would put the feature engine back in
scope — and feature parity is a 1e-9 RELATIVE gate, which is enormous next to a
boolean. A cell 1e-12 from ``adx > 25`` would land on opposite sides on the two
panels and the run would go red for a reason that is not the predicate. Here
both sides read the same double, so:

    *** TOLERANCE IS ZERO. A MASK IS A DECISION, NOT A MEASUREMENT. ***

One differing cell out of 62 x 699 x 5 x 2 is a failure. There is no
"close enough" for "does this bar trade".

The regime NAMES fed to the reference are the CODED ones the live stack carries
(``r029_and_r001_and_r073_long``); ``apply_filter_mask`` decodes them itself via
``decode_regime_tolerant``, exactly as production does. The C++ side is handed
the same regimes as ARRAYS OF uint16 CODES, which is what crosses ICore. Neither
side is ever shown a real regime name.

THE TRAP THIS HARNESS EXISTS TO AVOID
-------------------------------------
53 of the 62 regimes in the deployed stack CANNOT FIRE LIVE. Their vol-quantile
cutoff columns (``price_range_pct_q80/q90/q95``) are
``rolling(700, min_periods=700)`` on a 699-row panel, hence NaN everywhere, and
``x > NaN`` is False. That is today's production behaviour under an open finding
(marvel PR #532, docs/findings/2026-08-19-vol-quantile-regimes-inert-live.md)
and this port reproduces it deliberately.

Which means a gate that returned all-False for EVERYTHING would agree with the
reference on 53 of 62 regimes and would look like a strong pass. So the harness
asserts the SHAPE of the answer, not only its equality:

  * every regime's mask must be exactly equal, cell for cell;
  * the 53 r073/r074/r075-gated regimes must be ALL-FALSE on both sides, and
    the q80/q90/q95 columns of the panel the gate READ must be all-NaN — the
    inertness is asserted at its cause, not only at its effect;
  * each of the 53 is ALSO evaluated with its vol-quantile atom STRIPPED (a
    "probe" regime, not deployed), and those must fire. That is the causal
    control: it separates "inert because the cutoff is NaN" from "inert because
    the gate is broken", which look identical from the mask alone;
  * the 9 firable regimes and every probe must be NON-TRIVIAL — never all-True
    (that is `baseline` under another name, removed forever on 2026-06-18) and
    not all-False on every scenario. All-False is judged ACROSS the five
    scenarios rather than within each: two of them are deliberately degenerate
    (injected NaN runs, a 26-bar flat run, a close of 1e-305), and requiring
    `bb_rebound` to fire on those would be asserting something about the
    synthetic data instead of about the gate.

Fail any of those and the gate is not doing its job even if the diff is clean.
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

# The scenario generator and the panel-width reader are REUSED from the feature
# gate rather than re-implemented: two generators would drift, and the whole
# point of running the gate on the PEPE scenario is that it is the same panel
# stage 2.2 proved the epsilon on.
import feature_parity as fp  # noqa: E402

PANEL_BARS = fp.PANEL_BARS


# ---------------------------------------------------------------------------
# THE STACK UNDER TEST.
#
# Read from a file rather than typed here, and defaulting to a committed copy of
# what is DEPLOYED. Retyping 62 regimes would make this harness grade a stack
# that no longer exists, which is the failure mode the whole exercise is about.
# ---------------------------------------------------------------------------
DEFAULT_STACK = HERE / "regime_stack_deployed.csv"

# The three trailing vol-quantile atoms. Every regime carrying one of them is
# inert live. Named as CODES because that is what the stack carries; the names
# are in research_filters._VOL_QUANTILE_ATOMS and are not needed here.
VOL_Q_ATOM_CODES = {73, 74, 75}


def load_stack(path: Path) -> list[tuple[str, str]]:
    """[(coded_regime_name, position)] from a filtered_optimal_regime_stack.csv."""
    df = pd.read_csv(path)
    for col in ("regime", "position"):
        if col not in df.columns:
            raise SystemExit(f"{path}: no '{col}' column — is this a regime stack?")
    # The pipeline appends a `__summary__` bookkeeping row. Dropped by NAME, not
    # by position: dropping "the last row" would silently eat a real regime the
    # day the summary moves or disappears.
    rows = [(str(r.regime), str(r.position)) for r in df.itertuples()
            if not str(r.regime).startswith("__")]
    if not rows:
        raise SystemExit(f"{path}: no regimes after dropping __summary__ rows")
    return rows


def atoms_of(coded_name: str) -> list[int]:
    """`r029_and_r001_and_r073_long` -> [29, 1, 73].

    Pure structure: split on the conjunction separator, strip the position
    suffix, read the digits. No name table, and none needed — this is the same
    parse the public strategy does before it calls setRegimeStack().
    """
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
                f"regime {coded_name!r} has non-coded atom {part!r}. The gate takes "
                "CODES; a stack carrying real names must be encoded before it "
                "reaches the core (and would leak them into the artifact).")
        out.append(int(part[1:]))
    return out


def spec_arg(stack: list[tuple[str, str]]) -> str:
    """The --regimes argument: `60.75:L,29.66.73:L,...`. Codes only."""
    parts = []
    for name, pos in stack:
        codes = ".".join(str(a) for a in atoms_of(name))
        parts.append(f"{codes}:{'L' if pos == 'long' else 'S'}")
    return ",".join(parts)


def run_driver(driver_cmd: str, raw: pd.DataFrame, regimes: str):
    """Feed the raw panel in; read the engineered panel AND the masks back."""
    buf = io.StringIO()
    raw.to_csv(buf, index=False, float_format="%.17g", na_rep="nan")
    cmd = shlex.split(driver_cmd) + ["--regimes", regimes]
    proc = subprocess.run(cmd, input=buf.getvalue(), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"driver failed (rc={proc.returncode}):\n{proc.stderr}")
    lines = proc.stdout.splitlines()
    try:
        i_panel = lines.index("#panel")
        i_masks = lines.index("#masks")
    except ValueError:
        raise SystemExit(
            "driver output has no #panel/#masks sections — refusing to guess "
            f"its layout:\n{proc.stdout[:400]}")

    panel_lines = lines[i_panel + 1:i_masks]
    # float() reads nan/inf/-inf; %.17g on the way out and this on the way in
    # both round-trip a double exactly, so the seam adds no error of its own.
    panel = pd.DataFrame(
        [[float(x) for x in ln.split(",")] for ln in panel_lines[1:]],
        columns=panel_lines[0].split(","))

    mask_lines = [ln for ln in lines[i_masks + 1:] if ln.strip()]
    masks = pd.DataFrame(
        [[int(x) for x in ln.split(",")] for ln in mask_lines[1:]],
        columns=mask_lines[0].split(","))
    return panel, masks


def decoder():
    """feature CODE -> real name, for handing the C++ panel to research_filters.

    The reverse of tests/feature_parity.py's `encoder`. Uncoded keys (`close`,
    `mvg1/2/3` — dc/obfuscation/map.json has no entry for them) pass through.
    """
    mp = json.loads((DC_ROOT / "obfuscation" / "map.json").read_text())["features"]
    rev = {v: k for k, v in mp.items()}
    if len(rev) != len(mp):
        raise SystemExit("feature map is not bijective — a code decodes two ways")
    return lambda c: rev.get(c, c)


def reference_masks(panel_named: pd.DataFrame,
                    stack: list[tuple[str, str]]) -> pd.DataFrame:
    """The REAL research_filters, over the C++ panel, one column per regime.

    Nothing about the predicates is re-implemented here: `apply_filter_mask` is
    imported and called with the CODED regime name and the position, which is
    exactly the call `AgamottoResearch.filter_signals` makes in production. A
    change in research_filters is therefore visible to this gate.
    """
    from agamotto.research_filters import apply_filter_mask  # noqa: PLC0415

    out = {}
    for idx, (name, pos) in enumerate(stack):
        m = apply_filter_mask(panel_named, name, pos, strict_filters=True)
        if not isinstance(m, pd.Series):
            raise SystemExit(
                f"reference returned {type(m).__name__} for {name!r}, not a "
                "per-row Series — a scalar here would broadcast to always-fire")
        if len(m) != len(panel_named):
            raise SystemExit(f"reference mask for {name!r} is {len(m)} rows, "
                             f"panel is {len(panel_named)}")
        out[idx] = m.to_numpy().astype(bool)
    return pd.DataFrame(out)


def parent_of(coded_name: str, position: str) -> str | None:
    """`r029_and_r066_and_r073_long` -> `r029_and_r066_long`, or None.

    THE CAUSAL PROBE. "The 53 are all-False on both sides" is satisfied by a
    gate that is broken in some completely different way — a typo'd column
    name, a predicate that always returns false, a position check that refuses
    everything. Evaluating each inert regime WITHOUT its vol-quantile atom and
    showing that it DOES fire pins the cause on the NaN cutoff specifically,
    which is the property marvel PR #532 documents and this port must preserve.
    """
    body = coded_name.strip().lower()
    for suffix in ("_long", "_short"):
        if body.endswith(suffix):
            body = body[: -len(suffix)]
            break
    kept = [p for p in body.split("_and_") if int(p[1:]) not in VOL_Q_ATOM_CODES]
    if len(kept) == len(body.split("_and_")) or not kept:
        return None
    return "_and_".join(kept) + ("_long" if position == "long" else "_short")


def compare(scenario: str, specs, cpp_masks: pd.DataFrame,
            ref_masks: pd.DataFrame, allfalse_ever_broken: set) -> int:
    """Return the number of FAILURES for one scenario (0 == pass).

    `allfalse_ever_broken` accumulates ACROSS scenarios: a regime is required to
    be non-trivial SOMEWHERE, not on every panel. Two of the five scenarios are
    deliberately degenerate (injected NaN runs, a 26-bar flat run, a close of
    1e-305), and demanding that `bb_rebound` fire on those would be demanding
    something about the synthetic data rather than about the gate.
    """
    print(f"\n--- scenario: {scenario} ---")
    failures = 0

    if cpp_masks.shape[1] != len(specs):
        print(f"=== FAIL: driver returned {cpp_masks.shape[1]} masks for "
              f"{len(specs)} regimes ===")
        return 1
    if len(cpp_masks) != len(ref_masks):
        print(f"=== FAIL: {len(cpp_masks)} C++ rows vs {len(ref_masks)} reference ===")
        return 1

    # ---- 1. EXACT equality, per regime, per row ---------------------------
    diffs = []
    for i, (name, pos, _kind) in enumerate(specs):
        a = cpp_masks.iloc[:, i].to_numpy().astype(bool)
        b = ref_masks.iloc[:, i].to_numpy().astype(bool)
        bad = np.flatnonzero(a != b)
        if bad.size:
            diffs.append((name, pos, bad))
    if diffs:
        failures += 1
        print(f"=== FAIL: {len(diffs)} regime(s) disagree with research_filters ===")
        for name, pos, bad in diffs[:10]:
            print(f"    {name} [{pos}]: {bad.size} row(s) differ, first at "
                  f"{bad[:6].tolist()}")
        if len(diffs) > 10:
            print(f"    ... and {len(diffs) - 10} more")
    else:
        print(f"  masks: {len(specs)} regimes x {len(cpp_masks)} rows, "
              f"EXACTLY equal (0 differing cells, zero tolerance)")

    inert = [i for i, (_n, _p, k) in enumerate(specs) if k == "inert"]
    live = [i for i, (_n, _p, k) in enumerate(specs) if k == "live"]
    probe = [i for i, (_n, _p, k) in enumerate(specs) if k == "probe"]

    # ---- 2. the vol-quantile regimes must be INERT ------------------------
    # Asserted on BOTH sides. "Both agree" is satisfied by two engines wrong in
    # the same way, and THIS wrongness — a gate firing where production's
    # cannot — is precisely the one Phase 3 was told to preserve.
    bad_inert = [specs[i][0] for i in inert
                 if cpp_masks.iloc[:, i].to_numpy().any()
                 or ref_masks.iloc[:, i].to_numpy().any()]
    if bad_inert:
        failures += 1
        print(f"=== FAIL: {len(bad_inert)} r07x-gated regime(s) FIRED: "
              f"{bad_inert[:6]} ===")
        print("    They compare price_range_pct against a rolling(700, "
              "min_periods=700) cutoff on a 699-row panel, which is NaN on "
              "every row. Firing means the all-NaN property broke — see marvel "
              "PR #532 / docs/findings/2026-08-19-vol-quantile-regimes-inert-live.md.")
    else:
        print(f"  inert:  {len(inert)} r07x-gated regime(s) ALL-FALSE on both sides")

    # ---- 3. no DEPLOYED regime may be all-TRUE ----------------------------
    # An all-True mask is `baseline` — the unconditional fire-on-every-bar
    # regime CLAUDE.md removed forever (2026-06-18) — arriving under another
    # name. Unlike all-False it is never a property of the data, so it fails in
    # the scenario it appears in rather than being deferred.
    #
    # DEPLOYED rows only. The PROBE rows are synthetic (a deployed regime with
    # its vol-quantile atom stripped) and one of them, the bare `r048` leg, is
    # all-True — see the FINDING printed below. Failing the port for that would
    # be failing it for the reference's own predicate, which the cell-for-cell
    # equality above already proves it reproduces exactly.
    all_true = [specs[i][0] for i in live
                if cpp_masks.iloc[:, i].to_numpy().astype(bool).all()]
    if all_true:
        failures += 1
        print(f"=== FAIL: {len(all_true)} DEPLOYED regime(s) are ALL-TRUE (a "
              f"`baseline` regime under another name): {all_true[:6]} ===")
    probe_all_true = [specs[i][0] for i in probe
                      if cpp_masks.iloc[:, i].to_numpy().astype(bool).all()]
    if probe_all_true:
        # Reported, not failed, and reported EVERY time rather than allowlisted
        # into silence: it is a property of the reference predicate that the
        # deployed stack happens to hide behind an inert atom today, and the day
        # PR #532 is resolved it stops being hidden.
        print(f"  FINDING: {len(probe_all_true)} probe regime(s) are ALL-TRUE on "
              f"this panel: {probe_all_true}")

    # ---- 4. record what DID vary, for the cross-scenario check -----------
    for i in live + probe:
        if cpp_masks.iloc[:, i].to_numpy().any():
            allfalse_ever_broken.add(specs[i][0])

    rates = [(specs[i][0], float(cpp_masks.iloc[:, i].mean())) for i in live]
    fired_here = sum(1 for _n, r in rates if r > 0.0)
    print(f"  firable: {len(live)} regime(s), {fired_here} fired here; rates "
          + ", ".join(f"{r:.3f}" for _n, r in rates))
    probe_rates = [float(cpp_masks.iloc[:, i].mean()) for i in probe]
    if probe_rates:
        print(f"  probes:  {len(probe)} r07x-stripped parent(s), "
              f"{sum(1 for r in probe_rates if r > 0.0)} fired "
              f"(min {min(probe_rates):.3f} max {max(probe_rates):.3f})")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True,
                    help="command that reads the raw CSV on stdin and prints "
                         "#panel/#masks (shlex-split, so a `docker run -i ...` "
                         "line works)")
    ap.add_argument("--stack", default=str(DEFAULT_STACK),
                    help="filtered_optimal_regime_stack.csv to grade against")
    args = ap.parse_args()

    stack = load_stack(Path(args.stack))

    # specs = (coded_name, position, kind). `kind` classifies what the row is
    # FOR, so the assertions below are about the right thing:
    #   inert  a stack regime carrying r073/r074/r075 — must never fire
    #   live   a stack regime that can fire — must not be trivial
    #   probe  an inert regime with its vol-quantile atom REMOVED — the causal
    #          control. It is not in the deployed stack; it exists to prove the
    #          53 are inert BECAUSE of the NaN cutoff and not because the gate
    #          is broken in some way that happens to look the same.
    specs = []
    for name, pos in stack:
        specs.append((name, pos, "inert" if set(atoms_of(name)) & VOL_Q_ATOM_CODES
                      else "live"))
    seen = {(n, p) for n, p, _k in specs}
    for name, pos, kind in list(specs):
        if kind != "inert":
            continue
        par = parent_of(name, pos)
        if par and (par, pos) not in seen:
            seen.add((par, pos))
            specs.append((par, pos, "probe"))

    regimes = spec_arg([(n, p) for n, p, _k in specs])

    n_atoms = sorted({a for name, _ in stack for a in atoms_of(name)})
    n_inert = sum(1 for _n, _p, k in specs if k == "inert")
    n_live = sum(1 for _n, _p, k in specs if k == "live")
    n_probe = sum(1 for _n, _p, k in specs if k == "probe")
    arity = pd.Series([len(atoms_of(n)) for n, _ in stack]).value_counts().sort_index()

    print(f"[regime_parity] pandas {pd.__version__}  numpy {np.__version__}")
    print(f"[regime_parity] PANEL_BARS={PANEL_BARS}")
    print(f"[regime_parity] stack: {args.stack}")
    print(f"[regime_parity] {len(stack)} regimes, {len(n_atoms)} distinct atoms "
          f"{['r%03d' % a for a in n_atoms]}")
    print("[regime_parity] arity: "
          + ", ".join(f"{k}-atom x{v}" for k, v in arity.items()))
    print(f"[regime_parity] {n_inert} carry r073/r074/r075 and CANNOT fire live; "
          f"{n_live} can")
    print(f"[regime_parity] + {n_probe} r07x-stripped PROBE regime(s) as the "
          f"causal control (not deployed)")
    print(f"[regime_parity] driver: {args.driver}")

    dec = decoder()
    failures = 0
    fired_somewhere: set = set()
    for name, price, seed, holes, leads, safe_branch in fp.SCENARIOS:
        raw = fp.make_panel(PANEL_BARS, price, seed, holes, leads, safe_branch)
        panel, cpp_masks = run_driver(args.driver, raw, regimes)

        # THE ALL-NaN CUTOFFS, ASSERTED ON THE PANEL THE GATE ACTUALLY READ.
        # Without this the inertness above could come from anywhere; with it,
        # the input to the comparison is pinned too.
        for code_col in ("f108", "f109", "f110"):
            if code_col not in panel.columns:
                print(f"=== FAIL: the C++ panel has no {code_col} column ===")
                failures += 1
            elif not panel[code_col].isna().all():
                print(f"=== FAIL: {code_col} is NOT all-NaN on the panel the gate "
                      f"read ({int(panel[code_col].notna().sum())} finite cells). "
                      "min_periods=700 on 699 rows must never be met — see "
                      "src/feature_engine.hpp VOL_Q_WINDOW. ===")
                failures += 1

        # The C++ panel, with its column CODES decoded, is what the reference is
        # driven over. Same numbers, both sides — which is what makes the mask
        # comparison EXACT rather than tolerant.
        panel_named = panel.rename(columns={c: dec(c) for c in panel.columns})
        ref_masks = reference_masks(panel_named, [(n, p) for n, p, _k in specs])
        failures += compare(name, specs, cpp_masks, ref_masks, fired_somewhere)

    # ---- the cross-scenario non-triviality check --------------------------
    # A gate hardwired to all-False agrees with the reference on the 53 inert
    # regimes and would sail through everything above. This is what stops that:
    # every firable regime, and every causal probe, must have fired on at least
    # one of the five panels.
    never = [n for n, _p, k in specs if k in ("live", "probe") and n not in fired_somewhere]
    if never:
        failures += 1
        print(f"\n=== FAIL: {len(never)} regime(s) NEVER fired on ANY scenario ===")
        for n in never[:10]:
            print(f"    {n}")
        print("    A gate that never fires passes a mask-equality test trivially. "
              "Either the predicate is dead or no scenario reaches it.")
    else:
        print(f"\n  non-triviality: all {n_live} firable + {n_probe} probe "
              f"regime(s) fired on at least one of the {len(fp.SCENARIOS)} scenarios")

    print()
    if failures:
        print(f"=== REGIME PARITY FAILED ({failures} failure groups) ===")
        return 1
    print(f"=== REGIME PARITY PASS: {len(specs)} regimes x {PANEL_BARS} rows x "
          f"{len(fp.SCENARIOS)} scenarios, EXACT ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
