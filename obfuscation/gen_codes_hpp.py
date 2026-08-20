#!/usr/bin/env python3
"""Emit `codes_generated.hpp` for a C++ core from dc/obfuscation/map.json.

WHY THIS EXISTS. `sentinel_core/src/codes_generated.hpp` has said
"GENERATED from dc/obfuscation/map.json — do not edit by hand" since
2026-07-30, but no generator was ever committed: the file was produced once by
hand and has drifted from the map ever since. CLAUDE.md is explicit that a
repeatable procedure must live in a committed script with args, never in a
chat transcript or in someone's memory. This is that script.

WHAT IT EMITS. Codes only — the C++ identifiers compile away, and the string
VALUES are the obfuscation codes, so no real regime or feature name survives in
the built .so for `strings` to recover. Features become `const char*` column
keys; regimes become `uint16_t` gate-dispatch ids (the integer part of `rNNN`).

Usage:
    python obfuscation/gen_codes_hpp.py --namespace agamotto \\
        --out agamotto_core/src/codes_generated.hpp
    python obfuscation/gen_codes_hpp.py --namespace mjolnir \\
        --out sentinel_core/src/codes_generated.hpp --check
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAP_PATH = HERE / "map.json"

NAMESPACES = ("mjolnir", "agamotto")

# The historical preamble, verbatim from the 2026-07-30 hand-written file. The
# "172 real names / 118 recoverable" sentence is a RECORD OF A MEASUREMENT made
# on the artifact this file replaced — it is deliberately NOT recomputed from
# the current map, because it describes what the OLD name-table binary leaked,
# not what the map happens to hold today.
PREAMBLE = """// GENERATED from dc/obfuscation/map.json — do not edit by hand.
//
// CODES ONLY. The identifiers below are C++ symbols (they compile away); the
// VALUES are the obfuscation codes. No real regime or feature name appears as a
// string literal, so none can be recovered from the built .so with `strings`.
//
// This replaces the earlier code->name / name->code tables, which embedded all
// 172 real names in the binary (118 were recoverable) — the artifact was no more
// opaque than the vendored map.json that ships with the Python packages.
//
// Passthrough columns (OHLCV, timestamps, depth_bid_L*, ofi_L*, bids_*/asks_*)
// are deliberately NOT coded: the obfuscation map does not cover them, they are
// universal market-data field names, and they carry no strategy information.
#pragma once
#include <cstdint>
"""

FEATURE_BANNER = "// ---- features: column keys used by the engine and the model ----------------"
REGIME_BANNER = "// ---- regimes: numeric codes for gate dispatch ------------------------------"

_SYM_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def symbol(prefix: str, name: str) -> str:
    sym = f"{prefix}{name.upper()}"
    if not _SYM_OK.match(sym):
        # A name with a hyphen or a dot would emit a header that does not
        # compile. Fail here, where the cause is obvious, rather than in a
        # build log three repos away.
        raise SystemExit(f"map.json name {name!r} does not form a C++ identifier ({sym!r})")
    return sym


def render(namespace: str, mapping: dict) -> str:
    features = mapping["features"]
    regimes = mapping["regimes"]

    lines = [PREAMBLE, f"namespace {namespace} {{", "namespace codes {", "", FEATURE_BANNER]

    seen: dict[str, str] = {}
    for name in sorted(features):
        code = features[name]
        if code in seen:
            raise SystemExit(f"map.json gives code {code!r} to both {seen[code]!r} and {name!r}")
        seen[code] = name
        lines.append(f'inline constexpr const char* {symbol("F_", name)} = "{code}";')

    lines += ["", REGIME_BANNER]
    seen.clear()
    for name in sorted(regimes):
        code = regimes[name]
        m = re.fullmatch(r"r(\d+)", code)
        if not m:
            raise SystemExit(
                f"regime {name!r} has code {code!r}; the gate dispatches on the "
                "integer part, so a non-rNNN code cannot be emitted as uint16_t")
        if code in seen:
            raise SystemExit(f"map.json gives code {code!r} to both {seen[code]!r} and {name!r}")
        seen[code] = name
        lines.append(f"inline constexpr uint16_t {symbol('R_', name)} = {int(m.group(1))};")

    lines += ["", "} // namespace codes", f"}} // namespace {namespace}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namespace", required=True, choices=NAMESPACES,
                    help="C++ namespace to wrap the codes in")
    ap.add_argument("--out", required=True, type=Path, help="header to write")
    ap.add_argument("--map", type=Path, default=MAP_PATH, help="map.json to read")
    ap.add_argument("--check", action="store_true",
                    help="do not write; diff against --out and exit 1 if it differs")
    args = ap.parse_args()

    mapping = json.loads(args.map.read_text())
    for key in ("features", "regimes"):
        if key not in mapping:
            raise SystemExit(f"{args.map} has no {key!r} section")
    text = render(args.namespace, mapping)

    if args.check:
        if not args.out.exists():
            print(f"CHECK FAILED: {args.out} does not exist")
            return 1
        current = args.out.read_text()
        if current == text:
            print(f"CHECK OK: {args.out} is byte-identical to the generator's output")
            return 0
        import difflib
        print(f"CHECK FAILED: {args.out} differs from the generator's output")
        sys.stdout.writelines(difflib.unified_diff(
            current.splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile=f"{args.out} (committed)", tofile="gen_codes_hpp.py (from map.json)"))
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}: {len(mapping['features'])} features, "
          f"{len(mapping['regimes'])} regimes, namespace {args.namespace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
