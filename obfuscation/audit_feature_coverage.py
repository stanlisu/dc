#!/usr/bin/env python3
"""Fail-loud audit: every feature column a dc algo PRODUCES must be obfuscatable.

A feature column that is neither in the obfuscation map nor an explicit
passthrough ships REAL-NAMED into parquet/meta.pkl — a silent leak and a
coded/real train-vs-predict schema split. This scans the feature-producing
modules for column assignments, reduces each to its base (stripping the TF
prefix and the generic _roll{w}_{stat} transform the codec preserves), and
reports any base that is neither mapped nor passthrough.

Dynamic f-string targets (e.g. df[f"depth_imbalance_L{n}"]) can't be expanded
statically; their literal prefix is reported under FSTRING so a human confirms
the concrete variants are mapped (they're listed in extract_inventory._TICK_PROMOTE).

Run:
    cd ~/Documents/sandbox/dc
    python obfuscation/build_map.py && python obfuscation/audit_feature_coverage.py
Exit 1 if any unmapped base is found.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DC_ROOT = HERE.parent

# Modules that PRODUCE feature columns (assign df["x"]/new_cols["x"]/roll_new["x"]).
_FEATURE_MODULES = [
    "mjolnir_pkg/src/mjolnir/core/features.py",
    "mjolnir_pkg/src/mjolnir/core/research.py",
    "stormbreaker_pkg/src/stormbreaker/core/research.py",
    "valkyrie_pkg/src/valkyrie/core/features.py",
    "valkyrie_pkg/src/valkyrie/core/research.py",
    "vomir_pkg/src/vomir/research.py",
    "agamotto_pkg/src/agamotto/research.py",
    "orb_pkg/src/orb/research.py",
]

_ASSIGN = re.compile(r'(?:df|new_cols|roll_new|out|feats)\[(f?)"([^"{]+)\{?[^"]*"\]\s*=')
_TF_PREFIX = re.compile(r"^\d+[smhd]_")
_ROLL_SUFFIX = re.compile(r"_roll\d+_(?:mean|std)$")


def _base(col: str) -> str:
    col = _TF_PREFIX.sub("", col)
    col = _ROLL_SUFFIX.sub("", col)
    return col


def main() -> int:
    m = json.loads((HERE / "map.json").read_text())
    inv = json.loads((HERE / "inventory.json").read_text())
    known = set(m["features"]) | set(inv.get("passthrough_excluded", []))

    unmapped: dict[str, str] = {}
    fstring: dict[str, str] = {}
    for rel in _FEATURE_MODULES:
        p = DC_ROOT / rel
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            for mm in _ASSIGN.finditer(line):
                is_f, raw = mm.group(1), mm.group(2)
                base = _base(raw)
                if is_f:  # dynamic — report prefix for human confirmation
                    fstring.setdefault(raw, f"{rel}:{i}")
                    continue
                if base and base not in known:
                    unmapped.setdefault(base, f"{rel}:{i}")

    if fstring:
        print("FSTRING (dynamic targets — confirm variants are mapped):")
        for k, v in sorted(fstring.items()):
            print(f"  {k}*  @ {v}")
    if unmapped:
        print(f"\nUNMAPPED feature bases ({len(unmapped)}) — would ship REAL-NAMED:")
        for k, v in sorted(unmapped.items()):
            print(f"  {k}  @ {v}")
        return 1
    print("\nOK: every statically-assigned feature base is mapped or passthrough.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
