#!/usr/bin/env python3
"""Build the reversible obfuscation map from inventory.json.

STABLE / APPEND-ONLY (2026-07-24): codes already assigned in the committed
map.json are PRESERVED forever — names new to the inventory get the next
unused code; names retired from the inventory KEEP their map entry (harmless,
still decodable, and their code is never reused). Deployed artifacts
(coded weights meta, filter parquet columns, stack CSVs) reference codes, so
renumbering is a production-breaking event. The previous sequential
enumeration renumbered every code after any inventory removal — caught
2026-07-24 when dropping oi_velocity/oi_acceleration shifted r053..r072 and
f062..f100 (conftest re-runs this script before every test session).

Run (after extract_inventory.py):
    cd ~/Documents/sandbox/dc
    python obfuscation/build_map.py
Writes obfuscation/map.json.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INVENTORY = HERE / "inventory.json"
MAP = HERE / "map.json"


def _stable_assign(names: list[str], existing: dict[str, str], prefix: str) -> dict[str, str]:
    """Existing names keep their codes; retired names stay; new names append."""
    merged = dict(existing)
    used = set(merged.values())
    next_i = 1
    for name in sorted(names):
        if name in merged:
            continue
        while f"{prefix}{next_i:03d}" in used:
            next_i += 1
        merged[name] = f"{prefix}{next_i:03d}"
        used.add(merged[name])
    return dict(sorted(merged.items()))


def main():
    inv = json.loads(INVENTORY.read_text())
    regimes = sorted(inv["regime_atoms"])
    features = sorted(inv["feature_cols"])

    prev = json.loads(MAP.read_text()) if MAP.exists() else {}
    regime_map = _stable_assign(regimes, prev.get("regimes", {}), "r")
    feature_map = _stable_assign(features, prev.get("features", {}), "f")

    # Bijection guard: codes unique, and every inventoried name is mapped
    # (map may be a SUPERSET of inventory — retired names keep their codes).
    assert len(regime_map) == len(set(regime_map.values()))
    assert len(feature_map) == len(set(feature_map.values()))
    assert set(regimes) <= set(regime_map)
    assert set(features) <= set(feature_map)
    # No code may collide across namespaces' string form vs a real name, and no
    # real name may look like a code (would make decode ambiguous).
    all_codes = set(regime_map.values()) | set(feature_map.values())
    assert not (set(regimes) | set(features)) & all_codes, "name/code collision"

    out = {
        "version": 1,
        "note": "PRIVATE. Reversible code<->name map for marvel obfuscation. "
                "Structure (_and_, _long/_short, TF prefixes) preserved by codec.",
        "regimes": regime_map,
        "features": feature_map,
    }
    MAP.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {MAP} : {len(regime_map)} regimes, {len(feature_map)} features")


if __name__ == "__main__":
    main()
