#!/usr/bin/env python3
"""Build the reversible obfuscation map from inventory.json.

Deterministic: sorted atom/feature names -> zero-padded codes, so the map is
reproducible and reviewable. Regime atoms -> r001.., features -> f001..
Structural tokens (_and_, _long/_short, TF prefixes) are NOT mapped here; the
codec preserves them at encode/decode time.

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


def main():
    inv = json.loads(INVENTORY.read_text())
    regimes = sorted(inv["regime_atoms"])
    features = sorted(inv["feature_cols"])

    regime_map = {name: f"r{i:03d}" for i, name in enumerate(regimes, start=1)}
    feature_map = {name: f"f{i:03d}" for i, name in enumerate(features, start=1)}

    # Bijection guard (codes unique by construction; assert names unique too).
    assert len(regime_map) == len(set(regime_map.values())) == len(regimes)
    assert len(feature_map) == len(set(feature_map.values())) == len(features)
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
