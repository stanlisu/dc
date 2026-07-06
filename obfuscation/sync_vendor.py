#!/usr/bin/env python3
"""Vendor the self-contained codec + map into every dc algo package.

The codec must be importable from inside each *_pkg at runtime on the servers,
but build_distribution.sh only builds/deploys the *_pkg dirs (not dc/obfuscation/).
Each package is built independently and several (mjolnir/stormbreaker/valkyrie)
do NOT depend on agamotto, so a single shared location won't do. We therefore
vendor a copy of codec.py + map.json into each package under <pkg>/_obf/.

Single source of truth: dc/obfuscation/{codec.py,map.json}. Re-run after
rebuilding the map; build_distribution.sh also calls this before pyarmor so a
deploy never ships a stale copy.

Import from inside a package:  from <pkg>._obf.codec import default

Run:
    cd ~/Documents/sandbox/dc
    python obfuscation/build_map.py && python obfuscation/sync_vendor.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DC_ROOT = HERE.parent
CODEC = HERE / "codec.py"
MAP = HERE / "map.json"

# pkg dir name -> top-level package module dir under src/
# (mjolnir/valkyrie nest their code under <pkg>/core, but _obf goes at the top
#  package level so the import path is always <pkg>._obf.codec)
_TARGETS = {
    "agamotto_pkg": "agamotto",
    "orb_pkg": "orb",
    "scepter_pkg": "scepter",
    "aether_pkg": "aether",
    "mjolnir_pkg": "mjolnir",
    "stormbreaker_pkg": "stormbreaker",
    "vomir_pkg": "vomir",
    "valkyrie_pkg": "valkyrie",
}

_INIT = '"""Vendored obfuscation codec (synced from dc/obfuscation). Do not edit here."""\n'


def main():
    if not MAP.exists():
        raise SystemExit("map.json missing — run build_map.py first")
    n = 0
    for pkg_dir, mod in _TARGETS.items():
        dest = DC_ROOT / pkg_dir / "src" / mod / "_obf"
        if not dest.parent.exists():
            print(f"  skip {pkg_dir}: {dest.parent} not found")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "__init__.py").write_text(_INIT)
        shutil.copy2(CODEC, dest / "codec.py")
        shutil.copy2(MAP, dest / "map.json")
        print(f"  vendored -> {dest.relative_to(DC_ROOT)}")
        n += 1
    print(f"synced codec+map into {n} packages")


if __name__ == "__main__":
    main()
