#!/usr/bin/env python3
"""Post-deploy gate: prove the bytes we just rsynced are the bytes python imports.

Run ON the target host, with that host's own interpreter, after the rsync +
install. Exits non-zero — loudly — if any package fails.

    ssh <host> '<py313> - /home/stan/sandbox/marvel --expect <algo>=<sha256>' \
        < dc/scripts/verify_deploy.py

WHY THIS EXISTS
---------------
`deploy_algo()` in build_distribution.sh only ever rsynced
`build_dist/<algo>_pkg/` -> `<host>:/home/stan/sandbox/marvel/<algo>_pkg/`.
Whether those bytes reach the import path is a property of how each package
happens to be *installed* on each host, which the deploy never controlled:

  - EDITABLE install  -> a .pth points at the rsynced tree; rsync is authoritative.
  - NON-EDITABLE      -> pip copied the tree into site-packages at install time
                         and that copy is frozen. rsync NEVER reaches it. The
                         deploy prints success and the old code keeps running.

That is not hypothetical. `dc/.github/workflows/deploy.yml` ran
`pip install -e A/ B/ C/ ...` — where pip binds `-e` only to the path that
immediately follows it — so `agamotto` was editable and the other EIGHT were
installed as frozen copies. Measured on shield 2026-08-09:
`mjolnir.core.ladder` resolved to a site-packages copy dated 2026-08-06 while
the deployed tree held the 2026-08-08 build; every dc deploy to shield since
2026-08-06 had been a silent no-op for 8 of 9 packages. The import smoke-test
in the old runbook could not catch it — the STALE copy imports perfectly.

So the check is NOT "does it import". It is:
  1. IDENTITY  — does `find_spec(algo)` resolve INSIDE the deployed tree?
  2. INTEGRITY — does the deployed tree digest match what we just built?
  3. LOADABILITY — does it actually import (pyarmor runtime binding)?
A pass on (3) alone is exactly the false green that hid this for three days.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys

ALL_ALGOS = ["agamotto", "orb", "aether", "scepter", "mjolnir",
             "stormbreaker", "vibranium", "valkyrie", "vomir"]

# A submodule worth importing per algo: the top-level __init__ is often a thin
# re-export that binds the pyarmor runtime lazily, so importing only the package
# can pass on a build whose real modules are unloadable.
DEEP_IMPORT = {
    "agamotto": "agamotto.research",
    "mjolnir": "mjolnir.core.ladder",
    "orb": "orb.research",
}

# vibranium is deliberately excluded from obfuscation/sync_vendor._TARGETS and
# ships no _obf/map.json — see dc/README.md "Vendoring". Absent != broken.
NO_OBF_MAP = {"vibranium"}

_SKIP_DIRS = {"__pycache__"}
_SKIP_SUFFIX = (".pyc", ".pyo")


def tree_digest(root: str) -> str | None:
    """sha256 over (relpath, filehash) for every file under `root`.

    Excludes __pycache__/*.pyc (written by whoever imports first, on the host
    only) and *.egg-info (written by the install, not by the build), so the
    digest compares the SHIPPED bytes and nothing else.
    """
    if not os.path.isdir(root):
        return None
    h = hashlib.sha256()
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in _SKIP_DIRS and not d.endswith(".egg-info"))
        for fn in sorted(filenames):
            if fn.endswith(_SKIP_SUFFIX):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            fh = hashlib.sha256()
            try:
                with open(full, "rb") as fp:
                    for chunk in iter(lambda: fp.read(1 << 20), b""):
                        fh.update(chunk)
            except OSError as exc:
                raise SystemExit("verify_deploy: cannot read %s: %s" % (full, exc))
            entries.append((rel, fh.hexdigest()))
    for rel, fhex in sorted(entries):
        h.update(rel.encode()); h.update(b"\0"); h.update(fhex.encode()); h.update(b"\0")
    return h.hexdigest()


def install_mode(algo: str) -> str:
    """How pip recorded this package: EDITABLE / HARD-INSTALLED / (none)."""
    try:
        import sysconfig
        purelib = sysconfig.get_paths()["purelib"]
    except Exception:
        return "unknown"
    editable = hard = False
    try:
        for name in os.listdir(purelib):
            low = name.lower()
            if low == "__editable__.%s" % algo or (
                    low.startswith("__editable__.") and low.split(".")[1].split("-")[0] == algo):
                editable = True
            if low.startswith(algo + "-") and low.endswith(".dist-info"):
                du = os.path.join(purelib, name, "direct_url.json")
                if os.path.exists(du):
                    try:
                        with open(du) as fp:
                            if json.load(fp).get("dir_info", {}).get("editable"):
                                editable = True
                    except Exception:
                        pass
        if os.path.isdir(os.path.join(purelib, algo)):
            hard = True
    except OSError:
        return "unknown"
    if hard and editable:
        return "MIXED"
    if hard:
        return "HARD-INSTALLED"
    if editable:
        return "EDITABLE"
    return "no-install-record"


def check(algo: str, marvel_root: str, expect: dict[str, str]) -> tuple[bool, list[str]]:
    """Return (ok, lines). Every failure line starts with 'FAIL'."""
    lines = []
    ok = True
    deployed = os.path.join(marvel_root, "%s_pkg" % algo, "src", algo)
    mode = install_mode(algo)

    if not os.path.isdir(deployed):
        return False, ["FAIL %-13s no deployed tree at %s" % (algo, deployed)]

    # (1) IDENTITY — the check the old import smoke-test could not make.
    spec = importlib.util.find_spec(algo)
    if spec is None:
        return False, ["FAIL %-13s not importable at all (install missing); "
                       "deployed tree exists at %s" % (algo, deployed)]
    locs = list(spec.submodule_search_locations or [])
    if not locs:
        return False, ["FAIL %-13s resolved to a non-package: %s" % (algo, spec.origin)]
    resolved = os.path.realpath(locs[0])
    if resolved != os.path.realpath(deployed):
        ok = False
        lines.append(
            "FAIL %-13s SHADOWED — python imports a different tree than we deployed\n"
            "       deployed: %s\n"
            "       imported: %s\n"
            "       install mode: %s  (a non-editable install froze a copy at install\n"
            "       time; rsync cannot reach it — reinstall with `pip install -e`)"
            % (algo, deployed, resolved, mode))

    # (2) INTEGRITY — the deployed tree must be exactly what we built.
    want = expect.get(algo)
    if want:
        src_root = os.path.join(marvel_root, "%s_pkg" % algo, "src")
        got = tree_digest(src_root)
        if got != want:
            ok = False
            lines.append(
                "FAIL %-13s DIGEST MISMATCH under %s\n"
                "       built:    %s\n"
                "       on host:  %s\n"
                "       (partial/interrupted rsync, or the host was written by "
                "another deploy)" % (algo, src_root, want, got))
        else:
            lines.append("       %-13s digest %s ok" % (algo, want[:16]))

    # (3) LOADABILITY — pyarmor runtime actually binds.
    for target in (algo, DEEP_IMPORT.get(algo)):
        if not target:
            continue
        try:
            importlib.import_module(target)
        except Exception as exc:
            ok = False
            lines.append("FAIL %-13s import %s raised %s: %s"
                         % (algo, target, type(exc).__name__, exc))

    # (4) the vendored codec DATA file survived the build + install.
    if algo not in NO_OBF_MAP:
        mp = os.path.join(resolved, "_obf", "map.json")
        if not os.path.exists(mp):
            ok = False
            lines.append("FAIL %-13s vendored _obf/map.json missing under %s "
                         "(pyarmor drops DATA files; pip drops unlisted "
                         "package_data)" % (algo, resolved))
        else:
            with open(mp, "rb") as fp:
                md5 = hashlib.md5(fp.read()).hexdigest()
            lines.append("       %-13s map.json md5 %s" % (algo, md5))

    if ok and not any(ln.startswith("FAIL") for ln in lines):
        lines.insert(0, "  OK   %-13s %-16s -> %s" % (algo, mode, resolved))
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # The builder computes the expected digest with THIS SAME script, so the two
    # sides of the comparison can never drift apart into a false pass.
    if len(sys.argv) >= 3 and sys.argv[1] == "--print-digest":
        d = tree_digest(sys.argv[2])
        if d is None:
            print("verify_deploy: no such dir: %s" % sys.argv[2], file=sys.stderr)
            return 1
        print(d)
        return 0
    ap.add_argument("marvel_root")
    ap.add_argument("--algos", default=",".join(ALL_ALGOS))
    ap.add_argument("--expect", action="append", default=[],
                    metavar="ALGO=SHA256",
                    help="expected tree digest of <root>/<algo>_pkg/src, from the builder")
    args = ap.parse_args()

    expect = {}
    for item in args.expect:
        if "=" not in item:
            ap.error("--expect wants ALGO=SHA256, got %r" % item)
        k, v = item.split("=", 1)
        expect[k] = v

    algos = [a for a in args.algos.split(",") if a]
    unknown = [a for a in algos if a not in ALL_ALGOS]
    if unknown:
        ap.error("unknown algos: %s" % ", ".join(unknown))

    print("verify_deploy on %s" % os.uname().nodename)
    print("  interpreter: %s (%s)" % (sys.executable, sys.version.split()[0]))
    print("  marvel root: %s" % args.marvel_root)
    print("")

    failed = []
    for algo in algos:
        ok, lines = check(algo, args.marvel_root, expect)
        for ln in lines:
            print(ln)
        if not ok:
            failed.append(algo)

    print("")
    if failed:
        print("=" * 72)
        print("DEPLOY VERIFICATION FAILED: %s" % ", ".join(failed))
        print("The rsync succeeded but those packages are NOT what python imports.")
        print("Do not treat this deploy as delivered. Fix the install, re-verify.")
        print("=" * 72)
        return 1
    print("verify_deploy: all %d package(s) OK — deployed bytes are the imported bytes."
          % len(algos))
    return 0


if __name__ == "__main__":
    sys.exit(main())
