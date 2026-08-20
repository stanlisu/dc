#!/usr/bin/env python3
"""Fail-loud gate: no distinctive regime/feature name may appear in a PUBLIC repo.

Companion to ``audit_feature_coverage.py``. That one asks "is every feature we
PRODUCE obfuscatable?"; this one asks "did a real name LEAK into a repo we may
share or publish?" — e.g. the ``sentinel`` C++ port, which vendors a third-party
SDK and is a candidate for publication.

WHY THIS LIVES IN dc (and not in the public repo's own CI)
----------------------------------------------------------
The gate needs the secret name list to grep for, and that list IS the secret.
Shipping ``map.json`` into the public repo to "let CI check itself" would leak
all 172 names in one commit — the exact thing being prevented. So the scanner
runs HERE, where the map already is, and points AT a public checkout:

    python obfuscation/audit_public_surface.py --repo ~/Documents/sandbox/sentinel

Exit 1 on any genuine hit. Wire it as (a) a pre-commit hook in the public repo
that shells out to this script, and (b) a dc-side CI job scanning the public
checkout. Do NOT try to make the public repo self-checking.

DISTINCTIVE vs COMMON
---------------------
A raw grep of all 172 mapped names over a real codebase is ~90% false positives:
``spread``, ``delta``, ``std`` are ordinary words, and ``rsi``/``macd``/``atr``
are public TA-Lib functions — knowing a strategy uses RSI reveals nothing. The
2026-07-01 audit hit this exactly (1111 raw hits -> 22 lines, all FP).

So bare single-token names that are standard TA indicators, option greeks, or
generic statistics are treated as NON-distinctive and skipped. Compound names
(``wide_spread``, ``depth_imbalance_L1``) still match in full — the allowlist
only suppresses the exact bare token.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Bare tokens that are NOT IP. Compound names containing them still match.
#   - standard TA-Lib indicator functions (public API, reveal nothing)
#   - option greeks / generic statistics / ordinary market vocabulary
_NON_DISTINCTIVE = {
    # TA-Lib
    "ad", "adx", "atr", "bop", "cci", "cmo", "dx", "macd", "macdhist", "mfi",
    "mom", "natr", "obv", "roc", "rsi", "sar", "trix", "ultosc", "willr",
    # greeks / stats / ordinary market terms
    "delta", "gamma", "theta", "vega", "skew", "kurt", "std", "spread",
    "notional", "pv", "microprice", "contango", "backwardation",
}

# Never scan these (binaries, vendored third-party SDK, VCS, build output).
_SKIP_DIRS = {".git", "build", "bin", "lib", "__pycache__", "node_modules", ".venv"}
_SKIP_SUFFIX = {
    ".so", ".o", ".a", ".out", ".bin", ".png", ".jpg", ".jpeg", ".gif", ".pdf",
    ".zip", ".gz", ".tar", ".parquet", ".pkl", ".pyc", ".ico", ".woff", ".woff2",
}


def load_distinctive(map_path: Path) -> list[str]:
    """Real names worth gating on, longest-first so overlaps report the specific one."""
    m = json.loads(map_path.read_text())
    names = set(m["regimes"]) | set(m["features"])
    distinctive = {n for n in names if n.lower() not in _NON_DISTINCTIVE}
    return sorted(distinctive, key=len, reverse=True)


# Printable-ASCII runs, the same rule `strings(1)` uses (default -n 4). Applied
# to files the text scanner cannot read, so a COMPILED ARTIFACT can be audited.
_STRINGS_RE = re.compile(rb"[\x20-\x7e]{4,}")


def scan_binary(path: Path, pattern: re.Pattern) -> list[tuple[int, str, str]]:
    """`strings` a binary and match the same pattern against each run.

    WHY THIS EXISTS. The text scanner below skips anything that is not valid
    UTF-8, and `_SKIP_SUFFIX` skips `.so`/`.o`/`.a` outright — which is correct
    for a source-tree audit and is exactly WRONG when the thing being audited is
    the .so itself. VERIFIED 2026-08-20: `sentinel_core/build_linux.sh:75` points
    this script at `build-linux/`, and `_SKIP_SUFFIX` (line 58-61) contains
    `.so`/`.o`/`.a` — so that run reads CMakeCache.txt and the build logs, prints
    "artifact leak audit: PASS", and never opens the library it just built. The
    gate has been hollow, not merely narrow.

    No leak has actually been observed in a shipped artifact: both
    `libmjolnir_core.so` and `libagamotto_core.so` PASS under `--binaries` as of
    2026-08-20. What is demonstrated is that the old run could not have found one
    — confirmed by planting a real name from map.json into a synthetic .so, which
    the scanner exits 1 on with `--binaries` and silently passes without it.

    String literals are where names re-enter a compiled artifact — exception
    text, log formats, column keys — and none of them look like data in review.
    """
    try:
        blob = path.read_bytes()
    except OSError:
        return []
    out = []
    for m in _STRINGS_RE.finditer(blob):
        run = m.group().decode("ascii")
        for found in pattern.finditer(run):
            out.append((m.start(), found.group(1), run[:110]))
    return out


def scan(repo: Path, names: list[str], extra_skip: list[str],
         binaries: bool = False) -> list[tuple[Path, int, str, str]]:
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
    hits: list[tuple[Path, int, str, str]] = []
    for path in sorted(repo.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if any(str(rel).startswith(s) for s in extra_skip):
            continue
        if path.suffix.lower() in _SKIP_SUFFIX:
            # --binaries turns the artifact itself back into a target. Only the
            # linkable/compiled kinds: there is nothing to recover from a .png.
            if binaries and path.suffix.lower() in {".so", ".o", ".a", ".bin", ".out"}:
                hits.extend((rel, off, name, run)
                            for off, name, run in scan_binary(path, pattern))
            continue
        try:
            text = path.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            # WHY: not text. With --binaries an unsuffixed EXECUTABLE still has
            # to be audited — an artifact that escapes because nobody gave it an
            # extension is the same leak.
            if binaries:
                hits.extend((rel, off, name, run)
                            for off, name, run in scan_binary(path, pattern))
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            # finditer, not search: a single line can carry several names
            # (e.g. a mapping table) and reporting only the first hides the rest.
            for found in pattern.finditer(line):
                hits.append((rel, lineno, found.group(1), line.strip()[:110]))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="path to the PUBLIC repo checkout to scan")
    ap.add_argument("--map", default=str(HERE / "map.json"), help="obfuscation map.json")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="repo-relative path prefixes to skip (vendored SDK, etc.)")
    ap.add_argument("--binaries", action="store_true",
                    help="ALSO scan compiled artifacts (.so/.o/.a/.bin/.out and "
                         "extensionless non-text files) by extracting printable "
                         "runs, as strings(1) does. Required when --repo points "
                         "at a BUILD DIRECTORY: without it the .so is skipped by "
                         "suffix and the run passes without ever opening the "
                         "thing it claims to audit.")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        raise SystemExit(f"--repo is not a directory: {repo}")
    map_path = Path(args.map).expanduser().resolve()
    if not map_path.is_file():
        raise SystemExit(f"map.json not found: {map_path}")

    names = load_distinctive(map_path)
    hits = scan(repo, names, args.skip, binaries=args.binaries)

    print(f"[audit_public_surface] repo={repo}")
    print(f"[audit_public_surface] gating on {len(names)} distinctive names "
          f"({len(_NON_DISTINCTIVE)} common tokens allowlisted)")
    if args.skip:
        print(f"[audit_public_surface] skipping prefixes: {args.skip}")
    if args.binaries:
        print("[audit_public_surface] compiled artifacts ARE being scanned "
              "(printable runs, as strings(1))")

    if not hits:
        print("=== PASS: no distinctive regime/feature name found ===")
        return 0

    print(f"\n=== FAIL: {len(hits)} leak line(s) ===\n")
    for rel, lineno, name, line in hits:
        # `lineno` is a line number for text and a BYTE OFFSET for a binary.
        print(f"  {rel}:{lineno}: [{name}]  {line}")
    print(f"\n{len(hits)} leak(s). Replace real names with their codes "
          f"(codec.encode_regime / encode_columns) or exclude the file.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
