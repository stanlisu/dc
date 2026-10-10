#!/usr/bin/env bash
# Negative controls for tests/ctx_tf_driver.cpp (ABI 6, orb context timeframes).
#
# A test that has never failed proves nothing. Each mutant below breaks ONE rule
# of the context path in src/core_impl.cpp or src/feature_engine.cpp; the driver
# must fail on every one. A survivor is a rule the driver does not actually check.
#
# Runs on the BUILD HOST (dev105): mutants are applied here with python3, and
# each one is compiled and run inside the build image, which has no python.
#
# Usage:  bash tests/run_ctx_tf_mutants.sh <sentinel-repo> [image]
set -uo pipefail
if [ -z "${1:-}" ]; then
    echo "usage: $0 <path-to-sentinel-repo>" >&2
    exit 2
fi
SENTINEL="$(cd "$1" && pwd)"
IMAGE="${2:-mjolnir-core-build:latest}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
# The image runs as root, so its build dirs are root-owned: clean them up
# through the image too, or the host cannot remove them.
trap 'docker run --rm -v "$WORK:/w" "$IMAGE" rm -rf /w/base /w/m >/dev/null 2>&1; rm -rf "$WORK"' EXIT

build_and_run() {   # $1 = source tree; exit status of the driver, 90 = no build
    local tree="$1"
    docker run --rm -v "$tree:/src" -v "$SENTINEL:/sentinel:ro" -w /src "$IMAGE" bash -c '
        cmake -S . -B build-mut -DSENTINEL_REPO=/sentinel \
              -DAGAMOTTO_CORE_GITSHA=mutant >/dev/null 2>&1 || exit 90
        cmake --build build-mut --target ctx_tf_driver -j"$(nproc)" >/dev/null 2>&1 || exit 90
        ./build-mut/ctx_tf_driver >/dev/null 2>&1' 
}

# THE BASELINE. Without it a driver that fails on the unmutated core would
# report every mutant as killed.
echo "=== baseline (unmutated) ==="
rsync -a --exclude "build*/" "$HERE/" "$WORK/base/"
build_and_run "$WORK/base"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "BASELINE FAILS (rc=$rc) -- fix the core before reading any mutant result"
    exit 1
fi
echo "  baseline passes"

killed=0
survived=0
mutate() {   # name, file (relative to agamotto_core), old, new
    local name="$1" file="$2" old="$3" new="$4"
    docker run --rm -v "$WORK:/w" "$IMAGE" rm -rf /w/m >/dev/null 2>&1
    rsync -a --exclude "build*/" "$WORK/base/" "$WORK/m/"
    if ! F="$WORK/m/$file" OLD="$old" NEW="$new" python3 - <<'PYEOF'
import os, pathlib, sys
p = pathlib.Path(os.environ["F"]); s = p.read_text()
old, new = os.environ["OLD"], os.environ["NEW"]
if s.count(old) != 1:
    sys.stderr.write("anchor count %d\n" % s.count(old)); sys.exit(9)
p.write_text(s.replace(old, new, 1))
PYEOF
    then
        echo "  ANCHOR MISSING  $name"; survived=$((survived + 1)); return
    fi
    build_and_run "$WORK/m"
    local rc=$?
    if [ "$rc" -eq 90 ]; then
        echo "  DID NOT BUILD   $name"; survived=$((survived + 1))
    elif [ "$rc" -eq 0 ]; then
        echo "  SURVIVED        $name"; survived=$((survived + 1))
    else
        echo "  killed          $name"; killed=$((killed + 1))
    fi
}

echo
echo "=== mutants ==="
mutate "the NEWEST context bar instead of the as-of one" src/context_asof.hpp \
    "const auto it = std::upper_bound(close_ms.begin(), close_ms.end(), base_open_ms);" \
    "const auto it = close_ms.end();"
mutate "no freshness check (a stale bar is used)" src/context_asof.hpp \
    "    if (*row != need) {" \
    "    if (false) {"
mutate "one-row shift (the as-of row minus one)" src/context_asof.hpp \
    "return static_cast<int>(row - close_ms.begin());" \
    "return static_cast<int>(row - close_ms.begin()) - 1;"
mutate "a context atom read off the BASE panel" src/core_impl.cpp \
    "atomMask(mCtxPanels.at(tf).panel, sp.atoms[k], sp.pos);" \
    "atomMask(mPanel, sp.atoms[k], sp.pos);"
mutate "context atoms routed through the base-only path" src/core_impl.cpp \
    "                sp.ctx = true;" \
    "                sp.ctx = false;"
mutate "the timeframe prefix dropped from the weights directory" src/core_impl.cpp \
    "dirs.push_back(prefixed ? regimeDirName(sp.atoms, sp.tfs, sp.pos)" \
    "dirs.push_back(false ? regimeDirName(sp.atoms, sp.tfs, sp.pos)"
mutate "a still-open context bar kept" src/core_impl.cpp \
    "if (b.bucket_open_ms + tf_ms > now_ms) {" \
    "if (false) {"
mutate "an unconfigured atom timeframe accepted" src/core_impl.cpp \
    "                if (mCtxTfs.count(tfi) == 0) {" \
    "                if (false) {"
mutate "a short (whole-history) context panel refused" src/feature_engine.cpp \
    "if (n < MIN_CONTEXT_BARS || n > PANEL_BARS)" \
    "if (n != PANEL_BARS)"

echo
echo "killed=$killed survived=$survived"
[ "$survived" -eq 0 ]
