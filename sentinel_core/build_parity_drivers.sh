#!/bin/bash
# Build the test drivers in tests/ inside the pinned build image.
#
# WHY THIS EXISTS
# ---------------
# Until now every driver was built from a hand-typed g++ line pasted between
# sessions. CMakeLists.txt has no add_executable for any of them, so the ONLY
# record of how to build the M1 gate lived in shell history. Two concrete costs:
#
#   * the source list is per-driver (model_parity needs LightGBM but not TA-Lib,
#     bar_parity needs neither) and a wrong list either fails to link or, worse,
#     links against a stale object and silently grades the wrong code;
#   * the pinned image is what guarantees TA-Lib 0.6.4 / LightGBM 4.6.0. A
#     driver built on the host against some other version is a silent parity
#     break, which is exactly the class of bug these drivers exist to catch.
#
# CLAUDE.md: a repeatable procedure lives in a committed script, never in
# reconstructed steps.
#
# Usage:
#   ./build_parity_drivers.sh                    # build every driver
#   ./build_parity_drivers.sh regime_parity      # build one (with or without
#   ./build_parity_drivers.sh regime_parity_driver #  the _driver suffix)
#   ./build_parity_drivers.sh --list             # show drivers and their sources
#
# Env:
#   SENTINEL_REPO   path to the public sentinel checkout (contract header)
#   IMAGE           override the build image
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENTINEL_REPO="${SENTINEL_REPO:-$(cd "$HERE/../../sentinel" 2>/dev/null && pwd || true)}"
IMAGE="${IMAGE:-mjolnir-core-build:latest}"
OUTDIR="$HERE/build-linux"

# Source sets, per driver. Kept EXPLICIT rather than derived from #include
# lines: an include graph tells you what compiles, not what must be linked, and
# a driver that links too little fails at link time while one that links a
# stale extra object can still produce numbers.
#
# Flags: TALIB pulls libta-lib (+ -lm, which it needs and does not pull in
# itself); LGBM pulls lib_lightgbm; SHA means the translation unit bakes in
# MJOLNIR_CORE_GITSHA.
driver_sources() {
    case "$1" in
        bar_parity_driver)       echo "src/bar_builder.cpp" ;;
        feature_parity_driver)   echo "src/bar_builder.cpp src/feature_engine.cpp src/talib_block.cpp|TALIB" ;;
        btc_cross_parity_driver) echo "src/bar_builder.cpp src/feature_engine.cpp src/talib_block.cpp|TALIB" ;;
        regime_parity_driver)    echo "src/bar_builder.cpp src/feature_engine.cpp src/regime_gate.cpp src/talib_block.cpp|TALIB" ;;
        model_parity_driver)     echo "src/model_runner.cpp src/regime_gate.cpp|LGBM" ;;
        # Header-only unit test: no extra TUs, no libs. NOSRC is the explicit
        # "zero sources" marker — an empty string is how driver_sources reports
        # "no recipe", so the two must not look alike.
        warmup_gate_driver)      echo "|NOSRC" ;;
        m1_gate_driver)          echo "src/bar_builder.cpp src/feature_engine.cpp src/regime_gate.cpp src/model_runner.cpp src/talib_block.cpp|TALIB,LGBM" ;;
        # Goes through the PUBLIC contract (makeCore), so it also needs
        # core_impl.cpp — and therefore the build-sha define.
        shadow_driver)           echo "src/core_impl.cpp src/bar_builder.cpp src/feature_engine.cpp src/regime_gate.cpp src/model_runner.cpp src/talib_block.cpp|TALIB,LGBM,SHA" ;;
        *)                       echo "" ;;
    esac
}

# Every driver in tests/ must have a recipe. A new driver that silently gets no
# entry recreates precisely the gap this script closes, so refuse to build.
ALL_DRIVERS=()
for f in "$HERE"/tests/*_driver.cpp; do
    [ -e "$f" ] || continue
    d="$(basename "$f" .cpp)"
    if [ -z "$(driver_sources "$d")" ]; then
        echo "FAIL: tests/$d.cpp has no entry in driver_sources()." >&2
        echo "      Add its source list rather than building it by hand." >&2
        exit 1
    fi
    ALL_DRIVERS+=("$d")
done
if [ ${#ALL_DRIVERS[@]} -eq 0 ]; then
    echo "FAIL: no tests/*_driver.cpp found under $HERE — nothing to build." >&2
    exit 1
fi

if [ "${1:-}" = "--list" ]; then
    for d in "${ALL_DRIVERS[@]}"; do
        printf '%-26s %s\n' "$d" "$(driver_sources "$d" | tr '|' ' ')"
    done
    exit 0
fi

# Select targets.
TARGETS=()
if [ $# -eq 0 ]; then
    TARGETS=("${ALL_DRIVERS[@]}")
else
    for want in "$@"; do
        d="$want"
        case "$d" in *_driver) ;; *) d="${d}_driver" ;; esac
        if [ -z "$(driver_sources "$d")" ] || [ ! -f "$HERE/tests/$d.cpp" ]; then
            echo "FAIL: unknown driver '$want'. Known:" >&2
            printf '  %s\n' "${ALL_DRIVERS[@]}" >&2
            exit 1
        fi
        TARGETS+=("$d")
    done
fi

# Preconditions — fail loud and specific, never build something subtly wrong.
if [ -z "$SENTINEL_REPO" ] || \
   [ ! -f "$SENTINEL_REPO/Strategy/ltp_release/ltp_strat_sdk/stan_code/mjolnir_core.hpp" ]; then
    echo "FAIL: contract header not found under SENTINEL_REPO=${SENTINEL_REPO:-<unset>}" >&2
    echo "      set SENTINEL_REPO=/path/to/sentinel" >&2
    exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "FAIL: build image '$IMAGE' not present." >&2
    echo "      docker build --platform linux/amd64 -f Dockerfile.build -t $IMAGE ." >&2
    echo "      It is what pins TA-Lib/LightGBM to the production versions;" >&2
    echo "      building on the host instead is a silent parity risk." >&2
    exit 1
fi

# core_impl bakes in the sha so a running core traces back to its source.
# Computed on the HOST: the container mounts only sentinel_core/, so dc's .git
# is absent and an in-container rev-parse would yield "unknown".
SHA="$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || true)"
if [ -n "$SHA" ] && [ -n "$(git -C "$HERE" status --porcelain -- "$HERE" 2>/dev/null)" ]; then
    SHA="${SHA}-dirty"
fi

mkdir -p "$OUTDIR"
# build_linux.sh builds the .so as root inside the container, which leaves
# build-linux/ root-owned. Since we deliberately compile as the host user (see
# --user below), reclaim the directory first — otherwise the link step fails
# with a bare "Permission denied" that reads like a docker problem.
if [ ! -w "$OUTDIR" ]; then
    echo "[build_parity_drivers] $OUTDIR is not writable — reclaiming ownership"
    docker run --rm --platform linux/amd64 -v "$HERE":/core "$IMAGE" \
        chown -R "$(id -u):$(id -g)" /core/build-linux
fi
echo "[build_parity_drivers] image=$IMAGE sentinel=$SENTINEL_REPO sha=${SHA:-<none>}"

# Emit one compile command per target, then run them all in a single container:
# container startup dominates a -O2 build of a handful of files.
SCRIPT="set -e"
for d in "${TARGETS[@]}"; do
    spec="$(driver_sources "$d")"
    srcs="${spec%%|*}"
    flags=""
    case "$spec" in *\|*) flags="${spec#*|}" ;; esac

    libs=""
    defs=""
    case ",$flags," in *,TALIB,*) libs="$libs /usr/local/lib/libta-lib.a -lm" ;; esac
    # SHARED, not static: the pinned image ships only lib_lightgbm.so — there is
    # no lib_lightgbm.a to link, and asking for one fails with "No such file or
    # directory" rather than falling back. CMakeLists.txt already resolves the
    # same .so for libmjolnir_core, so the drivers now match the library.
    # No -lgomp/-lpthread here: those were the STATIC lib's undefined symbols.
    # lib_lightgbm.so carries libgomp.so.1 and libpthread.so.0 as DT_NEEDED, so
    # the loader pulls them in, and no driver source uses OpenMP directly.
    case ",$flags," in *,LGBM,*)  libs="$libs /usr/local/lib/lib_lightgbm.so" ;; esac
    case ",$flags," in
        *,SHA,*)
            if [ -z "$SHA" ]; then
                echo "FAIL: $d links core_impl.cpp, which bakes in MJOLNIR_CORE_GITSHA," >&2
                echo "      but the dc sha could not be resolved. Refusing to build an" >&2
                echo "      untraceable core into a driver." >&2
                exit 1
            fi
            defs="-DMJOLNIR_CORE_GITSHA='\"$SHA\"'"
            ;;
    esac

    SCRIPT="$SCRIPT
echo '  -> $d'
g++ -std=c++17 -O2 -Wall -Wextra \
    -I/sentinel/Strategy/ltp_release/ltp_strat_sdk/stan_code \
    $defs \
    tests/$d.cpp $srcs $libs \
    -o /core/build-linux/$d"
done

# --user: without it the container writes the binaries as ROOT, and the host
# user then cannot delete or replace them. That is not cosmetic — it means a
# stale driver survives an `rm` that appears to have worked, and a stale driver
# silently grades the wrong code, which is the exact failure class these tests
# exist to catch.
docker run --rm --platform linux/amd64 \
    --user "$(id -u):$(id -g)" \
    -v "$HERE":/core \
    -v "$SENTINEL_REPO":/sentinel:ro \
    -w /core "$IMAGE" bash -c "$SCRIPT"

# Prove each requested binary actually landed — a build step that reports
# success without producing the artifact is the failure this whole file is
# about.
for d in "${TARGETS[@]}"; do
    if [ ! -x "$OUTDIR/$d" ]; then
        echo "FAIL: $OUTDIR/$d missing after build." >&2
        exit 1
    fi
done
echo "[build_parity_drivers] built ${#TARGETS[@]} driver(s) into $OUTDIR"
ls -1 "${TARGETS[@]/#/$OUTDIR/}"
