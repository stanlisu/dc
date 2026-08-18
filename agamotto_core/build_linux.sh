#!/bin/bash
# Build libagamotto_core.so for Linux x86_64 in the OLDEST-TARGET image.
#
# Mirrors ../sentinel_core/build_linux.sh, and for the same reason. Building in
# devbox-v5.1 (the vendor SDK image) produces a .so that requires
# GLIBCXX_3.4.29: it loads on hydra, and fails on dev105 and any Rocky 8-class
# host. Measured 2026-08-18 — the devbox build needs 3.4.29, the rockylinux:8
# build needs none of it and resolves on every target.
#
# Only the built .so ever leaves this machine; dc source is local-only by
# policy, the same discipline the pyarmor packages follow.
#
# Usage:
#   ./build_linux.sh                    # build
#   ./build_linux.sh --deploy dev105    # build, then scp the .so to a host
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENTINEL_REPO="${SENTINEL_REPO:-$(cd "$HERE/../../sentinel" && pwd)}"
IMAGE="mjolnir-core-build:latest"
BUILD_DIR="$HERE/build-linux"
OUT="$BUILD_DIR/libagamotto_core.so"

DEPLOY_HOST=""
if [ "${1:-}" = "--deploy" ]; then
    DEPLOY_HOST="${2:?--deploy needs a host}"
fi

CONTRACT="$SENTINEL_REPO/Strategy/ltp_release/ltp_strat_sdk/stan_code/agamotto_core.hpp"
if [ ! -f "$CONTRACT" ]; then
    echo "FAIL: contract header not found at $CONTRACT" >&2
    echo "      set SENTINEL_REPO=/path/to/sentinel" >&2
    exit 1
fi

# Computed on the HOST: the container sees only agamotto_core/, so dc's .git is
# absent and an in-container rev-parse would silently produce "unknown",
# losing the trace from a running strategy back to the core that built it.
SHA="$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || true)"
if [ -z "$SHA" ]; then
    echo "FAIL: cannot resolve dc git sha — refusing to build an untraceable core." >&2
    exit 1
fi
if [ -n "$(git -C "$HERE" status --porcelain -- "$HERE" 2>/dev/null)" ]; then
    SHA="${SHA}-dirty"
fi

echo "=== building libagamotto_core.so in $IMAGE (sha=$SHA) ==="
docker run --rm \
    -v "$HERE:/src" \
    -v "$SENTINEL_REPO:/sentinel:ro" \
    -w /src "$IMAGE" bash -c "
        set -e
        cmake -S . -B build-linux -DSENTINEL_REPO=/sentinel \
              -DAGAMOTTO_CORE_GITSHA='$SHA' >/dev/null
        cmake --build build-linux -j\$(nproc)
        ./build-linux/kline_parity_driver --selftest | tail -2
    "

[ -f "$OUT" ] || { echo "FAIL: $OUT was not produced" >&2; exit 1; }

# The whole point of this image: refuse to ship a .so that needs a libstdc++
# newer than the oldest deploy target. Catch it here, not at dlopen time on a
# host where the failure reads as a missing plugin.
if strings "$OUT" | grep -qE 'GLIBCXX_3\.4\.(2[9]|[3-9][0-9])'; then
    echo "FAIL: $OUT requires GLIBCXX_3.4.29+ — it was not built in $IMAGE" >&2
    strings "$OUT" | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -uV | tail -3 >&2
    exit 1
fi

echo "=== built $OUT (sha=$SHA), portable to the oldest target ==="
if [ -n "$DEPLOY_HOST" ]; then
    scp "$OUT" "$DEPLOY_HOST:~/agamotto_core_lib/libagamotto_core.so"
    echo "=== deployed to $DEPLOY_HOST:~/agamotto_core_lib/ ==="
fi
