#!/bin/bash
# Cross-build libmjolnir_core.so for Linux x86_64 WITHOUT dc source leaving this
# machine.
#
# dc is local-only by policy: the source of truth for regime/feature IP stays
# here, and only BUILT artifacts deploy — the same discipline the pyarmor
# packages already follow. This builds inside a rockylinux:8 container (glibc
# 2.28 / gcc 8.5, matching the oldest deploy target) so the resulting .so loads
# on Rocky 8 and on newer-glibc hosts alike. Only the .so is ever shipped.
#
# Usage:
#   ./build_linux.sh                       # build
#   ./build_linux.sh --deploy dev105       # build, then scp the .so to a host
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENTINEL_REPO="${SENTINEL_REPO:-$(cd "$HERE/../../sentinel" && pwd)}"
IMAGE="rockylinux/rockylinux:8"
OUT="$HERE/build-linux/libmjolnir_core.so"

DEPLOY_HOST=""
if [ "${1:-}" = "--deploy" ]; then
    DEPLOY_HOST="${2:?--deploy needs a host}"
fi

if [ ! -f "$SENTINEL_REPO/Strategy/ltp_release/ltp_strat_sdk/stan_code/mjolnir_core.hpp" ]; then
    echo "FAIL: contract header not found under SENTINEL_REPO=$SENTINEL_REPO" >&2
    exit 1
fi

# Computed on the HOST: the container sees only sentinel_core/, so dc's .git is
# absent and an in-container rev-parse would silently produce "unknown".
SHA="$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || true)"
if [ -z "$SHA" ]; then
    echo "FAIL: cannot resolve dc git sha — refusing to build an untraceable core." >&2
    exit 1
fi
if [ -n "$(git -C "$HERE" status --porcelain -- "$HERE" 2>/dev/null)" ]; then
    SHA="${SHA}-dirty"
fi
echo "[build_linux] sha=$SHA sentinel=$SENTINEL_REPO"

docker run --rm --platform linux/amd64 \
    -v "$HERE":/core \
    -v "$SENTINEL_REPO":/sentinel:ro \
    "$IMAGE" bash -c "
        set -e
        dnf -q install -y gcc-c++ cmake make >/dev/null 2>&1
        cd /core && rm -rf build-linux
        cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release \
              -DSENTINEL_REPO=/sentinel -DMJOLNIR_CORE_GITSHA='$SHA'
        cmake --build build-linux -j\$(nproc)
    "

echo "[build_linux] built: $OUT"

# The artifact is the thing that leaves this machine — prove it carries no
# distinctive name before it does.
if ! python3 "$HERE/../obfuscation/audit_public_surface.py" --repo "$HERE/build-linux" >/dev/null 2>&1; then
    echo "FAIL: leak audit found a distinctive name in the build output." >&2
    python3 "$HERE/../obfuscation/audit_public_surface.py" --repo "$HERE/build-linux" >&2 || true
    exit 1
fi
echo "[build_linux] artifact leak audit: PASS"

if [ -n "$DEPLOY_HOST" ]; then
    echo "[build_linux] deploying .so ONLY to $DEPLOY_HOST (no source)"
    ssh "$DEPLOY_HOST" 'mkdir -p ~/mjolnir_core_lib'
    scp -q "$OUT" "$DEPLOY_HOST:~/mjolnir_core_lib/"
    ssh "$DEPLOY_HOST" 'ls -l ~/mjolnir_core_lib/libmjolnir_core.so'
fi
