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
IMAGE="mjolnir-core-build:latest"
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
        cd /core && rm -rf build-linux
        cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release \
              -DSENTINEL_REPO=/sentinel -DMJOLNIR_CORE_GITSHA='$SHA'
        cmake --build build-linux -j\$(nproc)
    "

echo "[build_linux] built: $OUT"

# The artifact is the thing that leaves this machine — prove it carries no
# distinctive name before it does.
#
# The scanner needs py3.7+ (it uses `from __future__ import annotations`), and a
# bare `python3` is 3.6 on some build hosts. A scanner that cannot RUN is not a
# leak — reporting it as one trains the operator to ignore the gate, which is
# how a real leak eventually ships. Separate the two outcomes explicitly.
AUDIT_PY=""
for c in "${AUDIT_PYTHON:-}" python3.13 python3.12 python3.11 python3 "$HOME/miniconda3/envs/py313/bin/python3" /opt/miniconda3/envs/py313/bin/python3; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,7) else 1)' 2>/dev/null; then
        AUDIT_PY="$c"; break
    fi
done
if [ -z "$AUDIT_PY" ]; then
    echo "BLOCKED: leak audit could not RUN — no python >= 3.7 found. This is NOT a pass." >&2
    echo "         Set AUDIT_PYTHON=/path/to/python3, or audit the .so on a host that has one." >&2
    exit 2
fi
AUDIT_OUT="$("$AUDIT_PY" "$HERE/../obfuscation/audit_public_surface.py" --repo "$HERE/build-linux" 2>&1)"; AUDIT_RC=$?
if [ "$AUDIT_RC" -ne 0 ]; then
    echo "FAIL: leak audit found a distinctive name in the build output." >&2
    echo "$AUDIT_OUT" >&2
    exit 1
fi
echo "[build_linux] artifact leak audit: PASS ($AUDIT_PY)"

if [ -n "$DEPLOY_HOST" ]; then
    echo "[build_linux] deploying .so ONLY to $DEPLOY_HOST (no source)"
    ssh "$DEPLOY_HOST" 'mkdir -p ~/mjolnir_core_lib'
    scp -q "$OUT" "$DEPLOY_HOST:~/mjolnir_core_lib/"
    ssh "$DEPLOY_HOST" 'ls -l ~/mjolnir_core_lib/libmjolnir_core.so'
fi
