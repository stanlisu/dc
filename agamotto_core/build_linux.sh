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
#
# AGAMOTTO_CORE_GITSHA in the ENVIRONMENT overrides it. That is not a fallback,
# it is the documented remote path: dc source is local-only by policy, so a
# build host (dev105) receives agamotto_core/ by scp into a directory that is
# NOT a git checkout and cannot resolve a sha of its own. Passing the sha the
# source actually came from is the only way that build stays traceable. With
# neither, this still refuses to build.
SHA="${AGAMOTTO_CORE_GITSHA:-}"
if [ -z "$SHA" ]; then
    SHA="$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || true)"
    if [ -n "$(git -C "$HERE" status --porcelain -- "$HERE" 2>/dev/null)" ]; then
        SHA="${SHA}-dirty"
    fi
fi
if [ -z "$SHA" ]; then
    echo "FAIL: cannot resolve dc git sha and AGAMOTTO_CORE_GITSHA is unset —" >&2
    echo "      refusing to build an untraceable core." >&2
    exit 1
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
        # Stage 2.6: the bar layer passing its own self-tests says nothing
        # about whether the feature engine is actually WIRED to it. Run the
        # integration gate here so a core that computes no panel cannot ship.
        ./build-linux/core_integration_driver | tail -12
        # Stage 3: the gate's own consistency sweep — atomIsKnown() (what
        # setRegimeStack validates against) must agree with atomMask() (what
        # runs every bar) on every code. Disagreement means a stack that
        # validates at boot and throws on the first warm panel, 7.3 days later.
        ./build-linux/regime_parity_driver --selftest
    "

[ -f "$OUT" ] || { echo "FAIL: $OUT was not produced" >&2; exit 1; }

# THE ARTIFACT LEAK AUDIT. The .so is the only thing that leaves this machine,
# so prove it carries no distinctive regime or feature name BEFORE it does.
#
# --binaries IS LOAD-BEARING. Without it the scanner skips `.so` by suffix and
# the run "passes" without ever opening the library — which is what
# ../sentinel_core/build_linux.sh has been doing since 2026-07 while printing
# "artifact leak audit: PASS". Measured 2026-08-20: this .so carried a real
# feature name in a `throw` message that a suffix-skipping audit could not see.
#
# A scanner that cannot RUN is NOT a pass, and is reported as its own outcome:
# treating it as a failure trains the operator to ignore the gate, which is how
# a real leak eventually ships.
AUDIT_PY=""
for c in "${AUDIT_PYTHON:-}" python3.13 python3.12 python3.11 python3 \
         "$HOME/miniconda3/envs/py313/bin/python3" /opt/miniconda3/envs/py313/bin/python3; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,7) else 1)' 2>/dev/null; then
        AUDIT_PY="$c"; break
    fi
done
AUDIT_SCRIPT="$HERE/../obfuscation/audit_public_surface.py"
if [ -z "$AUDIT_PY" ]; then
    echo "BLOCKED: leak audit could not RUN — no python >= 3.7 found. This is NOT a pass." >&2
    echo "         Set AUDIT_PYTHON=/path/to/python3, or audit the .so on a host that has one." >&2
    exit 2
elif [ ! -f "$AUDIT_SCRIPT" ]; then
    # dc source is local-only by policy, so a build host that received only
    # agamotto_core/ by scp has no obfuscation/ next to it. Say so plainly
    # instead of silently shipping unaudited.
    echo "BLOCKED: leak audit could not RUN — $AUDIT_SCRIPT is absent (this host" >&2
    echo "         has agamotto_core/ without dc/obfuscation/). This is NOT a pass:" >&2
    echo "         audit $OUT on a host that has the map before shipping it." >&2
    exit 2
else
    if ! AUDIT_OUT="$("$AUDIT_PY" "$AUDIT_SCRIPT" --repo "$BUILD_DIR" --binaries 2>&1)"; then
        echo "FAIL: leak audit found a distinctive name in the build output." >&2
        echo "$AUDIT_OUT" >&2
        exit 1
    fi
    echo "[build_linux] artifact leak audit: PASS ($AUDIT_PY, binaries scanned)"
fi

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
