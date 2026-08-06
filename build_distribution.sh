#!/bin/bash
set -e

# Build all algo packages with PyArmor (cross-compile for linux.x86_64)
# and deploy to remote servers via rsync.
#
# Run locally on Mac:
#   cd ~/Documents/sandbox/dc
#   ./build_distribution.sh               # build + deploy all algos
#   ./build_distribution.sh orb           # build + deploy single algo
#   ./build_distribution.sh --build-only  # build without deploying
#   ./build_distribution.sh --xmen        # build mjolnir + sync into the local
#                                         #   xmen clone (then commit + PR + pull;
#                                         #   xmen is git-deployed, never rsynced)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="$SCRIPT_DIR/build_dist"

# Remote hosts (edit user/host as needed)
HYDRA_HOST="hydra"
SHIELD_HOST="shield"
SHIELD2_HOST="shield2"

# Remote base path
REMOTE_BASE="/home/stan/sandbox/marvel"

# Deploy target: marvel root only (algo_pkg dirs live at REMOTE_BASE/)

# ---- xmen (private Knull mirror with the real Mjolnir grafted in) ----
# xmen vendors ONLY mjolnir_pkg, and — unlike marvel — it TRACKS that build IN GIT
# (imported via PYTHONPATH="$XROOT:$XROOT/mjolnir_pkg/src", see launch_xmen_bot.sh).
# So xmen is deployed by `git pull` on the host, NOT by rsync.
#
# We therefore sync into the LOCAL xmen clone and stop: rsyncing straight into a
# host's ~/sandbox/xmen would leave the live checkout permanently dirty, bypass
# review, and be silently reverted by the next `git pull`. Commit + PR + pull is
# the deploy path. See xmen README "Refreshing the vendored mjolnir build".
#
# The refresh went undocumented for a release once already: xmen sat on a dc #22
# build until 2026-07-29 while dc had moved to #23 (the UNSIGNED-target fix), and
# nothing in this script pointed at xmen. That is why this target exists.
XMEN_LOCAL="${XMEN_LOCAL:-$HOME/Documents/sandbox/xmen}"
XMEN_ALGO="mjolnir"

# All algo packages
ALL_ALGOS=(agamotto orb aether scepter mjolnir stormbreaker vibranium valkyrie vomir)

BUILD_ONLY=false
XMEN_ONLY=false
ALGOS=("${ALL_ALGOS[@]}")
ALGOS_EXPLICIT=false

# Parse args
for arg in "$@"; do
    if [ "$arg" = "--build-only" ]; then
        BUILD_ONLY=true
    elif [ "$arg" = "--xmen" ]; then
        XMEN_ONLY=true
    elif [[ " ${ALL_ALGOS[*]} " =~ " ${arg} " ]]; then
        ALGOS=("$arg")
        ALGOS_EXPLICIT=true
    else
        echo "Unknown argument: $arg"
        echo "Usage: $0 [--build-only] [--xmen] [algo_name]"
        echo "Algos: ${ALL_ALGOS[*]}"
        exit 1
    fi
done

if [ "$BUILD_ONLY" = true ] && [ "$XMEN_ONLY" = true ]; then
    echo "ERROR: --build-only and --xmen are mutually exclusive"
    echo "       (--xmen exists to place the build into the xmen clone)"
    exit 1
fi

# xmen vendors ONLY mjolnir_pkg — building the other 8 algos for it is pure waste,
# so --xmen narrows the build unless an algo was named explicitly (in which case
# an explicit non-mjolnir choice is a mistake worth failing on, not overriding).
if [ "$XMEN_ONLY" = true ]; then
    if [ "$ALGOS_EXPLICIT" = true ] && [ "${ALGOS[0]}" != "$XMEN_ALGO" ]; then
        echo "ERROR: --xmen only vendors '$XMEN_ALGO', but '${ALGOS[0]}' was requested"
        exit 1
    fi
    ALGOS=("$XMEN_ALGO")
fi

build_algo() {
    local algo="$1"
    local pkg_dir="$SCRIPT_DIR/${algo}_pkg"
    local build_dir="$BUILD_ROOT/${algo}_pkg"

    if [ ! -d "$pkg_dir" ]; then
        echo "ERROR: $pkg_dir not found, skipping $algo"
        return 1
    fi

    echo "=========================================="
    echo "Building $algo..."
    echo "=========================================="

    # Clean build dir for this algo
    rm -rf "$build_dir"
    mkdir -p "$build_dir"

    # Obfuscate source with PyArmor, targeting linux.x86_64 for remote servers
    echo "  Obfuscating ${algo} (target: linux.x86_64)..."
    if ! pyarmor_out=$(pyarmor gen --platform linux.x86_64 -O "$build_dir/src" -r "$pkg_dir/src/$algo" 2>&1); then
        echo "$pyarmor_out"
        echo "ERROR: pyarmor gen returned non-zero exit code"
        exit 1
    fi
    if echo "$pyarmor_out" | grep -q "ERROR"; then
        echo "$pyarmor_out"
        echo "ERROR: pyarmor gen output contained an ERROR (e.g. out of license)"
        exit 1
    fi
    
    # Verify the runtime package was actually generated
    local runtime_dirs=("$build_dir/src"/pyarmor_runtime_*)
    if [ ! -e "${runtime_dirs[0]}" ]; then
        echo "$pyarmor_out"
        echo "ERROR: pyarmor_runtime missing. Obfuscation failed silently."
        exit 1
    fi

    # Carry the vendored obfuscation codec map (a DATA file — pyarmor gen only
    # processes .py and drops it). codec.py loads map.json from next to itself,
    # so it must sit beside the obfuscated codec in the build output.
    _obf_map="$pkg_dir/src/$algo/_obf/map.json"
    if [ -f "$_obf_map" ]; then
        cp "$_obf_map" "$build_dir/src/$algo/_obf/map.json"
        echo "  Copied _obf/map.json"
    fi

    # Copy setup.py and patch it to include binary files + the _obf/map.json
    # codec data (pip drops non-listed data files; without "*.json" the vendored
    # map never reaches site-packages and codec.py fails to load).
    echo "  Copying setup files..."
    cp "$pkg_dir/setup.py" "$build_dir/"
    sed -i '' 's/python_requires=">=3.7",/python_requires=">=3.7", package_data={"": ["*.so", "*.dylib", "*.dll", "*.json"]}, zip_safe=False,/g' "$build_dir/setup.py"

    # Copy README if exists
    if [ -f "$pkg_dir/README.md" ]; then
        cp "$pkg_dir/README.md" "$build_dir/"
    fi

    # Copy requirements if exists
    if [ -f "$pkg_dir/requirements.txt" ]; then
        cp "$pkg_dir/requirements.txt" "$build_dir/"
    fi

    echo "  Done building $algo."
}

deploy_algo() {
    local algo="$1"
    local host="$2"
    local build_dir="$BUILD_ROOT/${algo}_pkg"

    if [ ! -d "$build_dir" ]; then
        echo "ERROR: Build dir $build_dir not found for $algo."
        return 1
    fi

    echo "  Deploying ${algo}_pkg -> $host..."

    local remote_path="$host:$REMOTE_BASE/${algo}_pkg/"
    rsync -az --delete "$build_dir/" "$remote_path"
    echo "    -> $remote_path"

}

# Sync the mjolnir build into the LOCAL xmen clone (git-tracked — see XMEN_LOCAL
# above). Does NOT commit: the change is left staged for review, because this
# replaces a live trading dependency and every file changes on every rebuild
# (PyArmor obfuscation is not deterministic), so the diff itself proves nothing.
# The verify step is what proves the build is correct.
deploy_xmen() {
    local build_dir="$BUILD_ROOT/${XMEN_ALGO}_pkg"

    if [ ! -d "$build_dir" ]; then
        echo "ERROR: no build for $XMEN_ALGO at $build_dir — build it first"
        echo "       (xmen vendors ONLY ${XMEN_ALGO}_pkg; run: $0 $XMEN_ALGO)"
        return 1
    fi
    if [ ! -d "$XMEN_LOCAL/.git" ]; then
        echo "ERROR: no xmen git clone at $XMEN_LOCAL"
        echo "       set XMEN_LOCAL=/path/to/xmen (it must be a git checkout —"
        echo "       xmen is deployed by 'git pull', not rsync)"
        return 1
    fi

    echo "  Syncing ${XMEN_ALGO}_pkg -> $XMEN_LOCAL (local clone) ..."
    rsync -a --delete "$build_dir/" "$XMEN_LOCAL/${XMEN_ALGO}_pkg/"
    echo "    -> $XMEN_LOCAL/${XMEN_ALGO}_pkg/"

    # Prove the build carries the current target convention BEFORE it can be
    # committed. Obfuscated builds cannot be grepped, so the checker calls the
    # real function and reads the sign; it exits non-zero on a stale build.
    local verifier="$XMEN_LOCAL/scripts/verify_mjolnir_build.py"
    if [ ! -f "$verifier" ]; then
        echo "  WARNING: $verifier missing — cannot verify the vendored build."
    elif [ "$(uname -s)" != "Linux" ]; then
        # The build targets linux.x86_64, so importing it here would die on the
        # .so ("slice is not valid mach-o file"). We deliberately do NOT run it
        # and swallow the error: a traceback followed by "expected on a Mac"
        # trains you to ignore exactly the output that would flag a STALE build.
        echo "  Verify SKIPPED — build is linux.x86_64, this host is $(uname -s)."
        echo "    Run it on the target host after 'git pull' (this is the gate):"
        echo "      PYTHONPATH=mjolnir_pkg/src python3 scripts/verify_mjolnir_build.py"
    else
        echo "  Verifying vendored build ..."
        if PYTHONPATH="$XMEN_LOCAL/${XMEN_ALGO}_pkg/src" python3 "$verifier"; then
            echo "    verify OK — target is UNSIGNED"
        else
            echo "    VERIFY FAILED — do NOT commit this build. See the output above."
            return 1
        fi
    fi

    echo ""
    echo "  xmen is NOT deployed by this script. Next steps:"
    echo "    cd $XMEN_LOCAL && git checkout -b chore/refresh-mjolnir && git add -A ${XMEN_ALGO}_pkg"
    echo "    git commit && git push github <branch>   # then PR -> master"
    echo "    ssh <host> 'cd ~/sandbox/xmen && git pull --ff-only'"
    echo "    ssh <host> 'cd ~/sandbox/xmen && PYTHONPATH=mjolnir_pkg/src python3 scripts/verify_mjolnir_build.py'"
}

# ---- Main ----
echo "PyArmor Algo Package Builder (local -> remote)"
echo "==============================================="
echo "Algos: ${ALGOS[*]}"
echo ""

# Refresh the vendored obfuscation codec+map inside each package before building,
# so a deploy never ships a stale map. Single source of truth: obfuscation/.
echo "Syncing vendored obfuscation codec+map into packages..."
python "$SCRIPT_DIR/obfuscation/build_map.py"
python "$SCRIPT_DIR/obfuscation/sync_vendor.py"
echo ""

# Build phase
for algo in "${ALGOS[@]}"; do
    build_algo "$algo"
done

echo ""
echo "=========================================="
echo "All builds complete."
echo "Artifacts: $BUILD_ROOT"
echo "=========================================="

if [ "$BUILD_ONLY" = true ]; then
    echo "Build-only mode. Skipping deploy."
    exit 0
fi

echo ""
if [ "$XMEN_ONLY" = true ]; then
    deploy_xmen
    echo ""
    echo "Done."
    exit $?
fi

echo "Deploy targets:"
echo "  1) hydra ($HYDRA_HOST)"
echo "  2) shield ($SHIELD_HOST)"
echo "  3) shield2 ($SHIELD2_HOST)"
echo "  4) all marvel hosts"
echo "  5) xmen local clone ($XMEN_LOCAL) — ${XMEN_ALGO}_pkg only, commit+PR to deploy"
echo "  6) all marvel hosts + xmen local clone"
echo "  7) skip"
echo ""
read -p "Choose [1-7]: " choice

deploy_marvel_all() {
    for algo in "${ALGOS[@]}"; do
        deploy_algo "$algo" "$HYDRA_HOST"
        deploy_algo "$algo" "$SHIELD_HOST"
        deploy_algo "$algo" "$SHIELD2_HOST"
    done
}

case "$choice" in
    1)
        for algo in "${ALGOS[@]}"; do deploy_algo "$algo" "$HYDRA_HOST"; done
        ;;
    2)
        for algo in "${ALGOS[@]}"; do deploy_algo "$algo" "$SHIELD_HOST"; done
        ;;
    3)
        for algo in "${ALGOS[@]}"; do deploy_algo "$algo" "$SHIELD2_HOST"; done
        ;;
    4)
        deploy_marvel_all
        ;;
    5)
        deploy_xmen
        ;;
    6)
        deploy_marvel_all
        deploy_xmen
        ;;
    7|*)
        echo "Skipping deploy."
        ;;
esac

echo ""
echo "Done."
