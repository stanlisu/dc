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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="$SCRIPT_DIR/build_dist"

# Remote hosts (edit user/host as needed)
HYDRA_HOST="hydra"
SHIELD_HOST="shield"

# Remote base path
REMOTE_BASE="/home/stan/sandbox/marvel"

# Deploy targets (subdirs under REMOTE_BASE that get each algo_pkg)
DEPLOY_DIRS=(gauntlet optimus ltp sumo)

# All algo packages
ALL_ALGOS=(agamotto orb aether scepter mjolnir stormbreaker vibranium valkyrie vomir)

BUILD_ONLY=false
ALGOS=("${ALL_ALGOS[@]}")

# Parse args
for arg in "$@"; do
    if [ "$arg" = "--build-only" ]; then
        BUILD_ONLY=true
    elif [[ " ${ALL_ALGOS[*]} " =~ " ${arg} " ]]; then
        ALGOS=("$arg")
    else
        echo "Unknown argument: $arg"
        echo "Usage: $0 [--build-only] [algo_name]"
        echo "Algos: ${ALL_ALGOS[*]}"
        exit 1
    fi
done

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
    pyarmor gen --platform linux.x86_64 -O "$build_dir/src" -r "$pkg_dir/src/$algo"

    # Copy setup.py and patch it to include binary files
    echo "  Copying setup files..."
    cp "$pkg_dir/setup.py" "$build_dir/"
    sed -i '' 's/python_requires=">=3.7",/python_requires=">=3.7", package_data={"": ["*.so", "*.dylib", "*.dll"]}, zip_safe=False,/g' "$build_dir/setup.py"

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

    for subdir in "${DEPLOY_DIRS[@]}"; do
        local remote_path="$host:$REMOTE_BASE/$subdir/${algo}_pkg/"
        rsync -az --delete "$build_dir/" "$remote_path"
        echo "    -> $remote_path"
    done

}

# ---- Main ----
echo "PyArmor Algo Package Builder (local -> remote)"
echo "==============================================="
echo "Algos: ${ALGOS[*]}"
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
echo "Deploy targets:"
echo "  1) hydra ($HYDRA_HOST)"
echo "  2) shield ($SHIELD_HOST)"
echo "  3) both"
echo "  4) skip"
echo ""
read -p "Choose [1-4]: " choice

case "$choice" in
    1)
        for algo in "${ALGOS[@]}"; do deploy_algo "$algo" "$HYDRA_HOST"; done
        ;;
    2)
        for algo in "${ALGOS[@]}"; do deploy_algo "$algo" "$SHIELD_HOST"; done
        ;;
    3)
        for algo in "${ALGOS[@]}"; do
            deploy_algo "$algo" "$HYDRA_HOST"
            deploy_algo "$algo" "$SHIELD_HOST"
        done
        ;;
    4|*)
        echo "Skipping deploy."
        ;;
esac

echo ""
echo "Done."
