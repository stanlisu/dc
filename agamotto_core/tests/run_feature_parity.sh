#!/bin/bash
# The stage-2.2 + 2.3 gate: the REAL research.py on one side, the C++ feature
# blocks and the TA-Lib indicator block on the other, every cell compared.
#
# TWO TOOLCHAINS, ALWAYS, and this is not belt-and-braces. Stage 2.1 was GREEN
# on rockylinux/gcc 8.5 and RED on macOS/clang from identical source: arm64 has
# FMA in the base ISA and Apple clang defaults to -ffp-contract=on, which
# contracted `mXY - mX*mY` in rollCorr into one fnmsub and flipped a NaN mask.
# x86-64 gcc had no FMA instruction to contract into and matched pandas by
# accident of the target ISA. A single-platform pass is therefore not evidence
# about the other platform; run both.
#
#   ./tests/run_feature_parity.sh              # macOS/clang -O2, host build
#   ./tests/run_feature_parity.sh --linux      # rocky8/gcc 8.5, in the build image
#   ./tests/run_feature_parity.sh --both
#   ./tests/run_feature_parity.sh --negative   # NEGATIVE CONTROL, see below
#
# The reference is pandas, which lives on the HOST — the build image has no
# python at all — so on --linux the DRIVER runs in docker (stdin is piped in
# with `docker run -i`) while the harness and research.py stay on the host.
#
# ---------------------------------------------------------------------------
# TA-LIB ON THE HOST
#
# From stage 2.3 the driver LINKS ta-lib, and it must link the SAME version the
# reference's Python wrapper calls (0.6.4, pinned in CMakeLists.txt and in
# sentinel_core/Dockerfile.build). Homebrew ships 0.7.1, which CMakeLists
# already refuses outright — so the host needs its own pinned build:
#
#   curl -sSL -o /tmp/ta-lib-0.6.4-src.tar.gz \
#     https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
#   mkdir -p ~/.local/src && tar xzf /tmp/ta-lib-0.6.4-src.tar.gz -C ~/.local/src
#   cd ~/.local/src/ta-lib-0.6.4 && \
#     ./configure --prefix=$HOME/.local/talib-0.6.4 --with-pic && make -j8 && make install
#
# Override the location with AGAMOTTO_TALIB_PREFIX. The version is VERIFIED
# below rather than assumed: a driver silently built against 0.7.1 would
# compute slightly different indicators and the gate would blame the port.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="mjolnir-core-build:latest"
PY="${AGAMOTTO_PARITY_PYTHON:-python3}"
TALIB_PREFIX="${AGAMOTTO_TALIB_PREFIX:-$HOME/.local/talib-0.6.4}"

MODE="${1:-host}"
case "$MODE" in
    ""|host)   MODE=host ;;
    --linux)   MODE=linux ;;
    --both)    MODE=both ;;
    --negative) MODE=negative ;;
    *) echo "usage: $0 [--linux|--both|--negative]" >&2; exit 2 ;;
esac

if ! "$PY" -c 'import pandas' >/dev/null 2>&1; then
    # Loud, not skipped. The reference IS research.py running on pandas; a gate
    # that quietly does nothing when its reference is missing prints a green
    # that means nothing.
    echo "FAIL: $PY has no pandas. research.py IS the reference — refusing to" >&2
    echo "      run a parity gate without it. Set AGAMOTTO_PARITY_PYTHON=..." >&2
    exit 1
fi

# The pin, read from the C++ side's own source of truth (feature_parity.py
# cross-checks it against sentinel_core/Dockerfile.build and against the
# reference wrapper's talib.__ta_version__).
PINNED="$(sed -n 's/^set(TALIB_PINNED_VERSION "\(.*\)")/\1/p' "$HERE/CMakeLists.txt")"
if [ -z "$PINNED" ]; then
    echo "FAIL: cannot read TALIB_PINNED_VERSION from $HERE/CMakeLists.txt" >&2
    exit 1
fi

check_host_talib() {
    local pc="$TALIB_PREFIX/lib/pkgconfig/ta-lib.pc"
    if [ ! -f "$TALIB_PREFIX/lib/libta-lib.a" ] || [ ! -f "$pc" ]; then
        echo "FAIL: no pinned ta-lib at $TALIB_PREFIX (need lib/libta-lib.a and" >&2
        echo "      lib/pkgconfig/ta-lib.pc). Build it — see the header of this" >&2
        echo "      script — or set AGAMOTTO_TALIB_PREFIX. Homebrew's 0.7.1 is" >&2
        echo "      NOT a substitute: different indicator internals, silently." >&2
        exit 1
    fi
    local got
    got="$(sed -n 's/^Version:[[:space:]]*//p' "$pc")"
    if [ "$got" != "$PINNED" ]; then
        echo "FAIL: ta-lib at $TALIB_PREFIX reports $got, this core is pinned to" >&2
        echo "      $PINNED. A version difference is a SILENT parity break." >&2
        exit 1
    fi
    echo "[run_feature_parity] host ta-lib $got at $TALIB_PREFIX"
}

# $1 = output binary, $2... = extra flags (e.g. a mutant -D)
build_host_driver() {
    local out="$1"; shift
    # Compiled directly rather than through CMake: CMakeLists needs the sentinel
    # contract header to configure at all, and this driver does not link it. The
    # flags mirror the CMake target exactly — -O2 and -ffp-contract=off.
    mkdir -p "$HERE/build"
    "${CXX:-c++}" -std=c++17 -O2 -ffp-contract=off -Wall -Wextra \
        -I"$HERE/src" -I"$TALIB_PREFIX/include" "$@" -o "$out" \
        "$HERE/tests/feature_parity_driver.cpp" \
        "$HERE/src/feature_engine.cpp" "$HERE/src/talib_block.cpp" \
        "$TALIB_PREFIX/lib/libta-lib.a" -lm
}

run_host() {
    echo "=== building the driver for the HOST toolchain ==="
    check_host_talib
    build_host_driver "$HERE/build/feature_parity_driver"
    "${CXX:-c++}" --version | head -1
    echo "=== host parity ==="
    "$PY" "$HERE/tests/feature_parity.py" --driver "$HERE/build/feature_parity_driver"
}

run_linux() {
    echo "=== building the driver in $IMAGE (rocky8 / gcc 8.5) ==="
    # g++ directly, not cmake: CMakeLists needs the sentinel contract header
    # (which is not in this image) to configure at all. ta-lib IS in the image,
    # pinned by Dockerfile.build, and is linked STATICALLY here so the driver
    # carries no runtime dependency on /usr/local/lib. -lm is required:
    # libta-lib references asin/exp/log/sqrt and does not pull it in itself.
    docker run --rm -v "$HERE:/src" -w /src "$IMAGE" bash -c "
        set -e
        mkdir -p build-linux
        g++ --version | head -1
        g++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Isrc \
            -o build-linux/feature_parity_driver \
            tests/feature_parity_driver.cpp src/feature_engine.cpp \
            src/talib_block.cpp /usr/local/lib/libta-lib.a -lm
    "
    echo "=== linux parity (driver in docker, pandas on the host) ==="
    "$PY" "$HERE/tests/feature_parity.py" --driver \
        "docker run --rm -i -v $HERE:/src -w /src $IMAGE ./build-linux/feature_parity_driver"
}

# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.
#
# A gate that has never been seen to FAIL is a gate nobody has tested. Each
# mutation below is a PLAUSIBLE wrong answer — the thing a careful person would
# actually have written — not a random corruption, and each is applied by
# rewriting ONE line in a copy of the source. The run INVERTS the exit code: a
# mutant that PASSES means the gate is not watching that quantity.
#
# Every mutation is guarded twice: the rewritten line must be PRESENT
# afterwards, and the original must be GONE. A sed whose pattern has drifted
# would otherwise produce an unmutated binary and report a green negative
# control, which is worse than no control at all.
#
# run_mutant <label> <src-file> <sed-program> <must-appear> <must-vanish> <why>
run_mutant() {
    local label="$1" src="$2" prog="$3" want="$4" gone="$5" why="$6"
    echo
    echo "=== NEGATIVE CONTROL: $label ==="
    local tmp base
    tmp="$(mktemp -d)"
    base="$(basename "$src" .cpp)"
    sed "$prog" "$HERE/src/$src" > "$tmp/${base}_mutant.cpp"
    if ! grep -qF "$want" "$tmp/${base}_mutant.cpp"; then
        echo "FAIL: the mutation did not apply to src/$src — the target line" >&2
        echo "      moved. Refusing to report a control that mutated nothing." >&2
        rm -rf "$tmp"; exit 1
    fi
    if grep -qF "$gone" "$tmp/${base}_mutant.cpp"; then
        echo "FAIL: the ORIGINAL line is still present in src/$src after the" >&2
        echo "      mutation — the sed matched somewhere else." >&2
        rm -rf "$tmp"; exit 1
    fi

    # The mutated copy replaces its original; the other translation units come
    # from src/ unchanged.
    local units=("$HERE/tests/feature_parity_driver.cpp")
    local u
    for u in feature_engine.cpp talib_block.cpp; do
        if [ "$u" = "$src" ]; then units+=("$tmp/${base}_mutant.cpp");
        else units+=("$HERE/src/$u"); fi
    done
    mkdir -p "$HERE/build"
    "${CXX:-c++}" -std=c++17 -O2 -ffp-contract=off -Wall -Wextra \
        -I"$HERE/src" -I"$TALIB_PREFIX/include" \
        -o "$HERE/build/feature_parity_driver_mutant" \
        "${units[@]}" "$TALIB_PREFIX/lib/libta-lib.a" -lm
    rm -rf "$tmp"

    if "$PY" "$HERE/tests/feature_parity.py" \
            --driver "$HERE/build/feature_parity_driver_mutant" >/dev/null 2>&1; then
        echo "=== NEGATIVE CONTROL FAILED: the mutant PASSED the gate ===" >&2
        echo "    $why" >&2
        exit 1
    fi
    echo "--- caught (gate went red) ---"
}

run_negative() {
    check_host_talib

    # STAGE 2.3. The single most plausible wrong answer: BBANDS at mjolnir's
    # timeperiod=5 instead of agamotto's 20 (research.py:549). Plausible
    # precisely because sentinel_core/src/talib_block.cpp is the file this one
    # was built from, and 5 is what it says.
    run_mutant "2.3 BBANDS timeperiod 20 -> 5 (mjolnir's value)" \
        talib_block.cpp \
        's/TA_BBANDS(0, n - bC - 1, c + bC, 20, 2\.0, 2\.0/TA_BBANDS(0, n - bC - 1, c + bC, 5, 2.0, 2.0/' \
        "c + bC, 5, 2.0, 2.0" "c + bC, 20, 2.0, 2.0" \
        "BBANDS at timeperiod=5 is not distinguished from 20, so bb_upper/bb_lower are unverified."

    # STAGE 2.4 (a). POPULATION skew instead of pandas' SAMPLE (bias-corrected)
    # G1. This is the textbook formula and the one every "rolling skewness"
    # snippet on the internet computes; it is 11.05% smaller at n=14
    # (factor (n-2)/sqrt(n(n-1)) = 12/sqrt(182)), which is far too small to look
    # like a bug in a plot and far too large to be rounding.
    run_mutant "2.4 skew: SAMPLE G1 -> POPULATION g1" \
        feature_engine.cpp \
        's|return (std::sqrt(dn \* (dn - 1\.0)) \* C) / ((dn - 2\.0) \* R \* R \* R);|return C / (R * R * R);  // MUTANT: population g1|' \
        "return C / (R * R * R);  // MUTANT" \
        "return (std::sqrt(dn * (dn - 1.0)) * C) / ((dn - 2.0) * R * R * R);" \
        "population skew is not distinguished from pandas' sample G1 — the skew column is unverified."

    # STAGE 2.4 (b). POPULATION excess kurtosis (D/B^2 - 3) instead of pandas'
    # SAMPLE EXCESS G2. Same reasoning; ~40% off at n=14.
    run_mutant "2.4 kurt: SAMPLE EXCESS G2 -> POPULATION excess g2" \
        feature_engine.cpp \
        's|return K / ((dn - 2\.0) \* (dn - 3\.0));|(void)K; return D / (B * B) - 3.0;  // MUTANT: population g2|' \
        "(void)K; return D / (B * B) - 3.0;  // MUTANT" \
        "return K / ((dn - 2.0) * (dn - 3.0));" \
        "population excess kurtosis is not distinguished from pandas' sample G2 — the kurt column is unverified."

    # STAGE 2.5 (a). obv_is_cumulative=True — i.e. `_flow` takes a SECOND
    # difference of an already-differenced obv/ad
    # (features_scalefree.py:66-69, research.py:538-539). This is the mutation
    # the reference's own docstring warns about, and it is silent: on a steady
    # flow the second difference is ~0 on every row while the column still
    # looks like a valid feature.
    run_mutant "2.5 obv/ad: obv_is_cumulative False -> True (a SECOND diff)" \
        feature_engine.cpp \
        's|safeDiv(t\.get(codes::F_OBV), vol_sum)|safeDiv(pdops::diffN(t.get(codes::F_OBV), SCALE_FREE_WINDOW), vol_sum)  /* MUTANT */|; s|safeDiv(t\.get(codes::F_AD), vol_sum)|safeDiv(pdops::diffN(t.get(codes::F_AD), SCALE_FREE_WINDOW), vol_sum)  /* MUTANT */|' \
        "pdops::diffN(t.get(codes::F_OBV), SCALE_FREE_WINDOW)" \
        "safeDiv(t.get(codes::F_OBV), vol_sum));" \
        "a second difference of obv/ad is not distinguished from none — obv_slope/ad_slope are unverified."

    # STAGE 2.5 (b). `_safe` WITHOUT its second step: `num / den.replace(0.0,
    # nan)` and nothing else. This is what `if (den == 0) return NaN;` gives
    # you, and it is the natural thing to write. It is caught ONLY because the
    # fifth scenario contains a close of 1e-305, which overflows a BTC-scale
    # numerator; without that bar this control would pass and step 2 would be
    # dead code as far as the gate is concerned.
    run_mutant "2.5 _safe: drop step 2 (replace([inf,-inf], nan))" \
        feature_engine.cpp \
        's|out\[i\] = std::isinf(r) ? NA : r;|out[i] = r;  // MUTANT: step 2 removed|' \
        "out[i] = r;  // MUTANT: step 2 removed" \
        "out[i] = std::isinf(r) ? NA : r;" \
        "an inf that _safe must map to NaN is surviving unnoticed — step 2 is unverified."

    echo
    echo "=== NEGATIVE CONTROLS OK: 5 mutants, all caught (gate went red) ==="
}

case "$MODE" in
    host)     run_host ;;
    linux)    run_linux ;;
    both)     run_host; echo; run_linux ;;
    negative) run_negative ;;
esac
