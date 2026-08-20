#!/bin/bash
# The PHASE 4 gate: the DEPLOYED sklearn pipeline on one side, the C++ model
# runner on the other, every regime x every row compared at 1e-9 relative.
#
#   ./tests/run_model_parity.sh              # macOS/clang -O2, host build
#   ./tests/run_model_parity.sh --linux      # rocky8/gcc 8.5, in the build image
#   ./tests/run_model_parity.sh --both
#   ./tests/run_model_parity.sh --negative   # NEGATIVE CONTROLS, exit INVERTED
#
# TWO TOOLCHAINS, for the same reason the feature and regime gates run both:
# stage 2.1 was GREEN on gcc 8.5/x86-64 and RED on clang/arm64 from identical
# source. This gate's kernel is a dot product, which is exactly the shape a
# fused multiply-add contracts — so it inherits that exposure directly, and
# -ffp-contract=off is applied here as it is everywhere else.
#
# TWO WEIGHT TREES, AND THEY MUST BE DIFFERENT DIRECTORIES:
#   AGAMOTTO_WEIGHTS      the export_agamotto_sentinel_weights.py OUTPUT
#                         (model.txt/scaler.txt/features.txt) — the C++ side
#   AGAMOTTO_RAW_WEIGHTS  the SOURCE window_YYYY_MM_DD of ridge_*.pkl — the
#                         DEPLOYED loader's side
# Grading text against text would compare the export with itself.
#
#   AGAMOTTO_MARVEL_ROOT  a marvel checkout, for utils.weights_io. That module
#                         IS the reference (the bot's and tesseract's own
#                         loader) and is imported, never reimplemented.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="mjolnir-core-build:latest"
PY="${AGAMOTTO_PARITY_PYTHON:-python3}"
TALIB_PREFIX="${AGAMOTTO_TALIB_PREFIX:-$HOME/.local/talib-0.6.4}"
STACK="${AGAMOTTO_REGIME_STACK:-$HERE/tests/regime_stack_deployed.csv}"
WEIGHTS="${AGAMOTTO_WEIGHTS:-$HOME/agamotto_test/weights}"
RAW_WEIGHTS="${AGAMOTTO_RAW_WEIGHTS:-$HOME/agamotto_test/raw_weights/window_2026_07_31}"
MARVEL="${AGAMOTTO_MARVEL_ROOT:-$HOME/Documents/sandbox/marvel}"
MODEL="${AGAMOTTO_MODEL_PREFIX:-ridge}"

MODE="${1:-host}"
case "$MODE" in
    ""|host)    MODE=host ;;
    --linux)    MODE=linux ;;
    --both)     MODE=both ;;
    --negative) MODE=negative ;;
    *) echo "usage: $0 [--linux|--both|--negative]" >&2; exit 2 ;;
esac

if ! "$PY" -c 'import pandas, sklearn' >/dev/null 2>&1; then
    echo "FAIL: $PY has no pandas/sklearn. utils.weights_io IS the reference —" >&2
    echo "      refusing to run a parity gate without it. Set" >&2
    echo "      AGAMOTTO_PARITY_PYTHON=..." >&2
    exit 1
fi
if [ ! -f "$STACK" ]; then
    echo "FAIL: no regime stack at $STACK. The gate grades the DEPLOYED stack;" >&2
    echo "      a hand-typed substitute would grade a stack that is not running." >&2
    exit 1
fi
if [ ! -d "$WEIGHTS" ]; then
    echo "FAIL: no exported weights at $WEIGHTS. Produce them with" >&2
    echo "      marvel/gauntlet/export_agamotto_sentinel_weights.py --weights" >&2
    echo "      <window dir> --out $WEIGHTS  (set AGAMOTTO_WEIGHTS to move it)." >&2
    exit 1
fi
if [ ! -d "$RAW_WEIGHTS" ]; then
    echo "FAIL: no source weights at $RAW_WEIGHTS. The REFERENCE side loads the" >&2
    echo "      ridge_*.pkl the bot loads; without them this gate would have to" >&2
    echo "      grade the export against itself. Set AGAMOTTO_RAW_WEIGHTS=..." >&2
    exit 1
fi
if [ ! -f "$MARVEL/utils/weights_io.py" ]; then
    echo "FAIL: $MARVEL has no utils/weights_io.py. Set AGAMOTTO_MARVEL_ROOT to a" >&2
    echo "      marvel checkout — that module is THE deployed loader and this" >&2
    echo "      gate refuses to grade against a reimplementation of it." >&2
    exit 1
fi

PINNED="$(sed -n 's/^set(TALIB_PINNED_VERSION "\(.*\)")/\1/p' "$HERE/CMakeLists.txt")"
[ -n "$PINNED" ] || { echo "FAIL: cannot read TALIB_PINNED_VERSION" >&2; exit 1; }

check_host_talib() {
    local pc="$TALIB_PREFIX/lib/pkgconfig/ta-lib.pc"
    if [ ! -f "$TALIB_PREFIX/lib/libta-lib.a" ] || [ ! -f "$pc" ]; then
        echo "FAIL: no pinned ta-lib at $TALIB_PREFIX. See the header of" >&2
        echo "      tests/run_feature_parity.sh for the build recipe." >&2
        exit 1
    fi
    local got; got="$(sed -n 's/^Version:[[:space:]]*//p' "$pc")"
    if [ "$got" != "$PINNED" ]; then
        echo "FAIL: ta-lib at $TALIB_PREFIX reports $got, pinned to $PINNED." >&2
        exit 1
    fi
    echo "[run_model_parity] host ta-lib $got at $TALIB_PREFIX"
}

# $1 = output binary, $2.. = translation units (the runner source may be a mutant)
build_host_driver() {
    local out="$1"; shift
    mkdir -p "$HERE/build"
    "${CXX:-c++}" -std=c++17 -O2 -ffp-contract=off -Wall -Wextra \
        -I"$HERE/src" -I"$TALIB_PREFIX/include" -o "$out" \
        "$HERE/tests/model_parity_driver.cpp" "$@" \
        "$HERE/src/regime_gate.cpp" "$HERE/src/feature_engine.cpp" \
        "$HERE/src/talib_block.cpp" \
        "$TALIB_PREFIX/lib/libta-lib.a" -lm
}

harness() {
    "$PY" "$HERE/tests/model_parity.py" --stack "$STACK" --weights "$WEIGHTS" \
        --raw-weights "$RAW_WEIGHTS" --marvel-root "$MARVEL" --model "$MODEL" "$@"
}

run_host() {
    echo "=== building the model driver for the HOST toolchain ==="
    check_host_talib
    build_host_driver "$HERE/build/model_parity_driver" "$HERE/src/model_runner.cpp"
    "${CXX:-c++}" --version | head -1
    # THE REFUSE-TO-LOAD TESTS, before the parity run. A core that loaded a
    # LightGBM dump, or a features.txt naming a column the panel lacks, would
    # boot clean and predict numbers — the failure mode with no symptom.
    echo "=== loader self-test (refuse-to-load + regimeDirName) ==="
    "$HERE/build/model_parity_driver" --selftest
    echo "=== host model parity ==="
    harness --driver "$HERE/build/model_parity_driver"
}

run_linux() {
    echo "=== building the model driver in $IMAGE (rocky8 / gcc 8.5) ==="
    docker run --rm -v "$HERE:/src" -w /src "$IMAGE" bash -c "
        set -e
        mkdir -p build-linux
        g++ --version | head -1
        g++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Isrc \
            -o build-linux/model_parity_driver \
            tests/model_parity_driver.cpp src/model_runner.cpp \
            src/regime_gate.cpp src/feature_engine.cpp src/talib_block.cpp \
            /usr/local/lib/libta-lib.a -lm
        ./build-linux/model_parity_driver --selftest --tmp /tmp
    "
    echo "=== linux model parity (driver in docker, sklearn on the host) ==="
    # The exported weights live outside $HERE, so they are bind-mounted at their
    # OWN absolute path inside the container: the harness passes one --weights
    # string to both toolchains, and remapping it for one of them is a place for
    # the two runs to silently grade different artifacts. READ-ONLY — a parity
    # driver has no business writing to a weight tree.
    harness --driver "docker run --rm -i -v $HERE:/src -v $WEIGHTS:$WEIGHTS:ro \
-w /src $IMAGE ./build-linux/model_parity_driver"
}

# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.
#
# The first two are DRIVER FLAGS rather than source mutants, because the thing
# they must prove absent is a whole missing STEP, not a wrong operator: an
# engine that never applies the scaler, and a coefficient that is not the one on
# disk. The rest are source mutations in the sed style the other gates use.
#
# Every control INVERTS the exit code: a mutant that PASSES means the gate is
# not watching that property.
# ---------------------------------------------------------------------------
run_negative() {
    check_host_talib
    build_host_driver "$HERE/build/model_parity_driver" "$HERE/src/model_runner.cpp"

    # (1) THE SCALER IS ACTUALLY APPLIED. `--no-scaler` predicts
    # intercept + sum(coef * x_raw), dropping (x - center)/scale entirely. On
    # features whose centre is near 0 and whose scale is near 1 the two are
    # nearly the same number, so without this control a runner that ignored the
    # scaler could look correct on a lucky regime.
    echo
    echo "=== NEGATIVE CONTROL: predict WITHOUT the scaler ==="
    if harness --driver "$HERE/build/model_parity_driver --no-scaler" \
               --expect-red >/dev/null 2>&1; then
        echo "--- caught (gate went red) ---"
    else
        echo "=== NEGATIVE CONTROL FAILED: the unscaled runner PASSED ===" >&2
        echo "    (x - center)/scale is unverified — the gate cannot tell an" >&2
        echo "    engine that applies the RobustScaler from one that does not." >&2
        exit 1
    fi

    # (2) THE COEFFICIENTS ARE THE ONES ON DISK. One coefficient of regime 0 is
    # nudged by 1e-6 — five orders of magnitude BELOW the deployed threshold
    # (~5.8e-4), so this is not a control that only catches vandalism; it pins
    # the gate's resolution well under anything that could change a decision.
    echo
    echo "=== NEGATIVE CONTROL: perturb ONE coefficient of ONE regime by 1e-6 ==="
    if harness --driver "$HERE/build/model_parity_driver --perturb-coef 0:0:1e-6" \
               --expect-red >/dev/null 2>&1; then
        echo "--- caught (gate went red) ---"
    else
        echo "=== NEGATIVE CONTROL FAILED: a perturbed coefficient PASSED ===" >&2
        echo "    The gate is not actually comparing against the deployed model." >&2
        exit 1
    fi

    # (3) THE SIGN OF THE CENTRE. `(x - center)` vs `(x + center)` is the single
    # most plausible transcription error in a RobustScaler port, and on a
    # centre near zero it is nearly invisible — which is exactly why it needs a
    # control rather than a reading.
    run_mutant "scaler: (x - center) -> (x + center)" \
        's|y += coef\[i\] \* ((x - center\[i\]) / scale\[i\]);|y += coef[i] * ((x + center[i]) / scale[i]);  /* MUTANT */|' \
        "((x + center[i]) / scale[i]);  /* MUTANT */" \
        "y += coef[i] * ((x - center[i]) / scale[i]);" \
        "the centring SIGN is unverified."

    # (4) MULTIPLY BY THE SCALE INSTEAD OF DIVIDING. Same family, and on a scale
    # near 1.0 — which SIX deployed rows have exactly (the zero-IQR
    # substitution) — it changes nothing at all on those rows. It must still be
    # caught by every other row.
    run_mutant "scaler: divide by scale -> multiply by scale" \
        's|y += coef\[i\] \* ((x - center\[i\]) / scale\[i\]);|y += coef[i] * ((x - center[i]) * scale[i]);  /* MUTANT */|' \
        "((x - center[i]) * scale[i]);  /* MUTANT */" \
        "y += coef[i] * ((x - center[i]) / scale[i]);" \
        "dividing by the IQR is not distinguished from multiplying by it."

    # (5) THE NaN FILL. trading.py:700 fills NaN with 0.0 BEFORE scaling, so a
    # filled cell contributes coef * ((0 - center)/scale) — emphatically not
    # zero contribution. Skipping the term instead is the natural "treat missing
    # as absent" reading and produces a different number on every NaN row.
    run_mutant "NaN fill: contribute coef*((0-center)/scale) -> skip the term" \
        's|            if (nan_filled != nullptr) ++\*nan_filled;|            if (nan_filled != nullptr) ++*nan_filled;\n            continue;  /* MUTANT */|' \
        "continue;  /* MUTANT */" \
        "" \
        "trading.py's NaN->0.0 fill is not distinguished from dropping the term."

    # (6) THE FEATURE ORDER. features.txt order IS the model's input order;
    # reversing it pairs every coefficient with the wrong column while keeping
    # the count, the scaler shape and the arithmetic all valid.
    run_mutant "feature order: reverse the resolved column indices" \
        's|        m.column_index.push_back(j);|        m.column_index.insert(m.column_index.begin(), j);  /* MUTANT */|' \
        "m.column_index.insert(m.column_index.begin(), j);  /* MUTANT */" \
        "        m.column_index.push_back(j);" \
        "features.txt ORDER is unverified — every coefficient could be paired with the wrong column."

    echo
    echo "=== NEGATIVE CONTROLS OK: 6 controls, all caught (gate went red) ==="
}

# run_mutant <label> <sed-program> <must-appear> <must-vanish|""> <why>
#
# Guarded twice (the new text must appear, the old must be gone) so a drifted
# sed cannot report a green control over an unmutated binary. `must-vanish` may
# be empty for an INSERTION, which does not remove its anchor line.
run_mutant() {
    local label="$1" prog="$2" want="$3" gone="$4" why="$5"
    echo
    echo "=== NEGATIVE CONTROL: $label ==="
    local tmp; tmp="$(mktemp -d)"
    sed "$prog" "$HERE/src/model_runner.cpp" > "$tmp/model_runner_mutant.cpp"
    if ! grep -qF "$want" "$tmp/model_runner_mutant.cpp"; then
        echo "FAIL: the mutation did not apply to src/model_runner.cpp — the" >&2
        echo "      target line moved. Refusing to report a control that" >&2
        echo "      mutated nothing." >&2
        rm -rf "$tmp"; exit 1
    fi
    if [ -n "$gone" ] && grep -qF "$gone" "$tmp/model_runner_mutant.cpp"; then
        echo "FAIL: the ORIGINAL line is still present after the mutation —" >&2
        echo "      the sed matched somewhere else." >&2
        rm -rf "$tmp"; exit 1
    fi
    build_host_driver "$HERE/build/model_parity_driver_mutant" \
        "$tmp/model_runner_mutant.cpp"
    rm -rf "$tmp"

    if harness --driver "$HERE/build/model_parity_driver_mutant" >/dev/null 2>&1; then
        echo "=== NEGATIVE CONTROL FAILED: the mutant PASSED the gate ===" >&2
        echo "    $why" >&2
        exit 1
    fi
    echo "--- caught (gate went red) ---"
}

case "$MODE" in
    host)     run_host ;;
    linux)    run_linux ;;
    both)     run_host; echo; run_linux ;;
    negative) run_negative ;;
esac
