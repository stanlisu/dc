#!/bin/bash
# The PHASE 5 gate: the REFERENCE decision path on one side, the C++ Decision on
# the other, fired/side compared EXACTLY over every regime x every row.
#
#   ./tests/run_decision_parity.sh              # macOS/clang -O2, host build
#   ./tests/run_decision_parity.sh --linux      # rocky8/gcc 8.5, in the build image
#   ./tests/run_decision_parity.sh --both
#   ./tests/run_decision_parity.sh --negative   # NEGATIVE CONTROLS, exit INVERTED
#
# TWO TOOLCHAINS, for the same reason every other gate here runs both: stage 2.1
# was GREEN on gcc 8.5/x86-64 and RED on clang/arm64 from identical source. This
# gate's kernel is a chain of `>` and `<` against a sum — the shape a fused
# multiply-add contracts — so -ffp-contract=off is applied here as everywhere.
#
# THE GATE COMES FROM setting.json, NOT FROM THIS SCRIPT. The harness imports
# gauntlet.thresholds and reads the DEPLOYED experiment's THRESHOLD_LONG /
# THRESHOLD_SHORT / THRESHOLD_CENTER_LONG / THRESHOLD_CENTER_SHORT / REVERSE.
# Hardcoding numbers here would grade a gate nobody is running.
#
#   AGAMOTTO_WEIGHTS      the export_agamotto_sentinel_weights.py OUTPUT
#   AGAMOTTO_MARVEL_ROOT  a marvel checkout, for gauntlet.thresholds — THE
#                         definition of the gate, imported and never rewritten
#   AGAMOTTO_SETTING      the deployed setting.json (default: the 15m_1 arm)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="mjolnir-core-build:latest"
PY="${AGAMOTTO_PARITY_PYTHON:-python3}"
TALIB_PREFIX="${AGAMOTTO_TALIB_PREFIX:-$HOME/.local/talib-0.6.4}"
STACK="${AGAMOTTO_REGIME_STACK:-$HERE/tests/regime_stack_deployed.csv}"
WEIGHTS="${AGAMOTTO_WEIGHTS:-$HOME/agamotto_test/weights}"
MARVEL="${AGAMOTTO_MARVEL_ROOT:-$HOME/Documents/sandbox/marvel}"
SETTING="${AGAMOTTO_SETTING:-$MARVEL/gauntlet/pred_agamotto.base.15m_1/setting.json}"

MODE="${1:-host}"
case "$MODE" in
    ""|host)    MODE=host ;;
    --linux)    MODE=linux ;;
    --both)     MODE=both ;;
    --negative) MODE=negative ;;
    *) echo "usage: $0 [--linux|--both|--negative]" >&2; exit 2 ;;
esac

if ! "$PY" -c 'import pandas' >/dev/null 2>&1; then
    echo "FAIL: $PY has no pandas." >&2; exit 1
fi
for f in "$STACK" "$SETTING"; do
    [ -f "$f" ] || { echo "FAIL: missing $f" >&2; exit 1; }
done
[ -d "$WEIGHTS" ] || { echo "FAIL: no exported weights at $WEIGHTS" >&2; exit 1; }
if [ ! -f "$MARVEL/gauntlet/thresholds.py" ]; then
    echo "FAIL: $MARVEL has no gauntlet/thresholds.py. That module IS the" >&2
    echo "      definition of the gate and this harness refuses to grade" >&2
    echo "      against a reimplementation of it. Set AGAMOTTO_MARVEL_ROOT." >&2
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
    [ "$got" = "$PINNED" ] || { echo "FAIL: ta-lib $got, pinned $PINNED" >&2; exit 1; }
    echo "[run_decision_parity] host ta-lib $got at $TALIB_PREFIX"
}

# $1 = output binary, $2.. = translation units (the rule source may be a mutant)
build_host_driver() {
    local out="$1"; shift
    mkdir -p "$HERE/build"
    "${CXX:-c++}" -std=c++17 -O2 -ffp-contract=off -Wall -Wextra \
        -I"$HERE/src" -I"$TALIB_PREFIX/include" -o "$out" \
        "$HERE/tests/decision_parity_driver.cpp" "$@" \
        "$HERE/src/model_runner.cpp" "$HERE/src/regime_gate.cpp" \
        "$HERE/src/feature_engine.cpp" "$HERE/src/talib_block.cpp" \
        "$TALIB_PREFIX/lib/libta-lib.a" -lm
}

harness() {
    "$PY" "$HERE/tests/decision_parity.py" --stack "$STACK" --weights "$WEIGHTS" \
        --marvel-root "$MARVEL" --setting "$SETTING" "$@"
}

run_host() {
    echo "=== building the decision driver for the HOST toolchain ==="
    check_host_talib
    build_host_driver "$HERE/build/decision_parity_driver" "$HERE/src/decision_rule.cpp"
    "${CXX:-c++}" --version | head -1
    # THE REFUSE-TO-LOAD TESTS, before the parity run. A gate that accepted a
    # sub-floor width, a zero width or a reverse of 0 would boot clean and
    # produce decisions — the failure mode with no symptom.
    echo "=== gate self-test (refuse-to-load + the deployed edges) ==="
    "$HERE/build/decision_parity_driver" --selftest
    echo "=== host decision parity ==="
    harness --driver "$HERE/build/decision_parity_driver"
}

run_linux() {
    echo "=== building the decision driver in $IMAGE (rocky8 / gcc 8.5) ==="
    docker run --rm -v "$HERE:/src" -w /src "$IMAGE" bash -c "
        set -e
        mkdir -p build-linux
        g++ --version | head -1
        g++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Isrc \
            -o build-linux/decision_parity_driver \
            tests/decision_parity_driver.cpp src/decision_rule.cpp \
            src/model_runner.cpp src/regime_gate.cpp src/feature_engine.cpp \
            src/talib_block.cpp /usr/local/lib/libta-lib.a -lm
        ./build-linux/decision_parity_driver --selftest
    "
    echo "=== linux decision parity (driver in docker, reference on the host) ==="
    harness --driver "docker run --rm -i -v $HERE:/src -v $WEIGHTS:$WEIGHTS:ro \
-w /src $IMAGE ./build-linux/decision_parity_driver"
}

# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.
#
# The first four are GATE overrides rather than source mutants, because what
# they must prove absent is a whole missing PROPERTY of the gate — its centre,
# its per-leg-ness, its direction, its floor — not a wrong operator. The rest
# are source mutations in the sed style the other gates use.
#
# Every control INVERTS the exit code: a mutant that PASSES means the gate is
# not watching that property.
# ---------------------------------------------------------------------------
run_negative() {
    check_host_talib
    build_host_driver "$HERE/build/decision_parity_driver" "$HERE/src/decision_rule.cpp"
    local D="$HERE/build/decision_parity_driver"

    # (1) THE CENTRE. A zero-centred gate is the pre-2026-08-08 behaviour and is
    # the single most likely thing for a port to have "simplified" away: on this
    # arm the centres are 1.2 and 3.1 bps against widths of 7.0 and 13.5, so a
    # dropped centre is a gate that is still plausible, still selective, and
    # wrong on every bar near the edge.
    control "drop the CENTRE (zero-centred gate)" --zero-center

    # (2) THE PER-LEG WIDTHS. Swapping them keeps both numbers, both signs and
    # the floor — everything a sanity check would look at — and moves both
    # edges. On this arm the short width is 1.9x the long's.
    control "SWAP the long and short widths" --swap-legs

    # (3) REVERSE. Inverting it flips every side while leaving every vote count
    # identical, so nothing but `side` can catch it.
    control "INVERT reverse" --invert-reverse

    # (4) THE 2 BPS FLOOR, at load. Not "does it go red" — the driver must
    # REFUSE to run at all, and refuse rather than clamp up to the floor.
    echo
    echo "=== NEGATIVE CONTROL: a SUB-FLOOR threshold must be REFUSED at load ==="
    if harness --driver "$D" --force-threshold 0.00001 --expect-refusal; then
        echo "--- caught (refused at load) ---"
    else
        echo "=== NEGATIVE CONTROL FAILED: a 0.1 bps gate was accepted ===" >&2
        exit 1
    fi

    # (5) THE SHORT LEG'S SIGN. `C-T` vs `C+T` is the 2026-06 bug verbatim:
    # comparing shorts against the long edge fires on nearly every bar. It is
    # the single highest-consequence transcription error in this file.
    run_mutant "signedThreshold: short gets C+T instead of C-T" \
        's|    return (pos == Position::LONG) ? (center + mag) : (center - mag);|    return center + mag;  /* MUTANT */|' \
        "return center + mag;  /* MUTANT */" \
        "return (pos == Position::LONG) ? (center + mag) : (center - mag);" \
        "the SHORT leg's sign is unverified — the 2026-06 always-firing bug."

    # (6) THE SHORT LEG'S COMPARISON. Same bug from the other end: `>` on a
    # short is the selective-vs-permissive flip that dual_gate_filter:48 exists
    # to pin down.
    run_mutant "legFires: short compares > instead of <" \
        's|    return (pos == Position::LONG) ? (y_pred > gate.edge) : (y_pred < gate.edge);|    return y_pred > gate.edge;  /* MUTANT */|' \
        "return y_pred > gate.edge;  /* MUTANT */" \
        "return (pos == Position::LONG) ? (y_pred > gate.edge) : (y_pred < gate.edge);" \
        "the short leg's COMPARISON is unverified."

    # (7) THE NET. `long + short` instead of `long - short` is the reading under
    # which two disagreeing regimes reinforce instead of cancelling. It agrees
    # with the reference on every bar where only one leg votes, which on this
    # arm is most of them — so it needs a control, not a reading.
    run_mutant "the vote: net = n_long + n_short instead of n_long - n_short" \
        's|    out.net_count = out.n_long - out.n_short;|    out.net_count = out.n_long + out.n_short;  /* MUTANT */|' \
        "out.net_count = out.n_long + out.n_short;  /* MUTANT */" \
        "out.net_count = out.n_long - out.n_short;" \
        "the vote NETS rather than sums, and nothing proves it."

    # (8) THE NON-FINITE GUARD. `inf > edge` is true, so without it a poisoned
    # feature column votes. The reference cannot even be asked (sklearn refuses
    # inf and trading.py:744 returns an empty frame), so the two disagree on
    # exactly those rows.
    run_mutant "legFires: allow a non-finite y_pred to vote" \
        's|    if (!std::isfinite(y_pred)) {\n        return false;\n    }||' \
        "" "" "" SKIP_IF_NO_INF

    # (9) THE REPRESENTATIVE REGIME. Picking over ALL voters instead of the
    # majority leg can name a SHORT regime as the winner of a LONG decision,
    # which makes the [AGDEC] log line say something untrue while every count
    # stays right.
    #
    # GRADED BY --selftest, NOT BY THE PARITY RUN, and the reason is measured:
    # across the 6990 real decisions in the suite only TWO rows have both legs
    # voting at once, and on neither does the minority leg hold the larger
    # |y_pred| — so this mutation is INVISIBLE to the parity comparison. The
    # driver's --selftest builds ballots that separate the two rules, and this
    # control requires the mutant to fail THAT. A control graded on a run that
    # cannot distinguish the two would read as evidence while proving nothing.
    run_mutant_selftest "representative regime: pick over ALL voters, not the majority leg" \
        's|        if (have_majority \&\& positions\[i\] != majority) continue;|        /* MUTANT: majority leg ignored */|' \
        "/* MUTANT: majority leg ignored */" \
        "if (have_majority && positions[i] != majority) continue;" \
        "the representative regime may come from the leg the decision went AGAINST."

    echo
    echo "=== NEGATIVE CONTROLS OK: all caught ==="
}

# control <label> <harness flags...>
control() {
    local label="$1"; shift
    echo
    echo "=== NEGATIVE CONTROL: $label ==="
    if harness --driver "$HERE/build/decision_parity_driver" "$@" --expect-red \
            >/dev/null 2>&1; then
        echo "--- caught (gate went red) ---"
    else
        echo "=== NEGATIVE CONTROL FAILED: '$label' PASSED the gate ===" >&2
        echo "    The gate is not watching that property." >&2
        exit 1
    fi
}

# Builds the mutant exactly as run_mutant does, then requires it to FAIL the
# driver's own --selftest. For properties the real-panel suite provably cannot
# exercise; see control (9) for the measurement that justifies it.
run_mutant_selftest() {
    local label="$1" prog="$2" want="$3" gone="$4" why="$5"
    echo
    echo "=== NEGATIVE CONTROL (graded by --selftest): $label ==="
    local tmp; tmp="$(mktemp -d)"
    sed "$prog" "$HERE/src/decision_rule.cpp" > "$tmp/decision_rule_mutant.cpp"
    if ! grep -qF "$want" "$tmp/decision_rule_mutant.cpp"; then
        echo "FAIL: the mutation did not apply — the target line moved." >&2
        rm -rf "$tmp"; exit 1
    fi
    if [ -n "$gone" ] && grep -qF "$gone" "$tmp/decision_rule_mutant.cpp"; then
        echo "FAIL: the ORIGINAL line survived the mutation." >&2
        rm -rf "$tmp"; exit 1
    fi
    build_host_driver "$HERE/build/decision_parity_driver_mutant" \
        "$tmp/decision_rule_mutant.cpp" 2>/dev/null
    rm -rf "$tmp"
    if "$HERE/build/decision_parity_driver_mutant" --selftest >/dev/null 2>&1; then
        echo "=== NEGATIVE CONTROL FAILED: the mutant PASSED --selftest ===" >&2
        echo "    $why" >&2
        exit 1
    fi
    echo "--- caught (--selftest went red) ---"
}

# run_mutant <label> <sed-program> <must-appear> <must-vanish|""> <why>
#
# Guarded twice (the new text must appear, the old must be gone) so a drifted
# sed cannot report a green control over an unmutated binary.
run_mutant() {
    local label="$1" prog="$2" want="$3" gone="$4" why="$5" skip="${6:-}"
    if [ "$skip" = "SKIP_IF_NO_INF" ]; then
        # This one needs a scenario whose SELECTED model columns actually carry
        # an inf. Left out of the automatic run rather than reported as a green
        # control over a mutation nothing could exercise — a control that cannot
        # fail is worse than no control, because it reads as evidence.
        echo
        echo "=== NEGATIVE CONTROL SKIPPED: $label ==="
        echo "    The five scenarios do not put a +/-inf into a SELECTED model"
        echo "    column, so this mutation is unobservable here. The property is"
        echo "    covered instead by the driver's --selftest (inf/NaN never vote)."
        return 0
    fi
    echo
    echo "=== NEGATIVE CONTROL: $label ==="
    local tmp; tmp="$(mktemp -d)"
    sed "$prog" "$HERE/src/decision_rule.cpp" > "$tmp/decision_rule_mutant.cpp"
    if ! grep -qF "$want" "$tmp/decision_rule_mutant.cpp"; then
        echo "FAIL: the mutation did not apply to src/decision_rule.cpp — the" >&2
        echo "      target line moved. Refusing to report a control that" >&2
        echo "      mutated nothing." >&2
        rm -rf "$tmp"; exit 1
    fi
    if [ -n "$gone" ] && grep -qF "$gone" "$tmp/decision_rule_mutant.cpp"; then
        echo "FAIL: the ORIGINAL line is still present after the mutation —" >&2
        echo "      the sed matched somewhere else." >&2
        rm -rf "$tmp"; exit 1
    fi
    build_host_driver "$HERE/build/decision_parity_driver_mutant" \
        "$tmp/decision_rule_mutant.cpp" 2>/dev/null
    rm -rf "$tmp"

    if harness --driver "$HERE/build/decision_parity_driver_mutant" \
            >/dev/null 2>&1; then
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
