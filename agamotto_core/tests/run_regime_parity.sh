#!/bin/bash
# The PHASE 3 gate: the REAL research_filters on one side, the C++ regime gate
# on the other, every regime x every row compared for EXACT boolean equality.
#
#   ./tests/run_regime_parity.sh              # macOS/clang -O2, host build
#   ./tests/run_regime_parity.sh --linux      # rocky8/gcc 8.5, in the build image
#   ./tests/run_regime_parity.sh --both
#   ./tests/run_regime_parity.sh --negative   # NEGATIVE CONTROLS, exit INVERTED
#
# TWO TOOLCHAINS, for the same reason tests/run_feature_parity.sh runs both:
# stage 2.1 was GREEN on gcc 8.5/x86-64 and RED on clang/arm64 from identical
# source, because FMA contraction flipped a NaN mask. This gate is MADE of
# comparisons against columns that carry NaN, so it inherits that exposure
# directly — and a mask flip is not a last-bit difference, it is a bar that
# trades or does not.
#
# The reference is research_filters, which needs pandas and the vendored codec,
# so on --linux the DRIVER runs in docker while the harness stays on the host —
# the same split run_feature_parity.sh uses. Host ta-lib is required for the
# same reason too (the gate reads adx/cci/macdhist/bb_*/sar/mom/mfi/bop/roc/
# rsi/stoch_*, all of which come out of the TA-Lib block).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="mjolnir-core-build:latest"
PY="${AGAMOTTO_PARITY_PYTHON:-python3}"
TALIB_PREFIX="${AGAMOTTO_TALIB_PREFIX:-$HOME/.local/talib-0.6.4}"
STACK="${AGAMOTTO_REGIME_STACK:-$HERE/tests/regime_stack_deployed.csv}"

MODE="${1:-host}"
case "$MODE" in
    ""|host)    MODE=host ;;
    --linux)    MODE=linux ;;
    --both)     MODE=both ;;
    --negative) MODE=negative ;;
    *) echo "usage: $0 [--linux|--both|--negative]" >&2; exit 2 ;;
esac

if ! "$PY" -c 'import pandas' >/dev/null 2>&1; then
    echo "FAIL: $PY has no pandas. research_filters IS the reference — refusing" >&2
    echo "      to run a parity gate without it. Set AGAMOTTO_PARITY_PYTHON=..." >&2
    exit 1
fi
if [ ! -f "$STACK" ]; then
    echo "FAIL: no regime stack at $STACK. The gate grades against the DEPLOYED" >&2
    echo "      stack; a hand-typed substitute would grade a stack that is not" >&2
    echo "      running. Set AGAMOTTO_REGIME_STACK=..." >&2
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
    echo "[run_regime_parity] host ta-lib $got at $TALIB_PREFIX"
}

# $1 = output binary, $2.. = translation units (the gate source may be a mutant)
build_host_driver() {
    local out="$1"; shift
    mkdir -p "$HERE/build"
    "${CXX:-c++}" -std=c++17 -O2 -ffp-contract=off -Wall -Wextra \
        -I"$HERE/src" -I"$TALIB_PREFIX/include" -o "$out" \
        "$HERE/tests/regime_parity_driver.cpp" "$@" \
        "$HERE/src/feature_engine.cpp" "$HERE/src/talib_block.cpp" \
        "$TALIB_PREFIX/lib/libta-lib.a" -lm
}

run_host() {
    echo "=== building the gate driver for the HOST toolchain ==="
    check_host_talib
    build_host_driver "$HERE/build/regime_parity_driver" "$HERE/src/regime_gate.cpp"
    "${CXX:-c++}" --version | head -1
    # The atomIsKnown/atomMask consistency sweep, BEFORE the parity run. A core
    # that accepts a stack it cannot evaluate would boot clean and only throw
    # 7.3 days later, when the first panel is warm.
    echo "=== gate self-test (atomIsKnown vs atomMask over all 4096 codes) ==="
    "$HERE/build/regime_parity_driver" --selftest
    echo "=== host regime parity ==="
    "$PY" "$HERE/tests/regime_parity.py" \
        --driver "$HERE/build/regime_parity_driver" --stack "$STACK"
}

run_linux() {
    echo "=== building the gate driver in $IMAGE (rocky8 / gcc 8.5) ==="
    docker run --rm -v "$HERE:/src" -w /src "$IMAGE" bash -c "
        set -e
        mkdir -p build-linux
        g++ --version | head -1
        g++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Isrc \
            -o build-linux/regime_parity_driver \
            tests/regime_parity_driver.cpp src/regime_gate.cpp \
            src/feature_engine.cpp src/talib_block.cpp \
            /usr/local/lib/libta-lib.a -lm
        ./build-linux/regime_parity_driver --selftest
    "
    echo "=== linux regime parity (driver in docker, pandas on the host) ==="
    "$PY" "$HERE/tests/regime_parity.py" --stack "$STACK" --driver \
        "docker run --rm -i -v $HERE:/src -w /src $IMAGE ./build-linux/regime_parity_driver"
}

# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS.
#
# Each mutation is a PLAUSIBLE wrong predicate — the thing a careful person
# would actually have written from the regime's NAME rather than from
# research_filters' body — applied by rewriting ONE line. The run INVERTS the
# exit code: a mutant that PASSES means the gate is not watching that predicate.
#
# Every mutation is guarded twice (the new text must appear, the old must be
# gone), so a drifted sed cannot report a green control over an unmutated
# binary.
#
# ON `>` vs `>=`, WHICH IS THE OBVIOUS MUTATION AND IS NOT ALWAYS OBSERVABLE:
# a boundary mutation only changes a mask on rows where the two operands are
# EXACTLY equal. On `adx > 25` against a continuous double that never happens,
# so `>=` there is genuinely indistinguishable on any data — a real limit of
# this gate, recorded rather than papered over. Where exact ties DO occur the
# mutation is caught, and the controls below are chosen to sit there: the
# scenarios carry a 26-bar FLAT run and a 25-bar ZERO-VOLUME run, on which
# `mom` is exactly 0, `close == mvg1 == mvg2 == mvg3`, and stoch_k == stoch_d.
#
# run_mutant <label> <sed-program> <must-appear> <must-vanish> <why>
run_mutant() {
    local label="$1" prog="$2" want="$3" gone="$4" why="$5"
    echo
    echo "=== NEGATIVE CONTROL: $label ==="
    local tmp; tmp="$(mktemp -d)"
    sed "$prog" "$HERE/src/regime_gate.cpp" > "$tmp/regime_gate_mutant.cpp"
    if ! grep -qF "$want" "$tmp/regime_gate_mutant.cpp"; then
        echo "FAIL: the mutation did not apply to src/regime_gate.cpp — the" >&2
        echo "      target line moved. Refusing to report a control that" >&2
        echo "      mutated nothing." >&2
        rm -rf "$tmp"; exit 1
    fi
    if grep -qF "$gone" "$tmp/regime_gate_mutant.cpp"; then
        echo "FAIL: the ORIGINAL line is still present after the mutation —" >&2
        echo "      the sed matched somewhere else." >&2
        rm -rf "$tmp"; exit 1
    fi
    build_host_driver "$HERE/build/regime_parity_driver_mutant" \
        "$tmp/regime_gate_mutant.cpp"
    rm -rf "$tmp"

    if "$PY" "$HERE/tests/regime_parity.py" --stack "$STACK" \
            --driver "$HERE/build/regime_parity_driver_mutant" >/dev/null 2>&1; then
        echo "=== NEGATIVE CONTROL FAILED: the mutant PASSED the gate ===" >&2
        echo "    $why" >&2
        exit 1
    fi
    echo "--- caught (gate went red) ---"
}

run_negative() {
    check_host_talib

    # (1) THE BOUNDARY MUTATION, placed where ties actually happen. `mom` is
    # `close - close[10]`, which is EXACTLY 0.0 across the 26-bar flat run in
    # scenario 5, so `> 0` and `>= 0` genuinely differ there. This is the
    # `>` -> `>=` control; on adx it would be unobservable (see the banner).
    run_mutant "mom_positive long: mom > 0 -> mom >= 0 (the boundary)" \
        's|return lng ? cmp1(panel.get(codes::F_MOM), \[\](double v) { return v > 0.0; })|return lng ? cmp1(panel.get(codes::F_MOM), [](double v) { return v >= 0.0; })  /* MUTANT */|' \
        "return v >= 0.0; })  /* MUTANT */" \
        "return lng ? cmp1(panel.get(codes::F_MOM), [](double v) { return v > 0.0; })" \
        "a >= boundary on mom_positive is not distinguished from > — the predicate is unverified."

    # (2) THE NAME-DRIVEN MUTATION. research_filters' SHORT branch for
    # `mom_positive` is `mom < 0`; writing `mom > 0` on both sides is what the
    # NAME says and what anyone porting from the regime list would produce.
    run_mutant "mom_positive short: mom < 0 -> mom > 0 (believing the name)" \
        's|: cmp1(panel.get(codes::F_MOM), \[\](double v) { return v < 0.0; });|: cmp1(panel.get(codes::F_MOM), [](double v) { return v > 0.0; });  /* MUTANT */|' \
        "return v > 0.0; });  /* MUTANT */" \
        ": cmp1(panel.get(codes::F_MOM), [](double v) { return v < 0.0; });" \
        "the short branch's inverted predicate is unverified — the name would have been believed."

    # (3) THE PINNED PRODUCTION PROPERTY. Reading q50 where the reference reads
    # q95 is the single most plausible "fix" for 53 regimes that never fire:
    # q50 is populated on every row, so the regimes would start firing and
    # everything would look healthier. The gate must go red for exactly that.
    run_mutant "high_vol_q95: read the q50 cutoff instead (the tempting 'fix')" \
        's|panel.get(codes::F_PRICE_RANGE_PCT_Q95),|panel.get(codes::F_PRICE_RANGE_PCT_Q50),  /* MUTANT */|' \
        "F_PRICE_RANGE_PCT_Q50),  /* MUTANT */" \
        "panel.get(codes::F_PRICE_RANGE_PCT_Q95)," \
        "a firing q95 gate is not distinguished from an inert one — PR #532's pinned property is unverified."

    # (4) THE COLUMN-ORDER MUTATION. research_filters' `_volume_ratio` prefers
    # quote_vol_ratio and falls back to vol_ratio. Both exist in a live panel
    # and they are DIFFERENT numbers, so the wrong order silently redefines
    # which bars are high-volume — and r029/r039/r069 appear in 45 of the 62
    # deployed regimes.
    run_mutant "volume ratio: prefer vol_ratio over quote_vol_ratio (wrong order)" \
        's|    if (t.has(codes::F_QUOTE_VOL_RATIO)) return t.get(codes::F_QUOTE_VOL_RATIO);|    if (t.has(codes::F_VOL_RATIO)) return t.get(codes::F_VOL_RATIO);  /* MUTANT */|' \
        "if (t.has(codes::F_VOL_RATIO)) return t.get(codes::F_VOL_RATIO);  /* MUTANT */" \
        "if (t.has(codes::F_QUOTE_VOL_RATIO)) return t.get(codes::F_QUOTE_VOL_RATIO);" \
        "reading the wrong volume-ratio column is not distinguished — 45 of 62 regimes are unverified."

    # (5) THE SIGNED-vs-ABSOLUTE MUTATION. `near_ma` reads as a proximity BAND
    # and is written as a `<` on a SIGNED distance, so on the long side every
    # bar BELOW mvg1 satisfies it. |x| < 0.02 is what the name suggests and is
    # a strictly TIGHTER gate, i.e. it removes trades rather than adding them —
    # the direction of error nobody notices.
    run_mutant "near_ma long: signed distance -> |distance| (believing the name)" \
        's|return lng ? cmp2(c, m1, \[\](double a, double b) { return (a - b) / b < 0.02; })|return lng ? cmp2(c, m1, [](double a, double b) { return std::fabs((a - b) / b) < 0.02; })  /* MUTANT */|' \
        "std::fabs((a - b) / b) < 0.02; })  /* MUTANT */" \
        "return lng ? cmp2(c, m1, [](double a, double b) { return (a - b) / b < 0.02; })" \
        "an absolute-value near_ma is not distinguished from the reference's signed one."

    echo
    echo "=== NEGATIVE CONTROLS OK: 5 mutants, all caught (gate went red) ==="
}

case "$MODE" in
    host)     run_host ;;
    linux)    run_linux ;;
    both)     run_host; echo; run_linux ;;
    negative) run_negative ;;
esac
