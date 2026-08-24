cmake_minimum_required(VERSION 3.16)
project(agamotto_core LANGUAGES CXX)


# ROUTE A runs the panel on a worker thread, so every target that links the
# core needs pthread. gcc 8 does not pull it in implicitly: without this the
# link fails on pthread_create, and only for the targets that instantiate
# std::thread -- so it would surface as one driver failing to link rather than
# as an obvious missing dependency.
find_package(Threads REQUIRED)

# Built as a shared lib and linked by the PUBLIC sentinel strategy via
#   -DAGAMOTTO_CORE_DIR=<this>/build
# The public repo never sees these sources. Mirrors dc/sentinel_core.
#
# Phase 2 stage 2.0/2.1 adds the feature-engine scaffolding and the numeric
# primitives, and wires ta-lib the way sentinel_core pins it. LightGBM still
# arrives with the model.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT SENTINEL_REPO)
    set(SENTINEL_REPO "$ENV{HOME}/Documents/sandbox/sentinel")
endif()
set(CONTRACT_DIR "${SENTINEL_REPO}/Strategy/ltp_release/ltp_strat_sdk/stan_code")
if(NOT EXISTS "${CONTRACT_DIR}/agamotto_core.hpp")
    # Fail loud: a stale or missing contract silently builds a core the
    # strategy cannot link against.
    message(FATAL_ERROR
        "contract header not found at ${CONTRACT_DIR}/agamotto_core.hpp\n"
        "  set -DSENTINEL_REPO=/path/to/sentinel")
endif()

if(NOT AGAMOTTO_CORE_GITSHA)
    execute_process(
        COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short HEAD
        OUTPUT_VARIABLE AGAMOTTO_CORE_GITSHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()
if(NOT AGAMOTTO_CORE_GITSHA)
    message(FATAL_ERROR
        "AGAMOTTO_CORE_GITSHA is empty and git is unavailable here.\n"
        "  Pass -DAGAMOTTO_CORE_GITSHA=<sha>. Refusing to build an untraceable core.")
endif()

# ---- TA-Lib ----------------------------------------------------------------
# Same pattern, and the same reasoning, as ../sentinel_core/CMakeLists.txt: the
# indicator block must be built against the SAME version production runs, or the
# indicators still compute — they just compute slightly different numbers than
# the model was trained on, which is a SILENT parity break.
#
# Prefer the STATIC archive: the shipped .so must not depend on /usr/local/lib
# at runtime. -lm is required, libta-lib references asin/exp/log/sqrt and does
# not pull it in itself.
#
# src/talib_block.cpp (stage 2.3) is the ONE translation unit that calls it —
# 25 indicator calls transcribed from research.py:501-553. `engineerFeatures`
# calls into it, so EVERY target that links src/feature_engine.cpp must also
# link src/talib_block.cpp and ta-lib; the driver targets below do.
set(TALIB_PINNED_VERSION "0.6.4")
find_library(TALIB_LIB NAMES libta-lib.a ta-lib ta_lib HINTS /usr/local/lib /opt/homebrew/lib)
find_path(TALIB_INC ta-lib/ta_libc.h HINTS /usr/local/include /opt/homebrew/include)
if(NOT TALIB_LIB OR NOT TALIB_INC)
    message(FATAL_ERROR
        "ta-lib not found. Build with ./build_linux.sh inside "
        "mjolnir-core-build:latest, which pins ${TALIB_PINNED_VERSION} "
        "(../sentinel_core/Dockerfile.build ARG TALIB_VERSION).")
endif()
# A REAL pin, not just a sentence in an error message: read the version the
# installed library declares and refuse a mismatch outright.
find_file(TALIB_PC ta-lib.pc HINTS /usr/local/lib/pkgconfig /opt/homebrew/lib/pkgconfig)
if(NOT TALIB_PC)
    message(FATAL_ERROR
        "ta-lib found at ${TALIB_LIB} but ta-lib.pc is missing, so its version "
        "cannot be verified. Refusing to build against an unidentified ta-lib.")
endif()
file(STRINGS "${TALIB_PC}" _talib_version_line REGEX "^Version:")
string(REGEX REPLACE "^Version:[ \t]*" "" TALIB_VERSION "${_talib_version_line}")
if(NOT TALIB_VERSION VERSION_EQUAL TALIB_PINNED_VERSION)
    message(FATAL_ERROR
        "ta-lib ${TALIB_VERSION} found at ${TALIB_LIB}, but this core is pinned "
        "to ${TALIB_PINNED_VERSION}. A different version changes indicator "
        "internals and breaks parity without failing anything.")
endif()
message(STATUS "agamotto_core: ta-lib ${TALIB_VERSION} ${TALIB_LIB}")

# ---------------------------------------------------------------------------
# FP contraction: OFF for every target that compiles src/feature_engine.cpp.
#
# pdops reproduces pandas' arithmetic operation for operation. A fused
# multiply-add keeps a product at infinite precision and rounds once, which is
# a DIFFERENT function — and at three sites in that file the result of a
# cancellation decides a NaN mask, not just a value. Apple clang on arm64
# defaults to -ffp-contract=on with FMA in the base ISA, so the parity gate
# was RED on macOS and GREEN on rockylinux/gcc 8.5 (baseline x86-64, no FMA
# instruction to contract into) from identical source and identical goldens.
#
# This flag is layer 3 of 3 and is a CONVENIENCE, not the guarantee. The
# correctness guarantee is source-level: pdRound() in feature_engine.cpp puts
# a volatile round-trip at every mask-deciding site, which no flag can undo.
# Dropping this flag costs at most a last-bit value difference (absorbed by
# the 1e-12 gate); it cannot resurrect the NaN-mask defect.
#
# Applied to the library AND the test driver so the gate exercises exactly
# what ships. -ffast-math/-Ofast is rejected outright by a #error in the
# source: it declares NaN and inf nonexistent, which no barrier can survive.
set(AGAMOTTO_FP_FLAGS -ffp-contract=off)

add_library(agamotto_core SHARED
    src/core_impl.cpp
    src/kline_builder.cpp
    src/feature_engine.cpp
    src/talib_block.cpp
    src/regime_gate.cpp
    src/model_runner.cpp
    src/decision_rule.cpp
)
target_include_directories(agamotto_core PRIVATE ${CONTRACT_DIR} src ${TALIB_INC})
target_compile_definitions(agamotto_core PRIVATE
    AGAMOTTO_CORE_GITSHA="${AGAMOTTO_CORE_GITSHA}")
target_link_libraries(agamotto_core PRIVATE ${TALIB_LIB} m Threads::Threads)
target_compile_options(agamotto_core PRIVATE -Wall -Wextra ${AGAMOTTO_FP_FLAGS})

message(STATUS "agamotto_core: contract=${CONTRACT_DIR} sha=${AGAMOTTO_CORE_GITSHA}")

# Parity/unit driver. Standalone: no SDK, no feed — it replays synthetic and
# recorded ticks through the SAME builder the live core uses.
add_executable(kline_parity_driver tests/kline_parity_driver.cpp src/kline_builder.cpp)
target_include_directories(kline_parity_driver PRIVATE ${CONTRACT_DIR} src)
target_compile_options(kline_parity_driver PRIVATE -Wall -Wextra)

# Numeric-primitive parity driver. Consumes the two CSVs tests/pdops_golden.py
# emits from pandas 2.3.3 and diffs every cell of every primitive.
#
# It exercises pdops ONLY and calls no indicator, but it links
# src/feature_engine.cpp, whose `engineerFeatures` calls talib_block::compute —
# so talib_block.cpp and ta-lib come along or the link fails. Linking them is
# the honest fix; stubbing the call out for this target would mean the pdops
# gate no longer builds the same code the library ships.
add_executable(pdops_parity_driver tests/pdops_parity_driver.cpp
                                   src/feature_engine.cpp src/talib_block.cpp)
target_include_directories(pdops_parity_driver PRIVATE src ${TALIB_INC})
target_link_libraries(pdops_parity_driver PRIVATE ${TALIB_LIB} m Threads::Threads)
target_compile_options(pdops_parity_driver PRIVATE -Wall -Wextra ${AGAMOTTO_FP_FLAGS})

# Stage 2.2 + 2.3 feature-column parity. Reads one symbol's raw OHLCV panel as
# CSV on stdin and prints the engineered panel; tests/feature_parity.py drives
# the REAL research.py alongside it and diffs every cell — NaN/inf
# classification and first-valid-index first, values after.
add_executable(feature_parity_driver tests/feature_parity_driver.cpp
                                     src/feature_engine.cpp src/talib_block.cpp)
target_include_directories(feature_parity_driver PRIVATE src ${TALIB_INC})
target_link_libraries(feature_parity_driver PRIVATE ${TALIB_LIB} m Threads::Threads)
target_compile_options(feature_parity_driver PRIVATE -Wall -Wextra ${AGAMOTTO_FP_FLAGS})

# Stage 2.6 INTEGRATION gate: ticks -> KlineBuilder -> RealCore ->
# engineerFeatures, in process. Links src/core_impl.cpp — the ONLY target that
# does other than the shared library itself — because the property under test is
# the WIRING, and a driver that called engineerFeatures directly would pass
# whether or not RealCore ever did.
#
# AGAMOTTO_CORE_GITSHA is required by core_impl.cpp's #error guard, and the same
# FP flags apply: this target compiles feature_engine.cpp and must be the code
# that ships.
add_executable(core_integration_driver tests/core_integration_driver.cpp
                                       src/core_impl.cpp
                                       src/kline_builder.cpp
                                       src/feature_engine.cpp
                                       src/talib_block.cpp
                                       src/regime_gate.cpp
                                       src/model_runner.cpp
                                       src/decision_rule.cpp)
target_include_directories(core_integration_driver PRIVATE ${CONTRACT_DIR} src ${TALIB_INC})
target_compile_definitions(core_integration_driver PRIVATE
    AGAMOTTO_CORE_GITSHA="${AGAMOTTO_CORE_GITSHA}")
target_link_libraries(core_integration_driver PRIVATE ${TALIB_LIB} m Threads::Threads)
target_compile_options(core_integration_driver PRIVATE -Wall -Wextra ${AGAMOTTO_FP_FLAGS})

# ORDER-PATH ARITHMETIC (sentinel side). Links NOTHING from src/ and no
# ta-lib: the unit under test is order_path_math.hpp, which includes only
# <cmath>/<cstdint> precisely so it can be tested without the vendor SDK.
# AgamottoStrategy itself is unlinkable in a driver (AlgoBase has out-of-line
# ctor/send_order and ltp_strat_sdk/lib/ is gitignored), so arithmetic left as
# a member function of the strategy would have NO test at all.
add_executable(order_path_driver tests/order_path_driver.cpp)
target_include_directories(order_path_driver PRIVATE ${CONTRACT_DIR})
target_compile_options(order_path_driver PRIVATE -Wall -Wextra ${AGAMOTTO_FP_FLAGS})

# PHASE 3 regime-gate parity. Reads the raw OHLCV panel on stdin, prints the
# ENGINEERED PANEL *and* the per-regime masks it computed from it;
# tests/regime_parity.py hands that same panel to the REAL research_filters and
# diffs the masks at ZERO tolerance.
#
# It links src/feature_engine.cpp (and therefore ta-lib) because the gate reads
# columns the TA-Lib block produces — adx, cci, macdhist, bb_upper/lower, sar,
# mom, mfi, bop, roc, rsi, stoch_k/d. A driver that stubbed those out would
# grade the predicates against numbers no live panel contains.
add_executable(regime_parity_driver tests/regime_parity_driver.cpp
                                    src/regime_gate.cpp
                                    src/feature_engine.cpp
                                    src/talib_block.cpp)
target_include_directories(regime_parity_driver PRIVATE src ${TALIB_INC})
target_link_libraries(regime_parity_driver PRIVATE ${TALIB_LIB} m Threads::Threads)
target_compile_options(regime_parity_driver PRIVATE -Wall -Wextra ${AGAMOTTO_FP_FLAGS})

# PHASE 4 model parity. Reads the raw OHLCV panel on stdin, prints the
# ENGINEERED PANEL *and* the per-regime y_pred it computed from it for EVERY row
# (not only the newest); tests/model_parity.py hands that same panel to the
# DEPLOYED sklearn loader (utils.weights_io.load_regime(...).predict) and diffs.
#
# EVERY ROW, deliberately. The core scores only the panel's last row, but a
# one-row-per-scenario comparison is five numbers per regime — far too few to
# distinguish a correct dot product from one that is right at the mean and wrong
# in the tails. The kernel under test is row-independent, so grading it on all
# 699 rows costs nothing and multiplies the evidence by 699.
#
# Links src/model_runner.cpp and src/feature_engine.cpp (hence ta-lib): the
# models read columns the TA-Lib block produces, and a driver that engineered a
# different panel than the one it scored would grade the dot product against
# numbers no live panel contains.
add_executable(model_parity_driver tests/model_parity_driver.cpp
                                   src/model_runner.cpp
                                   src/regime_gate.cpp
                                   src/feature_engine.cpp
                                   src/talib_block.cpp)
target_include_directories(model_parity_driver PRIVATE src ${TALIB_INC})
target_link_libraries(model_parity_driver PRIVATE ${TALIB_LIB} m Threads::Threads)
target_compile_options(model_parity_driver PRIVATE -Wall -Wextra ${AGAMOTTO_FP_FLAGS})

# PHASE 5 decision parity. Reads the raw OHLCV panel on stdin, prints the
# ENGINEERED PANEL, the per-regime y_pred, *and* the DECISION it took on every
# row; tests/decision_parity.py hands that same panel to the deployed sklearn
# loader, applies the REAL `gauntlet.thresholds.signed_threshold` and the REAL
# `agamotto.trading.dual_gate_filter`, and diffs side/fired EXACTLY.
#
# It links src/decision_rule.cpp (the rule under test) plus the model, gate and
# feature engine, because a decision is only meaningful on the numbers the same
# panel produced — a driver that decided on synthetic y_pred would grade the
# comparison operators and nothing else.
add_executable(decision_parity_driver tests/decision_parity_driver.cpp
                                      src/decision_rule.cpp
                                      src/model_runner.cpp
                                      src/regime_gate.cpp
                                      src/feature_engine.cpp
                                      src/talib_block.cpp)
target_include_directories(decision_parity_driver PRIVATE src ${TALIB_INC})
target_link_libraries(decision_parity_driver PRIVATE ${TALIB_LIB} m Threads::Threads)
target_compile_options(decision_parity_driver PRIVATE -Wall -Wextra ${AGAMOTTO_FP_FLAGS})

# The ta-lib LINK PROOF. find_library above only proves the file exists; this
# target is the one that actually resolves TA_* symbols, so a broken or
# ABI-mismatched archive fails the build here rather than at Phase 2.2.
add_executable(talib_bench tests/talib_bench.cpp)
target_include_directories(talib_bench PRIVATE ${TALIB_INC})
target_link_libraries(talib_bench PRIVATE ${TALIB_LIB} m Threads::Threads)
target_compile_options(talib_bench PRIVATE -Wall -Wextra)
