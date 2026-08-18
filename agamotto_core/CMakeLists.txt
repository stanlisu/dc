cmake_minimum_required(VERSION 3.16)
project(agamotto_core LANGUAGES CXX)

# Built as a shared lib and linked by the PUBLIC sentinel strategy via
#   -DAGAMOTTO_CORE_DIR=<this>/build
# The public repo never sees these sources. Mirrors dc/sentinel_core.
#
# Phase 1 is the bar layer only, so there is deliberately no ta-lib and no
# LightGBM dependency yet — they arrive with the feature engine and the model
# in Phase 2, pinned the same way sentinel_core pins them.

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

add_library(agamotto_core SHARED
    src/core_impl.cpp
    src/kline_builder.cpp
)
target_include_directories(agamotto_core PRIVATE ${CONTRACT_DIR} src)
target_compile_definitions(agamotto_core PRIVATE
    AGAMOTTO_CORE_GITSHA="${AGAMOTTO_CORE_GITSHA}")
target_compile_options(agamotto_core PRIVATE -Wall -Wextra)

message(STATUS "agamotto_core: contract=${CONTRACT_DIR} sha=${AGAMOTTO_CORE_GITSHA}")

# Parity/unit driver. Standalone: no SDK, no feed — it replays synthetic and
# recorded ticks through the SAME builder the live core uses.
add_executable(kline_parity_driver tests/kline_parity_driver.cpp src/kline_builder.cpp)
target_include_directories(kline_parity_driver PRIVATE ${CONTRACT_DIR} src)
target_compile_options(kline_parity_driver PRIVATE -Wall -Wextra)
