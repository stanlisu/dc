cmake_minimum_required(VERSION 3.16)
project(mjolnir_core LANGUAGES CXX)

# Built as a shared lib and linked by the PUBLIC sentinel strategy via
#   -DMJOLNIR_CORE_DIR=<this>/build
# The public repo never sees these sources.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT SENTINEL_REPO)
    set(SENTINEL_REPO "$ENV{HOME}/Documents/sandbox/sentinel")
endif()
set(CONTRACT_DIR "${SENTINEL_REPO}/Strategy/ltp_release/ltp_strat_sdk/stan_code")
if(NOT EXISTS "${CONTRACT_DIR}/mjolnir_core.hpp")
    # Fail loud: a stale or missing contract silently builds a core that the
    # strategy cannot link against.
    message(FATAL_ERROR
        "contract header not found at ${CONTRACT_DIR}/mjolnir_core.hpp\n"
        "  set -DSENTINEL_REPO=/path/to/sentinel")
endif()

# Build tag so any run can be traced to the exact core that produced it.
execute_process(
    COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short HEAD
    OUTPUT_VARIABLE MJOLNIR_CORE_GITSHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT MJOLNIR_CORE_GITSHA)
    set(MJOLNIR_CORE_GITSHA "unknown")
endif()

add_library(mjolnir_core SHARED
    src/core_impl.cpp
    # M1 TODO — the IP modules, added as each lands:
    #   src/bar_builder.cpp
    #   src/feature_engine.cpp
    #   src/regime_gate.cpp
    #   src/model_runner.cpp
)

target_include_directories(mjolnir_core PRIVATE ${CONTRACT_DIR} src)
target_compile_definitions(mjolnir_core PRIVATE
    MJOLNIR_CORE_GITSHA="${MJOLNIR_CORE_GITSHA}")
target_compile_options(mjolnir_core PRIVATE -Wall -Wextra)

message(STATUS "mjolnir_core: contract=${CONTRACT_DIR} sha=${MJOLNIR_CORE_GITSHA}")
