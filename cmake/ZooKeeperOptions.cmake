include_guard(GLOBAL)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}" ZOO_PROJECT_IS_TOP_LEVEL)

option(ZOO_BUILD_TESTS "Build test suite" OFF)
option(ZOO_BUILD_INTEGRATION_TESTS "Build integration test suite" OFF)
option(ZOO_BUILD_EXAMPLES "Build examples" OFF)
option(ZOO_BUILD_BENCHMARKS "Build local performance harnesses" OFF)
option(ZOO_BUILD_DOCS "Build API documentation with Doxygen" OFF)
option(ZOO_ENABLE_COVERAGE "Enable coverage instrumentation" OFF)
option(ZOO_ENABLE_SANITIZERS "Enable ASan/UBSan" OFF)
option(ZOO_WARNINGS_AS_ERRORS "Treat warnings in zoo-owned targets as errors" OFF)
option(ZOO_ENABLE_INSTALL "Generate install and package metadata" ${ZOO_PROJECT_IS_TOP_LEVEL})
option(ZOO_ENABLE_METAL "Enable Metal acceleration (macOS)" ${APPLE})
option(ZOO_ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(ZOO_BUILD_HUB "Build the hub layer (GGUF inspection, HuggingFace, model store)" OFF)
option(ZOO_ENABLE_LOGGING "Enable debug logging to stderr" OFF)
option(ZOO_ENABLE_CRAP "Compute CRAP scores (complexity × coverage) via lizard + gcovr" OFF)
set(ZOO_CRAP_THRESHOLD "30" CACHE STRING
    "CRAP score threshold — functions above this value cause a non-zero exit (default: 30)")

if(ZOO_ENABLE_CRAP AND NOT ZOO_PROJECT_IS_TOP_LEVEL)
    message(FATAL_ERROR
        "ZOO_ENABLE_CRAP is a project-internal metric and must only be set when zoo-keeper "
        "is the top-level project. Disable it when consuming zoo-keeper from a parent build.")
endif()

if(ZOO_ENABLE_CRAP)
    set(ZOO_BUILD_TESTS ON CACHE BOOL "Build test suite (implied by ZOO_ENABLE_CRAP)" FORCE)
    set(ZOO_ENABLE_COVERAGE ON CACHE BOOL "Coverage instrumentation (implied by ZOO_ENABLE_CRAP)" FORCE)
endif()

if(ZOO_ENABLE_COVERAGE AND ZOO_ENABLE_SANITIZERS)
    message(FATAL_ERROR
        "ZOO_ENABLE_COVERAGE and ZOO_ENABLE_SANITIZERS are mutually exclusive: the combination "
        "produces gcov runtime errors. Enable one at a time.")
endif()
# Known-good SHA256 for the pinned default ZOO_LLAMA_TAG. When the user does
# not override either ZOO_LLAMA_TAG or ZOO_LLAMA_ARCHIVE_SHA256, this hash is
# used automatically so the build is fail-closed against supply-chain
# tampering. Bumping ZOO_LLAMA_DEFAULT_TAG REQUIRES updating
# ZOO_LLAMA_DEFAULT_SHA256 too. See cmake/ZooKeeperDependencies.cmake for the
# resolution logic.
set(ZOO_LLAMA_DEFAULT_TAG "b8992")
set(ZOO_LLAMA_DEFAULT_SHA256
    "942c56b7e7389edfd19150f886794e3f54fe7d51001ebb072c805c5d05016a48")

set(ZOO_LLAMA_TAG "${ZOO_LLAMA_DEFAULT_TAG}" CACHE STRING
    "llama.cpp release tag used by FetchContent")
set(ZOO_LLAMA_ARCHIVE_BASE_URL "https://github.com/ggml-org/llama.cpp/archive/refs/tags" CACHE STRING
    "Base URL for llama.cpp release archives used by FetchContent")
set(ZOO_LLAMA_ARCHIVE_SHA256 "" CACHE STRING
    "SHA256 of the llama.cpp source archive. Leave empty to inherit the \
baked-in hash for the default ZOO_LLAMA_TAG; required when overriding the tag.")

set(ZOO_INTEGRATION_MODEL "" CACHE FILEPATH "Path to a GGUF model used by live integration smoke tests")
