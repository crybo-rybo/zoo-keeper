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

if(ZOO_ENABLE_CRAP)
    set(ZOO_BUILD_TESTS ON CACHE BOOL "Build test suite (implied by ZOO_ENABLE_CRAP)" FORCE)
    set(ZOO_ENABLE_COVERAGE ON CACHE BOOL "Coverage instrumentation (implied by ZOO_ENABLE_CRAP)" FORCE)
endif()
set(ZOO_LLAMA_DEFAULT_TAG "b9296")
set(ZOO_LLAMA_DEFAULT_SHA256 "cb8ee6b9326a0263daaa9c61cedb42e575d37d7cc4a3b9a64915b4056c1a15fe")
set(ZOO_LLAMA_DEFAULT_ARCHIVE_BASE_URL "https://github.com/ggerganov/llama.cpp/archive/refs/tags")
set(ZOO_LLAMA_TAG "${ZOO_LLAMA_DEFAULT_TAG}" CACHE STRING
    "llama.cpp release tag used by FetchContent")
set(ZOO_LLAMA_SHA256 "${ZOO_LLAMA_DEFAULT_SHA256}" CACHE STRING
    "Expected SHA-256 digest for the llama.cpp release archive")
set(ZOO_LLAMA_ARCHIVE_BASE_URL "${ZOO_LLAMA_DEFAULT_ARCHIVE_BASE_URL}" CACHE STRING
    "Base URL for llama.cpp release archives used by FetchContent")
set(ZOO_INTEGRATION_MODEL "" CACHE FILEPATH "Path to a GGUF model used by live integration smoke tests")
