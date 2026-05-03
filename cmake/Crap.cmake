include_guard(GLOBAL)

# Defines the 'crap' build target, which:
#   1. Clears stale .gcda files
#   2. Runs the test suite to generate fresh coverage data
#   3. Calls scripts/crap_report.py to compute CRAP scores via lizard + gcovr
#
# Prerequisites (enforced by ZooKeeperOptions.cmake):
#   ZOO_BUILD_TESTS=ON, ZOO_ENABLE_COVERAGE=ON, ZOO_ENABLE_CRAP=ON

if(NOT ZOO_ENABLE_CRAP)
    return()
endif()

if(NOT ZOO_BUILD_TESTS)
    message(FATAL_ERROR "ZOO_ENABLE_CRAP requires ZOO_BUILD_TESTS=ON")
endif()

if(NOT ZOO_ENABLE_COVERAGE)
    message(FATAL_ERROR "ZOO_ENABLE_CRAP requires ZOO_ENABLE_COVERAGE=ON")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_target(crap
    COMMAND ${CMAKE_COMMAND}
        "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
        -P "${CMAKE_SOURCE_DIR}/cmake/zoo_clear_gcda.cmake"
    COMMAND ${CMAKE_CTEST_COMMAND}
        --test-dir "${CMAKE_BINARY_DIR}"
        --output-on-failure
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/scripts/crap_report.py"
        "--build-dir"   "${CMAKE_BINARY_DIR}"
        "--source-dir"  "${CMAKE_SOURCE_DIR}"
        "--threshold"   "${ZOO_CRAP_THRESHOLD}"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Computing CRAP scores (complexity × coverage)..."
    VERBATIM
)

if(TARGET zoo_tests)
    add_dependencies(crap zoo_tests)
endif()
