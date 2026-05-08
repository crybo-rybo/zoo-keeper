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
        "-DBUILD_DIR=${PROJECT_BINARY_DIR}"
        -P "${CMAKE_CURRENT_LIST_DIR}/zoo_clear_gcda.cmake"
    COMMAND ${CMAKE_CTEST_COMMAND}
        --test-dir "${PROJECT_BINARY_DIR}"
        --output-on-failure
    COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/scripts/crap_report.py"
        "--build-dir"   "${PROJECT_BINARY_DIR}"
        "--source-dir"  "${PROJECT_SOURCE_DIR}"
        "--threshold"   "${ZOO_CRAP_THRESHOLD}"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    COMMENT "Computing CRAP scores (complexity × coverage)..."
    VERBATIM
)

foreach(test_target IN ITEMS zoo_tests zoo_integration_tests)
    if(TARGET ${test_target})
        add_dependencies(crap ${test_target})
    endif()
endforeach()
