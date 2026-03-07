cmake_minimum_required(VERSION 3.18)

include(GoogleTest)

add_executable(zoo_tests
    unit/test_types.cpp
    unit/test_tool_registry.cpp
    unit/test_tool_parser.cpp
    unit/test_error_recovery.cpp
    unit/test_tool_interceptor.cpp
)

target_link_libraries(zoo_tests
    PRIVATE
        zoo
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
)

target_include_directories(zoo_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
)

gtest_discover_tests(zoo_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES
        LABELS "unit"
)

zoo_set_warnings(zoo_tests PRIVATE)
zoo_enable_coverage(zoo_tests)
zoo_enable_sanitizers(zoo_tests)
