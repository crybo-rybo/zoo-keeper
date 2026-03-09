cmake_minimum_required(VERSION 3.18)

add_executable(zoo_integration_tests
    test_model_agent.cpp
)

target_link_libraries(zoo_integration_tests
    PRIVATE
        zoo
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
)

target_include_directories(zoo_integration_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/tests/fixtures
)

target_compile_definitions(zoo_integration_tests PRIVATE
    ZOO_PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(ZOO_INTEGRATION_MODEL)
    target_compile_definitions(zoo_integration_tests PRIVATE
        ZOO_INTEGRATION_MODEL_PATH="${ZOO_INTEGRATION_MODEL}"
    )
endif()

gtest_discover_tests(zoo_integration_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES
        LABELS "integration"
)

zoo_set_warnings(zoo_integration_tests PRIVATE)
zoo_enable_coverage(zoo_integration_tests)
zoo_enable_sanitizers(zoo_integration_tests)
