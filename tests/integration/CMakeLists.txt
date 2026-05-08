add_executable(zoo_integration_tests
    test_model_agent.cpp
    test_extract.cpp
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
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/fixtures
)

target_compile_definitions(zoo_integration_tests PRIVATE
    ZOO_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

if(ZOO_INTEGRATION_MODEL)
    target_compile_definitions(zoo_integration_tests PRIVATE
        ZOO_INTEGRATION_MODEL_PATH="${ZOO_INTEGRATION_MODEL}"
    )
endif()

gtest_discover_tests(zoo_integration_tests
    PROPERTIES
        LABELS "integration"
)

zoo_apply_strict_target_options(zoo_integration_tests)
