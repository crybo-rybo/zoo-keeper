cmake_minimum_required(VERSION 3.18)

include(GoogleTest)

add_executable(zoo_tests
    unit/test_types.cpp
    unit/test_tool_registry.cpp
    unit/test_tool_parser.cpp
    unit/test_schema_grammar.cpp
    unit/test_error_recovery.cpp
    unit/test_tool_interceptor.cpp
    unit/test_batch.cpp
    unit/test_prompt_bookkeeping.cpp
    unit/test_agent_mailbox.cpp
    unit/test_agent_runtime.cpp
    unit/test_extraction.cpp
    unit/test_request_tracker.cpp
    unit/test_model_tool_calling.cpp
    unit/test_token_accounting.cpp
    unit/test_streaming_filter.cpp
    unit/test_callback_dispatcher.cpp
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

zoo_mark_llama_includes_as_system(zoo_tests)

gtest_discover_tests(zoo_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES
        LABELS "unit"
)

zoo_set_warnings(zoo_tests PRIVATE)
zoo_enable_warnings_as_errors(zoo_tests)
zoo_enable_coverage(zoo_tests)
zoo_enable_sanitizers(zoo_tests)

if(ZOO_BUILD_INTEGRATION_TESTS)
    add_subdirectory(integration)
endif()
