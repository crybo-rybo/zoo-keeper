if(NOT ZOO_BUILD_TESTS AND NOT ZOO_BUILD_INTEGRATION_TESTS)
    return()
endif()

include(GoogleTest)

if(ZOO_BUILD_TESTS)
    set(ZOO_TEST_SOURCES
        unit/test_types.cpp
        unit/test_tool_registry.cpp
        unit/test_tool_parser.cpp
        unit/test_schema_grammar.cpp
        unit/test_error_recovery.cpp
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

    if(ZOO_BUILD_HUB)
        list(APPEND ZOO_TEST_SOURCES unit/test_hub.cpp)
    endif()

    add_executable(zoo_tests ${ZOO_TEST_SOURCES})

    target_link_libraries(zoo_tests
        PRIVATE
            zoo
            $<BUILD_INTERFACE:llama-common>
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
    )

    target_include_directories(zoo_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
            ${PROJECT_SOURCE_DIR}/src
    )
    target_compile_definitions(zoo_tests
        PRIVATE
            ZOO_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    )

    gtest_discover_tests(zoo_tests
        PROPERTIES
            LABELS "unit"
    )

    zoo_apply_strict_target_options(zoo_tests)
endif()

if(ZOO_BUILD_INTEGRATION_TESTS)
    add_subdirectory(integration)
endif()
