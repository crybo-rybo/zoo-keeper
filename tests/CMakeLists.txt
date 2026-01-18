cmake_minimum_required(VERSION 3.18)

# GoogleTest is fetched by parent CMakeLists.txt
include(GoogleTest)

# Test executable
add_executable(zoo_tests
    unit/test_types.cpp
    unit/test_request_queue.cpp
    unit/test_history_manager.cpp
    unit/test_template_engine.cpp
    unit/test_agent.cpp
    mocks/mock_backend.cpp
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
        ${CMAKE_CURRENT_SOURCE_DIR}/mocks
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
)

# Discover tests
gtest_discover_tests(zoo_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES
        LABELS "unit"
)

# Add compile options
if(MSVC)
    target_compile_options(zoo_tests PRIVATE /W4)
else()
    target_compile_options(zoo_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Coverage support
if(ZOO_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(zoo_tests PRIVATE --coverage)
        target_link_options(zoo_tests PRIVATE --coverage)
    endif()
endif()

# Sanitizers
if(ZOO_ENABLE_SANITIZERS)
    if(NOT MSVC)
        set(SANITIZER_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer")
        target_compile_options(zoo_tests PRIVATE ${SANITIZER_FLAGS})
        target_link_options(zoo_tests PRIVATE ${SANITIZER_FLAGS})
    endif()
endif()
