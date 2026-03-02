include(FetchContent)
set(FETCHCONTENT_QUIET OFF CACHE BOOL "" FORCE)

# tl::expected (C++23 std::expected backport)
FetchContent_Declare(
    tl_expected
    GIT_REPOSITORY https://github.com/TartanLlama/expected.git
    GIT_TAG v1.3.1
    GIT_SHALLOW TRUE
)
set(EXPECTED_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tl_expected)

# nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(json)

# llama.cpp (from submodule)
set(GGML_METAL ${ZOO_ENABLE_METAL})
set(LLAMA_CUDA ${ZOO_ENABLE_CUDA})
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/extern/llama.cpp/CMakeLists.txt)
    add_subdirectory(extern/llama.cpp)
else()
    message(WARNING "llama.cpp submodule not found. Run: git submodule update --init --recursive")
endif()

# SQLite3 - use FetchContent on Windows, find_package elsewhere
if(WIN32)
    FetchContent_Declare(
        sqlite3
        URL https://www.sqlite.org/2024/sqlite-amalgamation-3450300.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(sqlite3)
    add_library(sqlite3_lib STATIC ${sqlite3_SOURCE_DIR}/sqlite3.c)
    target_include_directories(sqlite3_lib PUBLIC ${sqlite3_SOURCE_DIR})
    add_library(SQLite::SQLite3 ALIAS sqlite3_lib)
else()
    find_package(SQLite3 REQUIRED)
endif()

# GoogleTest (only if building tests)
if(ZOO_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    # For Windows: Prevent overriding parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()
