include(FetchContent)
set(FETCHCONTENT_QUIET OFF CACHE BOOL "" FORCE)

# nlohmann/json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

function(zoo_configure_llama_build_options)
    set(GGML_METAL ${ZOO_ENABLE_METAL} CACHE BOOL "" FORCE)
    set(LLAMA_CUDA ${ZOO_ENABLE_CUDA} CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
    set(LLAMA_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
    set(GGML_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endfunction()

function(zoo_add_llama_subdirectory source_dir binary_dir provider)
    zoo_configure_llama_build_options()
    add_subdirectory("${source_dir}" "${binary_dir}")
    set(ZOO_LLAMA_SOURCE_DIR "${source_dir}" PARENT_SCOPE)
    set(ZOO_LLAMA_PROVIDER "${provider}" PARENT_SCOPE)
endfunction()

set(ZOO_LLAMA_SOURCE_DIR "")
set(ZOO_LLAMA_PROVIDER "")

if(TARGET llama OR TARGET common)
    if(TARGET llama AND TARGET common)
        set(ZOO_LLAMA_PROVIDER "parent project targets")
        message(STATUS "Zoo-Keeper: using llama.cpp from ${ZOO_LLAMA_PROVIDER}")
    else()
        message(FATAL_ERROR
            "Zoo-Keeper requires both `llama` and `common` targets when a parent "
            "project provides llama.cpp. Provide both targets or neither.")
    endif()
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/llama.cpp/CMakeLists.txt")
    zoo_add_llama_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/extern/llama.cpp"
        "${CMAKE_CURRENT_BINARY_DIR}/extern/llama.cpp"
        "vendored submodule"
    )
    message(STATUS "Zoo-Keeper: using llama.cpp from ${ZOO_LLAMA_PROVIDER}")
elseif(ZOO_FETCH_LLAMA)
    zoo_configure_llama_build_options()
    FetchContent_Declare(
        llama_cpp
        GIT_REPOSITORY "${ZOO_LLAMA_REPOSITORY}"
        GIT_TAG "${ZOO_LLAMA_TAG}"
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(llama_cpp)
    set(ZOO_LLAMA_SOURCE_DIR "${llama_cpp_SOURCE_DIR}")
    set(ZOO_LLAMA_PROVIDER "FetchContent (${ZOO_LLAMA_REPOSITORY} @ ${ZOO_LLAMA_TAG})")
    message(STATUS "Zoo-Keeper: using llama.cpp from ${ZOO_LLAMA_PROVIDER}")
else()
    message(FATAL_ERROR
        "llama.cpp sources are unavailable. Initialize the vendored submodule "
        "with `git submodule update --init --recursive`, provide both `llama` "
        "and `common` targets from the parent project, or enable "
        "`-DZOO_FETCH_LLAMA=ON`.")
endif()

# GoogleTest (needed by unit tests, integration tests, or both)
if(ZOO_BUILD_TESTS OR ZOO_BUILD_INTEGRATION_TESTS)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()
