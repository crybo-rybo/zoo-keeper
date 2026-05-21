include_guard(GLOBAL)

include(FetchContent)
include("${CMAKE_CURRENT_LIST_DIR}/ZooKeeperLlama.cmake")
set(FETCHCONTENT_QUIET OFF CACHE BOOL "" FORCE)

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
    if(ZOO_LLAMA_TAG MATCHES "^b([0-9]+)$")
        set(LLAMA_BUILD_NUMBER "${CMAKE_MATCH_1}" CACHE STRING "" FORCE)
    else()
        set(LLAMA_BUILD_NUMBER 0 CACHE STRING "" FORCE)
    endif()
    set(LLAMA_BUILD_COMMIT "${ZOO_LLAMA_TAG}" CACHE STRING "" FORCE)
endfunction()

set(ZOO_LLAMA_SOURCE_DIR "")

if(TARGET llama OR TARGET llama-common)
    if(TARGET llama AND TARGET llama-common)
        message(STATUS "Zoo-Keeper: using llama.cpp from parent project targets")
    else()
        message(FATAL_ERROR
            "Zoo-Keeper requires both `llama` and `llama-common` targets when a parent "
            "project provides llama.cpp. Provide both targets or neither.")
    endif()
else()
    zoo_configure_llama_build_options()
    set(ZOO_LLAMA_ARCHIVE_URL "${ZOO_LLAMA_ARCHIVE_BASE_URL}/${ZOO_LLAMA_TAG}.tar.gz")
    FetchContent_Declare(
        llama_cpp
        URL "${ZOO_LLAMA_ARCHIVE_URL}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(llama_cpp)
    zoo_apply_llama_common_workarounds()
    set(ZOO_LLAMA_SOURCE_DIR "${llama_cpp_SOURCE_DIR}")
    message(STATUS "Zoo-Keeper: using llama.cpp from FetchContent archive (${ZOO_LLAMA_ARCHIVE_URL})")
endif()

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
