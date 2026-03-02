cmake_minimum_required(VERSION 3.18)
project(zoo-keeper VERSION 0.1.0 LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(ZOO_BUILD_TESTS "Build test suite" OFF)
option(ZOO_BUILD_EXAMPLES "Build examples" OFF)
option(ZOO_ENABLE_COVERAGE "Enable coverage instrumentation" OFF)
option(ZOO_ENABLE_SANITIZERS "Enable ASan/TSan/UBSan" OFF)
if(APPLE)
    option(ZOO_ENABLE_METAL "Enable Metal acceleration (macOS)" ON)
else()
    option(ZOO_ENABLE_METAL "Enable Metal acceleration (macOS)" OFF)
endif()
option(ZOO_ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(ZOO_ENABLE_MCP "Enable MCP (Model Context Protocol) client support" OFF)

# Include dependencies (SQLite3 is handled inside FetchDependencies.cmake)
include(cmake/FetchDependencies.cmake)

# Include platform-specific utility modules
include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/Coverage.cmake)

# Header-only library target
add_library(zoo INTERFACE)
target_include_directories(zoo INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(zoo INTERFACE cxx_std_17)
target_link_libraries(zoo INTERFACE nlohmann_json::nlohmann_json tl::expected SQLite::SQLite3)

# MCP support compile definition
if(ZOO_ENABLE_MCP)
    target_compile_definitions(zoo INTERFACE ZOO_ENABLE_MCP)
    # Add subprocess.h include path from llama.cpp vendor
    target_include_directories(zoo INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/extern/llama.cpp/vendor/sheredom>
    )
endif()

# Llama Backend (requires linking)
add_library(zoo_backend STATIC src/backend/llama_backend.cpp)
target_link_libraries(zoo_backend PUBLIC zoo llama)
target_include_directories(zoo_backend PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Compiler warnings
zoo_set_warnings(zoo INTERFACE)
zoo_set_warnings(zoo_backend PRIVATE)

# Sanitizers (ASan + UBSan; TSan is mutually exclusive with ASan)
zoo_enable_sanitizers(zoo_backend)

# Coverage
zoo_enable_coverage(zoo_backend)

# Tests
if(ZOO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(ZOO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install rules
# Note: We only install zoo_backend as it's the main target users need
# Dependencies (llama, json, expected) are fetched/built separately
install(TARGETS zoo_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

# Config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/ZooKeeperConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperConfig.cmake
    INSTALL_DESTINATION lib/cmake/ZooKeeper
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperConfigVersion.cmake
    DESTINATION lib/cmake/ZooKeeper
)
