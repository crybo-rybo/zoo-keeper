cmake_minimum_required(VERSION 3.18)
project(zoo-keeper VERSION 0.2.0 LANGUAGES CXX)

# C++23 required
set(CMAKE_CXX_STANDARD 23)
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

# Include dependencies
include(cmake/FetchDependencies.cmake)

# Include platform-specific utility modules
include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/Coverage.cmake)

# Header-only library target (core types + tools)
add_library(zoo INTERFACE)
target_include_directories(zoo INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(zoo INTERFACE cxx_std_23)
target_link_libraries(zoo INTERFACE nlohmann_json::nlohmann_json)

# Model library (no llama.cpp dependency - can be tested with mocks)
add_library(zoo_model STATIC src/core/model.cpp)
target_link_libraries(zoo_model PUBLIC zoo)

# Backend library (llama.cpp integration)
add_library(zoo_backend STATIC src/core/llama_backend.cpp)
target_link_libraries(zoo_backend PUBLIC zoo_model llama)
target_include_directories(zoo_backend PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Compiler warnings
zoo_set_warnings(zoo INTERFACE)
zoo_set_warnings(zoo_model PRIVATE)
zoo_set_warnings(zoo_backend PRIVATE)

# Sanitizers
zoo_enable_sanitizers(zoo_model)
zoo_enable_sanitizers(zoo_backend)

# Coverage
zoo_enable_coverage(zoo_model)
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
