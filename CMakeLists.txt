cmake_minimum_required(VERSION 3.18)
project(zoo-keeper VERSION 0.2.0 LANGUAGES CXX)

# C++23 required
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(ZOO_BUILD_TESTS "Build test suite" OFF)
option(ZOO_BUILD_EXAMPLES "Build examples" OFF)
option(ZOO_BUILD_DOCS "Build API documentation with Doxygen" OFF)
option(ZOO_ENABLE_COVERAGE "Enable coverage instrumentation" OFF)
option(ZOO_ENABLE_SANITIZERS "Enable ASan/TSan/UBSan" OFF)
if(APPLE)
    option(ZOO_ENABLE_METAL "Enable Metal acceleration (macOS)" ON)
else()
    option(ZOO_ENABLE_METAL "Enable Metal acceleration (macOS)" OFF)
endif()
option(ZOO_ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(ZOO_ENABLE_LOGGING "Enable debug logging to stderr" OFF)

# Include dependencies
include(cmake/FetchDependencies.cmake)

# Include platform-specific utility modules
include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/Coverage.cmake)

# Generate version header into the build tree.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/zoo/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp
    @ONLY
)

# Header-only library target (core types + tools)
add_library(zoo INTERFACE)
target_include_directories(zoo INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(zoo INTERFACE cxx_std_23)
target_link_libraries(zoo INTERFACE nlohmann_json::nlohmann_json)

if(ZOO_ENABLE_LOGGING)
    target_compile_definitions(zoo INTERFACE ZOO_LOGGING_ENABLED)
endif()

# Core library (Model + llama.cpp)
add_library(zoo_core STATIC src/core/model.cpp)
target_link_libraries(zoo_core PUBLIC zoo llama)

# Compiler warnings
zoo_set_warnings(zoo INTERFACE)
zoo_set_warnings(zoo_core PRIVATE)

# Sanitizers
zoo_enable_sanitizers(zoo_core)

# Coverage
zoo_enable_coverage(zoo_core)

# Tests
if(ZOO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(ZOO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Documentation
if(ZOO_BUILD_DOCS)
    include(cmake/Doxygen.cmake)
endif()

# Install rules
install(TARGETS zoo_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include
    PATTERN "*.in" EXCLUDE
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp
    DESTINATION include/zoo
)

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
