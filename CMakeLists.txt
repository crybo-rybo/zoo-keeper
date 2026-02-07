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

# Include dependencies
include(cmake/FetchDependencies.cmake)

# Header-only library target
add_library(zoo INTERFACE)
target_include_directories(zoo INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(zoo INTERFACE cxx_std_17)
target_link_libraries(zoo INTERFACE nlohmann_json::nlohmann_json tl::expected)

# Llama Backend (requires linking)
add_library(zoo_backend STATIC src/backend/llama_backend.cpp)
target_link_libraries(zoo_backend PUBLIC zoo llama)
target_include_directories(zoo_backend PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Compiler warnings
if(MSVC)
    target_compile_options(zoo INTERFACE /W4)
    target_compile_options(zoo_backend PRIVATE /W4)
else()
    target_compile_options(zoo INTERFACE -Wall -Wextra -Wpedantic)
    target_compile_options(zoo_backend PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Sanitizers
if(ZOO_ENABLE_SANITIZERS)
    if(NOT MSVC)
        set(SANITIZER_FLAGS "-fsanitize=address,undefined,thread -fno-omit-frame-pointer")
        target_compile_options(zoo_backend PRIVATE ${SANITIZER_FLAGS})
        target_link_options(zoo_backend PRIVATE ${SANITIZER_FLAGS})
    endif()
endif()

# Coverage
if(ZOO_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(zoo_backend PRIVATE --coverage)
        target_link_options(zoo_backend PRIVATE --coverage)
    endif()
endif()

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
