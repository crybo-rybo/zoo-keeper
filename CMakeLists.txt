cmake_minimum_required(VERSION 3.18)
project(zoo-keeper VERSION 0.2.0 LANGUAGES CXX)

include(GNUInstallDirs)

# C++23 required
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(ZOO_BUILD_TESTS "Build test suite" OFF)
option(ZOO_BUILD_INTEGRATION_TESTS "Build integration test suite" OFF)
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
set(ZOO_INTEGRATION_MODEL "" CACHE FILEPATH "Path to a GGUF model used by live integration smoke tests")

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

# Primary public library target
add_library(zoo STATIC
    src/agent.cpp
    src/core/model.cpp
)
target_include_directories(zoo PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<BUILD_INTERFACE:${json_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(zoo PUBLIC cxx_std_23)
target_link_libraries(zoo PRIVATE llama)
target_link_libraries(zoo INTERFACE
    $<BUILD_INTERFACE:llama>
    $<INSTALL_INTERFACE:llama>
)

if(ZOO_ENABLE_LOGGING)
    target_compile_definitions(zoo PRIVATE ZOO_LOGGING_ENABLED)
endif()

# Backward-compatible compatibility target.
add_library(zoo_core INTERFACE)
target_link_libraries(zoo_core INTERFACE zoo)

# Compiler warnings
zoo_set_warnings(zoo PRIVATE)

# Sanitizers
zoo_enable_sanitizers(zoo)

# Coverage
zoo_enable_coverage(zoo)

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
install(TARGETS zoo zoo_core
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/zoo DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "internal" EXCLUDE
    PATTERN "*.in" EXCLUDE
)
install(DIRECTORY ${json_SOURCE_DIR}/include/nlohmann
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zoo
)

# Config file
include(CMakePackageConfigHelpers)
set(ZOOKEEPER_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(ZOOKEEPER_LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
configure_package_config_file(
    cmake/ZooKeeperConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ZooKeeper
    PATH_VARS
        ZOOKEEPER_INCLUDE_INSTALL_DIR
        ZOOKEEPER_LIB_INSTALL_DIR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ZooKeeper
)


set(ZOO_PKGCONFIG_LIBS "-L\${libdir} -lzoo_core -llama -lggml -lggml-base -lggml-cpu")
if(TARGET ggml-blas)
    string(APPEND ZOO_PKGCONFIG_LIBS " -lggml-blas")
endif()
if(TARGET ggml-metal)
    string(APPEND ZOO_PKGCONFIG_LIBS " -lggml-metal")
endif()
if(UNIX)
    string(APPEND ZOO_PKGCONFIG_LIBS " -lm")
endif()
if(APPLE)
    if(TARGET ggml-blas)
        string(APPEND ZOO_PKGCONFIG_LIBS " -framework Accelerate")
    endif()
    if(TARGET ggml-metal)
        string(APPEND ZOO_PKGCONFIG_LIBS " -framework Foundation -framework Metal -framework MetalKit")
    endif()
endif()

configure_file(
    cmake/zoo-keeper.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/zoo-keeper.pc
    @ONLY
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/zoo-keeper.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
