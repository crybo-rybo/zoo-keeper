cmake_minimum_required(VERSION 3.18)
project(zoo-keeper VERSION 1.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

# C++23 required
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(ZOO_BUILD_TESTS "Build test suite" OFF)
option(ZOO_BUILD_INTEGRATION_TESTS "Build integration test suite" OFF)
option(ZOO_BUILD_EXAMPLES "Build examples" OFF)
option(ZOO_BUILD_BENCHMARKS "Build local performance harnesses" OFF)
option(ZOO_BUILD_DOCS "Build API documentation with Doxygen" OFF)
option(ZOO_ENABLE_COVERAGE "Enable coverage instrumentation" OFF)
option(ZOO_ENABLE_SANITIZERS "Enable ASan/TSan/UBSan" OFF)
option(ZOO_WARNINGS_AS_ERRORS "Treat warnings in zoo-owned targets as errors" OFF)
if(APPLE)
    option(ZOO_ENABLE_METAL "Enable Metal acceleration (macOS)" ON)
else()
    option(ZOO_ENABLE_METAL "Enable Metal acceleration (macOS)" OFF)
endif()
option(ZOO_ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(ZOO_FETCH_LLAMA "Fetch llama.cpp when vendored sources are unavailable" OFF)
option(ZOO_BUILD_HUB "Build the hub layer (GGUF inspection, HuggingFace, model store)" OFF)
option(ZOO_ENABLE_LOGGING "Enable debug logging to stderr" OFF)
set(ZOO_LLAMA_REPOSITORY "https://github.com/ggerganov/llama.cpp.git" CACHE STRING
    "llama.cpp repository used when ZOO_FETCH_LLAMA=ON")
set(ZOO_LLAMA_TAG "d1b4757dedbb60a811c8d7012249a96b1b702606" CACHE STRING
    "llama.cpp git tag or commit used when ZOO_FETCH_LLAMA=ON")
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

set(ZOO_BUILD_LLAMA_LINK_LIBRARIES
    "$<TARGET_FILE:common>"
    "$<TARGET_FILE:llama>"
    "$<TARGET_FILE:ggml>"
    "$<TARGET_FILE:ggml-base>"
    "$<TARGET_FILE:ggml-cpu>"
)
if(TARGET ggml-blas)
    list(APPEND ZOO_BUILD_LLAMA_LINK_LIBRARIES "$<TARGET_FILE:ggml-blas>")
endif()
if(TARGET ggml-metal)
    list(APPEND ZOO_BUILD_LLAMA_LINK_LIBRARIES "$<TARGET_FILE:ggml-metal>")
endif()
if(UNIX)
    list(APPEND ZOO_BUILD_LLAMA_LINK_LIBRARIES "m")
endif()
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    list(APPEND ZOO_BUILD_LLAMA_LINK_LIBRARIES "${CMAKE_DL_LIBS}")
endif()
if(APPLE)
    if(TARGET ggml-blas)
        list(APPEND ZOO_BUILD_LLAMA_LINK_LIBRARIES "-framework Accelerate")
    endif()
    if(TARGET ggml-metal)
        list(APPEND ZOO_BUILD_LLAMA_LINK_LIBRARIES
            "-framework Foundation"
            "-framework Metal"
            "-framework MetalKit"
        )
    endif()
endif()

# Primary public library target
add_library(zoo STATIC
    src/agent/facade.cpp
    src/agent/backend_model.cpp
    src/agent/runtime.cpp
    src/agent/runtime_tool_loop.cpp
    src/agent/runtime_extraction.cpp
    src/core/model.cpp
    src/core/model_init.cpp
    src/core/model_inference.cpp
    src/core/model_prompt.cpp
    src/core/model_history.cpp
    src/core/model_sampling.cpp
    src/core/model_tool_calling.cpp
    src/core/stream_filter.cpp
    src/log_callback.cpp
)
target_include_directories(zoo PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<BUILD_INTERFACE:${json_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(zoo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(zoo PUBLIC cxx_std_23)
# nlohmann/json is part of the installed public header surface via
# `include/zoo/core/json.hpp`. Keep the build-tree include path explicit for
# subdirectory/FetchContent consumers, and publish the install-tree dependency
# through the target interface so installed-package consumers inherit it.
target_link_libraries(zoo PRIVATE
    $<BUILD_INTERFACE:llama>
    $<BUILD_INTERFACE:common>
    $<INSTALL_INTERFACE:ZooKeeper::llama>
)
set_property(TARGET zoo APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES "$<INSTALL_INTERFACE:ZooKeeper::nlohmann_json>"
)

if(ZOO_BUILD_HUB)
    target_sources(zoo PRIVATE
        src/hub/inspector.cpp
        src/hub/huggingface.cpp
        src/hub/store.cpp
    )
    target_compile_definitions(zoo PUBLIC ZOO_HUB_ENABLED)
endif()

if(ZOO_ENABLE_LOGGING)
    target_compile_definitions(zoo PRIVATE ZOO_LOGGING_ENABLED)
endif()

# Backward-compatible compatibility target.
add_library(zoo_core INTERFACE)
target_link_libraries(zoo_core INTERFACE zoo)

# Namespaced aliases so FetchContent / add_subdirectory consumers can use the
# same target names as installed-package consumers (ZooKeeper::zoo).
add_library(ZooKeeper::zoo ALIAS zoo)
add_library(ZooKeeper::zoo_core ALIAS zoo_core)

# Compiler warnings
zoo_set_warnings(zoo PRIVATE)
zoo_enable_warnings_as_errors(zoo)
zoo_mark_llama_includes_as_system(zoo)

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

# Benchmarks
if(ZOO_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Documentation
if(ZOO_BUILD_DOCS)
    include(cmake/Doxygen.cmake)
endif()

# Install rules
install(TARGETS zoo zoo_core
    EXPORT ZooKeeperTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# llama.cpp's common library is not exported by upstream, but zoo.a depends
# on it at link time.  Install it alongside our own archives so the install-
# tree config can reference it.
install(FILES $<TARGET_FILE:common>
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT ZooKeeperTargets
    FILE ZooKeeperTargets.cmake
    NAMESPACE ZooKeeper::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ZooKeeper
)

install(DIRECTORY include/zoo DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "internal" EXCLUDE
    PATTERN "*.in" EXCLUDE
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zoo
)

include(CMakePackageConfigHelpers)
set(ZOO_BUILD_ZOO_LOCATION "$<TARGET_FILE:zoo>")
set(ZOO_BUILD_INTERFACE_INCLUDE_DIRECTORIES "$<TARGET_PROPERTY:zoo,INTERFACE_INCLUDE_DIRECTORIES>")
set(ZOO_BUILD_GGML_OPENMP_ENABLED "${GGML_OPENMP_ENABLED}")
set(ZOO_BUILD_GGML_BLAS_ENABLED "${GGML_BLAS}")
set(ZOO_BUILD_BLAS_LIBRARIES "${BLAS_LIBRARIES}")
set(ZOO_BUILD_BLAS_LINKER_FLAGS "${BLAS_LINKER_FLAGS}")
configure_package_config_file(
    cmake/ZooKeeperBuildTreeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperBuildTreeConfig.cmake.in
    INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperConfig.cmake"
    INPUT "${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperBuildTreeConfig.cmake.in"
)

# Install-tree config — uses find_dependency to locate llama and nlohmann_json.
configure_package_config_file(
    cmake/ZooKeeperConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/ZooKeeperConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ZooKeeper
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/ZooKeeperConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ZooKeeperConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ZooKeeper
)


set(ZOO_PKGCONFIG_LIBS "-L\${libdir} -lzoo -lcommon -llama -lggml -lggml-base -lggml-cpu")
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
