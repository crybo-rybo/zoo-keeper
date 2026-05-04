cmake_minimum_required(VERSION 3.18)
project(zoo-keeper VERSION 1.1.4 LANGUAGES CXX)

include(GNUInstallDirs)
include(cmake/ZooKeeperOptions.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)
include(cmake/Coverage.cmake)
include(cmake/ZooKeeperDependencies.cmake)
include(cmake/ZooKeeperTargets.cmake)

if(ZOO_BUILD_TESTS OR ZOO_BUILD_INTEGRATION_TESTS)
    enable_testing()
endif()

add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(benchmarks)

if(ZOO_BUILD_DOCS)
    include(cmake/Doxygen.cmake)
endif()

if(ZOO_ENABLE_CRAP)
    include(cmake/Crap.cmake)
endif()

include(cmake/ZooKeeperInstall.cmake)
include(cmake/ZooKeeperPackage.cmake)
