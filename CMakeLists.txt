cmake_minimum_required(VERSION 3.18)
project(zoo-keeper VERSION 1.1.4 LANGUAGES CXX)

include(GNUInstallDirs)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/ZooKeeperOptions.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/CompilerWarnings.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/Sanitizers.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/Coverage.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/ZooKeeperDependencies.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/ZooKeeperTargets.cmake")

if(ZOO_BUILD_TESTS OR ZOO_BUILD_INTEGRATION_TESTS)
    enable_testing()
endif()

add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(benchmarks)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/Doxygen.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/Crap.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/ZooKeeperInstall.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/ZooKeeperPackage.cmake")
