cmake_minimum_required(VERSION 3.18)
project(zoo_fetchcontent_consumer_smoke LANGUAGES CXX)

include(FetchContent)

if(NOT DEFINED ZOO_KEEPER_SOURCE_DIR)
    message(FATAL_ERROR "ZOO_KEEPER_SOURCE_DIR must point to a zoo-keeper source snapshot")
endif()

set(ZOO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZOO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZOO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    zoo_keeper
    SOURCE_DIR "${ZOO_KEEPER_SOURCE_DIR}"
)
FetchContent_MakeAvailable(zoo_keeper)

add_executable(zoo_fetchcontent_consumer_smoke main.cpp)
target_link_libraries(zoo_fetchcontent_consumer_smoke PRIVATE ZooKeeper::zoo)
