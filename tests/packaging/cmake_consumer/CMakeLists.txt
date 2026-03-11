cmake_minimum_required(VERSION 3.18)
project(zoo_cmake_consumer_smoke LANGUAGES CXX)

find_package(ZooKeeper CONFIG REQUIRED)

add_executable(zoo_cmake_consumer_smoke main.cpp)
target_link_libraries(zoo_cmake_consumer_smoke PRIVATE ZooKeeper::zoo)
