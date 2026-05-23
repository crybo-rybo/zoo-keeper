cmake_minimum_required(VERSION 3.18)
project(zoo_instrumentation_probe LANGUAGES CXX)

include("${ZOO_PROJECT_SOURCE_DIR}/cmake/Sanitizers.cmake")
include("${ZOO_PROJECT_SOURCE_DIR}/cmake/Coverage.cmake")

set(ZOO_ENABLE_SANITIZERS ON)
set(ZOO_ENABLE_COVERAGE ON)

add_library(zoo_owned STATIC dummy.cpp)
zoo_enable_sanitizers(zoo_owned)
zoo_enable_coverage(zoo_owned)

get_target_property(interface_link_options zoo_owned INTERFACE_LINK_OPTIONS)
if(interface_link_options MATCHES "sanitize|coverage")
    message(FATAL_ERROR
        "Instrumentation flags leaked into INTERFACE_LINK_OPTIONS: "
        "${interface_link_options}")
endif()

get_target_property(private_link_options zoo_owned LINK_OPTIONS)
if(NOT private_link_options MATCHES "sanitize")
    message(FATAL_ERROR "Sanitizer link option was not applied privately")
endif()
if(NOT private_link_options MATCHES "coverage")
    message(FATAL_ERROR "Coverage link option was not applied privately")
endif()
