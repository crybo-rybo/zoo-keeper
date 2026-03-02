# cmake/Coverage.cmake
# Provides zoo_enable_coverage(target) to apply coverage instrumentation flags.

function(zoo_enable_coverage target)
    if(ZOO_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE --coverage)
        target_link_options(${target} PRIVATE --coverage)
    endif()
endfunction()
