# cmake/CompilerWarnings.cmake
# Provides helpers for owned-target warning policy and third-party include handling.

function(zoo_set_warnings target visibility)
    if(MSVC)
        target_compile_options(${target} ${visibility} /W4)
    else()
        target_compile_options(${target} ${visibility} -Wall -Wextra -Wpedantic)
    endif()
endfunction()

function(zoo_enable_warnings_as_errors target)
    if(ZOO_WARNINGS_AS_ERRORS)
        if(MSVC)
            target_compile_options(${target} PRIVATE /WX)
        else()
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

function(zoo_mark_llama_includes_as_system target)
    if(NOT DEFINED ZOO_LLAMA_SOURCE_DIR OR ZOO_LLAMA_SOURCE_DIR STREQUAL "")
        return()
    endif()

    target_include_directories(${target} SYSTEM PRIVATE
        ${ZOO_LLAMA_SOURCE_DIR}/common
        ${ZOO_LLAMA_SOURCE_DIR}/vendor
        ${ZOO_LLAMA_SOURCE_DIR}/include
        ${ZOO_LLAMA_SOURCE_DIR}/ggml/include
    )
endfunction()
