include_guard(GLOBAL)

function(zoo_set_warnings target visibility)
    target_compile_options(${target} ${visibility}
        "$<$<CXX_COMPILER_ID:MSVC>:/W4>"
        "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall>"
        "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wextra>"
        "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wpedantic>"
    )
endfunction()

function(zoo_enable_warnings_as_errors target)
    target_compile_options(${target} PRIVATE
        "$<$<AND:$<BOOL:${ZOO_WARNINGS_AS_ERRORS}>,$<CXX_COMPILER_ID:MSVC>>:/WX>"
        "$<$<AND:$<BOOL:${ZOO_WARNINGS_AS_ERRORS}>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>"
    )
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

function(zoo_apply_owned_target_options target)
    set_target_properties(${target} PROPERTIES
        CXX_EXTENSIONS OFF
    )
    zoo_set_warnings(${target} PRIVATE)
    zoo_enable_sanitizers(${target})
endfunction()

function(zoo_apply_strict_target_options target)
    zoo_apply_owned_target_options(${target})
    zoo_enable_warnings_as_errors(${target})
    zoo_enable_coverage(${target})
    zoo_mark_llama_includes_as_system(${target})
endfunction()
