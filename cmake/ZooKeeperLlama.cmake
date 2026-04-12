include_guard(GLOBAL)

function(zoo_collect_llama_build_link_libraries output_var)
    set(link_libraries
        "$<TARGET_FILE:common>"
        "$<TARGET_FILE:llama>"
        "$<TARGET_FILE:ggml>"
        "$<TARGET_FILE:ggml-base>"
        "$<TARGET_FILE:ggml-cpu>"
    )
    if(TARGET ggml-blas)
        list(APPEND link_libraries "$<TARGET_FILE:ggml-blas>")
    endif()
    if(TARGET ggml-metal)
        list(APPEND link_libraries "$<TARGET_FILE:ggml-metal>")
    endif()
    if(UNIX)
        list(APPEND link_libraries "m")
    endif()
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        list(APPEND link_libraries "${CMAKE_DL_LIBS}")
    endif()
    if(APPLE)
        if(TARGET ggml-blas)
            list(APPEND link_libraries "-framework Accelerate")
        endif()
        if(TARGET ggml-metal)
            list(APPEND link_libraries
                "-framework Foundation"
                "-framework Metal"
                "-framework MetalKit"
            )
        endif()
    endif()

    set(${output_var} "${link_libraries}" PARENT_SCOPE)
endfunction()

function(zoo_collect_llama_pkgconfig_libs output_var)
    set(libs "-L\${libdir} -lzoo -lcommon -llama -lggml -lggml-base -lggml-cpu")
    if(TARGET ggml-blas)
        string(APPEND libs " -lggml-blas")
    endif()
    if(TARGET ggml-metal)
        string(APPEND libs " -lggml-metal")
    endif()
    if(UNIX)
        string(APPEND libs " -lm")
    endif()
    if(APPLE)
        if(TARGET ggml-blas)
            string(APPEND libs " -framework Accelerate")
        endif()
        if(TARGET ggml-metal)
            string(APPEND libs " -framework Foundation -framework Metal -framework MetalKit")
        endif()
    endif()

    set(${output_var} "${libs}" PARENT_SCOPE)
endfunction()
