include_guard(GLOBAL)

# All glue that ties zoo to llama.cpp's `common` static archive lives here.
# `common` is a load-bearing dependency (Jinja chat templates, tool-call parsing
# for 29+ formats) but upstream does not expose it through the `llama` CMake
# package's export set. Until that changes, zoo links it as a build-tree target
# and installs the archive next to its own libs. See ZooKeeperConfig.cmake.in
# for the matching consumer-side wiring that pulls libcommon.a back in via
# `find_package(ZooKeeper)`.

# Apply build-tree linkage to a zoo internal target. INSTALL_INTERFACE points at
# `ZooKeeper::llama`, which the installed config file recreates with libcommon.a
# folded in.
function(zoo_target_link_llama target)
    target_link_libraries(${target} PRIVATE
        $<BUILD_INTERFACE:llama>
        $<BUILD_INTERFACE:common>
        $<INSTALL_INTERFACE:ZooKeeper::llama>
    )
endfunction()

# Install llama.cpp's `common` static archive alongside zoo's own libraries.
function(zoo_install_llama_common)
    install(FILES $<TARGET_FILE:common>
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endfunction()

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
