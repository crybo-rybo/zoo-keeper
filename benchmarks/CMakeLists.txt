if(NOT ZOO_BUILD_BENCHMARKS)
    return()
endif()

add_executable(zoo_benchmarks
    perf_harness.cpp
)

target_link_libraries(zoo_benchmarks
    PRIVATE
        zoo
)

zoo_apply_owned_target_options(zoo_benchmarks)
zoo_mark_llama_includes_as_system(zoo_benchmarks)

if(ZOO_ENABLE_INSTALL)
    install(TARGETS zoo_benchmarks
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
